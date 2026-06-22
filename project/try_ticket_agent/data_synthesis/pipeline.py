from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

from .api_client import ApiUsage, JsonResponse, add_usage
from .blueprints import execute_reference_actions
from .schemas import EpisodeBlueprint
from .validators import validate_candidate


@dataclass(frozen=True)
class PipelinePaths:
    dataset: Path
    rejected: Path
    progress: Path
    summary: Path

    @classmethod
    def in_directory(cls, directory: Path) -> "PipelinePaths":
        return cls(
            dataset=directory / "accepted.jsonl",
            rejected=directory / "rejected.jsonl",
            progress=directory / "progress.jsonl",
            summary=directory / "summary.json",
        )


@dataclass(frozen=True)
class ProcessResult:
    accepted: bool
    attempts: int
    api_calls: int
    usage: ApiUsage
    record: dict[str, Any] | None = None
    reasons: tuple[str, ...] = ()


class TicketSynthesisPipeline:
    def __init__(self, client: Any, *, max_attempts: int = 3, rewrite_temperature: float = 0.3):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.client = client
        self.max_attempts = max_attempts
        self.rewrite_temperature = rewrite_temperature

    def process_blueprint(self, blueprint: EpisodeBlueprint) -> ProcessResult:
        feedback: list[str] = []
        usage = ApiUsage()
        api_calls = 0
        source = blueprint.to_dict()
        for attempt in range(1, self.max_attempts + 1):
            rewrite_response: JsonResponse = self.client.rewrite(
                source, prior_failures=feedback, temperature=self.rewrite_temperature
            )
            api_calls += 1
            usage = add_usage(usage, rewrite_response.usage)
            payload = rewrite_response.payload
            if set(payload) != {"user_request"} or not isinstance(payload.get("user_request"), str):
                feedback = ["Rewrite JSON schema error: expected exactly {'user_request': str}"]
                continue
            user_request = payload["user_request"].strip()
            validation = validate_candidate(blueprint, user_request)
            if not validation.ok:
                feedback = list(validation.messages)
                continue

            judge_response: JsonResponse = self.client.judge(source, user_request)
            api_calls += 1
            usage = add_usage(usage, judge_response.usage)
            try:
                accepted, reasons = _parse_judge(judge_response.payload)
            except ValueError as exc:
                feedback = [f"Judge JSON schema error: {exc}"]
                continue
            if not accepted:
                feedback = reasons or ["Judge rejected the candidate without a reason"]
                continue

            verification = execute_reference_actions(replace(blueprint, user_request=user_request))
            if not verification.success:
                feedback = ["Reference verification failed: " + ", ".join(verification.failure_codes)]
                continue
            record = blueprint.to_dict()
            record["user_request"] = user_request
            record.pop("canonical_request", None)
            return ProcessResult(True, attempt, api_calls, usage, record)
        return ProcessResult(False, self.max_attempts, api_calls, usage, reasons=tuple(feedback))

    def run(
        self,
        blueprints: list[EpisodeBlueprint],
        *,
        paths: PipelinePaths,
        resume: bool = True,
        concurrency: int = 1,
    ) -> dict[str, Any]:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        for path in (paths.dataset, paths.rejected, paths.progress, paths.summary):
            path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            for path in (paths.progress, paths.rejected):
                if path.exists():
                    path.unlink()
        completed = _load_progress(paths.progress) if resume else {}
        indexed = [(index, item) for index, item in enumerate(blueprints) if item.episode_id not in completed]
        invocation_api_calls = 0

        def save(index: int, blueprint: EpisodeBlueprint, result: ProcessResult) -> None:
            nonlocal invocation_api_calls
            invocation_api_calls += result.api_calls
            progress = {
                "source_index": index,
                "episode_id": blueprint.episode_id,
                "status": "accepted" if result.accepted else "rejected",
                "attempts": result.attempts,
                "api_calls": result.api_calls,
                "usage": _usage_dict(result.usage),
                "reasons": list(result.reasons),
                "record": result.record,
            }
            _append_jsonl(paths.progress, progress)
            completed[blueprint.episode_id] = progress
            if not result.accepted:
                _append_jsonl(paths.rejected, {key: progress[key] for key in progress if key != "record"})

        if concurrency == 1:
            for index, blueprint in indexed:
                save(index, blueprint, self.process_blueprint(blueprint))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures: dict[Future[ProcessResult], tuple[int, EpisodeBlueprint]] = {
                    executor.submit(self.process_blueprint, blueprint): (index, blueprint)
                    for index, blueprint in indexed
                }
                for future in as_completed(futures):
                    index, blueprint = futures[future]
                    save(index, blueprint, future.result())

        ordered = sorted(completed.values(), key=lambda item: int(item["source_index"]))
        accepted_records = [item["record"] for item in ordered if item["status"] == "accepted"]
        rejected_count = sum(item["status"] == "rejected" for item in ordered)
        usage = ApiUsage()
        for item in ordered:
            usage = add_usage(usage, _usage_from_dict(item.get("usage", {})))
        _atomic_write_text(
            paths.dataset,
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted_records),
        )
        summary = {
            "processed": len(ordered),
            "accepted": len(accepted_records),
            "rejected": rejected_count,
            "acceptance_rate": len(accepted_records) / len(ordered) if ordered else 0.0,
            "api_calls": invocation_api_calls,
            "usage": _usage_dict(usage),
        }
        _atomic_write_text(paths.summary, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return summary


def _parse_judge(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    if set(payload) != {"accepted", "reasons"}:
        raise ValueError("expected exactly accepted and reasons")
    if not isinstance(payload["accepted"], bool):
        raise ValueError("accepted must be bool")
    reasons = payload["reasons"]
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        raise ValueError("reasons must be list[str]")
    return payload["accepted"], reasons


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            completed[str(payload["episode_id"])] = payload
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Invalid progress record at {path}:{line_number}") from exc
    return completed


def _usage_dict(usage: ApiUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _usage_from_dict(payload: dict[str, Any]) -> ApiUsage:
    return ApiUsage(
        int(payload.get("prompt_tokens", 0)),
        int(payload.get("completion_tokens", 0)),
        int(payload.get("total_tokens", 0)),
    )
