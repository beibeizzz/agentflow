from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schemas import (
    ApiUsage,
    JudgeDecision,
    RewriteCandidate,
)
from .validators import validate_rewrite


@dataclass(frozen=True)
class PipelinePaths:
    dataset: Path
    rejected: Path
    progress: Path
    summary: Path

    @classmethod
    def in_directory(cls, directory: Path) -> "PipelinePaths":
        return cls(
            dataset=directory / "gsm8k_train_calculator_structured.json",
            rejected=directory / "rejected.jsonl",
            progress=directory / "progress.jsonl",
            summary=directory / "summary.json",
        )


@dataclass(frozen=True)
class ProcessResult:
    accepted: bool
    attempts: int
    usage: ApiUsage
    record: dict[str, Any] | None = None
    reasons: tuple[str, ...] = ()


class DatasetRewriter:
    def __init__(
        self,
        client: Any,
        *,
        max_attempts: int = 3,
        rewrite_temperature: float = 0.3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.client = client
        self.max_attempts = max_attempts
        self.rewrite_temperature = rewrite_temperature

    def process_record(self, source: dict[str, Any]) -> ProcessResult:
        feedback: list[str] = []
        usage = ApiUsage()
        for attempt in range(1, self.max_attempts + 1):
            rewrite_response = self.client.rewrite(
                source,
                prior_failures=feedback,
                temperature=self.rewrite_temperature,
            )
            usage = _add_usage(usage, rewrite_response.usage)
            try:
                candidate = RewriteCandidate.from_payload(rewrite_response.payload)
            except ValueError as exc:
                feedback = [f"Rewrite JSON schema error: {exc}"]
                continue

            validation = validate_rewrite(source, candidate.rewritten_question)
            if not validation.ok:
                feedback = list(validation.messages)
                continue

            judge_response = self.client.judge(source, candidate.rewritten_question)
            usage = _add_usage(usage, judge_response.usage)
            try:
                decision = JudgeDecision.from_payload(judge_response.payload)
            except ValueError as exc:
                feedback = [f"Judge JSON schema error: {exc}"]
                continue
            if not decision.accepted:
                feedback = list(decision.reasons) or ["Judge rejected the candidate without a reason."]
                continue

            record = dict(source)
            record["question"] = candidate.rewritten_question
            record["query"] = _build_query(candidate.rewritten_question)
            return ProcessResult(
                accepted=True,
                attempts=attempt,
                usage=usage,
                record=record,
            )

        return ProcessResult(
            accepted=False,
            attempts=self.max_attempts,
            usage=usage,
            reasons=tuple(feedback),
        )

    def run(
        self,
        records: list[dict[str, Any]],
        *,
        paths: PipelinePaths,
        start: int = 0,
        limit: int | None = None,
        resume: bool = True,
        concurrency: int = 1,
    ) -> dict[str, Any]:
        if start < 0:
            raise ValueError("start must be non-negative")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        paths.dataset.parent.mkdir(parents=True, exist_ok=True)
        paths.rejected.parent.mkdir(parents=True, exist_ok=True)
        paths.progress.parent.mkdir(parents=True, exist_ok=True)
        paths.summary.parent.mkdir(parents=True, exist_ok=True)

        if not resume:
            for path in (paths.progress, paths.rejected):
                if path.exists():
                    path.unlink()

        completed = _load_progress(paths.progress) if resume else {}
        stop = len(records) if limit is None else min(len(records), start + max(0, limit))
        selected_indices = range(min(start, len(records)), stop)

        pending_indices = [source_index for source_index in selected_indices if source_index not in completed]

        def save_result(source_index: int, result: ProcessResult) -> None:
            source = records[source_index]
            progress_record = {
                "source_index": source_index,
                "pid": source.get("pid"),
                "status": "accepted" if result.accepted else "rejected",
                "attempts": result.attempts,
                "usage": _usage_dict(result.usage),
                "reasons": list(result.reasons),
                "record": result.record,
            }
            _append_jsonl(paths.progress, progress_record)
            completed[source_index] = progress_record
            if not result.accepted:
                _append_jsonl(
                    paths.rejected,
                    {
                        "source_index": source_index,
                        "pid": source.get("pid"),
                        "question": source.get("question"),
                        "attempts": result.attempts,
                        "reasons": list(result.reasons),
                        "usage": _usage_dict(result.usage),
                    },
                )

        if concurrency == 1:
            for source_index in pending_indices:
                save_result(source_index, self.process_record(records[source_index]))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures: dict[Future[ProcessResult], int] = {
                    executor.submit(self.process_record, records[source_index]): source_index
                    for source_index in pending_indices
                }
                for future in as_completed(futures):
                    save_result(futures[future], future.result())

        accepted_records = [
            item["record"]
            for _, item in sorted(completed.items())
            if item.get("status") == "accepted" and isinstance(item.get("record"), dict)
        ]
        rejected_count = sum(1 for item in completed.values() if item.get("status") == "rejected")
        usage = ApiUsage()
        for item in completed.values():
            usage = _add_usage(usage, _usage_from_dict(item.get("usage") or {}))
        summary = {
            "processed": len(completed),
            "accepted": len(accepted_records),
            "rejected": rejected_count,
            "acceptance_rate": len(accepted_records) / len(completed) if completed else 0.0,
            "usage": _usage_dict(usage),
            "dataset": str(paths.dataset),
            "progress": str(paths.progress),
            "rejected_file": str(paths.rejected),
        }
        _atomic_write_json(paths.dataset, accepted_records)
        _atomic_write_json(paths.summary, summary)
        return summary


def _build_query(question: str) -> str:
    return f"\nYou should focus on your responsibility mentioned.\n\nProblem:\n{question}"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _load_progress(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                source_index = int(payload["source_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid progress record at {path}:{line_number}") from exc
            completed[source_index] = payload
    return completed


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _usage_dict(usage: ApiUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _usage_from_dict(payload: dict[str, Any]) -> ApiUsage:
    return ApiUsage(
        prompt_tokens=int(payload.get("prompt_tokens", 0)),
        completion_tokens=int(payload.get("completion_tokens", 0)),
        total_tokens=int(payload.get("total_tokens", 0)),
    )


def _add_usage(left: ApiUsage, right: ApiUsage) -> ApiUsage:
    return ApiUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )
