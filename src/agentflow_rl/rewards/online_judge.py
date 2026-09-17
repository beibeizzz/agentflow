from __future__ import annotations

import hashlib
import json
import asyncio
from pathlib import Path
from typing import Any, Callable

from agentflow_rl.runtime.errors import InfrastructureError
from agentflow_rl.runtime.parsing import strict_json_object
from agentflow_rl.backends.concurrency import CrossProcessSlotLimiter

from .schemas import ProcessScore, ProcessTransition
from .transition_view import render_process_transition
from .rubric import PROCESS_RUBRIC_REVISION, render_judge_system


class JsonlScoreCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._scores: dict[str, ProcessScore] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._scores[row["key"]] = ProcessScore.model_validate(row["score"])

    def get(self, key: str) -> ProcessScore | None:
        return self._scores.get(key)

    def reload(self) -> None:
        self._scores.clear()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._scores[row["key"]] = ProcessScore.model_validate(row["score"])

    def put(self, key: str, score: ProcessScore) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"key": key, "score": score.model_dump(mode="json")}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._scores[key] = score


class OpenAICompatibleProcessJudge:
    """Label public Planner transitions through a versioned chat-completion API."""

    rubric_revision = PROCESS_RUBRIC_REVISION

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        revision: str,
        cache: JsonlScoreCache | None = None,
        max_concurrency: int = 8,
        max_attempts: int = 3,
        backoff_s: float = 0.5,
        coordination_dir: str | Path | None = None,
        transition_renderer: Callable[[ProcessTransition], str] | None = None,
    ) -> None:
        if max_concurrency <= 0 or max_attempts <= 0 or backoff_s < 0:
            raise ValueError("Judge retry and concurrency settings are invalid")
        self.client = client
        self.model = model
        self.revision = revision
        self.cache = cache
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self._render_transition = transition_renderer or render_process_transition
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._cache_lock = asyncio.Lock()
        coordination_root = coordination_dir or self._default_coordination_dir()
        self._shared_limiter = CrossProcessSlotLimiter(
            coordination_root,
            name="online-process-judge",
            limit=max_concurrency,
        )
        self._shared_cache_limiter = CrossProcessSlotLimiter(
            coordination_root,
            name="online-process-judge-cache",
            limit=1,
        )
        self.requests = 0
        self.cache_hits = 0
        self.failures = 0
        self.retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.queue_wait_ms = 0.0

    @staticmethod
    def _default_coordination_dir() -> Path:
        import tempfile

        return Path(tempfile.gettempdir()) / "agentflow-locks"

    async def score(self, transition: ProcessTransition) -> ProcessScore:
        if transition.tool_result is not None and not transition.tool_result.valid_for_training:
            raise InfrastructureError("Infrastructure-invalid transitions are excluded from process labeling")
        key = self._cache_key(transition)
        if self.cache and (cached := self.cache.get(key)) is not None and cached.rubric_revision == self.rubric_revision:
            self.cache_hits += 1
            return cached
        async with self._semaphore:
            if self.cache and (cached := self.cache.get(key)) is not None and cached.rubric_revision == self.rubric_revision:
                self.cache_hits += 1
                return cached
            last_error: Exception | None = None
            for attempt in range(self.max_attempts):
                self.requests += 1
                try:
                    async with self._shared_limiter.slot(timeout_s=120.0) as wait_ms:
                        self.queue_wait_ms += wait_ms
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                            {
                                "role": "system",
                                "content": render_judge_system(transition.task.task_name),
                            },
                            {
                                "role": "user",
                                "content": self._render_transition(transition),
                            },
                            ],
                            temperature=0.0,
                            max_tokens=512,
                            response_format={"type": "json_object"},
                        )
                    usage = getattr(response, "usage", None)
                    self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                    self.completion_tokens += int(
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                    content = response.choices[0].message.content
                    payload = strict_json_object(content)
                    score = ProcessScore(
                        score=payload["score"],
                        confidence=payload["confidence"],
                        reason=payload["reason"],
                        failure_code=payload.get("failure_code"),
                        scorer_revision=self.revision,
                        rubric_revision=self.rubric_revision,
                    )
                    if self.cache:
                        async with self._cache_lock:
                            async with self._shared_cache_limiter.slot(
                                timeout_s=30.0
                            ):
                                self.cache.reload()
                                if (cached := self.cache.get(key)) is not None and cached.rubric_revision == self.rubric_revision:
                                    return cached
                                self.cache.put(key, score)
                    return score
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < self.max_attempts:
                        self.retries += 1
                        await asyncio.sleep(self.backoff_s * (2**attempt))
            self.failures += 1
            raise InfrastructureError("online process judge failed") from last_error

    def metrics(self) -> dict[str, float]:
        return {
            "requests": float(self.requests),
            "cache_hits": float(self.cache_hits),
            "failures": float(self.failures),
            "retries": float(self.retries),
            "prompt_tokens": float(self.prompt_tokens),
            "completion_tokens": float(self.completion_tokens),
            "queue_wait_ms": self.queue_wait_ms,
        }

    def _cache_key(self, transition: ProcessTransition) -> str:
        body = self._render_transition(transition)
        request = {"model": self.model, "scorer_revision": self.revision,
                   "rubric_revision": self.rubric_revision,
                   "system": render_judge_system(transition.task.task_name), "body": body,
                   "temperature": 0.0, "max_tokens": 512, "response_format": "json_object"}
        return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


__all__ = ["JsonlScoreCache", "OpenAICompatibleProcessJudge"]
