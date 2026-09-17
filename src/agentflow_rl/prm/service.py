from __future__ import annotations

import asyncio
import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from agentflow_rl.rewards.schemas import ProcessScore, ProcessTransition
from agentflow_rl.rewards.rubric import PROCESS_RUBRIC_REVISION
from agentflow_rl.runtime.errors import RevisionMismatchError


class ProcessRewardBackend(Protocol):
    revision: str

    async def predict(self, transition: ProcessTransition) -> float: ...

    async def predict_many(
        self, transitions: tuple[ProcessTransition, ...]
    ) -> tuple[float, ...]: ...


@dataclass
class _PendingTransition:
    transition: ProcessTransition
    future: asyncio.Future[float]


class AsyncProcessRewardBatcher:
    """Merge concurrent PRM requests into bounded backend batches."""

    def __init__(
        self,
        backend: ProcessRewardBackend,
        *,
        max_batch_size: int = 32,
        batch_wait_ms: float = 10.0,
        max_queue: int = 256,
    ) -> None:
        if max_batch_size <= 0 or batch_wait_ms < 0 or max_queue < max_batch_size:
            raise ValueError("PRM batch settings are invalid")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.max_queue = max_queue
        self._queue: asyncio.Queue[_PendingTransition] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.batch_count = 0
        self.scored_items = 0
        self.max_observed_batch_size = 0
        self.peak_pending = 0
        self.total_backend_latency_ms = 0.0
        self.max_backend_latency_ms = 0.0

    def _ensure_worker(self) -> asyncio.Queue[_PendingTransition]:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self._worker is not None and not self._worker.done():
                raise RuntimeError("PRM batcher cannot move between active event loops")
            self._loop = loop
            self._queue = asyncio.Queue(maxsize=self.max_queue)
            self._worker = loop.create_task(self._run())
        assert self._queue is not None
        return self._queue

    async def predict(self, transition: ProcessTransition) -> float:
        return (await self.predict_many((transition,)))[0]

    async def predict_many(
        self, transitions: tuple[ProcessTransition, ...]
    ) -> tuple[float, ...]:
        if not transitions:
            return ()
        queue = self._ensure_worker()
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in transitions]
        for transition, future in zip(transitions, futures, strict=True):
            await queue.put(_PendingTransition(transition, future))
            self.peak_pending = max(self.peak_pending, queue.qsize())
        return tuple(await asyncio.gather(*futures))

    async def _run(self) -> None:
        assert self._queue is not None
        queue = self._queue
        while True:
            first = await queue.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.batch_wait_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
                except TimeoutError:
                    break
            try:
                started = monotonic()
                predictor = getattr(self.backend, "predict_many", None)
                if predictor is None:
                    values = await asyncio.gather(
                        *(self.backend.predict(item.transition) for item in batch)
                    )
                else:
                    values = await predictor(
                        tuple(item.transition for item in batch)
                    )
                if len(values) != len(batch):
                    raise ValueError("PRM backend batch response length mismatch")
                for item, value in zip(batch, values, strict=True):
                    if not item.future.done():
                        item.future.set_result(float(value))
                self.batch_count += 1
                self.scored_items += len(batch)
                self.max_observed_batch_size = max(
                    self.max_observed_batch_size, len(batch)
                )
                elapsed_ms = (monotonic() - started) * 1000.0
                self.total_backend_latency_ms += elapsed_ms
                self.max_backend_latency_ms = max(
                    self.max_backend_latency_ms, elapsed_ms
                )
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
            finally:
                for _ in batch:
                    queue.task_done()

    async def close(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    def metrics(self) -> dict[str, float]:
        return {
            "configured_max_batch_size": float(self.max_batch_size),
            "configured_batch_wait_ms": self.batch_wait_s * 1000.0,
            "configured_max_queue": float(self.max_queue),
            "batch_count": float(self.batch_count),
            "scored_items": float(self.scored_items),
            "mean_batch_size": (
                self.scored_items / self.batch_count if self.batch_count else 0.0
            ),
            "max_observed_batch_size": float(self.max_observed_batch_size),
            "peak_pending": float(self.peak_pending),
            "pending": float(self._queue.qsize() if self._queue is not None else 0),
            "mean_backend_latency_ms": (
                self.total_backend_latency_ms / self.batch_count
                if self.batch_count
                else 0.0
            ),
            "max_backend_latency_ms": self.max_backend_latency_ms,
        }


class LearnedProcessScorer:
    """Expose a trained PRM behind the shared process-scorer contract."""

    def __init__(self, backend: ProcessRewardBackend) -> None:
        self.backend = backend
        self.revision = backend.revision
        self.rubric_revision = getattr(backend, "rubric_revision", "legacy")

    async def score(self, transition: ProcessTransition) -> ProcessScore:
        raw_score = await self.backend.predict(transition)
        score = min(1.0, max(0.0, float(raw_score)))
        return ProcessScore(
            score=score,
            confidence=1.0,
            reason="learned process reward",
            scorer_revision=self.revision,
            rubric_revision=self.rubric_revision,
        )

    async def score_many(
        self, transitions: tuple[ProcessTransition, ...]
    ) -> tuple[ProcessScore, ...]:
        if not transitions:
            return ()
        predictor = getattr(self.backend, "predict_many", None)
        if predictor is None:
            raw_scores = await asyncio.gather(
                *(self.backend.predict(transition) for transition in transitions)
            )
        else:
            raw_scores = await predictor(transitions)
        return tuple(
            ProcessScore(
                score=min(1.0, max(0.0, float(raw_score))),
                confidence=1.0,
                reason="learned process reward",
                scorer_revision=self.revision,
                rubric_revision=self.rubric_revision,
            )
            for raw_score in raw_scores
        )


HttpPost = Callable[[str, bytes, float], bytes]


def _post(url: str, body: bytes, timeout_s: float) -> bytes:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


class HttpProcessRewardBackend:
    rubric_revision = PROCESS_RUBRIC_REVISION
    def __init__(
        self,
        endpoint: str,
        *,
        revision: str,
        timeout_s: float = 30.0,
        http_post: HttpPost = _post,
    ) -> None:
        self.endpoint = endpoint
        self.revision = revision
        self.timeout_s = timeout_s
        self.http_post = http_post

    def _validate_identity(self, payload: dict) -> None:
        if payload.get("revision") != self.revision or payload.get("rubric_revision") != self.rubric_revision:
            raise RevisionMismatchError("PRM response model/rubric revision mismatch")

    async def predict(self, transition: ProcessTransition) -> float:
        body = json.dumps(
            {"transition": transition.model_dump(mode="json")},
            ensure_ascii=False,
        ).encode("utf-8")
        raw = await asyncio.to_thread(
            self.http_post, self.endpoint, body, self.timeout_s
        )
        payload = json.loads(raw)
        self._validate_identity(payload)
        return float(payload["score"])

    async def predict_many(
        self, transitions: tuple[ProcessTransition, ...]
    ) -> tuple[float, ...]:
        if not transitions:
            return ()
        endpoint = self.endpoint.rstrip("/")
        if endpoint.endswith("/score"):
            endpoint = endpoint[: -len("/score")]
        body = json.dumps(
            {
                "transitions": [
                    transition.model_dump(mode="json") for transition in transitions
                ]
            },
            ensure_ascii=False,
        ).encode("utf-8")
        raw = await asyncio.to_thread(
            self.http_post, f"{endpoint}/score_batch", body, self.timeout_s
        )
        payload = json.loads(raw)
        self._validate_identity(payload)
        scores = tuple(float(score) for score in payload["scores"])
        if len(scores) != len(transitions):
            raise ValueError("PRM batch response length mismatch")
        return scores


__all__ = [
    "AsyncProcessRewardBatcher",
    "HttpProcessRewardBackend",
    "LearnedProcessScorer",
    "ProcessRewardBackend",
]
