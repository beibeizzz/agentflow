from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, Field

from agentflow_rl.backends.wikipedia_hybrid import HybridWikipediaBackend
from agentflow_rl.backends.wikipedia_local import (
    ArrowWikipediaStore,
    E5FaissRetriever,
    PyseriniBM25Retriever,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class ReadRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    start_passage: int = Field(default=0, ge=0)
    max_passages: int = Field(default=5, ge=1, le=50)


@dataclass
class _PendingSearch:
    query: str
    top_k: int
    future: asyncio.Future


class _SearchBatcher:
    def __init__(
        self,
        backend,
        *,
        slots: asyncio.Semaphore,
        max_batch_size: int,
        wait_s: float,
        max_pending: int,
    ) -> None:
        self.backend = backend
        self.slots = slots
        self.max_batch_size = max_batch_size
        self.wait_s = wait_s
        self.max_pending = max_pending
        self.queue: asyncio.Queue[_PendingSearch] = asyncio.Queue()
        self.pending = 0
        self.pending_lock = asyncio.Lock()
        self.dispatcher: asyncio.Task | None = None
        self.executions: set[asyncio.Task] = set()
        self.batch_count = 0
        self.query_count = 0
        self.max_observed_batch_size = 0
        self.peak_pending = 0

    def start(self) -> None:
        if self.dispatcher is None or self.dispatcher.done():
            self.dispatcher = asyncio.create_task(self._dispatch())

    async def close(self) -> None:
        if self.dispatcher is not None:
            self.dispatcher.cancel()
            await asyncio.gather(self.dispatcher, return_exceptions=True)
        if self.executions:
            await asyncio.gather(*tuple(self.executions), return_exceptions=True)

    async def submit(self, query: str, top_k: int):
        self.start()
        async with self.pending_lock:
            if self.pending >= self.max_pending:
                raise OverflowError("retrieval queue is full")
            self.pending += 1
            self.peak_pending = max(self.peak_pending, self.pending)
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(_PendingSearch(query=query, top_k=top_k, future=future))
        try:
            return await future
        finally:
            async with self.pending_lock:
                self.pending -= 1

    async def _dispatch(self) -> None:
        while True:
            first = await self.queue.get()
            if self.wait_s:
                await asyncio.sleep(self.wait_s)
            batch = [first]
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            task = asyncio.create_task(self._execute(tuple(batch)))
            self.executions.add(task)
            task.add_done_callback(self.executions.discard)

    async def _execute(self, batch: tuple[_PendingSearch, ...]) -> None:
        self.batch_count += 1
        self.query_count += len(batch)
        self.max_observed_batch_size = max(self.max_observed_batch_size, len(batch))
        try:
            async with self.slots:
                search_many = getattr(self.backend, "search_many", None)
                if search_many is None:
                    results = await asyncio.gather(
                        *(
                            self.backend.search(query=item.query, top_k=item.top_k)
                            for item in batch
                        )
                    )
                else:
                    results = await search_many(
                        tuple((item.query, item.top_k) for item in batch)
                    )
            if len(results) != len(batch):
                raise ValueError("Wikipedia batch response length mismatch")
        except BaseException as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
        else:
            for item, result in zip(batch, results, strict=True):
                if not item.future.done():
                    item.future.set_result(result)

    def metrics(self) -> dict[str, int | float]:
        return {
            "batch_count": self.batch_count,
            "query_count": self.query_count,
            "mean_batch_size": (
                self.query_count / self.batch_count if self.batch_count else 0.0
            ),
            "max_observed_batch_size": self.max_observed_batch_size,
            "peak_pending": self.peak_pending,
        }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_digest(name: str) -> str:
    value = os.environ[name].lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")
    return value


def _require_arrow_fingerprint(actual: str) -> str:
    expected = os.environ["WIKIPEDIA_ARROW_FINGERPRINT"]
    if not expected or actual != expected:
        raise ValueError("Wikipedia Arrow fingerprint differs from its manifest")
    return expected


def build_backend_from_env() -> HybridWikipediaBackend:
    corpus_sha256 = _required_digest("WIKIPEDIA_CORPUS_SHA256")
    corpus_path = os.environ["WIKIPEDIA_CORPUS_PATH"]
    if os.environ.get("WIKIPEDIA_VERIFY_HASHES", "true").lower() in {"1", "true", "yes"}:
        if _sha256_file(corpus_path) != corpus_sha256:
            raise ValueError("Wikipedia corpus hash differs from its manifest")
    corpus = ArrowWikipediaStore(
        corpus_path,
        cache_dir=os.environ.get("WIKIPEDIA_ARROW_CACHE_DIR"),
        corpus_sha256=corpus_sha256,
    )
    _require_arrow_fingerprint(corpus.arrow_fingerprint)
    sparse = PyseriniBM25Retriever.from_index(
        os.environ["WIKIPEDIA_BM25_INDEX"], corpus
    )
    index_type = os.environ.get("WIKIPEDIA_INDEX_TYPE", "hnsw64").strip().lower()
    hnsw_m = int(os.environ.get("WIKIPEDIA_HNSW_M", "64"))
    hnsw_ef_search = int(os.environ.get("WIKIPEDIA_HNSW_EF_SEARCH", "256"))
    index_sha256 = _required_digest("WIKIPEDIA_INDEX_SHA256")
    if os.environ.get("WIKIPEDIA_VERIFY_HASHES", "true").lower() in {"1", "true", "yes"}:
        if _sha256_file(os.environ["WIKIPEDIA_FAISS_INDEX"]) != index_sha256:
            raise ValueError("Wikipedia FAISS index hash differs from its manifest")
    dense = E5FaissRetriever.from_files(
        model_name_or_path=os.environ["WIKIPEDIA_E5_MODEL"],
        index_path=os.environ["WIKIPEDIA_FAISS_INDEX"],
        corpus=corpus,
        device=os.environ.get("WIKIPEDIA_E5_DEVICE", "cpu"),
        memory_map=os.environ.get("WIKIPEDIA_FAISS_MMAP", "false").lower()
        in {"1", "true", "yes"},
        index_type=index_type,
        hnsw_m=hnsw_m,
        hnsw_ef_search=hnsw_ef_search,
        index_sha256=index_sha256,
        corpus_sha256=corpus_sha256,
        arrow_fingerprint=corpus.arrow_fingerprint,
        encoder_revision=os.environ["WIKIPEDIA_E5_REVISION"],
    )
    return HybridWikipediaBackend(
        sparse=sparse,
        dense=dense,
        documents=corpus,
        sparse_weight=float(os.environ.get("WIKIPEDIA_SPARSE_WEIGHT", "0.5")),
        revision=os.environ["WIKIPEDIA_INDEX_REVISION"],
    )


def create_app(
    backend=None,
    *,
    max_concurrency: int | None = None,
    max_queue: int | None = None,
    queue_timeout_s: float | None = None,
    batch_max_size: int | None = None,
    batch_wait_ms: float | None = None,
):
    from fastapi import FastAPI, HTTPException

    retrieval = backend or build_backend_from_env()
    limit = (
        int(os.environ.get("WIKIPEDIA_MAX_CONCURRENCY", "4"))
        if max_concurrency is None
        else max_concurrency
    )
    queue_limit = (
        int(os.environ.get("WIKIPEDIA_MAX_QUEUE", "40"))
        if max_queue is None
        else max_queue
    )
    queue_timeout = (
        float(os.environ.get("WIKIPEDIA_QUEUE_TIMEOUT_S", "30"))
        if queue_timeout_s is None
        else queue_timeout_s
    )
    batch_size = (
        int(os.environ.get("WIKIPEDIA_BATCH_MAX_SIZE", "16"))
        if batch_max_size is None
        else batch_max_size
    )
    batch_wait = (
        float(os.environ.get("WIKIPEDIA_BATCH_WAIT_MS", "5"))
        if batch_wait_ms is None
        else batch_wait_ms
    )
    if (
        limit <= 0
        or queue_limit < 0
        or queue_timeout <= 0
        or not 1 <= batch_size <= 16
        or batch_wait < 0
    ):
        raise ValueError("Wikipedia capacity settings are invalid")
    slots = asyncio.Semaphore(limit)
    batcher = _SearchBatcher(
        retrieval,
        slots=slots,
        max_batch_size=batch_size,
        wait_s=batch_wait / 1000.0,
        max_pending=limit + queue_limit,
    )

    @asynccontextmanager
    async def lifespan(_app):
        batcher.start()
        yield
        await batcher.close()

    app = FastAPI(
        title="AgentFlow Wikipedia Retrieval", version="1", lifespan=lifespan
    )

    @asynccontextmanager
    async def read_capacity():
        acquired = False
        try:
            try:
                await asyncio.wait_for(slots.acquire(), timeout=queue_timeout)
                acquired = True
            except asyncio.TimeoutError as exc:
                raise HTTPException(
                    status_code=503, detail="retrieval queue wait timed out"
                ) from exc
            yield
        finally:
            if acquired:
                slots.release()

    @app.get("/health")
    async def health():
        dense = getattr(retrieval, "dense", None)
        return {
            "status": "ok",
            "revision": retrieval.revision,
            "max_concurrency": limit,
            "max_queue": queue_limit,
            "batch_max_size": batch_size,
            "batch_wait_ms": batch_wait,
            "batch_metrics": batcher.metrics(),
            **dict(getattr(dense, "index_metadata", {})),
        }

    @app.post("/search")
    async def search(request: SearchRequest):
        try:
            hits = await asyncio.wait_for(
                batcher.submit(request.query, request.top_k), timeout=queue_timeout
            )
        except OverflowError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=503, detail="retrieval queue wait timed out"
            ) from exc
        return {
            "hits": [item.model_dump(mode="json") for item in hits],
            "revision": retrieval.revision,
        }

    @app.post("/read")
    async def read(request: ReadRequest):
        async with read_capacity():
            document = await retrieval.read(
                doc_id=request.doc_id,
                start_passage=request.start_passage,
                max_passages=request.max_passages,
            )
        return {
            "document": document.model_dump(mode="json"),
            "revision": retrieval.revision,
        }

    return app


__all__ = ["ReadRequest", "SearchRequest", "build_backend_from_env", "create_app"]
