from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class BackendModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchHit(BackendModel):
    result_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    snippet: str = ""
    rank: int = Field(ge=1)
    published_at: str | None = None


class PageDocument(BackendModel):
    result_id: str = Field(min_length=1)
    canonical_url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    passages: tuple[str, ...]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WikipediaHit(BackendModel):
    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    passage_id: str = Field(min_length=1)
    snippet: str = ""
    rank: int = Field(ge=1)
    fused_score: float
    bm25_score: float | None = None
    dense_score: float | None = None


class WikipediaDocument(BackendModel):
    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    passages: tuple[str, ...]
    passage_ids: tuple[str, ...]

    def model_post_init(self, __context: object) -> None:
        if len(self.passages) != len(self.passage_ids):
            raise ValueError("Wikipedia passages and passage_ids must align")


class SandboxRequest(BackendModel):
    code: str = Field(min_length=1)
    mode: str = Field(pattern=r"^(run|test)$")
    stdin: str = ""
    tests: tuple[dict[str, Any], ...] = ()
    timeout_s: float = Field(gt=0.0)


class SandboxResult(BackendModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    passed: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    failures: tuple[dict[str, Any], ...] = ()
    admission_wait_ms: float = Field(default=0.0, ge=0.0)
    queue_wait_ms: float = Field(default=0.0, ge=0.0)
    execution_ms: float = Field(default=0.0, ge=0.0)

    def model_post_init(self, __context: object) -> None:
        if (self.passed is None) != (self.total is None):
            raise ValueError("sandbox passed and total must be provided together")
        if self.passed is not None and self.total is not None and self.passed > self.total:
            raise ValueError("sandbox passed count exceeds total")


class BaseGeneratorBackend(Protocol):
    revision: str

    async def generate(self, *, prompt: str, max_tokens: int) -> str: ...


class SandboxBackend(Protocol):
    revision: str

    async def execute(self, request: SandboxRequest) -> SandboxResult: ...


class SearchBackend(Protocol):
    revision: str

    async def search(self, *, query: str, top_k: int) -> tuple[SearchHit, ...]: ...


class PageReaderBackend(Protocol):
    revision: str

    async def open(self, *, url: str, result_id: str | None = None) -> PageDocument: ...


class WikipediaBackend(Protocol):
    revision: str

    async def search(self, *, query: str, top_k: int) -> tuple[WikipediaHit, ...]: ...

    async def read(
        self,
        *,
        doc_id: str,
        start_passage: int,
        max_passages: int,
    ) -> WikipediaDocument: ...


__all__ = [
    "BaseGeneratorBackend",
    "PageDocument",
    "PageReaderBackend",
    "SandboxBackend",
    "SandboxRequest",
    "SandboxResult",
    "SearchBackend",
    "SearchHit",
    "WikipediaBackend",
    "WikipediaDocument",
    "WikipediaHit",
]
