from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from .contracts import WikipediaDocument, WikipediaHit


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    title: str
    passage_id: str
    snippet: str
    score: float


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int) -> tuple[RetrievalHit, ...]: ...


class WikipediaDocumentStore(Protocol):
    def read(self, doc_id: str) -> WikipediaDocument: ...


def _minmax(hits: tuple[RetrievalHit, ...]) -> dict[str, float]:
    if not hits:
        return {}
    values = [item.score for item in hits]
    low, high = min(values), max(values)
    if high == low:
        return {item.passage_id: 1.0 for item in hits}
    return {item.passage_id: (item.score - low) / (high - low) for item in hits}


class HybridWikipediaBackend:
    def __init__(
        self,
        *,
        sparse: Retriever,
        dense: Retriever,
        documents: WikipediaDocumentStore,
        sparse_weight: float = 0.5,
        revision: str,
    ) -> None:
        if not 0.0 <= sparse_weight <= 1.0:
            raise ValueError("sparse_weight must be within [0, 1]")
        self.sparse = sparse
        self.dense = dense
        self.documents = documents
        self.sparse_weight = sparse_weight
        self.revision = revision

    async def search(self, *, query: str, top_k: int) -> tuple[WikipediaHit, ...]:
        return (await self.search_many(((query, top_k),)))[0]

    def _fuse(
        self,
        sparse_hits: tuple[RetrievalHit, ...],
        dense_hits: tuple[RetrievalHit, ...],
        *,
        top_k: int,
    ) -> tuple[WikipediaHit, ...]:
        sparse_norm = _minmax(sparse_hits)
        dense_norm = _minmax(dense_hits)
        by_id = {item.passage_id: item for item in [*sparse_hits, *dense_hits]}
        sparse_raw = {item.passage_id: item.score for item in sparse_hits}
        dense_raw = {item.passage_id: item.score for item in dense_hits}
        ranked = []
        for passage_id, item in by_id.items():
            fused = (
                self.sparse_weight * sparse_norm.get(passage_id, 0.0)
                + (1.0 - self.sparse_weight) * dense_norm.get(passage_id, 0.0)
            )
            ranked.append((fused, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].passage_id))
        return tuple(
            WikipediaHit(
                doc_id=item.doc_id,
                title=item.title,
                passage_id=item.passage_id,
                snippet=item.snippet,
                rank=rank,
                fused_score=fused,
                bm25_score=sparse_raw.get(item.passage_id),
                dense_score=dense_raw.get(item.passage_id),
            )
            for rank, (fused, item) in enumerate(ranked[:top_k], start=1)
        )

    async def search_many(
        self, requests: tuple[tuple[str, int], ...]
    ) -> tuple[tuple[WikipediaHit, ...], ...]:
        if not requests:
            return ()
        candidate_ks = tuple(max(top_k, top_k * 2) for _, top_k in requests)
        max_candidate_k = max(candidate_ks)

        def sparse_searches() -> tuple[tuple[RetrievalHit, ...], ...]:
            return tuple(
                self.sparse.search(query, top_k=candidate_k)
                for (query, _), candidate_k in zip(requests, candidate_ks, strict=True)
            )

        dense_many = getattr(self.dense, "search_many", None)
        if dense_many is None:
            def dense_searches() -> tuple[tuple[RetrievalHit, ...], ...]:
                return tuple(
                    self.dense.search(query, top_k=candidate_k)
                    for (query, _), candidate_k in zip(requests, candidate_ks, strict=True)
                )
        else:
            def dense_searches() -> tuple[tuple[RetrievalHit, ...], ...]:
                rows = dense_many(
                    tuple(query for query, _ in requests), top_k=max_candidate_k
                )
                return tuple(
                    tuple(row[:candidate_k])
                    for row, candidate_k in zip(rows, candidate_ks, strict=True)
                )

        sparse_batches, dense_batches = await asyncio.gather(
            asyncio.to_thread(sparse_searches),
            asyncio.to_thread(dense_searches),
        )
        return tuple(
            self._fuse(sparse_hits, dense_hits, top_k=top_k)
            for sparse_hits, dense_hits, (_, top_k) in zip(
                sparse_batches, dense_batches, requests, strict=True
            )
        )

    async def read(
        self,
        *,
        doc_id: str,
        start_passage: int,
        max_passages: int,
    ) -> WikipediaDocument:
        document = await asyncio.to_thread(self.documents.read, doc_id)
        stop = start_passage + max_passages
        return WikipediaDocument(
            doc_id=document.doc_id,
            title=document.title,
            passages=document.passages[start_passage:stop],
            passage_ids=document.passage_ids[start_passage:stop],
        )


__all__ = [
    "HybridWikipediaBackend",
    "RetrievalHit",
    "Retriever",
    "WikipediaDocumentStore",
]
