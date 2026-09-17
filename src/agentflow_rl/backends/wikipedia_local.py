from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from .contracts import WikipediaDocument
from .wikipedia_hybrid import RetrievalHit


class PassageCorpus(Protocol):
    def __len__(self) -> int: ...

    def get_by_index(self, row_index: int) -> RetrievalHit: ...

    def read(self, doc_id: str) -> WikipediaDocument: ...


def _split_contents(row: dict[str, Any]) -> tuple[str, str]:
    contents = str(row.get("contents", "")).strip()
    if "\n" in contents:
        raw_title, text = contents.split("\n", 1)
        title = raw_title.strip().strip('"')
        return title or str(row.get("id", "")), text.strip()
    title = str(row.get("title", "")).strip().strip('"')
    text = str(row.get("text", contents)).strip()
    return title or str(row.get("id", "")), text


class ArrowWikipediaStore:
    """Read the canonical wiki-18 JSONL through its memory-mapped Arrow cache.

    Search-R1 indexes preserve JSONL row order. Lucene document IDs and FAISS
    result indices therefore address the same Arrow row. The row's original
    ``id`` remains the public passage identifier, while ``doc_id`` is the stable
    integer row position used by the bounded read endpoint.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        dataset: Any | None = None,
        corpus_sha256: str = "unknown",
    ) -> None:
        if dataset is None:
            if path is None:
                raise ValueError("a Wikipedia JSONL path is required")
            from datasets import load_dataset

            resolved = Path(path).resolve()
            dataset = load_dataset(
                "json",
                data_files=str(resolved),
                split="train",
                cache_dir=str(Path(cache_dir).resolve()) if cache_dir else None,
            )
            self.path: Path | None = resolved
        else:
            self.path = Path(path).resolve() if path is not None else None
        if len(dataset) <= 0:
            raise ValueError("Wikipedia corpus must contain at least one passage")
        first = dict(dataset[0])
        if "id" not in first or "contents" not in first:
            raise ValueError("Wikipedia corpus rows require id and contents fields")
        self.dataset = dataset
        self.corpus_sha256 = corpus_sha256
        self.arrow_fingerprint = str(getattr(dataset, "_fingerprint", "unknown"))

    def __len__(self) -> int:
        return len(self.dataset)

    def _row(self, row_index: int) -> dict[str, Any]:
        if row_index < 0 or row_index >= len(self):
            raise KeyError(str(row_index))
        return dict(self.dataset[row_index])

    @lru_cache(maxsize=65_536)
    def get_by_index(self, row_index: int) -> RetrievalHit:
        row = self._row(row_index)
        title, text = _split_contents(row)
        return RetrievalHit(
            doc_id=str(row_index),
            title=title,
            passage_id=str(row["id"]),
            snippet=text[:500],
            score=0.0,
        )

    def read(self, doc_id: str) -> WikipediaDocument:
        try:
            row_index = int(doc_id)
        except ValueError as exc:
            raise KeyError(doc_id) from exc
        row = self._row(row_index)
        title, text = _split_contents(row)
        return WikipediaDocument(
            doc_id=str(row_index),
            title=title,
            passage_ids=(str(row["id"]),),
            passages=(text,),
        )


class PyseriniBM25Retriever:
    def __init__(self, searcher: Any, corpus: PassageCorpus) -> None:
        self.searcher = searcher
        self.corpus = corpus

    @classmethod
    def from_index(cls, index_path: str | Path, corpus: PassageCorpus):
        from pyserini.search.lucene import LuceneSearcher

        return cls(LuceneSearcher(str(index_path)), corpus)

    def search(self, query: str, *, top_k: int) -> tuple[RetrievalHit, ...]:
        results = []
        for hit in self.searcher.search(query, k=top_k):
            item = self.corpus.get_by_index(int(hit.docid))
            results.append(
                RetrievalHit(
                    doc_id=item.doc_id,
                    title=item.title,
                    passage_id=item.passage_id,
                    snippet=item.snippet,
                    score=float(hit.score),
                )
            )
        return tuple(results)


class E5FaissRetriever:
    def __init__(
        self,
        *,
        encoder: Any,
        index: Any,
        corpus: PassageCorpus,
        index_type: str = "flat",
        hnsw_m: int | None = None,
        hnsw_ef_search: int | None = None,
        faiss_version: str = "unknown",
        index_sha256: str = "unknown",
        corpus_sha256: str = "unknown",
        arrow_fingerprint: str = "unknown",
        encoder_revision: str = "unknown",
    ) -> None:
        if int(index.ntotal) != len(corpus):
            raise ValueError("FAISS rows and Arrow corpus rows must align")
        normalized_type = index_type.strip().lower()
        if normalized_type not in {"flat", "hnsw64"}:
            raise ValueError(f"unsupported FAISS index type: {index_type}")
        graph = getattr(index, "hnsw", None)
        if normalized_type == "flat" and graph is not None:
            raise ValueError("flat retrieval configuration received an HNSW index")
        if normalized_type == "hnsw64":
            if graph is None:
                raise ValueError("hnsw64 retrieval requires an HNSW FAISS index")
            if hnsw_m != 64 or hnsw_ef_search is None or hnsw_ef_search <= 0:
                raise ValueError("hnsw64 requires M=64 and a positive efSearch")
            neighbors = getattr(graph, "nb_neighbors", None)
            if callable(neighbors) and int(neighbors(1)) != hnsw_m:
                raise ValueError("loaded HNSW index M differs from configured M=64")
            graph.efSearch = int(hnsw_ef_search)
        self.encoder = encoder
        self.index = index
        self.corpus = corpus
        self.index_metadata = {
            "index_type": normalized_type,
            "passage_count": len(corpus),
            "hnsw_m": hnsw_m,
            "hnsw_ef_search": hnsw_ef_search,
            "faiss_version": faiss_version,
            "index_sha256": index_sha256,
            "corpus_sha256": corpus_sha256,
            "arrow_fingerprint": arrow_fingerprint,
            "encoder_revision": encoder_revision,
        }

    @classmethod
    def from_files(
        cls,
        *,
        model_name_or_path: str,
        index_path: str | Path,
        corpus: PassageCorpus,
        device: str = "cpu",
        memory_map: bool = True,
        index_type: str = "flat",
        hnsw_m: int | None = None,
        hnsw_ef_search: int | None = None,
        index_sha256: str = "unknown",
        corpus_sha256: str = "unknown",
        arrow_fingerprint: str = "unknown",
        encoder_revision: str = "unknown",
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        normalized_type = index_type.strip().lower()
        if normalized_type == "hnsw64" and memory_map:
            raise ValueError("HNSW indexes must be loaded into RAM; disable FAISS mmap")
        flags = 0
        if memory_map:
            flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        return cls(
            encoder=SentenceTransformer(model_name_or_path, device=device),
            index=faiss.read_index(str(index_path), flags),
            corpus=corpus,
            index_type=normalized_type,
            hnsw_m=hnsw_m,
            hnsw_ef_search=hnsw_ef_search,
            faiss_version=str(getattr(faiss, "__version__", "unknown")),
            index_sha256=index_sha256,
            corpus_sha256=corpus_sha256,
            arrow_fingerprint=arrow_fingerprint,
            encoder_revision=encoder_revision,
        )

    def search(self, query: str, *, top_k: int) -> tuple[RetrievalHit, ...]:
        return self.search_many((query,), top_k=top_k)[0]

    def search_many(
        self, queries: tuple[str, ...], *, top_k: int
    ) -> tuple[tuple[RetrievalHit, ...], ...]:
        if not queries:
            return ()
        vectors = self.encoder.encode(
            [f"query: {query}" for query in queries],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores, indices = self.index.search(vectors, top_k)
        batches = []
        for query_scores, query_indices in zip(scores, indices, strict=True):
            results = []
            for score, index in zip(query_scores, query_indices, strict=True):
                if int(index) < 0:
                    continue
                item = self.corpus.get_by_index(int(index))
                results.append(
                    RetrievalHit(
                        doc_id=item.doc_id,
                        title=item.title,
                        passage_id=item.passage_id,
                        snippet=item.snippet,
                        score=float(score),
                    )
                )
            batches.append(tuple(results))
        return tuple(batches)


# Compatibility alias retained for callers selecting an exact Flat index.
E5FlatRetriever = E5FaissRetriever


__all__ = [
    "ArrowWikipediaStore",
    "E5FaissRetriever",
    "E5FlatRetriever",
    "PyseriniBM25Retriever",
]
