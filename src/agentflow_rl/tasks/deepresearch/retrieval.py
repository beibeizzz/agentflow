from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .schemas import DeepResearchExample


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class ResearchDocument:
    doc_id: str
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    score: float
    snippet: str


class ResearchIndex(Protocol):
    def search(self, query: str, *, top_k: int) -> list[SearchHit]: ...
    def read(self, doc_id: str) -> ResearchDocument: ...


class InMemoryBM25Index:
    def __init__(self, documents: Iterable[ResearchDocument], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = {document.doc_id: document for document in documents}
        self.k1 = k1
        self.b = b
        self._tokens = {key: tokenize(doc.title + " " + doc.text) for key, doc in self.documents.items()}
        self._tf = {key: Counter(tokens) for key, tokens in self._tokens.items()}
        self._avg_len = sum(map(len, self._tokens.values())) / max(1, len(self._tokens))
        self._df: Counter[str] = Counter()
        for tokens in self._tokens.values():
            self._df.update(set(tokens))

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        terms = tokenize(query)
        count = len(self.documents)
        scored: list[tuple[float, ResearchDocument]] = []
        for doc_id, document in self.documents.items():
            length = len(self._tokens[doc_id])
            score = 0.0
            for term in terms:
                frequency = self._tf[doc_id][term]
                if frequency == 0:
                    continue
                idf = math.log(1.0 + (count - self._df[term] + 0.5) / (self._df[term] + 0.5))
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length / self._avg_len)
                score += idf * frequency * (self.k1 + 1.0) / denominator
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].doc_id))
        return [
            SearchHit(doc.doc_id, doc.title, score, doc.text[:320])
            for score, doc in scored[:top_k]
        ]

    def read(self, doc_id: str) -> ResearchDocument:
        try:
            return self.documents[doc_id]
        except KeyError as exc:
            raise KeyError(f"unknown research document: {doc_id}") from exc


class PyseriniResearchIndex:
    def __init__(self, index_path: str | Path) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ModuleNotFoundError as exc:
            raise RuntimeError("Pyserini is required for the remote FullWiki index") from exc
        try:
            self.searcher = LuceneSearcher(str(index_path))
        except Exception as exc:
            raise RuntimeError(f"Lucene index initialization failed: {index_path}") from exc

    @property
    def document_count(self) -> int:
        return int(self.searcher.num_docs)

    def _document(self, doc_id: str) -> ResearchDocument:
        try:
            raw = self.searcher.doc(doc_id)
        except Exception as exc:
            raise RuntimeError(f"Lucene document read failed: {doc_id}") from exc
        if raw is None:
            raise KeyError(f"unknown research document: {doc_id}")
        try:
            payload = json.loads(raw.raw())
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Lucene stored document is invalid JSON: {doc_id}") from exc
        sentences = payload.get("sentences") or [payload.get("contents", payload.get("text", ""))]
        return ResearchDocument(
            doc_id=str(payload.get("id", doc_id)),
            title=str(payload.get("title", doc_id)),
            sentences=tuple(str(value) for value in sentences),
        )

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        try:
            hits = self.searcher.search(query, k=top_k)
        except Exception as exc:
            raise RuntimeError(f"Lucene search failed for query: {query!r}") from exc
        return [
            SearchHit(hit.docid, doc.title, float(hit.score), doc.text[:320])
            for hit in hits
            for doc in [self._document(hit.docid)]
        ]

    def read(self, doc_id: str) -> ResearchDocument:
        return self._document(doc_id)


def load_jsonl_documents(path: str | Path) -> list[ResearchDocument]:
    documents = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            documents.append(ResearchDocument(
                doc_id=str(row["id"]),
                title=str(row["title"]),
                sentences=tuple(str(value) for value in row["sentences"]),
            ))
    return documents


def normalized_title(value: str) -> str:
    return " ".join(value.casefold().split())


def normalized_sentence(value: str) -> str:
    return " ".join(value.casefold().split())


def expected_supporting_sentence(
    example: DeepResearchExample, title: str, sentence_id: int
) -> str | None:
    wanted_title = normalized_title(title)
    for row in example.metadata.get("retrieval_documents", ()):
        if normalized_title(str(row.get("title", ""))) != wanted_title:
            continue
        sentences = row.get("sentences", ())
        if sentence_id < len(sentences):
            return str(sentences[sentence_id])
    return None


def validate_supporting_fact_coverage(
    index: ResearchIndex,
    examples: Iterable[DeepResearchExample],
    *,
    top_k: int = 10,
) -> int:
    checked_facts = 0
    for example in examples:
        for fact in example.supporting_facts:
            hits = index.search(fact.title, top_k=top_k)
            hit = next(
                (item for item in hits if normalized_title(item.title) == normalized_title(fact.title)),
                None,
            )
            if hit is None:
                raise RuntimeError(
                    f"supporting title is absent from retrieval results: {fact.title!r}"
                )
            document = index.read(hit.doc_id)
            if fact.sentence_id >= len(document.sentences):
                raise RuntimeError(
                    f"supporting sentence is out of range: {fact.title!r}#{fact.sentence_id} "
                    f"with {len(document.sentences)} sentences"
                )
            expected = expected_supporting_sentence(
                example, fact.title, fact.sentence_id
            )
            if expected is not None and normalized_sentence(
                document.sentences[fact.sentence_id]
            ) != normalized_sentence(expected):
                raise RuntimeError(
                    f"supporting sentence differs from benchmark context: "
                    f"{fact.title!r}#{fact.sentence_id}"
                )
            checked_facts += 1
    return checked_facts


__all__ = [
    "InMemoryBM25Index",
    "PyseriniResearchIndex",
    "ResearchDocument",
    "ResearchIndex",
    "SearchHit",
    "load_jsonl_documents",
    "normalized_title",
    "normalized_sentence",
    "validate_supporting_fact_coverage",
]
