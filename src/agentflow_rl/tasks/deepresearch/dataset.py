from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .retrieval import ResearchDocument, normalized_title
from .schemas import DeepResearchExample


HOTPOT_RETRIEVAL_STAGES = {"hotpot_distractor", "hotpot_fullwiki"}


def deterministic_subset(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("subset count must be positive")
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(str(row["episode_id"]).encode("utf-8")).digest(),
    )[:count]


def allows_retrieval_stage_reuse(
    previous_stage: str,
    current_stage: str,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    return (
        {previous_stage, current_stage} == HOTPOT_RETRIEVAL_STAGES
        and previous.get("dataset") == current.get("dataset") == "hotpotqa"
        and previous.get("episode_id") == current.get("episode_id")
        and previous.get("question") == current.get("question")
        and previous.get("answer") == current.get("answer")
        and previous.get("supporting_facts") == current.get("supporting_facts")
    )


def standardize_example(row: dict[str, Any], *, dataset: str) -> dict[str, Any]:
    normalized = dict(row)
    normalized["dataset"] = dataset
    example = DeepResearchExample.from_row(normalized)
    result = example.model_dump(mode="json")
    documents = context_documents(row)
    if documents:
        result["metadata"] = {
            **result.get("metadata", {}),
            "retrieval_documents": [
                {
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "sentences": document.sentences,
                }
                for document in documents
            ],
        }
    return result


def context_documents(row: dict[str, Any]) -> list[ResearchDocument]:
    context = row.get("context", ())
    if isinstance(context, dict):
        context = zip(context.get("title", ()), context.get("sentences", ()), strict=False)
    documents = []
    for index, item in enumerate(context):
        title, sentences = item
        digest = hashlib.sha1(str(title).encode("utf-8")).hexdigest()[:16]
        documents.append(ResearchDocument(
            doc_id=f"{digest}-{index}",
            title=str(title),
            sentences=tuple(str(sentence) for sentence in sentences),
        ))
    return documents


def deduplicate_documents(documents: Iterable[ResearchDocument]) -> list[ResearchDocument]:
    by_title: dict[str, ResearchDocument] = {}
    for document in documents:
        key = normalized_title(document.title)
        current = by_title.get(key)
        if current is None or len(document.text) > len(current.text):
            by_title[key] = document
    return sorted(by_title.values(), key=lambda document: document.title)


def write_documents_jsonl(documents: Iterable[ResearchDocument], target: str | Path) -> int:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(
            {
                "id": document.doc_id,
                "title": document.title,
                "sentences": document.sentences,
                "contents": f"{document.title} {document.text}",
            },
            ensure_ascii=False,
        )
        for document in documents
    ]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def source_context_corpora(
    rows_by_dataset: dict[str, Iterable[ResearchDocument]],
    root: str | Path,
) -> dict[str, int]:
    output_root = Path(root)
    counts: dict[str, int] = {}
    for dataset, documents in sorted(rows_by_dataset.items()):
        counts[dataset] = write_documents_jsonl(
            deduplicate_documents(documents),
            output_root / dataset / "documents.jsonl",
        )
    return counts


__all__ = [
    "allows_retrieval_stage_reuse",
    "context_documents",
    "deduplicate_documents",
    "deterministic_subset",
    "standardize_example",
    "source_context_corpora",
    "write_documents_jsonl",
]
