from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentflow_rl.tasks.deepresearch.retrieval import (
    PyseriniResearchIndex,
    validate_supporting_fact_coverage,
)
from agentflow_rl.tasks.deepresearch.schemas import DeepResearchExample


def probe_query(paths: list[Path]) -> str:
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                example = DeepResearchExample.from_row(json.loads(line))
                if example.supporting_facts:
                    return example.supporting_facts[0].title
                return example.question
    return "United States"


def check_supporting_fact_coverage(
    index: PyseriniResearchIndex,
    paths: list[Path],
    *,
    max_examples: int,
    top_k: int,
) -> tuple[int, int]:
    checked_examples = 0
    checked_facts = 0
    for path in paths:
        path_examples = 0
        examples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            example = DeepResearchExample.from_row(json.loads(line))
            examples.append(example)
            checked_examples += 1
            path_examples += 1
            if path_examples >= max_examples:
                break
        checked_facts += validate_supporting_fact_coverage(
            index, examples, top_k=top_k
        )
    return checked_examples, checked_facts


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Lucene search and document retrieval")
    parser.add_argument("--index", default="data/indexes/hotpotqa")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--examples", type=Path, action="append", default=[])
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--min-documents", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.top_k <= 0 or args.max_examples <= 0 or args.min_documents <= 0:
        parser.error("--top-k, --max-examples, and --min-documents must be positive")

    index = PyseriniResearchIndex(args.index)
    if index.document_count < args.min_documents:
        raise RuntimeError(
            f"research index has {index.document_count} documents; "
            f"this backend gate requires at least {args.min_documents}"
        )
    query = args.query or probe_query(args.examples)
    hits = index.search(query, top_k=args.top_k)
    if not hits:
        raise RuntimeError(f"research index returned no results for {query!r}")
    document = index.read(hits[0].doc_id)
    if not document.sentences:
        raise RuntimeError(f"research document has no sentences: {document.doc_id}")
    checked_examples, checked_facts = check_supporting_fact_coverage(
        index,
        args.examples,
        max_examples=args.max_examples,
        top_k=max(10, args.top_k),
    )
    print(json.dumps({
        "ready": True,
        "index": args.index,
        "query": query,
        "hits": len(hits),
        "first_doc_id": document.doc_id,
        "first_title": document.title,
        "sentence_count": len(document.sentences),
        "document_count": index.document_count,
        "checked_examples": checked_examples,
        "checked_supporting_facts": checked_facts,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
