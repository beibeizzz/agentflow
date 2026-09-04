from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentflow_rl.tasks.deepresearch.dataset import (
    allows_retrieval_stage_reuse,
    context_documents,
    deduplicate_documents,
    deterministic_subset,
    source_context_corpora,
    standardize_example,
    write_documents_jsonl,
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("rows", payload.get("examples")))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array or JSONL file: {path}")
    return [dict(row) for row in payload]


def parse_source(value: str) -> tuple[str, str, Path]:
    try:
        descriptor, raw_path = value.split("=", 1)
        dataset, output_split = descriptor.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source must use DATASET:OUTPUT_SPLIT=PATH") from exc
    if dataset not in {"hotpotqa", "2wiki"}:
        raise argparse.ArgumentTypeError("dataset must be hotpotqa or 2wiki")
    return dataset, output_split, Path(raw_path)


def write_jsonl(rows: list[dict[str, Any]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def parse_limit(value: str) -> tuple[str, int]:
    try:
        split, raw_count = value.split("=", 1)
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must use OUTPUT_SPLIT=COUNT") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("limit count must be positive")
    return split, count


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA/2Wiki data and a BM25 corpus")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Repeat DATASET:OUTPUT_SPLIT=PATH, for example hotpotqa:hotpot_distractor=raw.json",
    )
    parser.add_argument("--output-root", default="data/deepresearch")
    parser.add_argument("--preflight-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        action="append",
        type=parse_limit,
        default=[],
        help="Repeat OUTPUT_SPLIT=COUNT for deterministic remote subsets",
    )
    args = parser.parse_args()
    if args.preflight_size <= 0:
        parser.error("--preflight-size must be positive")

    root = Path(args.output_root)
    limits = dict(args.limit)
    documents = []
    documents_by_dataset: dict[str, list] = {}
    prepared: dict[str, list[dict[str, Any]]] = {}
    available_rows: dict[str, int] = {}
    seen_ids: dict[str, tuple[str, dict[str, Any]]] = {}
    seen_questions: dict[str, tuple[str, dict[str, Any]]] = {}
    for raw_source in args.source:
        dataset, output_split, path = parse_source(raw_source)
        rows = load_rows(path)
        normalized_rows = []
        for row in rows:
            normalized = standardize_example(row, dataset=dataset)
            identity = normalized["episode_id"]
            prior_id = seen_ids.get(identity)
            if prior_id is not None and not allows_retrieval_stage_reuse(
                prior_id[0], output_split, prior_id[1], normalized
            ):
                raise ValueError(f"duplicate DeepResearch episode_id: {identity}")
            seen_ids.setdefault(identity, (output_split, normalized))
            question_key = " ".join(normalized["question"].lower().split())
            owner = seen_questions.get(question_key)
            if owner is not None and not allows_retrieval_stage_reuse(
                owner[0], output_split, owner[1], normalized
            ):
                raise ValueError(
                    f"cross-split DeepResearch duplicate question: {owner[0]} and {output_split}"
                )
            seen_questions.setdefault(question_key, (output_split, normalized))
            normalized["metadata"] = {
                **normalized.get("metadata", {}),
                "curriculum_stage": output_split,
                "source_path": str(path),
            }
            normalized_rows.append(normalized)
            row_documents = context_documents(row)
            documents.extend(row_documents)
            documents_by_dataset.setdefault(dataset, []).extend(row_documents)
        available_rows[output_split] = len(normalized_rows)
        if output_split in limits:
            normalized_rows = deterministic_subset(
                normalized_rows, limits[output_split]
            )
        prepared[output_split] = normalized_rows
        write_jsonl(normalized_rows, root / f"{output_split}.jsonl")

    training_order = ["hotpot_distractor", "2wiki", "hotpot_fullwiki"]
    unknown_limits = set(limits) - set(prepared)
    if unknown_limits:
        raise ValueError(f"limits reference unknown output splits: {sorted(unknown_limits)}")
    training_rows = [row for split in training_order for row in prepared.get(split, ())]
    if len(training_rows) < args.preflight_size:
        raise ValueError(f"need at least {args.preflight_size} training rows for preflight")
    write_jsonl(training_rows[: args.preflight_size], root / "preflight.jsonl")
    count = write_documents_jsonl(
        deduplicate_documents(documents), root / "corpus" / "documents.jsonl"
    )
    context_corpora = source_context_corpora(
        documents_by_dataset, root / "context_corpus"
    )
    manifest = {
        "sources": args.source,
        "available_rows": available_rows,
        "limits": limits,
        "splits": {name: len(rows) for name, rows in prepared.items()},
        "preflight": args.preflight_size,
        "corpus_documents": count,
        "context_corpus_documents": context_corpora,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
