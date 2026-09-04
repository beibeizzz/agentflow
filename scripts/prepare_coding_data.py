from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentflow_rl.tasks.coding.dataset import (
    deterministic_limit,
    split_verified_rows,
    standardize_taco_row,
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


def write_jsonl(rows: list[dict[str, Any]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def prepare_split(
    path: Path,
    *,
    public_fraction: float,
    max_tests: int,
    max_test_payload_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = []
    filtered = {
        "unsupported_or_invalid": 0,
        "duplicate_question": 0,
        "too_many_tests": 0,
        "test_payload_too_large": 0,
    }
    fingerprints: set[str] = set()
    for row in load_rows(path):
        raw_tests = row.get("input_output", "")
        serialized_tests = (
            raw_tests if isinstance(raw_tests, str)
            else json.dumps(raw_tests, ensure_ascii=False, default=str)
        )
        if len(serialized_tests.encode("utf-8")) > max_test_payload_bytes:
            filtered["test_payload_too_large"] += 1
            continue
        normalized = standardize_taco_row(row, public_fraction=public_fraction)
        if normalized is None:
            filtered["unsupported_or_invalid"] += 1
            continue
        test_count = len(normalized["public_tests"]) + len(normalized["hidden_tests"])
        if test_count > max_tests:
            filtered["too_many_tests"] += 1
            continue
        fingerprint = normalized["metadata"]["question_fingerprint"]
        if fingerprint in fingerprints:
            filtered["duplicate_question"] += 1
            continue
        fingerprints.add(fingerprint)
        output.append(normalized)
    output.sort(key=lambda row: row["episode_id"])
    return output, filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare TACO-Verified Easy/Medium coding data")
    parser.add_argument("--source", type=Path, help="One TACO-Verified source split")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/coding"))
    parser.add_argument("--public-fraction", type=float, default=0.2)
    parser.add_argument("--preflight-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-tests", type=int, default=64)
    parser.add_argument("--max-test-payload-bytes", type=int, default=2_000_000)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    args = parser.parse_args()
    if not 0.0 < args.public_fraction < 1.0:
        parser.error("--public-fraction must be within (0, 1)")
    if args.preflight_size <= 0:
        parser.error("--preflight-size must be positive")
    explicit_sources = (args.train, args.validation, args.test)
    if args.source and any(explicit_sources):
        parser.error("use --source or the complete --train/--validation/--test set")
    if args.source is None and not all(explicit_sources):
        parser.error("provide --source or all of --train/--validation/--test")
    if args.validation_fraction <= 0 or args.test_fraction <= 0:
        parser.error("validation and test fractions must be positive")
    if args.validation_fraction + args.test_fraction >= 1:
        parser.error("validation and test fractions must sum to less than one")
    if args.max_tests < 2 or args.max_test_payload_bytes <= 0:
        parser.error("test count and payload limits must be positive")
    if any(
        value is not None and value <= 0
        for value in (args.train_limit, args.validation_limit, args.test_limit)
    ):
        parser.error("split limits must be positive")

    prepared: dict[str, list[dict[str, Any]]]
    filtered: dict[str, dict[str, int]]
    source_manifest: dict[str, str]
    if args.source is not None:
        source_rows, source_filtered = prepare_split(
            args.source,
            public_fraction=args.public_fraction,
            max_tests=args.max_tests,
            max_test_payload_bytes=args.max_test_payload_bytes,
        )
        prepared = split_verified_rows(
            source_rows,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )
        filtered = {"source": source_filtered}
        source_manifest = {"source": str(args.source)}
    else:
        prepared = {}
        filtered = {}
        source_manifest = {
            "train": str(args.train),
            "validation": str(args.validation),
            "test": str(args.test),
        }
        split_owner: dict[str, str] = {}
        for split, source in zip(("train", "validation", "test"), explicit_sources, strict=True):
            assert source is not None
            prepared[split], filtered[split] = prepare_split(
                source,
                public_fraction=args.public_fraction,
                max_tests=args.max_tests,
                max_test_payload_bytes=args.max_test_payload_bytes,
            )
            for row in prepared[split]:
                fingerprint = row["metadata"]["question_fingerprint"]
                owner = split_owner.get(fingerprint)
                if owner is not None:
                    raise ValueError(f"cross-split coding duplicate: {owner} and {split}: {fingerprint}")
                split_owner[fingerprint] = split
    available_rows = {name: len(rows) for name, rows in prepared.items()}
    limits = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "test": args.test_limit,
    }
    prepared = {
        name: deterministic_limit(rows, limits[name])
        for name, rows in prepared.items()
    }
    for split in ("train", "validation", "test"):
        write_jsonl(prepared[split], args.output_root / f"{split}.jsonl")
    if len(prepared["train"]) < args.preflight_size:
        raise ValueError(f"need at least {args.preflight_size} training rows for preflight")
    easy = [row for row in prepared["train"] if row["difficulty"] == "EASY"]
    medium = [row for row in prepared["train"] if row["difficulty"] == "MEDIUM"]
    write_jsonl(easy, args.output_root / "easy.jsonl")
    write_jsonl(medium, args.output_root / "medium.jsonl")
    write_jsonl(prepared["train"][: args.preflight_size], args.output_root / "preflight.jsonl")
    manifest = {
        "source": source_manifest,
        "available_rows": available_rows,
        "limits": limits,
        "rows": {**{name: len(rows) for name, rows in prepared.items()}, "easy": len(easy), "medium": len(medium)},
        "filtered_or_duplicate": filtered,
        "public_fraction": args.public_fraction,
        "preflight": args.preflight_size,
        "resource_limits": {
            "max_tests": args.max_tests,
            "max_test_payload_bytes": args.max_test_payload_bytes,
        },
        "split": {
            "strategy": "seeded_question_hash" if args.source is not None else "provided_sources",
            "seed": args.split_seed if args.source is not None else None,
            "validation_fraction": args.validation_fraction if args.source is not None else None,
            "test_fraction": args.test_fraction if args.source is not None else None,
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
