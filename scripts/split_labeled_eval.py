from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentflow_rl.tasks.deepresearch.eval_split import split_labeled_rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("rows", payload.get("examples")))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array or JSONL file: {path}")
    return [dict(row) for row in payload]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create disjoint labeled validation/test subsets by stable hash"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = load_rows(args.input)
    validation, test = split_labeled_rows(
        rows,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    write_jsonl(validation, args.validation_output)
    write_jsonl(test, args.test_output)
    print(json.dumps({
        "input_rows": len(rows),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "seed": args.seed,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
