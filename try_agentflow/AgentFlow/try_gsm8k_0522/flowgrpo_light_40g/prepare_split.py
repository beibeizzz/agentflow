from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("try_gsm8k_0522/data/gsm8k_test.json")
DEFAULT_TRAIN_OUTPUT = Path("try_gsm8k_0522/data/gsm8k_test_train_1000.json")
DEFAULT_EVAL_OUTPUT = Path("try_gsm8k_0522/data/gsm8k_test_eval_rest.json")


def split_rows(rows: list[dict[str, Any]], train_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if train_size >= len(rows):
        raise ValueError(f"train_size ({train_size}) must be smaller than row count ({len(rows)})")
    return rows[:train_size], rows[train_size:]


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected a JSON list in {path}")
    return rows


def write_json_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split converted GSM8K JSON into train/eval files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--train-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_json_rows(args.input)
    train_rows, eval_rows = split_rows(rows, args.train_size)
    write_json_rows(args.train_output, train_rows)
    write_json_rows(args.eval_output, eval_rows)
    print(f"Wrote {len(train_rows)} train rows to {args.train_output}")
    print(f"Wrote {len(eval_rows)} eval rows to {args.eval_output}")


if __name__ == "__main__":
    main()

