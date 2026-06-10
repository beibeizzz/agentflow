from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parquet_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        pid = row["pid"]
        converted.append(
            {
                "id": f"gsm8k-{pid}",
                "question": str(row["question"]),
                "result": str(row["gold_answer"]),
                "extra_info": {
                    "idx": pid,
                    "source": "gsm8k",
                    "answer": str(row["answer"]),
                    "gold_answer": str(row["gold_answer"]),
                },
            }
        )
    return converted


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise SystemExit(f"Expected a JSON list in {path}")
    return payload


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyarrow is required to write verl parquet data.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def resolve_cli_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GSM8K JSON rows to Flow-GRPO parquet rows.")
    parser.add_argument("--input", type=Path, default=Path("../data/gsm8k_smoke_50.json"))
    parser.add_argument("--train-output", type=Path, default=Path("data/train/gsm8k_smoke_train.parquet"))
    parser.add_argument("--val-output", type=Path, default=Path("data/val/gsm8k_smoke_val.parquet"))
    parser.add_argument("--val-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = resolve_cli_path(args.input)
    train_output = resolve_cli_path(args.train_output)
    val_output = resolve_cli_path(args.val_output)

    rows = build_parquet_rows(load_json_rows(input_path))
    if args.val_size < 1 or args.val_size >= len(rows):
        raise SystemExit(f"--val-size must be between 1 and {len(rows) - 1}")

    write_parquet(train_output, rows[:-args.val_size])
    write_parquet(val_output, rows[-args.val_size :])
    print(f"Wrote {len(rows) - args.val_size} train rows to {train_output}")
    print(f"Wrote {args.val_size} val rows to {val_output}")


if __name__ == "__main__":
    main()
