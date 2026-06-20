from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from gsm8k_utils import extract_gold_answer


DEFAULT_INPUT = Path("../data/gsm8k/main/test-00000-of-00001.parquet")
DEFAULT_OUTPUT_DIR = Path("data")
PROMPT_TEMPLATE = """
You should focus on your responsibility mentioned.

Problem:
{question}"""


def read_parquet_rows(path: Path) -> list[dict[str, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it with "
            "`uv pip install --python /home/north/vllm_test/.venv/bin/python pyarrow`."
        ) from exc

    table = pq.read_table(path)
    return table.to_pylist()


def convert_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    converted = []
    for idx, row in enumerate(rows):
        question = str(row["question"])
        answer = str(row["answer"])
        converted.append(
            {
                "pid": idx,
                "question": question,
                "query": PROMPT_TEMPLATE.format(question=question),
                "answer": answer,
                "gold_answer": extract_gold_answer(answer),
            }
        )
    return converted


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GSM8K parquet test split to AgentFlow JSON.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to GSM8K parquet file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON outputs.")
    parser.add_argument("--smoke-size", type=int, default=20, help="Number of examples for smoke JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = (script_dir / args.input).resolve() if not args.input.is_absolute() else args.input
    output_dir = (script_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir

    rows = read_parquet_rows(input_path)
    converted = convert_rows(rows)
    if len(converted) != 1319:
        raise SystemExit(f"Expected 1319 GSM8K test rows, found {len(converted)} in {input_path}")
    if not 0 < args.smoke_size <= len(converted):
        raise SystemExit(f"--smoke-size must be between 1 and {len(converted)}")

    full_path = output_dir / "gsm8k_test.json"
    smoke_path = output_dir / f"gsm8k_smoke_{args.smoke_size}.json"
    write_json(full_path, converted)
    seed = random.Random(42)
    smoke_rows = seed.sample(converted, args.smoke_size)
    write_json(smoke_path, smoke_rows)

    print(f"Wrote {len(converted)} rows to {full_path}")
    print(f"Wrote {args.smoke_size} rows to {smoke_path}")


if __name__ == "__main__":
    main()
