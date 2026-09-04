from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one Hugging Face dataset split as JSONL")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, args.name, split=args.split, streaming=args.streaming)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
            count += 1
            if args.limit is not None and count >= args.limit:
                break
    print(json.dumps({"dataset": args.dataset, "name": args.name, "split": args.split, "rows": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
