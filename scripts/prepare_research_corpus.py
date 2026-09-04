from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentflow_rl.tasks.deepresearch.corpus import normalize, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a sentence-aligned Wiki corpus for Pyserini")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/deepresearch/corpus/documents.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for source in args.input:
            for row in rows(source):
                handle.write(json.dumps(normalize(row), ensure_ascii=False) + "\n")
                count += 1
    print(json.dumps({"output": str(args.output), "documents": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
