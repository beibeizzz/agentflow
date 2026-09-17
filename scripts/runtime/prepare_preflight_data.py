from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from agentflow_rl.integrations.preflight import TRAIN_TASKS, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=32)
    args = parser.parse_args()
    if args.per_task < 32:
        raise ValueError("real preflight requires at least 32 prompts per task")

    table = pq.read_table(args.input)
    rows = table.to_pylist()
    selected_indices = []
    counts = {task.value: 0 for task in TRAIN_TASKS}
    ordered = sorted(
        enumerate(rows),
        key=lambda pair: (
            str(pair[1]["extra_info"]["task_name"]),
            str(pair[1].get("uid", pair[1]["extra_info"].get("task_id", ""))),
        ),
    )
    for index, row in ordered:
        task = str(row["extra_info"]["task_name"])
        if task in counts and counts[task] < args.per_task:
            selected_indices.append(index)
            counts[task] += 1
    if any(value != args.per_task for value in counts.values()):
        raise ValueError(f"insufficient preflight data: {counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(selected_indices), args.output)
    payload = {
        "schema_version": 1,
        "source_path": str(args.input),
        "source_sha256": sha256_file(args.input),
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
        "prompt_counts": counts,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
