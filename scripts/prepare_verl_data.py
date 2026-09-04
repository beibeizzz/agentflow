from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentflow_rl.verl.data import convert_file


DATASETS = {
    "ticket": ("train", "validation", "test", "smoke"),
    "gsm8k": ("train", "test", "smoke"),
    "deepresearch": (
        "hotpot_distractor",
        "2wiki",
        "hotpot_fullwiki",
        "hotpot_validation",
        "hotpot_test",
        "2wiki_validation",
        "2wiki_test",
        "preflight",
    ),
    "coding": ("train", "easy", "medium", "validation", "test", "preflight"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert AgentFlow data to veRL parquet")
    parser.add_argument("--task", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    root = Path(args.data_root)
    tasks = DATASETS if args.task == "all" else {args.task: DATASETS[args.task]}
    manifest = []
    for task, splits in tasks.items():
        suffix = ".json" if task == "gsm8k" else ".jsonl"
        for split in splits:
            manifest.append(convert_file(
                task,
                root / task / f"{split}{suffix}",
                root / "verl" / task / f"{split}.parquet",
            ))
    manifest_path = root / "verl" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
