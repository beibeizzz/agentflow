from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Mapping

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if "agentflow" not in sys.modules:
    agentflow_core = PROJECT_DIR / "agentflow" / "agentflow"
    agentflow_package = types.ModuleType("agentflow")
    agentflow_package.__path__ = [str(agentflow_core)]
    agentflow_package.__file__ = str(agentflow_core / "__init__.py")
    sys.modules["agentflow"] = agentflow_package

from try_ticket_agent.data_synthesis.blueprints import (
    GENERATOR_VERSION,
    execute_reference_actions,
    generate_blueprint,
    validate_blueprint_collection,
)


SPLIT_ORDER = ("smoke", "train", "validation", "test")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_blueprint_dataset(
    *, output_dir: Path, seed: int, counts: Mapping[str, int]
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_counts = {split: int(counts.get(split, 0)) for split in SPLIT_ORDER}
    if any(count < 0 for count in normalized_counts.values()):
        raise ValueError("split counts must be non-negative")

    all_items = []
    hashes: dict[str, str] = {}
    reference_failures: list[dict[str, object]] = []
    for split in SPLIT_ORDER:
        items = [
            generate_blueprint(seed=seed, split=split, index=index)
            for index in range(normalized_counts[split])
        ]
        all_items.extend(items)
        content = "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for item in items
        )
        path = output_dir / f"{split}.jsonl"
        _atomic_write(path, content)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        for item in items:
            result = execute_reference_actions(item)
            if not result.success:
                reference_failures.append(
                    {"episode_id": item.episode_id, "failure_codes": result.failure_codes}
                )

    collection_report = validate_blueprint_collection(all_items)
    curriculum = Counter(item.curriculum_mode for item in all_items)
    lookups = Counter(item.lookup_mode for item in all_items)
    reference_report = {
        "ok": not reference_failures,
        "failures": reference_failures,
        "curriculum_counts": dict(sorted(curriculum.items())),
        "lookup_counts": dict(sorted(lookups.items())),
    }
    manifest: dict[str, object] = {
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "counts": normalized_counts,
        "sha256": hashes,
        "collection_validation": collection_report,
        "reference_validation": reference_report,
    }
    if not collection_report["ok"] or not reference_report["ok"]:
        raise ValueError("generated blueprint dataset failed validation")
    _atomic_write(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Ticket episode blueprints.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", type=int, default=32)
    parser.add_argument("--train", type=int, default=2500)
    parser.add_argument("--validation", type=int, default=256)
    parser.add_argument("--test", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_blueprint_dataset(
        output_dir=args.output_dir,
        seed=args.seed,
        counts={split: getattr(args, split) for split in SPLIT_ORDER},
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
