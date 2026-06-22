from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any

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
    execute_reference_actions,
    validate_blueprint_collection,
)
from try_ticket_agent.data_synthesis.schemas import EpisodeBlueprint
from try_ticket_agent.data_synthesis.validators import validate_candidate
from try_ticket_agent.scripts.generate_blueprints import SPLIT_ORDER


def validate_blueprint_directory(directory: Path) -> dict[str, Any]:
    items: list[EpisodeBlueprint] = []
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    parse_errors: list[dict[str, object]] = []
    for split in SPLIT_ORDER:
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            parse_errors.append({"split": split, "error": "missing_file"})
            counts[split] = 0
            continue
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        split_count = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = EpisodeBlueprint.from_dict(json.loads(line))
                if item.split != split:
                    parse_errors.append(
                        {
                            "split": split,
                            "line": line_number,
                            "error": f"row split is {item.split!r}",
                        }
                    )
                items.append(item)
                split_count += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                parse_errors.append({"split": split, "line": line_number, "error": str(exc)})
        counts[split] = split_count

    collection = validate_blueprint_collection(items)
    reference_failures = []
    for item in items:
        result = execute_reference_actions(item)
        if not result.success:
            reference_failures.append(
                {"episode_id": item.episode_id, "failure_codes": result.failure_codes}
            )
    report: dict[str, Any] = {
        **collection,
        "counts": counts,
        "sha256": hashes,
        "parse_errors": parse_errors,
        "reference_failures": reference_failures,
    }
    report["ok"] = bool(collection["ok"] and not parse_errors and not reference_failures)
    return report


def validate_synthesized_directory(directory: Path) -> dict[str, Any]:
    items: list[EpisodeBlueprint] = []
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    parse_errors: list[dict[str, object]] = []
    candidate_failures: list[dict[str, object]] = []
    for split in SPLIT_ORDER:
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            continue
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        count = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                payload["canonical_request"] = payload["user_request"]
                item = EpisodeBlueprint.from_dict(payload)
                if item.split != split:
                    raise ValueError(f"row split is {item.split!r}")
                validation = validate_candidate(item, item.user_request)
                if not validation.ok:
                    candidate_failures.append(
                        {"episode_id": item.episode_id, "codes": list(validation.codes)}
                    )
                items.append(item)
                count += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                parse_errors.append({"split": split, "line": line_number, "error": str(exc)})
        counts[split] = count

    collection = validate_blueprint_collection(items)
    reference_failures = []
    for item in items:
        result = execute_reference_actions(item)
        if not result.success:
            reference_failures.append(
                {"episode_id": item.episode_id, "failure_codes": result.failure_codes}
            )
    report: dict[str, Any] = {
        **collection,
        "counts": counts,
        "sha256": hashes,
        "parse_errors": parse_errors,
        "candidate_failures": candidate_failures,
        "reference_failures": reference_failures,
    }
    report["ok"] = bool(
        items
        and collection["ok"]
        and not parse_errors
        and not candidate_failures
        and not reference_failures
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deterministic Ticket blueprints.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--blueprints", type=Path)
    inputs.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    report = (
        validate_blueprint_directory(args.blueprints)
        if args.blueprints is not None
        else validate_synthesized_directory(args.dataset)
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
