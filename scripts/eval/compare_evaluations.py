from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentflow_rl.evaluation import (
    ACTIVE_EVALUATION_CONDITIONS,
    EvaluationCondition,
    EvaluationRecord,
    build_run_manifest,
    compare_condition_records,
    paired_transitions,
    read_run_manifest,
    read_shared_manifest,
)


def read_records(path: Path) -> list[EvaluationRecord]:
    return [
        EvaluationRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the active E0-E3 evaluation matrix")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups: dict[str, list[EvaluationRecord]] = {}
    shared_hash: str | None = None
    for directory in args.run_dir:
        shared = read_shared_manifest(directory / "evaluation_shared_manifest.json")
        run = read_run_manifest(directory / "evaluation_run_manifest.json")
        records = read_records(directory / "eval_samples.jsonl")
        expected = build_run_manifest(
            shared,
            condition=run.condition,
            planner_revision=run.planner_revision,
            sandbox_service_url=str(run.sandbox_service_url),
            records=records,
        )
        if run != expected:
            raise ValueError(f"evaluation artifacts fail integrity checks: {directory}")
        if run.planner_revision != shared.condition_planner_revisions[run.condition]:
            raise ValueError(f"Planner revision differs from shared manifest: {directory}")
        if shared_hash is None:
            shared_hash = run.shared_manifest_sha256
        elif run.shared_manifest_sha256 != shared_hash:
            raise ValueError("evaluation conditions use different shared manifests")
        if run.condition.value in groups:
            raise ValueError(f"duplicate evaluation condition: {run.condition.value}")
        groups[run.condition.value] = records
    expected_conditions = {item.value for item in ACTIVE_EVALUATION_CONDITIONS}
    if set(groups) != expected_conditions:
        missing = sorted(expected_conditions - set(groups))
        raise ValueError(f"complete active E0-E3 runs are required; missing={missing}")
    direct = groups[EvaluationCondition.DIRECT.value]
    report = compare_condition_records(groups)
    report["shared_manifest_sha256"] = shared_hash
    report["transitions_from_E0"] = {
        name: paired_transitions(direct, records)
        for name, records in groups.items()
        if name != EvaluationCondition.DIRECT.value
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
