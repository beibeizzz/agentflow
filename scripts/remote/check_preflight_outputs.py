from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a completed AgentFlow preflight")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    metric_rows = read_jsonl([args.root / "metrics.jsonl"])
    if not metric_rows:
        raise RuntimeError("preflight metrics are empty")
    metrics = [row.get("data", row) for row in metric_rows]
    update_rows = [
        row for row in metrics
        if float(row.get("agentflow/actor_update_skipped", 1.0)) == 0.0
    ]
    if not update_rows:
        raise RuntimeError("preflight produced no trainable actor update")
    for row in update_rows:
        if float(row.get("agentflow/actor_trainable_turn_count", 0.0)) <= 0:
            raise RuntimeError("actor update has no trainable Planner turns")
        for key, value in row.items():
            if ("loss" in key or "grad_norm" in key) and isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"non-finite training metric: {key}={value}")
    if not any("actor/grad_norm" in row for row in update_rows):
        raise RuntimeError("preflight did not record actor/grad_norm")

    rollout_rows = read_jsonl(sorted((args.root / "rollouts").glob("*.jsonl")))
    if not rollout_rows:
        raise RuntimeError("preflight rollout dump is empty")
    if not any("kind=tool_event" in str(row.get("input", "")) for row in rollout_rows):
        raise RuntimeError("rollout dump contains no post-tool Planner turn")

    session_rewards: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rollout_rows:
        key = str(row.get("uid", ""))
        parts = key.rsplit("_", 2)
        if len(parts) == 3:
            session_rewards[parts[0]][parts[1]] = float(row.get("score", 0.0))
    if not any(len(set(rewards.values())) > 1 for rewards in session_rewards.values()):
        raise RuntimeError("preflight contains no query group with reward variance")

    checkpoints = [path for path in args.root.glob("global_step_*") if path.is_dir()]
    if not checkpoints:
        raise RuntimeError("preflight checkpoint is missing")
    print(json.dumps({
        "ready": True,
        "root": str(args.root),
        "metric_rows": len(metric_rows),
        "actor_update_rows": len(update_rows),
        "rollout_rows": len(rollout_rows),
        "checkpoint_count": len(checkpoints),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
