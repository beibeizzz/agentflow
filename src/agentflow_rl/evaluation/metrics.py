from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from .runner import EvaluationRecord


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    data = list(values)
    if not data or samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap inputs are invalid")
    if len(data) == 1:
        return data[0], data[0]
    generator = random.Random(seed)
    means = sorted(
        sum(generator.choice(data) for _ in data) / len(data) for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    low = means[int(tail * (samples - 1))]
    high = means[int((1.0 - tail) * (samples - 1))]
    return low, high


def _task_means(records: Iterable[EvaluationRecord]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.task_id].append(record.reward)
    return [sum(values) / len(values) for _, values in sorted(grouped.items())]


def clustered_reward_interval(
    records: Iterable[EvaluationRecord],
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap task clusters so repeated seeds remain dependent observations."""
    means = _task_means(records)
    return bootstrap_mean_interval(
        means, confidence=confidence, samples=samples, seed=seed
    )


def aggregate_records(records: Iterable[EvaluationRecord]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.condition.value, record.task_name.value)].append(record)
    result = {}
    for (condition, task), values in sorted(grouped.items()):
        valid = [item for item in values if item.valid]
        rewards = [item.reward for item in valid]
        task_means = _task_means(valid)
        low, high = clustered_reward_interval(valid) if valid else (0.0, 0.0)
        key = f"{condition}/{task}"
        result[key] = {
            "count": float(len(values)),
            "valid_count": float(len(valid)),
            "task_count": float(len({item.task_id for item in values})),
            "valid_task_count": float(len(task_means)),
            "success_rate": (
                sum(item.success for item in valid) / len(valid) if valid else 0.0
            ),
            "mean_reward": (
                sum(task_means) / len(task_means) if task_means else 0.0
            ),
            "reward_ci_low": low,
            "reward_ci_high": high,
            "infrastructure_failure_rate": 1.0 - len(valid) / len(values),
            "mean_planner_turns": (
                sum(item.planner_turns for item in valid) / len(valid) if valid else 0.0
            ),
            "mean_tool_calls": (
                sum(item.tool_calls for item in valid) / len(valid) if valid else 0.0
            ),
            "mean_planner_tokens": (
                sum(item.planner_tokens for item in valid) / len(valid) if valid else 0.0
            ),
            "mean_latency_ms": (
                sum(item.latency_ms for item in valid) / len(valid) if valid else 0.0
            ),
        }
    return result


def paired_transitions(
    baseline: Iterable[EvaluationRecord], candidate: Iterable[EvaluationRecord]
) -> dict[str, int]:
    left = {(item.task_id, item.seed): item for item in baseline if item.valid}
    right = {(item.task_id, item.seed): item for item in candidate if item.valid}
    keys = sorted(left.keys() & right.keys())
    observed = set(left) | set(right)
    return {
        "paired": len(keys),
        "baseline_valid": len(left),
        "candidate_valid": len(right),
        "missing_baseline": len(observed - set(left)),
        "missing_candidate": len(observed - set(right)),
        "failure_to_success": sum(not left[key].success and right[key].success for key in keys),
        "success_to_failure": sum(left[key].success and not right[key].success for key in keys),
        "stable_success": sum(left[key].success and right[key].success for key in keys),
        "stable_failure": sum(not left[key].success and not right[key].success for key in keys),
    }


def compare_condition_records(
    groups: dict[str, Iterable[EvaluationRecord]],
) -> dict[str, object]:
    """Score every condition on the same valid task/seed pairs."""
    indexed: dict[str, dict[tuple[str, int], EvaluationRecord]] = {}
    for name, records in groups.items():
        rows: dict[tuple[str, int], EvaluationRecord] = {}
        for record in records:
            key = (record.task_id, record.seed)
            if key in rows:
                raise ValueError(f"duplicate evaluation sample key in {name}: {key}")
            rows[key] = record
        indexed[name] = rows
    if not indexed:
        raise ValueError("at least one evaluation condition is required")
    observed = set().union(*(set(rows) for rows in indexed.values()))
    common_valid = set.intersection(
        *(
            {key for key, record in rows.items() if record.valid}
            for rows in indexed.values()
        )
    )
    results: dict[str, dict[str, float]] = {}
    for name, rows in indexed.items():
        selected = [rows[key] for key in sorted(common_valid)]
        task_means = _task_means(selected)
        low, high = clustered_reward_interval(selected) if selected else (0.0, 0.0)
        results[name] = {
            "observed_count": float(len(rows)),
            "valid_count": float(sum(record.valid for record in rows.values())),
            "missing_or_invalid_pair_count": float(len(observed - {
                key for key, record in rows.items() if record.valid
            })),
            "common_valid_count": float(len(selected)),
            "common_valid_task_count": float(len(task_means)),
            "common_success_rate": (
                sum(record.success for record in selected) / len(selected)
                if selected
                else 0.0
            ),
            "common_mean_reward": (
                sum(task_means) / len(task_means) if task_means else 0.0
            ),
            "common_reward_ci_low": low,
            "common_reward_ci_high": high,
        }
    return {
        "planned_pair_union_count": len(observed),
        "common_valid_pair_count": len(common_valid),
        "conditions": results,
    }


__all__ = [
    "aggregate_records",
    "bootstrap_mean_interval",
    "clustered_reward_interval",
    "compare_condition_records",
    "paired_transitions",
]
