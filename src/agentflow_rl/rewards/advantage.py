from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from agentflow_rl.runtime.contracts import TaskName


@dataclass(frozen=True)
class TurnReward:
    key: str
    task_id: str
    task_name: TaskName
    trajectory_id: str
    turn_index: int
    terminal_reward: float
    process_score: float | None = None
    valid_for_training: bool = True
    prompt_group_id: str | None = None

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")
        if not 0.0 <= self.terminal_reward <= 1.0:
            raise ValueError("terminal_reward must be within [0, 1]")
        if self.process_score is not None and not 0.0 <= self.process_score <= 1.0:
            raise ValueError("process_score must be within [0, 1]")

    @property
    def group_id(self) -> str:
        return self.prompt_group_id or self.task_id


@dataclass(frozen=True)
class TurnAdvantageMetrics:
    prompt_group_count: int
    trajectory_count: int
    valid_trajectory_count: int
    invalid_trajectory_count: int
    turn_count: int
    trainable_turn_count: int
    skipped_turn_count: int
    invalid_turn_count: int
    terminal_zero_variance_group_count: int
    process_supported_cell_count: int
    process_unsupported_cell_count: int
    process_missing_turn_count: int
    clipped_turn_count: int


@dataclass(frozen=True)
class AdvantageResult:
    combined: dict[str, float]
    terminal: dict[str, float]
    process: dict[str, float]
    trainable_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    invalid_keys: tuple[str, ...]
    metrics: TurnAdvantageMetrics


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, sqrt(variance)


def _normalize(values: list[float], epsilon: float) -> list[float] | None:
    if len(values) < 2:
        return None
    mean, std = _mean_std(values)
    if std < epsilon:
        return None
    return [(value - mean) / (std + epsilon) for value in values]


def compute_turn_advantages(
    rows: Iterable[TurnReward],
    *,
    lambda_process: float = 0.0,
    max_advantage: float = 5.0,
    epsilon: float = 1e-6,
) -> AdvantageResult:
    if lambda_process < 0.0:
        raise ValueError("lambda_process must be non-negative")
    if max_advantage <= 0.0 or epsilon <= 0.0:
        raise ValueError("max_advantage and epsilon must be positive")
    items = tuple(rows)
    if len({item.key for item in items}) != len(items):
        raise ValueError("turn reward keys must be unique")

    trajectories: dict[tuple[str, str], list[TurnReward]] = {}
    for item in items:
        trajectories.setdefault((item.task_id, item.trajectory_id), []).append(item)
    for trajectory_rows in trajectories.values():
        rewards = {item.terminal_reward for item in trajectory_rows}
        validity = {item.valid_for_training for item in trajectory_rows}
        if len(rewards) != 1 or len(validity) != 1:
            raise ValueError("trajectory turns must share terminal reward and validity")

    terminal_by_trajectory: dict[tuple[str, str], float] = {}
    terminal_zero_variance = 0
    prompt_groups: dict[str, list[tuple[tuple[str, str], TurnReward]]] = {}
    invalid_trajectories = 0
    for trajectory, trajectory_rows in trajectories.items():
        representative = trajectory_rows[0]
        prompt_groups.setdefault(representative.group_id, []).append((trajectory, representative))

    for prompt_id, group in prompt_groups.items():
        valid = [item for item in group if item[1].valid_for_training]
        invalid_trajectories += len(group) - len(valid)
        normalized = _normalize([item.terminal_reward for _, item in valid], epsilon)
        if normalized is None:
            terminal_zero_variance += 1
            normalized = [0.0] * len(valid)
        for (trajectory, _), advantage in zip(valid, normalized, strict=True):
            terminal_by_trajectory[trajectory] = advantage

    process_by_key = {item.key: 0.0 for item in items}
    process_cells: dict[tuple[str, int], list[TurnReward]] = {}
    process_missing = 0
    for item in items:
        if not item.valid_for_training:
            continue
        if item.process_score is None:
            process_missing += 1
            continue
        process_cells.setdefault((item.group_id, item.turn_index), []).append(item)

    supported_cells = 0
    unsupported_cells = 0
    for cell in process_cells.values():
        normalized = _normalize([float(item.process_score) for item in cell], epsilon)
        if normalized is None:
            unsupported_cells += 1
            continue
        supported_cells += 1
        for item, advantage in zip(cell, normalized, strict=True):
            process_by_key[item.key] = advantage

    terminal_by_key: dict[str, float] = {}
    combined: dict[str, float] = {}
    trainable: list[str] = []
    skipped: list[str] = []
    invalid: list[str] = []
    clipped = 0
    for item in items:
        if not item.valid_for_training:
            terminal_by_key[item.key] = 0.0
            combined[item.key] = 0.0
            invalid.append(item.key)
            continue
        terminal_value = terminal_by_trajectory.get((item.task_id, item.trajectory_id), 0.0)
        terminal_by_key[item.key] = terminal_value
        raw = terminal_value + lambda_process * process_by_key[item.key]
        value = min(max(raw, -max_advantage), max_advantage)
        clipped += int(value != raw)
        combined[item.key] = value
        if abs(value) >= epsilon:
            trainable.append(item.key)
        else:
            skipped.append(item.key)

    return AdvantageResult(
        combined=combined,
        terminal=terminal_by_key,
        process=process_by_key,
        trainable_keys=tuple(trainable),
        skipped_keys=tuple(skipped),
        invalid_keys=tuple(invalid),
        metrics=TurnAdvantageMetrics(
            prompt_group_count=len(prompt_groups),
            trajectory_count=len(trajectories),
            valid_trajectory_count=len(trajectories) - invalid_trajectories,
            invalid_trajectory_count=invalid_trajectories,
            turn_count=len(items),
            trainable_turn_count=len(trainable),
            skipped_turn_count=len(skipped),
            invalid_turn_count=len(invalid),
            terminal_zero_variance_group_count=terminal_zero_variance,
            process_supported_cell_count=supported_cells,
            process_unsupported_cell_count=unsupported_cells,
            process_missing_turn_count=process_missing,
            clipped_turn_count=clipped,
        ),
    )


__all__ = [
    "AdvantageResult",
    "TurnAdvantageMetrics",
    "TurnReward",
    "compute_turn_advantages",
]
