from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from agentflow_rl.runtime.contracts import TaskName


ADVANTAGE_REVISION = "terminal_prm_rtg_loo_v1"


@dataclass(frozen=True)
class TurnReward:
    key: str
    task_id: str
    task_name: TaskName | str
    trajectory_id: str
    turn_index: int
    terminal_reward: float
    process_score: float | None = None
    valid_for_training: bool = True
    prompt_group_id: str | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "task_name", TaskName(self.task_name))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid task_name: {self.task_name!r}") from error
        if not self.key or not self.task_id or not self.trajectory_id:
            raise ValueError(
                "reward key and task/trajectory identities must be non-empty"
            )
        if self.prompt_group_id == "":
            raise ValueError("prompt_group_id must be non-empty when provided")
        if (
            isinstance(self.turn_index, bool)
            or not isinstance(self.turn_index, int)
            or self.turn_index < 0
        ):
            raise ValueError("turn_index must be a non-negative integer")
        if not isfinite(self.terminal_reward) or not 0.0 <= self.terminal_reward <= 1.0:
            raise ValueError("terminal_reward must be finite and within [0, 1]")
        if self.process_score is not None and (
            not isfinite(self.process_score) or not 0.0 <= self.process_score <= 1.0
        ):
            raise ValueError("process_score must be finite and within [0, 1]")

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
    singleton_group_count: int
    process_complete_group_count: int
    process_fallback_group_count: int
    process_rtg_turn_count: int
    process_missing_turn_count: int
    clipped_turn_count: int


@dataclass(frozen=True)
class AdvantageResult:
    combined: dict[str, float]
    terminal: dict[str, float]
    process: dict[str, float]
    process_return: dict[str, float | None]
    total_return: dict[str, float | None]
    terminal_baseline: dict[str, float]
    process_baseline: dict[str, float | None]
    weighted_process: dict[str, float]
    raw_combined: dict[str, float]
    raw_process_score: dict[str, float | None]
    trajectory_utility: dict[tuple[str, str], float | None]
    process_group_status: dict[str, str]
    process_complete_groups_by_task: dict[TaskName, int]
    process_fallback_groups_by_task: dict[TaskName, int]
    trainable_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    invalid_keys: tuple[str, ...]
    metrics: TurnAdvantageMetrics


def _validate_parameters(
    *,
    lambda_process: float,
    max_advantage: float,
    epsilon: float,
    max_turns: int,
    advantage_revision: str,
) -> None:
    if not isfinite(lambda_process) or lambda_process < 0.0:
        raise ValueError("lambda_process must be finite and non-negative")
    if not isfinite(max_advantage) or max_advantage <= 0.0:
        raise ValueError("max_advantage must be finite and positive")
    if not isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
        raise ValueError("max_turns must be a positive integer")
    if advantage_revision != ADVANTAGE_REVISION:
        raise ValueError(
            f"unsupported advantage revision {advantage_revision!r}; "
            f"expected {ADVANTAGE_REVISION!r}"
        )


def compute_turn_advantages(
    rows: Iterable[TurnReward],
    *,
    max_turns: int,
    lambda_process: float = 0.0,
    max_advantage: float = 5.0,
    epsilon: float = 1e-6,
    advantage_revision: str = ADVANTAGE_REVISION,
) -> AdvantageResult:
    """Construct prompt-group RTG+LOO advantages before actor mini-batching."""
    _validate_parameters(
        lambda_process=lambda_process,
        max_advantage=max_advantage,
        epsilon=epsilon,
        max_turns=max_turns,
        advantage_revision=advantage_revision,
    )
    items = tuple(rows)
    if len({item.key for item in items}) != len(items):
        raise ValueError("turn reward keys must be unique")

    trajectory_rows: dict[tuple[str, str], list[TurnReward]] = {}
    prompt_groups: dict[str, list[tuple[str, str]]] = {}
    for item in items:
        identity = (item.group_id, item.trajectory_id)
        if identity not in trajectory_rows:
            prompt_groups.setdefault(item.group_id, []).append(identity)
        trajectory_rows.setdefault(identity, []).append(item)

    for identity, trajectory in trajectory_rows.items():
        trajectory.sort(key=lambda item: item.turn_index)
        turn_indices = [item.turn_index for item in trajectory]
        if len(turn_indices) != len(set(turn_indices)):
            raise ValueError(f"trajectory {identity!r} has duplicate turn indices")
        if turn_indices != list(range(len(turn_indices))):
            raise ValueError(
                f"trajectory {identity!r} turn indices must be contiguous from zero"
            )
        if len(trajectory) > max_turns:
            raise ValueError(
                f"trajectory {identity!r} has {len(trajectory)} turns; max_turns={max_turns}"
            )
        consistency = {
            "terminal reward": {item.terminal_reward for item in trajectory},
            "group identity": {item.group_id for item in trajectory},
            "task identity": {(item.task_id, item.task_name) for item in trajectory},
            "validity": {item.valid_for_training for item in trajectory},
        }
        inconsistent = [
            name for name, values in consistency.items() if len(values) != 1
        ]
        if inconsistent:
            raise ValueError(f"trajectory turns must share {', '.join(inconsistent)}")

    for group_id, identities in prompt_groups.items():
        task_identities = {
            (
                trajectory_rows[identity][0].task_id,
                trajectory_rows[identity][0].task_name,
            )
            for identity in identities
        }
        if len(task_identities) > 1:
            raise ValueError(
                f"prompt group {group_id!r} contains multiple task identities"
            )

    combined = {item.key: 0.0 for item in items}
    terminal = {item.key: 0.0 for item in items}
    process = {item.key: 0.0 for item in items}
    process_return: dict[str, float | None] = {item.key: None for item in items}
    total_return: dict[str, float | None] = {item.key: None for item in items}
    terminal_baseline = {item.key: 0.0 for item in items}
    process_baseline: dict[str, float | None] = {item.key: None for item in items}
    weighted_process = {item.key: 0.0 for item in items}
    raw_combined = {item.key: 0.0 for item in items}
    raw_process_score = {item.key: item.process_score for item in items}
    trajectory_utility: dict[tuple[str, str], float | None] = {}
    process_group_status: dict[str, str] = {}
    process_complete_groups_by_task = {task: 0 for task in TaskName}
    process_fallback_groups_by_task = {task: 0 for task in TaskName}

    invalid: list[str] = []
    trainable: list[str] = []
    skipped: list[str] = []
    invalid_trajectory_count = 0
    singleton_group_count = 0
    complete_group_count = 0
    fallback_group_count = 0
    process_rtg_turn_count = 0
    process_missing_turn_count = 0
    clipped_turn_count = 0

    for group_id, identities in prompt_groups.items():
        valid_identities = [
            identity
            for identity in identities
            if trajectory_rows[identity][0].valid_for_training
        ]
        invalid_identities = [
            identity
            for identity in identities
            if not trajectory_rows[identity][0].valid_for_training
        ]
        invalid_trajectory_count += len(invalid_identities)
        for identity in invalid_identities:
            invalid.extend(item.key for item in trajectory_rows[identity])

        if not valid_identities:
            process_group_status[group_id] = "no_valid_trajectories"
            continue
        if len(valid_identities) == 1:
            singleton_group_count += 1

        task_name = trajectory_rows[valid_identities[0]][0].task_name
        missing_in_group = sum(
            item.process_score is None
            for identity in valid_identities
            for item in trajectory_rows[identity]
        )
        process_missing_turn_count += missing_in_group
        process_complete = missing_in_group == 0

        rtg_by_trajectory: dict[tuple[str, str], list[float]] = {}
        if process_complete:
            process_group_status[group_id] = "complete"
            complete_group_count += 1
            process_complete_groups_by_task[task_name] += 1
            for identity in valid_identities:
                trajectory = trajectory_rows[identity]
                returns = [0.0] * max_turns
                running = 0.0
                for item in reversed(trajectory):
                    running += float(item.process_score)
                    returns[item.turn_index] = running / max_turns
                rtg_by_trajectory[identity] = returns
                trajectory_utility[identity] = (
                    trajectory[0].terminal_reward + lambda_process * returns[0]
                )
                for item in trajectory:
                    process_return[item.key] = returns[item.turn_index]
                    total_return[item.key] = (
                        item.terminal_reward + lambda_process * returns[item.turn_index]
                    )
                    process_rtg_turn_count += 1
        else:
            process_group_status[group_id] = "terminal_only_missing_process_score"
            fallback_group_count += 1
            process_fallback_groups_by_task[task_name] += 1
            for identity in valid_identities:
                trajectory_utility[identity] = None

        terminal_sum = sum(
            trajectory_rows[identity][0].terminal_reward
            for identity in valid_identities
        )
        process_sums = (
            [
                sum(rtg_by_trajectory[identity][turn] for identity in valid_identities)
                for turn in range(max_turns)
            ]
            if process_complete
            else None
        )
        denominator = len(valid_identities) - 1

        for identity in valid_identities:
            trajectory = trajectory_rows[identity]
            terminal_reward = trajectory[0].terminal_reward
            out_baseline = (
                (terminal_sum - terminal_reward) / denominator
                if denominator > 0
                else 0.0
            )
            out_advantage = terminal_reward - out_baseline
            for item in trajectory:
                terminal[item.key] = out_advantage
                terminal_baseline[item.key] = out_baseline
                proc_advantage = 0.0
                if process_complete:
                    own_return = rtg_by_trajectory[identity][item.turn_index]
                    proc_baseline = (
                        (process_sums[item.turn_index] - own_return) / denominator
                        if denominator > 0
                        else 0.0
                    )
                    process_baseline[item.key] = proc_baseline
                    proc_advantage = own_return - proc_baseline
                process[item.key] = proc_advantage
                weighted_process[item.key] = lambda_process * proc_advantage
                raw = out_advantage + weighted_process[item.key]
                raw_combined[item.key] = raw
                value = min(max(raw, -max_advantage), max_advantage)
                if value != raw:
                    clipped_turn_count += 1
                combined[item.key] = value
                if abs(value) >= epsilon:
                    trainable.append(item.key)
                else:
                    skipped.append(item.key)

    return AdvantageResult(
        combined=combined,
        terminal=terminal,
        process=process,
        process_return=process_return,
        total_return=total_return,
        terminal_baseline=terminal_baseline,
        process_baseline=process_baseline,
        weighted_process=weighted_process,
        raw_combined=raw_combined,
        raw_process_score=raw_process_score,
        trajectory_utility=trajectory_utility,
        process_group_status=process_group_status,
        process_complete_groups_by_task=process_complete_groups_by_task,
        process_fallback_groups_by_task=process_fallback_groups_by_task,
        trainable_keys=tuple(trainable),
        skipped_keys=tuple(skipped),
        invalid_keys=tuple(invalid),
        metrics=TurnAdvantageMetrics(
            prompt_group_count=len(prompt_groups),
            trajectory_count=len(trajectory_rows),
            valid_trajectory_count=len(trajectory_rows) - invalid_trajectory_count,
            invalid_trajectory_count=invalid_trajectory_count,
            turn_count=len(items),
            trainable_turn_count=len(trainable),
            skipped_turn_count=len(skipped),
            invalid_turn_count=len(invalid),
            singleton_group_count=singleton_group_count,
            process_complete_group_count=complete_group_count,
            process_fallback_group_count=fallback_group_count,
            process_rtg_turn_count=process_rtg_turn_count,
            process_missing_turn_count=process_missing_turn_count,
            clipped_turn_count=clipped_turn_count,
        ),
    )


__all__ = [
    "ADVANTAGE_REVISION",
    "AdvantageResult",
    "TurnAdvantageMetrics",
    "TurnReward",
    "compute_turn_advantages",
]
