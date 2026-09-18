from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Iterable

from agentflow_rl.runtime.contracts import TaskName


ADVANTAGE_REVISION = "terminal_broadcast_mixed_turn_grpo_v1"


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
    process_value_turn_count: int
    process_missing_turn_count: int
    clipped_turn_count: int


@dataclass(frozen=True)
class AdvantageResult:
    combined: dict[str, float]
    terminal: dict[str, float]
    process: dict[str, float]
    process_value: dict[str, float | None]
    mixed_reward: dict[str, float | None]
    terminal_mean: dict[str, float]
    process_mean: dict[str, float | None]
    weighted_process: dict[str, float]
    raw_combined: dict[str, float]
    raw_process_score: dict[str, float | None]
    normalization_mean: dict[str, float]
    normalization_std: dict[str, float]
    normalization_scope: dict[str, str]
    terminal_reference: dict[str, float]
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
    """Normalize complete prompt groups before selection and actor mini-batching.

    E2 (lambda_process=0) standardizes one reward per trajectory, then
    broadcasts. E3 standardizes R + lambda*p over all real valid turns.
    Any missing score in an E3 group triggers the exact E2 fallback.
    Population standard deviation is used; std <= epsilon produces zeros.
    Terminal/process components in E3 share the mixed-reward denominator.
    terminal_reference always records the counterfactual E2 advantage.
    """
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
    process_value: dict[str, float | None] = {item.key: None for item in items}
    mixed_reward: dict[str, float | None] = {item.key: None for item in items}
    terminal_mean = {item.key: 0.0 for item in items}
    process_mean: dict[str, float | None] = {item.key: None for item in items}
    weighted_process = {item.key: 0.0 for item in items}
    raw_combined = {item.key: 0.0 for item in items}
    raw_process_score = {item.key: item.process_score for item in items}
    normalization_mean = {item.key: 0.0 for item in items}
    normalization_std = {item.key: 0.0 for item in items}
    normalization_scope = {item.key: "invalid" for item in items}
    terminal_reference = {item.key: 0.0 for item in items}
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
    process_value_turn_count = 0
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

        use_process = lambda_process > 0.0 and process_complete
        if lambda_process == 0.0:
            process_group_status[group_id] = "disabled"
        elif process_complete:
            process_group_status[group_id] = "complete"
            complete_group_count += 1
            process_complete_groups_by_task[task_name] += 1
        else:
            process_group_status[group_id] = "terminal_only_missing_process_score"
            fallback_group_count += 1
            process_fallback_groups_by_task[task_name] += 1

        group_rows = [item for identity in valid_identities for item in trajectory_rows[identity]]
        rewards = [trajectory_rows[identity][0].terminal_reward for identity in valid_identities]

        def moments(values: list[float]) -> tuple[float, float]:
            mean = fsum(values) / len(values)
            return mean, sqrt(fsum((value - mean) ** 2 for value in values) / len(values))

        reward_mean, reward_std = moments(rewards)
        if use_process:
            values = [item.terminal_reward + lambda_process * float(item.process_score) for item in group_rows]
            group_mean, group_std = moments(values)
            terminal_center = fsum(item.terminal_reward for item in group_rows) / len(group_rows)
            process_center = fsum(float(item.process_score) for item in group_rows) / len(group_rows)
        else:
            group_mean, group_std = reward_mean, reward_std
            terminal_center, process_center = reward_mean, 0.0

        for identity in valid_identities:
            for item in trajectory_rows[identity]:
                terminal_reference[item.key] = (
                    (item.terminal_reward - reward_mean) / reward_std if reward_std > epsilon else 0.0
                )
                normalization_mean[item.key] = group_mean
                normalization_std[item.key] = group_std
                normalization_scope[item.key] = "all_real_turns" if use_process else "trajectories"
                terminal_mean[item.key] = terminal_center
                own_process = float(item.process_score) if use_process else 0.0
                mixed_reward[item.key] = item.terminal_reward + lambda_process * own_process
                if use_process:
                    process_value[item.key] = own_process
                    process_mean[item.key] = process_center
                    process_value_turn_count += 1
                terminal[item.key] = (item.terminal_reward - terminal_center) / group_std if group_std > epsilon else 0.0
                process[item.key] = (own_process - process_center) / group_std if use_process and group_std > epsilon else 0.0
                weighted_process[item.key] = lambda_process * process[item.key]
                raw = (mixed_reward[item.key] - group_mean) / group_std if group_std > epsilon else 0.0
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
        process_value=process_value,
        mixed_reward=mixed_reward,
        terminal_mean=terminal_mean,
        process_mean=process_mean,
        weighted_process=weighted_process,
        raw_combined=raw_combined,
        raw_process_score=raw_process_score,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        normalization_scope=normalization_scope,
        terminal_reference=terminal_reference,
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
            process_value_turn_count=process_value_turn_count,
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
