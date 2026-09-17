from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from agentflow_rl.rewards.advantage import TurnReward
from agentflow_rl.runtime.contracts import TaskName


TRAIN_TASKS = (TaskName.AIME, TaskName.TWOWIKI, TaskName.TACO)


@dataclass(frozen=True)
class DynamicRolloutGroup:
    group_id: str
    task_name: TaskName
    turns: tuple[TurnReward, ...]

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("rollout group requires Planner turns")
        if {item.group_id for item in self.turns} != {self.group_id}:
            raise ValueError("rollout group ID must match every turn prompt group")
        if {item.task_name for item in self.turns} != {self.task_name}:
            raise ValueError("rollout group task must match every turn")


class GroupProvider(Protocol):
    async def generate(
        self,
        *,
        task_name: TaskName,
        count: int,
        round_index: int,
    ) -> tuple[DynamicRolloutGroup, ...]: ...


@dataclass(frozen=True)
class DynamicSamplingMetrics:
    rounds: int
    candidate_groups: int
    kept_groups: int
    dropped_groups: int
    duplicate_groups: int
    per_task_kept: dict[TaskName, int]
    per_task_requested: dict[TaskName, int]


@dataclass(frozen=True)
class DynamicSamplingResult:
    groups: tuple[DynamicRolloutGroup, ...]
    complete: bool
    metrics: DynamicSamplingMetrics


@dataclass(frozen=True)
class DynamicBatchSelection:
    selected_group_ids: tuple[str, ...]
    trainable_keys: tuple[str, ...]
    complete: bool
    candidate_group_count: int
    informative_group_count: int
    filtered_group_count: int
    discarded_informative_group_count: int
    dropped_group_count: int
    target_group_count: int
    per_task_selected: dict[TaskName, int]
    per_task_requested: dict[TaskName, int]


def select_dapo_groups(
    rows: Sequence[TurnReward],
    *,
    trainable_keys: Sequence[str],
    per_task_quota: dict[TaskName, int],
    target_group_count: int,
    expected_trajectories_per_group: int,
) -> DynamicBatchSelection:
    """Filter, accumulate, and align a DAPO prompt-group training batch.

    Community DAPO implementations keep groups with non-zero reward variance,
    continue generation until ``train_batch_size`` qualified groups are
    available, and trim qualified overflow. AgentFlow additionally applies a
    rotating per-task quota so the unified three-task mixture remains balanced.
    """
    if expected_trajectories_per_group <= 1:
        raise ValueError("expected trajectories per group must exceed one")
    if target_group_count <= 0:
        raise ValueError("DAPO target group count must be positive")
    if sum(per_task_quota.values()) != target_group_count:
        raise ValueError("DAPO task quotas must sum to data.train_batch_size")
    by_group: dict[str, list[TurnReward]] = {}
    for row in rows:
        by_group.setdefault(row.group_id, []).append(row)

    def complete_group(group_rows: Sequence[TurnReward]) -> bool:
        trajectories: dict[str, list[TurnReward]] = {}
        for row in group_rows:
            trajectories.setdefault(row.trajectory_id, []).append(row)
        if len(trajectories) != expected_trajectories_per_group:
            return False
        for trajectory_rows in trajectories.values():
            indices = sorted(row.turn_index for row in trajectory_rows)
            if any(not row.valid_for_training for row in trajectory_rows):
                return False
            if indices != list(range(len(indices))):
                return False
        return True

    def terminal_reward_has_variance(group_rows: Sequence[TurnReward]) -> bool:
        by_trajectory: dict[str, float] = {}
        for row in group_rows:
            previous = by_trajectory.setdefault(row.trajectory_id, row.terminal_reward)
            if previous != row.terminal_reward:
                raise ValueError("terminal reward must be constant within a trajectory")
        values = tuple(by_trajectory.values())
        return bool(values) and max(values) != min(values)

    informative = {
        group_id
        for group_id, group_rows in by_group.items()
        if complete_group(group_rows)
        and terminal_reward_has_variance(group_rows)
    }
    selected: list[str] = []
    selected_counts = {task: 0 for task in per_task_quota}
    for group_id, group_rows in by_group.items():
        task = group_rows[0].task_name
        if task not in per_task_quota or group_id not in informative:
            continue
        if selected_counts[task] < per_task_quota[task]:
            selected.append(group_id)
            selected_counts[task] += 1
    complete = selected_counts == per_task_quota
    selected_set = set(selected) if complete else set()
    training_allowed = bool(trainable_keys)
    selected_keys = (
        tuple(
            row.key
            for row in rows
            if row.group_id in selected_set and row.valid_for_training
        )
        if training_allowed
        else ()
    )
    filtered_count = len(by_group) - len(informative)
    discarded_informative = len(informative) - len(selected)
    return DynamicBatchSelection(
        selected_group_ids=tuple(selected) if complete else (),
        trainable_keys=selected_keys,
        complete=complete,
        candidate_group_count=len(by_group),
        informative_group_count=len(informative),
        filtered_group_count=filtered_count,
        discarded_informative_group_count=discarded_informative,
        dropped_group_count=len(by_group) - len(selected_set),
        target_group_count=target_group_count,
        per_task_selected=selected_counts,
        per_task_requested=dict(per_task_quota),
    )


class DynamicSampler:
    def __init__(
        self,
        *,
        target_per_task: int,
        oversampling_factor: int = 2,
        max_rounds: int = 3,
        lambda_process: float = 0.0,
        epsilon: float = 1e-6,
        expected_trajectories_per_group: int = 6,
    ) -> None:
        if (
            target_per_task <= 0
            or oversampling_factor <= 0
            or max_rounds <= 0
            or expected_trajectories_per_group <= 1
        ):
            raise ValueError("dynamic sampling counts must be positive")
        self.target_per_task = target_per_task
        self.oversampling_factor = oversampling_factor
        self.max_rounds = max_rounds
        self.lambda_process = lambda_process
        self.epsilon = epsilon
        self.expected_trajectories_per_group = expected_trajectories_per_group

    def informative(self, group: DynamicRolloutGroup) -> bool:
        trajectories = {row.trajectory_id for row in group.turns}
        if len(trajectories) != self.expected_trajectories_per_group:
            return False
        if any(not row.valid_for_training for row in group.turns):
            return False
        rewards: dict[str, float] = {}
        for row in group.turns:
            previous = rewards.setdefault(row.trajectory_id, row.terminal_reward)
            if previous != row.terminal_reward:
                raise ValueError("terminal reward must be constant within a trajectory")
        values = tuple(rewards.values())
        return bool(values) and max(values) - min(values) > self.epsilon

    async def collect(self, provider: GroupProvider) -> DynamicSamplingResult:
        kept: dict[TaskName, list[DynamicRolloutGroup]] = {
            task: [] for task in TRAIN_TASKS
        }
        seen: set[str] = set()
        candidates = dropped = duplicates = 0
        rounds = 0
        for round_index in range(self.max_rounds):
            deficits = {
                task: self.target_per_task - len(kept[task]) for task in TRAIN_TASKS
            }
            if all(value == 0 for value in deficits.values()):
                break
            rounds = round_index + 1
            for task, deficit in deficits.items():
                if deficit <= 0:
                    continue
                batch = await provider.generate(
                    task_name=task,
                    count=deficit * self.oversampling_factor,
                    round_index=round_index,
                )
                candidates += len(batch)
                for group in batch:
                    if group.task_name is not task:
                        raise ValueError("group provider returned the wrong task")
                    if group.group_id in seen:
                        duplicates += 1
                        continue
                    seen.add(group.group_id)
                    if len(kept[task]) >= self.target_per_task:
                        continue
                    if self.informative(group):
                        kept[task].append(group)
                    else:
                        dropped += 1
        selected = tuple(
            group
            for task in TRAIN_TASKS
            for group in kept[task]
        )
        per_task_kept = {task: len(kept[task]) for task in TRAIN_TASKS}
        requested = {task: self.target_per_task for task in TRAIN_TASKS}
        return DynamicSamplingResult(
            groups=selected,
            complete=per_task_kept == requested,
            metrics=DynamicSamplingMetrics(
                rounds=rounds,
                candidate_groups=candidates,
                kept_groups=len(selected),
                dropped_groups=dropped,
                duplicate_groups=duplicates,
                per_task_kept=per_task_kept,
                per_task_requested=requested,
            ),
        )


__all__ = [
    "DynamicBatchSelection",
    "DynamicRolloutGroup",
    "DynamicSampler",
    "DynamicSamplingMetrics",
    "DynamicSamplingResult",
    "GroupProvider",
    "TRAIN_TASKS",
    "select_dapo_groups",
]
