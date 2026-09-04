from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class TrajectoryTurn:
    key: str
    reward: float
    valid_for_training: bool = True

    @property
    def identity(self) -> tuple[str, str, int]:
        uid, session_id, turn_index = self.key.rsplit("_", 2)
        if not session_id.isdigit() or not turn_index.isdigit():
            raise ValueError(f"invalid veRL trajectory-turn key: {self.key!r}")
        return uid, session_id, int(turn_index)


@dataclass(frozen=True)
class AdvantageMetrics:
    group_count: int
    trajectory_count: int
    valid_trajectory_count: int
    invalid_trajectory_count: int
    turn_count: int
    trainable_turn_count: int
    invalid_turn_count: int
    skipped_turn_count: int
    zero_variance_group_count: int
    skipped_group_count: int
    reward_mean: float | None
    reward_std: float | None
    advantage_mean: float | None
    advantage_std: float | None


@dataclass(frozen=True)
class AdvantageResult:
    advantages: dict[str, float]
    trainable_keys: tuple[str, ...]
    invalid_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    metrics: AdvantageMetrics


def _population(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, sqrt(variance)


def normalize_trajectory_turns(
    rows: Iterable[TrajectoryTurn], *, epsilon: float = 1e-6
) -> AdvantageResult:
    """Normalize final trajectory rewards per query, then broadcast to turns."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    items = tuple(rows)
    sessions: dict[tuple[str, str], list[tuple[int, TrajectoryTurn]]] = {}
    for item in items:
        uid, session_id, turn_index = item.identity
        sessions.setdefault((uid, session_id), []).append((turn_index, item))

    final_rows: dict[tuple[str, str], TrajectoryTurn] = {
        session: max(turns, key=lambda pair: pair[0])[1]
        for session, turns in sessions.items()
    }
    groups: dict[str, list[tuple[tuple[str, str], TrajectoryTurn]]] = {}
    for session, final in final_rows.items():
        groups.setdefault(session[0], []).append((session, final))

    session_advantages: dict[tuple[str, str], float] = {
        session: 0.0 for session in sessions
    }
    trainable_sessions: set[tuple[str, str]] = set()
    invalid_sessions: set[tuple[str, str]] = set()
    skipped_sessions: set[tuple[str, str]] = set()
    valid_rewards: list[float] = []
    valid_advantages: list[float] = []
    zero_variance_groups = 0
    skipped_groups = 0
    valid_trajectories = 0
    invalid_trajectories = 0

    for group in groups.values():
        valid = [(session, final) for session, final in group if final.valid_for_training]
        invalid_sessions.update(session for session, final in group if not final.valid_for_training)
        invalid_trajectories += len(group) - len(valid)
        valid_trajectories += len(valid)
        rewards = [float(final.reward) for _, final in valid]
        valid_rewards.extend(rewards)
        if len(valid) < 2:
            skipped_groups += 1
            skipped_sessions.update(session for session, _ in valid)
            valid_advantages.extend(0.0 for _ in valid)
            continue
        mean, std = _population(rewards)
        assert mean is not None and std is not None
        if std < epsilon:
            zero_variance_groups += 1
            skipped_groups += 1
            skipped_sessions.update(session for session, _ in valid)
            advantages = [0.0] * len(valid)
        else:
            trainable_sessions.update(session for session, _ in valid)
            advantages = [(reward - mean) / std for reward in rewards]
        for (session, _), advantage in zip(valid, advantages, strict=True):
            session_advantages[session] = advantage
        valid_advantages.extend(advantages)

    advantages = {
        item.key: session_advantages[(item.identity[0], item.identity[1])]
        if item.valid_for_training else 0.0
        for item in items
    }
    trainable_keys = tuple(
        item.key for item in items
        if (item.identity[0], item.identity[1]) in trainable_sessions
    )
    invalid_keys = tuple(
        item.key for item in items
        if (item.identity[0], item.identity[1]) in invalid_sessions
    )
    skipped_keys = tuple(
        item.key for item in items
        if (item.identity[0], item.identity[1]) in skipped_sessions
    )
    reward_mean, reward_std = _population(valid_rewards)
    advantage_mean, advantage_std = _population(valid_advantages)
    return AdvantageResult(
        advantages=advantages,
        trainable_keys=trainable_keys,
        invalid_keys=invalid_keys,
        skipped_keys=skipped_keys,
        metrics=AdvantageMetrics(
            group_count=len(groups),
            trajectory_count=len(sessions),
            valid_trajectory_count=valid_trajectories,
            invalid_trajectory_count=invalid_trajectories,
            turn_count=len(items),
            trainable_turn_count=len(trainable_keys),
            invalid_turn_count=len(invalid_keys),
            skipped_turn_count=len(skipped_keys),
            zero_variance_group_count=zero_variance_groups,
            skipped_group_count=skipped_groups,
            reward_mean=reward_mean,
            reward_std=reward_std,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
        ),
    )
