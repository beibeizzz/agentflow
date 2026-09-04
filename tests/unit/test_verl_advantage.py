from __future__ import annotations

from math import isclose

from agentflow_rl.verl.advantage import TrajectoryTurn, normalize_trajectory_turns


def row(key: str, reward: float, *, valid: bool = True) -> TrajectoryTurn:
    return TrajectoryTurn(key=key, reward=reward, valid_for_training=valid)


def test_population_advantage_uses_each_trajectory_once_then_broadcasts_to_turns() -> None:
    result = normalize_trajectory_turns([
        row("query_with_under_score_0_0", 0.0),
        row("query_with_under_score_0_1", 0.0),
        row("query_with_under_score_1_0", 1.0),
        row("query_with_under_score_1_1", 1.0),
        row("query_with_under_score_1_2", 1.0),
    ])

    assert result.advantages == {
        "query_with_under_score_0_0": -1.0,
        "query_with_under_score_0_1": -1.0,
        "query_with_under_score_1_0": 1.0,
        "query_with_under_score_1_1": 1.0,
        "query_with_under_score_1_2": 1.0,
    }
    assert result.metrics.trajectory_count == 2
    assert result.metrics.turn_count == 5
    assert result.trainable_keys == (
        "query_with_under_score_0_0",
        "query_with_under_score_0_1",
        "query_with_under_score_1_0",
        "query_with_under_score_1_1",
        "query_with_under_score_1_2",
    )
    assert result.invalid_keys == ()
    assert result.skipped_keys == ()
    assert isclose(result.metrics.reward_mean, 0.5)
    assert isclose(result.metrics.reward_std, 0.5)


def test_zero_variance_group_gets_zero_advantages() -> None:
    result = normalize_trajectory_turns([
        row("q_0_0", 1.0), row("q_0_1", 1.0), row("q_1_0", 1.0)
    ])

    assert set(result.advantages.values()) == {0.0}
    assert result.metrics.zero_variance_group_count == 1
    assert result.metrics.skipped_group_count == 1
    assert result.trainable_keys == ()
    assert result.skipped_keys == ("q_0_0", "q_0_1", "q_1_0")


def test_infrastructure_invalid_session_is_excluded_from_mean_std_and_training() -> None:
    result = normalize_trajectory_turns([
        row("q_0_0", 0.0), row("q_1_0", 1.0),
        row("q_2_0", 0.0, valid=False), row("q_2_1", 0.0, valid=False),
    ])

    assert result.advantages["q_0_0"] == -1.0
    assert result.advantages["q_1_0"] == 1.0
    assert result.advantages["q_2_0"] == 0.0
    assert result.advantages["q_2_1"] == 0.0
    assert result.metrics.valid_trajectory_count == 2
    assert result.metrics.invalid_trajectory_count == 1
    assert result.trainable_keys == ("q_0_0", "q_1_0")
    assert result.invalid_keys == ("q_2_0", "q_2_1")


def test_query_with_fewer_than_two_valid_trajectories_is_skipped() -> None:
    result = normalize_trajectory_turns([
        row("q_0_0", 1.0), row("q_1_0", 0.0, valid=False)
    ])

    assert set(result.advantages.values()) == {0.0}
    assert result.metrics.skipped_group_count == 1
    assert result.trainable_keys == ()
    assert result.skipped_keys == ("q_0_0",)
    assert result.invalid_keys == ("q_1_0",)


def test_multiple_queries_are_normalized_independently() -> None:
    result = normalize_trajectory_turns([
        row("a_0_0", 0.0), row("a_1_0", 1.0),
        row("b_0_0", 1.0), row("b_1_0", 1.0),
    ])

    assert result.advantages["a_0_0"] == -1.0
    assert result.advantages["a_1_0"] == 1.0
    assert result.advantages["b_0_0"] == 0.0
    assert result.advantages["b_1_0"] == 0.0
    assert result.metrics.group_count == 2
    assert result.metrics.zero_variance_group_count == 1
    assert result.trainable_keys == ("a_0_0", "a_1_0")
    assert result.skipped_keys == ("b_0_0", "b_1_0")
