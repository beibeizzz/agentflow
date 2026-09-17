import math

import pytest

from agentflow_rl.rewards import (
    ADVANTAGE_REVISION,
    TurnReward,
    compute_turn_advantages,
)
from agentflow_rl.runtime.contracts import TaskName


def row(
    trajectory: str,
    turn: int,
    terminal: float,
    process: float | None = None,
    *,
    valid: bool = True,
    task_id: str = "prompt-1",
    prompt_group_id: str | None = "group-1",
    task_name: TaskName | str = TaskName.AIME,
    key: str | None = None,
) -> TurnReward:
    return TurnReward(
        key=key or f"{trajectory}:{turn}",
        task_id=task_id,
        task_name=task_name,
        trajectory_id=trajectory,
        turn_index=turn,
        terminal_reward=terminal,
        process_score=process,
        valid_for_training=valid,
        prompt_group_id=prompt_group_id,
    )


def advantages(rows, **overrides):
    parameters = {
        "max_turns": 5,
        "lambda_process": 0.3,
        "max_advantage": 5.0,
        "advantage_revision": ADVANTAGE_REVISION,
    }
    parameters.update(overrides)
    return compute_turn_advantages(rows, **parameters)


def test_required_numeric_example_matches_rtg_loo_definition() -> None:
    result = advantages(
        (
            row("a", 0, 1.0, 0.2),
            row("a", 1, 1.0, 0.8),
            row("b", 0, 0.0, 0.4),
        )
    )
    assert result.process_return["a:0"] == pytest.approx(0.20)
    assert result.process_return["a:1"] == pytest.approx(0.16)
    assert result.process_return["b:0"] == pytest.approx(0.08)
    assert result.terminal["a:0"] == pytest.approx(1.0)
    assert result.terminal["a:1"] == pytest.approx(1.0)
    assert result.terminal["b:0"] == pytest.approx(-1.0)
    assert result.process["a:0"] == pytest.approx(0.12)
    assert result.process["a:1"] == pytest.approx(0.16)
    assert result.process["b:0"] == pytest.approx(-0.12)
    assert result.combined == pytest.approx({"a:0": 1.036, "a:1": 1.048, "b:0": -1.036})
    assert result.total_return["a:0"] == pytest.approx(1.06)
    assert result.total_return["a:1"] == pytest.approx(1.048)


def test_future_process_scores_assign_delayed_credit_to_earlier_turn() -> None:
    result = advantages(
        (
            row("a", 0, 0.0, 0.2),
            row("a", 1, 0.0, 0.8),
            row("b", 0, 0.0, 0.2),
            row("b", 1, 0.0, 0.0),
        )
    )
    assert result.process["a:0"] == pytest.approx(0.16)
    assert result.process["b:0"] == pytest.approx(-0.16)


def test_ended_trajectories_contribute_zero_absorbing_tail() -> None:
    result = advantages(
        (
            row("long", 0, 0.0, 0.1),
            row("long", 1, 0.0, 0.9),
            row("short", 0, 0.0, 0.1),
        )
    )
    assert result.process_baseline["long:1"] == 0.0
    assert result.process["long:1"] == pytest.approx(0.18)


def test_raw_process_scale_is_preserved() -> None:
    base = advantages((row("a", 0, 0.0, 0.2), row("b", 0, 0.0, 0.1)))
    scaled = advantages((row("a", 0, 0.0, 0.4), row("b", 0, 0.0, 0.2)))
    assert scaled.process["a:0"] == pytest.approx(2 * base.process["a:0"])
    assert scaled.process["b:0"] == pytest.approx(2 * base.process["b:0"])


def test_terminal_reward_is_included_once_in_every_real_turn_return() -> None:
    result = advantages(
        (
            row("a", 0, 0.7, 0.5),
            row("a", 1, 0.7, 0.5),
            row("b", 0, 0.2, 0.0),
        )
    )
    assert result.total_return["a:0"] == pytest.approx(0.7 + 0.3 * 0.2)
    assert result.total_return["a:1"] == pytest.approx(0.7 + 0.3 * 0.1)
    assert result.total_return["b:0"] == pytest.approx(0.2)


def test_input_order_does_not_change_keyed_results() -> None:
    rows = (
        row("a", 1, 1.0, 0.8),
        row("b", 0, 0.0, 0.4),
        row("a", 0, 1.0, 0.2),
    )
    forward = advantages(rows)
    reverse = advantages(reversed(rows))
    assert forward.combined == reverse.combined
    assert forward.process_return == reverse.process_return


def test_prompt_groups_are_isolated_when_task_id_matches() -> None:
    rows = (
        row("a", 0, 1.0, 0.6, prompt_group_id="g1", key="g1-a"),
        row("b", 0, 0.0, 0.2, prompt_group_id="g1", key="g1-b"),
        row("c", 0, 0.5, 0.7, prompt_group_id="g2", key="g2-c"),
        row("d", 0, 0.5, 0.7, prompt_group_id="g2", key="g2-d"),
    )
    result = advantages(rows)
    assert result.terminal["g1-a"] == 1.0
    assert result.terminal["g1-b"] == -1.0
    assert result.terminal["g2-c"] == 0.0


def test_missing_score_falls_back_entire_group_to_terminal_loo() -> None:
    result = advantages(
        (
            row("a", 0, 1.0, 0.8),
            row("a", 1, 1.0, None),
            row("b", 0, 0.0, 0.2),
        )
    )
    assert result.process == {"a:0": 0.0, "a:1": 0.0, "b:0": 0.0}
    assert all(value is None for value in result.process_return.values())
    assert result.raw_process_score == {"a:0": 0.8, "a:1": None, "b:0": 0.2}
    assert result.combined == {"a:0": 1.0, "a:1": 1.0, "b:0": -1.0}
    assert result.process_group_status["group-1"] == (
        "terminal_only_missing_process_score"
    )
    assert result.metrics.process_fallback_group_count == 1


def test_valid_zero_process_score_remains_available() -> None:
    result = advantages((row("a", 0, 1.0, 0.0), row("b", 0, 0.0, 0.0)))
    assert result.process_group_status["group-1"] == "complete"
    assert result.metrics.process_missing_turn_count == 0
    assert result.process_return["a:0"] == 0.0


def test_lambda_zero_is_terminal_loo_even_when_scores_are_missing() -> None:
    result = advantages(
        (row("a", 0, 1.0, None), row("b", 0, 0.0, None)),
        lambda_process=0.0,
    )
    assert result.combined == {"a:0": 1.0, "b:0": -1.0}


def test_singleton_uses_zero_baselines_without_division() -> None:
    result = advantages((row("a", 0, 0.7, 0.5),))
    assert result.terminal_baseline["a:0"] == 0.0
    assert result.process_baseline["a:0"] == 0.0
    assert result.combined["a:0"] == pytest.approx(0.7 + 0.3 * 0.1)
    assert result.metrics.singleton_group_count == 1


def test_invalid_trajectory_is_excluded_from_both_baselines() -> None:
    result = advantages(
        (
            row("a", 0, 1.0, 1.0),
            row("b", 0, 0.0, 0.0),
            row("infra", 0, 0.5, 0.5, valid=False),
        )
    )
    assert result.terminal["a:0"] == 1.0
    assert result.terminal["b:0"] == -1.0
    assert result.combined["infra:0"] == 0.0
    assert result.invalid_keys == ("infra:0",)


def test_combined_advantage_records_raw_value_and_clipping() -> None:
    result = advantages(
        (row("a", 0, 1.0, 1.0), row("b", 0, 0.0, 0.0)),
        lambda_process=10.0,
        max_advantage=2.0,
    )
    assert result.raw_combined == pytest.approx({"a:0": 3.0, "b:0": -3.0})
    assert result.combined == {"a:0": 2.0, "b:0": -2.0}
    assert result.metrics.clipped_turn_count == 2


@pytest.mark.parametrize(
    "rows, message",
    [
        ((row("a", 0, 1.0), row("b", 0, 0.0, key="a:0")), "keys"),
        ((row("a", 0, 1.0), row("a", 0, 1.0, key="duplicate-index")), "indices"),
        ((row("a", 0, 1.0), row("a", 2, 1.0)), "contiguous"),
        ((row("a", 0, 1.0), row("a", 1, 0.0)), "terminal reward"),
    ],
)
def test_structural_errors_are_rejected(rows, message) -> None:
    with pytest.raises(ValueError, match=message):
        advantages(rows)


def test_trajectory_longer_than_fixed_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        advantages(tuple(row("a", turn, 1.0, 0.5) for turn in range(6)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"lambda_process": math.nan},
        {"max_advantage": math.inf},
        {"epsilon": 0.0},
        {"max_turns": 0},
        {"advantage_revision": "legacy"},
    ],
)
def test_algorithm_parameters_are_validated(overrides) -> None:
    with pytest.raises(ValueError):
        advantages((), **overrides)


def test_task_name_strings_are_normalized_and_invalid_values_fail() -> None:
    assert row("a", 0, 1.0, task_name="aime").task_name is TaskName.AIME
    with pytest.raises(ValueError, match="invalid task_name"):
        row("a", 0, 1.0, task_name="unknown")


def test_empty_input_returns_empty_diagnostics() -> None:
    result = advantages(())
    assert result.combined == {}
    assert result.trainable_keys == ()
    assert result.metrics.prompt_group_count == 0
    assert all(math.isfinite(value) for value in result.combined.values())
