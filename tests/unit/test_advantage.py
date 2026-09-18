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
        "lambda_process": 0.5,
        "max_advantage": 5.0,
        "advantage_revision": ADVANTAGE_REVISION,
    }
    parameters.update(overrides)
    return compute_turn_advantages(rows, **parameters)


def example_rows():
    scores = [[.5,.1,.1,.9,.6],[.5,.5],[.5,0],[.5,0],[.5]]
    return [row(chr(97+i), t, float(i == 0), p) for i, ps in enumerate(scores) for t, p in enumerate(ps)]


def test_user_twelve_turn_numeric_example():
    result = advantages(example_rows())
    expected = [1.202861,.825493,.825493,1.580230,1.297203,-.683980,-.683980,-.683980,-1.155690,-.683980,-1.155690,-.683980]
    assert [result.combined[r.key] for r in example_rows()] == pytest.approx(expected, abs=5e-7)
    assert result.normalization_mean['a:0'] == pytest.approx(.6125)
    assert result.normalization_std['a:0'] == pytest.approx(.5299862419597953)
    assert sum(result.combined.values()) == pytest.approx(0, abs=1e-12)
    assert sum(v*v for v in result.combined.values())/12 == pytest.approx(1)
    for key in result.combined:
        assert result.terminal[key]+result.weighted_process[key] == pytest.approx(result.raw_combined[key])


def test_e2_normalizes_trajectories_before_broadcast_not_expanded_rows():
    result = advantages(example_rows(), lambda_process=0)
    for r in example_rows():
        assert result.combined[r.key] == pytest.approx(2 if r.trajectory_id == 'a' else -.5)
        assert result.normalization_scope[r.key] == 'trajectories'
    assert result.metrics.process_fallback_group_count == 0


def test_e3_no_suffix_accumulation_or_absorbing_tail():
    rows = [row('a',0,1,.2),row('a',1,1,.8),row('b',0,0,.4)]
    result = advantages(rows)
    assert result.mixed_reward == pytest.approx({'a:0':1.1,'a:1':1.4,'b:0':.2})
    assert result.process_value == pytest.approx({'a:0':.2,'a:1':.8,'b:0':.4})
    assert result.normalization_mean['a:0'] == pytest.approx(.9)
    assert advantages(rows,max_turns=9).combined == result.combined


def test_missing_score_falls_back_whole_group_to_exact_e2():
    rows = example_rows()
    from dataclasses import replace
    rows[1] = replace(rows[1], process_score=None)
    result = advantages(rows)
    assert result.combined == advantages(rows,lambda_process=0).combined
    assert result.process_group_status['group-1'] == 'terminal_only_missing_process_score'
    assert result.metrics.process_fallback_group_count == 1
    assert all(value is None for value in result.process_value.values())
    assert result.raw_process_score['a:0'] == .5


def test_zero_is_valid_and_singleton_or_constant_population_is_finite():
    result = advantages([row('a',0,1,0),row('b',0,0,0)])
    assert result.process_group_status['group-1'] == 'complete'
    assert result.metrics.process_missing_turn_count == 0
    assert result.combined == {'a:0':1,'b:0':-1}
    for rows in ([row('a',0,.7,.5)], [row('a',0,.7,.5),row('b',0,.7,.5)]):
        result = advantages(rows)
        assert all(v == 0 for v in result.combined.values())
        assert result.trainable_keys == ()


def test_invalid_trajectory_and_other_groups_do_not_change_statistics():
    base = example_rows()
    extended = base + [row('invalid',0,0,None,valid=False),row('other',0,0,0,prompt_group_id='g2',key='other')]
    result = advantages(extended)
    assert {r.key:result.combined[r.key] for r in base} == advantages(base).combined
    assert result.invalid_keys == ('invalid:0',)
    assert result.metrics.process_fallback_group_count == 0


def test_keyed_values_are_order_invariant():
    rows=example_rows()
    assert advantages(rows).combined == advantages(reversed(rows)).combined


def test_clipping_is_after_normalization_and_auditable():
    result = advantages(example_rows(),max_advantage=1)
    assert result.raw_combined['a:0'] > 1
    assert result.combined['a:0'] == 1
    assert result.metrics.clipped_turn_count == 5


def test_near_constant_rewards_do_not_amplify_roundoff():
    result=advantages([row('a',0,.5,.5),row('b',0,.5+1e-10,.5)])
    assert result.combined == {'a:0':0,'b:0':0}


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
