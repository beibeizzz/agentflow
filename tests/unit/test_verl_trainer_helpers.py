import asyncio
import sys
from pathlib import Path

import pytest
import yaml

from agentflow_rl.integrations.verl_trainer import (
    advantage_audit_records,
    advantage_state_identity,
    build_turn_rewards_from_metadata,
    bind_policy_update_identity,
    build_training_selection,
    balanced_padded_key_order,
    dynamic_quota_for_step,
    fixed_turn_mini_batch_layout,
    policy_freshness_metrics,
    process_availability_gate,
    training_state_path,
    validate_advantage_state_identity,
    write_step_metrics,
    zero_response_aligned_training_fields,
)
from agentflow_rl.integrations.policy_identity import read_policy_update_id
from agentflow_rl.rewards import ADVANTAGE_REVISION, TurnReward
from agentflow_rl.runtime.contracts import TaskName


def metadata(trajectory: str, turn: int, reward: float, process: float | None):
    return {
        "task_id": "aime-1",
        "uid": "collection-prompt-1",
        "task_name": "aime",
        "trajectory_id": trajectory,
        "turn_index": turn,
        "terminal_reward": reward,
        "process_score": process,
        "valid_for_training": True,
        "rollout_policy_revision": "checkpoint-3",
        "rollout_policy_update_id": "3",
    }


def test_build_turn_rewards_uses_explicit_metadata() -> None:
    rows = build_turn_rewards_from_metadata(
        keys=["row-1"], extra_fields=[metadata("trajectory-1", 0, 1.0, 0.8)]
    )
    assert rows[0].terminal_reward == 1.0
    assert rows[0].process_score == 0.8
    assert rows[0].group_id == "collection-prompt-1"
    assert rows[0].task_name is TaskName.AIME


@pytest.mark.parametrize(
    ("turn_count", "padded_count", "padding_count", "optimizer_steps"),
    [
        (20, 32, 12, 1),
        (32, 32, 0, 1),
        (33, 64, 31, 2),
        (63, 64, 1, 2),
        (71, 96, 25, 3),
        (80, 96, 16, 3),
        (100, 128, 28, 4),
    ],
)
def test_fixed_turn_mini_batch_layout_preserves_real_rows_and_bounds_steps(
    turn_count, padded_count, padding_count, optimizer_steps
):
    layout = fixed_turn_mini_batch_layout(turn_count=turn_count, mini_batch_size=32)
    assert layout.real_turn_count == turn_count
    assert layout.padded_turn_count == padded_count
    assert layout.padding_turn_count == padding_count
    assert layout.mini_batch_size == 32
    assert layout.optimizer_step_count == optimizer_steps
    assert layout.padded_turn_count % layout.mini_batch_size == 0
    assert layout.maximum_real_turns_per_step - layout.minimum_real_turns_per_step <= 1


@pytest.mark.parametrize("turn_count, mini_batch_size", [(0, 32), (10, 0), (-1, 32)])
def test_fixed_turn_mini_batch_layout_rejects_nonpositive_sizes(
    turn_count, mini_batch_size
):
    with pytest.raises(ValueError, match="positive"):
        fixed_turn_mini_batch_layout(
            turn_count=turn_count, mini_batch_size=mini_batch_size
        )


@pytest.mark.parametrize("include_reference", [False, True])
def test_padding_training_fields_are_zero_and_response_aligned(
    include_reference, monkeypatch
):
    class FakeTensor:
        def __init__(self, shape, dtype, value):
            self.shape = shape
            self.dtype = dtype
            self.value = value

        def clone(self):
            return FakeTensor(self.shape, self.dtype, self.value)

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def zeros_like(source, dtype):
            return FakeTensor(source.shape, dtype, 0)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    response_mask = FakeTensor((7, 1), "int64", 0)
    result = zero_response_aligned_training_fields(
        response_mask, include_reference=include_reference
    )
    expected = {"old_log_probs", "advantages", "returns"}
    if include_reference:
        expected.add("ref_log_prob")
    assert set(result) == expected
    for value in result.values():
        assert value.shape == response_mask.shape
        assert value.dtype == FakeTorch.float32
        assert value.value == 0


def test_formal_config_uses_fixed_32_turn_batches_and_40960_token_budget():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "configs/train/unified_terminal_prm.yaml").read_text(encoding="utf-8")
    )
    actor = config["actor_rollout_ref"]["actor"]
    assert actor["ppo_mini_batch_size"] == 32
    assert actor["shuffle"] is False
    tools = config["agentflow"]["tools"]
    assert tools["sandbox_cpus"] == 1
    assert tools["sandbox_max_concurrency"] == 6
    assert tools["sandbox_max_queue"] == 34
    assert tools["serper_max_concurrency"] == 40
    assert tools["wikipedia_max_concurrency"] == 4
    assert tools["sandbox_memory"] == "1g"
    assert tools["web_reader_max_concurrency"] == 12
    assert tools["wikipedia_batch_max_size"] == 16
    assert tools["wikipedia_index_type"] == "hnsw64"
    assert tools["wikipedia_hnsw_m"] == 64
    assert tools["wikipedia_hnsw_ef_search"] == 256
    assert config["agentflow"]["turn_mini_batch_size"] == 32
    assert config["agentflow"]["advantage_revision"] == ADVANTAGE_REVISION
    assert actor["use_dynamic_bsz"] is True
    assert actor["ppo_max_token_len_per_gpu"] == 40960

    catalog = yaml.safe_load(
        (root / "configs/experiments/metrics_catalog.yaml").read_text(encoding="utf-8")
    )
    metrics = {item["id"]: item for item in catalog["metrics"]}
    assert metrics["train.counters"]["native_mapping"]["O_per_update"] == (
        "agentflow/actor_optimizer_step_count"
    )
    row_mapping = metrics["train.actual_rows"]["native_mapping"]
    assert row_mapping["real_actor"] == "agentflow/actor_real_turn_count"
    assert row_mapping["padding"] == "agentflow/actor_padding_turn_count"


@pytest.mark.parametrize("turn_count", [20, 33, 63, 71, 80, 100])
def test_balanced_padding_spreads_real_turns_across_optimizer_steps(turn_count):
    layout = fixed_turn_mini_batch_layout(turn_count=turn_count, mini_batch_size=32)
    real_keys = tuple(f"real-{index}" for index in range(turn_count))
    padding_keys = tuple(f"pad-{index}" for index in range(layout.padding_turn_count))
    ordered = balanced_padded_key_order(
        real_keys=real_keys,
        padding_keys=padding_keys,
        mini_batch_size=32,
    )
    assert len(ordered) == layout.padded_turn_count
    assert {key for key in ordered if key.startswith("real-")} == set(real_keys)
    assert {key for key in ordered if key.startswith("pad-")} == set(padding_keys)
    real_counts = [
        sum(key.startswith("real-") for key in ordered[start : start + 32])
        for start in range(0, len(ordered), 32)
    ]
    assert min(real_counts) == layout.minimum_real_turns_per_step
    assert max(real_counts) == layout.maximum_real_turns_per_step
    assert max(real_counts) - min(real_counts) <= 1


def test_policy_freshness_detects_mixed_rollout_revisions() -> None:
    rows = [metadata("t1", 0, 1.0, None), metadata("t2", 0, 0.0, None)]
    assert (
        policy_freshness_metrics(rows, expected_update_id="3")[
            "agentflow/rollout_policy_fresh"
        ]
        == 1.0
    )
    rows[1]["rollout_policy_revision"] = "checkpoint-2"
    assert (
        policy_freshness_metrics(rows, expected_update_id="3")[
            "agentflow/rollout_policy_fresh"
        ]
        == 0.0
    )


def test_policy_freshness_requires_server_update_identity() -> None:
    rows = [metadata("t1", 0, 1.0, None)]
    assert (
        policy_freshness_metrics(rows, expected_update_id="3")[
            "agentflow/rollout_policy_fresh"
        ]
        == 1.0
    )
    assert (
        policy_freshness_metrics(rows, expected_update_id="4")[
            "agentflow/rollout_policy_fresh"
        ]
        == 0.0
    )


def test_policy_update_binding_uses_successful_update_count() -> None:
    class Manager:
        def __init__(self):
            self.calls = []

        def update_weights(self, global_steps=None):
            self.calls.append(global_steps)
            return global_steps

    manager = Manager()
    count = 2
    bind_policy_update_identity(manager, lambda: count)
    assert manager.update_weights() == 2
    count = 3
    assert manager.update_weights() == 3
    with pytest.raises(RuntimeError, match="conflicting"):
        manager.update_weights(global_steps=2)


def test_policy_identity_is_published_only_after_successful_sync(tmp_path) -> None:
    class Manager:
        def __init__(self):
            self.fail = False

        def update_weights(self, global_steps=None):
            if self.fail:
                raise RuntimeError("sync failed")
            return global_steps

    path = tmp_path / "policy.json"
    manager = Manager()
    update_id = 1
    bind_policy_update_identity(manager, lambda: update_id, identity_path=path)
    assert manager.update_weights() == 1
    assert read_policy_update_id(path) == "1"
    update_id = 2
    manager.fail = True
    with pytest.raises(RuntimeError, match="sync failed"):
        manager.update_weights()
    assert read_policy_update_id(path) == "1"


def test_policy_identity_waits_for_async_sync_completion(tmp_path) -> None:
    class Manager:
        async def update_weights(self, global_steps=None):
            await asyncio.sleep(0)
            return global_steps

    path = tmp_path / "policy.json"
    manager = Manager()
    bind_policy_update_identity(manager, lambda: 3, identity_path=path)
    assert not path.exists()
    assert asyncio.run(manager.update_weights()) == 3
    assert read_policy_update_id(path) == "3"


def test_training_selection_is_fixed_before_old_log_prob() -> None:
    keys = []
    fields = []
    for task in ("aime", "twowiki", "taco"):
        for session, reward in ((0, 0.0), (1, 1.0)):
            key = f"{task}-{session}"
            keys.append(key)
            fields.append(
                {
                    **metadata(f"{task}-trajectory-{session}", 0, reward, 0.5),
                    "task_id": f"{task}-task",
                    "uid": f"{task}-group",
                    "task_name": task,
                }
            )
    rows = build_turn_rewards_from_metadata(keys=keys, extra_fields=fields)
    plan = build_training_selection(
        rows,
        fields,
        expected_update_id="3",
        lambda_process=0.3,
        max_advantage=5.0,
        max_turns=5,
        advantage_revision=ADVANTAGE_REVISION,
        process_required=True,
        max_missing_rate=0.05,
        require_each_task=True,
        dynamic_enabled=True,
        per_task_quota={
            TaskName.AIME: 1,
            TaskName.TWOWIKI: 1,
            TaskName.TACO: 1,
        },
        train_prompt_group_count=3,
        expected_trajectories_per_group=2,
    )
    assert set(plan.trainable_keys) == set(keys)
    assert plan.metrics["agentflow/dynamic_sampling_complete"] == 1.0

    shuffled = list(zip(reversed(keys), reversed(fields), strict=True))
    shuffled_keys = [key for key, _ in shuffled]
    shuffled_fields = [field for _, field in shuffled]
    shuffled_plan = build_training_selection(
        build_turn_rewards_from_metadata(
            keys=shuffled_keys, extra_fields=shuffled_fields
        ),
        shuffled_fields,
        expected_update_id="3",
        lambda_process=0.3,
        max_advantage=5.0,
        max_turns=5,
        advantage_revision=ADVANTAGE_REVISION,
        process_required=True,
        max_missing_rate=0.05,
        require_each_task=True,
        dynamic_enabled=True,
        per_task_quota={
            TaskName.AIME: 1,
            TaskName.TWOWIKI: 1,
            TaskName.TACO: 1,
        },
        train_prompt_group_count=3,
        expected_trajectories_per_group=2,
    )
    assert set(shuffled_plan.trainable_keys) == set(keys)
    assert shuffled_plan.advantages.combined == plan.advantages.combined


def test_training_selection_rejects_stale_policy_before_inference() -> None:
    fields = [metadata("trajectory-1", 0, 1.0, None)]
    rows = build_turn_rewards_from_metadata(keys=["a"], extra_fields=fields)
    plan = build_training_selection(
        rows,
        fields,
        expected_update_id="4",
        lambda_process=0.0,
        max_advantage=5.0,
        max_turns=5,
        advantage_revision=ADVANTAGE_REVISION,
        process_required=False,
        max_missing_rate=0.05,
        require_each_task=True,
        dynamic_enabled=False,
        per_task_quota={TaskName.AIME: 1},
        train_prompt_group_count=1,
        expected_trajectories_per_group=2,
    )
    assert plan.trainable_keys == ()


def test_turn_reward_metadata_requires_identity() -> None:
    with pytest.raises(KeyError):
        build_turn_rewards_from_metadata(keys=["row"], extra_fields=[{}])


def test_dynamic_quota_balances_three_training_steps() -> None:
    totals = {task: 0 for task in (TaskName.AIME, TaskName.TWOWIKI, TaskName.TACO)}
    for step in (1, 2, 3):
        for task, count in dynamic_quota_for_step(step).items():
            totals[task] += count
    assert set(totals.values()) == {4}


def test_process_availability_gate_enforces_rate_and_task_support() -> None:
    rows = build_turn_rewards_from_metadata(
        keys=["a", "b"],
        extra_fields=[
            metadata("trajectory-1", 0, 1.0, 0.8),
            metadata("trajectory-2", 0, 0.0, None),
        ],
    )
    assert process_availability_gate(
        rows, max_missing_rate=0.5, require_each_task=True
    ).passed
    assert not process_availability_gate(
        rows, max_missing_rate=0.49, require_each_task=True
    ).passed


def test_process_availability_gate_rejects_missing_task_support() -> None:
    row = build_turn_rewards_from_metadata(
        keys=["a"], extra_fields=[metadata("trajectory-1", 0, 1.0, None)]
    )[0]
    unsupported = row.__class__(**{**row.__dict__, "task_name": TaskName.TACO})
    result = process_availability_gate(
        [row, unsupported], max_missing_rate=1.0, require_each_task=True
    )
    assert result.passed is False
    assert result.per_task_available[TaskName.TACO] == 0


def test_process_gate_detects_four_percent_missing_task_after_string_normalization() -> (
    None
):
    rows = [
        TurnReward(
            key=f"aime-{index}",
            task_id="aime-prompt",
            task_name="aime",
            trajectory_id=f"aime-{index}",
            turn_index=0,
            terminal_reward=float(index % 2),
            process_score=0.0,
            prompt_group_id=f"aime-group-{index}",
        )
        for index in range(96)
    ]
    rows.extend(
        TurnReward(
            key=f"taco-{index}",
            task_id="taco-prompt",
            task_name="taco",
            trajectory_id=f"taco-{index}",
            turn_index=0,
            terminal_reward=0.0,
            process_score=None,
            prompt_group_id=f"taco-group-{index}",
        )
        for index in range(4)
    )
    strict = process_availability_gate(
        rows, max_missing_rate=0.05, require_each_task=True
    )
    rate_only = process_availability_gate(
        rows, max_missing_rate=0.05, require_each_task=False
    )
    assert strict.missing_rate == pytest.approx(0.04)
    assert strict.passed is False
    assert rate_only.passed is True
    assert strict.per_task_available[TaskName.AIME] == 96
    assert strict.per_task_available[TaskName.TACO] == 0


def test_terminal_only_selection_skips_process_availability_gate() -> None:
    fields = [
        metadata("a", 0, 1.0, None),
        metadata("b", 0, 0.0, None),
    ]
    rows = build_turn_rewards_from_metadata(keys=["a", "b"], extra_fields=fields)
    plan = build_training_selection(
        rows,
        fields,
        expected_update_id="3",
        lambda_process=0.0,
        max_advantage=5.0,
        max_turns=5,
        advantage_revision=ADVANTAGE_REVISION,
        process_required=True,
        max_missing_rate=0.0,
        require_each_task=True,
        dynamic_enabled=False,
        per_task_quota={TaskName.AIME: 1},
        train_prompt_group_count=1,
        expected_trajectories_per_group=2,
    )
    assert set(plan.trainable_keys) == {"a", "b"}
    assert "agentflow/process_availability_gate_passed" not in plan.metrics


def test_advantage_audit_preserves_components_and_retained_membership() -> None:
    fields = [metadata("a", 0, 1.0, 0.8), metadata("b", 0, 0.0, 0.2)]
    rows = build_turn_rewards_from_metadata(keys=["a", "b"], extra_fields=fields)
    plan = build_training_selection(
        rows,
        fields,
        expected_update_id="3",
        lambda_process=0.3,
        max_advantage=5.0,
        max_turns=5,
        advantage_revision=ADVANTAGE_REVISION,
        process_required=True,
        max_missing_rate=0.05,
        require_each_task=True,
        dynamic_enabled=False,
        per_task_quota={TaskName.AIME: 1},
        train_prompt_group_count=1,
        expected_trajectories_per_group=2,
    )
    records = advantage_audit_records(
        plan.advantages,
        rows,
        retained_keys=("a",),
        advantage_revision=ADVANTAGE_REVISION,
        max_turns=5,
        lambda_process=0.3,
    )
    assert records[0]["raw_process_score"] == 0.8
    assert records[0]["process_return"] == pytest.approx(0.16)
    assert records[0]["retained_for_actor"] is True
    assert records[1]["retained_for_actor"] is False
    assert records[0]["raw_combined_advantage"] == pytest.approx(1.036)


def test_training_state_path_tracks_collection_checkpoint(tmp_path) -> None:
    assert training_state_path(tmp_path, 7) == (
        tmp_path / "global_step_7" / "agentflow_state.json"
    )


def test_checkpoint_advantage_identity_rejects_legacy_and_mismatch() -> None:
    identity = advantage_state_identity(
        advantage_revision=ADVANTAGE_REVISION,
        max_turns=5,
        lambda_process=0.3,
    )
    validate_advantage_state_identity(
        identity,
        advantage_revision=ADVANTAGE_REVISION,
        max_turns=5,
        lambda_process=0.3,
    )
    with pytest.raises(RuntimeError, match=r"lacks RTG\+LOO"):
        validate_advantage_state_identity(
            {},
            advantage_revision=ADVANTAGE_REVISION,
            max_turns=5,
            lambda_process=0.3,
        )
    with pytest.raises(RuntimeError, match="does not match"):
        validate_advantage_state_identity(
            identity,
            advantage_revision=ADVANTAGE_REVISION,
            max_turns=4,
            lambda_process=0.3,
        )


def test_step_metrics_writer_keeps_scalar_acceptance_metrics(tmp_path) -> None:
    path = write_step_metrics(
        tmp_path,
        3,
        {"agentflow/dynamic_sampling_complete": 1.0, "nested": {"skip": True}},
    )
    assert path.name == "step_3.json"
    assert '"agentflow/dynamic_sampling_complete": 1.0' in path.read_text()
    assert "nested" not in path.read_text()
