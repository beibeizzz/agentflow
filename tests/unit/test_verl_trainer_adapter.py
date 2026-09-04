from __future__ import annotations


def test_build_turn_rows_uses_verl_keys_rewards_and_validity() -> None:
    from agentflow_rl.verl.trainer import build_trajectory_turns

    rows = build_trajectory_turns(
        keys=["query_a_0_0", "query_a_0_1", "query_a_1_0"],
        rewards=[1.0, 1.0, 0.0],
        extra_fields=[
            {"valid_for_training": True},
            {"valid_for_training": True},
            {"valid_for_training": False},
        ],
    )

    assert [row.key for row in rows] == ["query_a_0_0", "query_a_0_1", "query_a_1_0"]
    assert rows[-1].valid_for_training is False
    assert rows[0].reward == 1.0


def test_advantage_metrics_are_namespaced_for_training_logs() -> None:
    from agentflow_rl.verl.advantage import normalize_trajectory_turns
    from agentflow_rl.verl.trainer import advantage_metrics_dict, build_trajectory_turns

    rows = build_trajectory_turns(
        keys=["q_0_0", "q_0_1", "q_1_0"],
        rewards=[1.0, 1.0, 0.0],
        extra_fields=[{}, {}, {}],
    )
    metrics = advantage_metrics_dict(normalize_trajectory_turns(rows).metrics)

    assert metrics["agentflow/trajectory_count"] == 2.0
    assert metrics["agentflow/turn_count"] == 3.0
    assert metrics["agentflow/trainable_turn_count"] == 3.0
    assert metrics["agentflow/reward_mean"] == 0.5
    assert metrics["agentflow/reward_std"] == 0.5
    assert metrics["agentflow/valid_trajectory_fraction"] == 1.0
    assert metrics["agentflow/zero_variance_group_fraction"] == 0.0


def test_actor_update_metadata_uses_actual_flattened_turn_count() -> None:
    from agentflow_rl.verl.trainer import actor_update_metadata

    metadata = actor_update_metadata(
        turn_count=11,
        mini_batch_size=8,
        ppo_epochs=2,
        seed=7,
        shuffle=True,
        temperature=1.2,
        calculate_entropy=False,
        distillation_use_topk=False,
    )

    assert metadata["global_batch_size"] == 8
    assert metadata["mini_batch_size"] == 8
    assert metadata["epochs"] == 2
    assert metadata["temperature"] == 1.2


def test_turn_mini_batch_is_bounded_and_evenly_divides_variable_turn_count() -> None:
    from agentflow_rl.verl.trainer import effective_turn_mini_batch_size

    assert effective_turn_mini_batch_size(turn_count=24, requested_size=8) == 8
    assert effective_turn_mini_batch_size(turn_count=18, requested_size=8) == 6
    assert effective_turn_mini_batch_size(turn_count=11, requested_size=8) == 1
    assert effective_turn_mini_batch_size(turn_count=3, requested_size=8) == 3


def test_trainer_only_requires_data_parallel_divisibility() -> None:
    from agentflow_rl.verl.trainer import AgentFlowPPOTrainer

    trainer = object.__new__(AgentFlowPPOTrainer)
    assert trainer._get_required_batch_multiple(1) == 1
    assert trainer._get_required_batch_multiple(2) == 2


def test_metadata_list_accepts_tensordict_linked_list_shape() -> None:
    from agentflow_rl.verl.trainer import as_metadata_list

    class LinkedListLike(list):
        pass

    metadata = LinkedListLike([{"valid_for_training": True}, {}])
    assert as_metadata_list(metadata) == [{"valid_for_training": True}, {}]


def test_put_batch_fields_returns_updated_transferqueue_metadata() -> None:
    from types import SimpleNamespace

    from agentflow_rl.verl.trainer import put_batch_fields

    old_batch = SimpleNamespace(keys=["a", "b"], partition_id="train", fields=["old"])
    updated_batch = SimpleNamespace(
        keys=old_batch.keys,
        partition_id=old_batch.partition_id,
        fields=["old", "advantages", "returns"],
    )
    calls = []

    def fake_put(**kwargs):
        calls.append(kwargs)
        return updated_batch

    result = put_batch_fields(
        put=fake_put,
        batch=old_batch,
        fields={"advantages": [1.0], "returns": [1.0]},
    )

    assert result is updated_batch
    assert calls == [
        {
            "keys": ["a", "b"],
            "partition_id": "train",
            "fields": {"advantages": [1.0], "returns": [1.0]},
        }
    ]


def test_old_log_prob_selection_excludes_infrastructure_invalid_rows() -> None:
    from agentflow_rl.verl.trainer import valid_training_keys

    assert valid_training_keys(
        keys=["q_0_0", "q_1_0", "q_2_0"],
        extra_fields=[
            {"valid_for_training": True},
            {"valid_for_training": False, "synthetic_diagnostic": True},
            {},
        ],
    ) == ("q_0_0", "q_2_0")


def test_unpadded_attention_mask_is_all_ones_for_dense_batch() -> None:
    import torch

    from agentflow_rl.verl.trainer import build_unpadded_attention_mask

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = build_unpadded_attention_mask(input_ids)

    assert mask.dtype == torch.int64
    assert mask.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_unpadded_attention_mask_pads_nested_batch_on_the_right() -> None:
    import torch

    from agentflow_rl.verl.trainer import build_unpadded_attention_mask

    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.tensor([1, 2, 3, 4, 5]),
        offsets=torch.tensor([0, 2, 5]),
    )
    mask = build_unpadded_attention_mask(input_ids)

    assert mask.tolist() == [[1, 1, 0], [1, 1, 1]]
