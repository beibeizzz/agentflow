from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentflow_rl.integrations.batch_lifecycle import (
    PaddingLifecycle,
    aligned_execution_size,
    close_padding_lifecycle,
    padding_keys,
    real_batch_view,
    real_metadata_view,
)
from agentflow_rl.integrations.verl_trainer import (
    build_turn_rewards_from_metadata,
    policy_freshness_metrics,
)


@dataclass
class Batch:
    keys: list[str]
    tags: list[dict]
    partition_id: str = "train"

    def select_keys(self, keys):
        positions = {key: index for index, key in enumerate(self.keys)}
        return Batch(
            list(keys),
            [self.tags[positions[key]] for key in keys],
            self.partition_id,
        )


class Store:
    def __init__(self, keys=()):
        self.keys = set(keys)
        self.clear_calls = []
        self.fail = False

    def kv_clear(self, *, keys, partition_id):
        self.clear_calls.append((partition_id, tuple(keys)))
        if self.fail:
            raise RuntimeError("clear failed")
        self.keys.difference_update(keys)


class Replay:
    def __init__(self, keys=()):
        self.keys = set(keys)
        self.remove_calls = []

    def remove(self, partition_id, keys):
        self.remove_calls.append((partition_id, tuple(keys)))
        self.keys.difference_update(keys)

    def add(self, partition_id, items):
        self.keys.update(items)


def execution_batch(real_count=5, padding_count=1):
    real = [f"real-{index}" for index in range(real_count)]
    pad = [f"pad-{index}" for index in range(padding_count)]
    before = Batch(real, [{} for _ in real])
    execution = Batch(
        real + pad,
        [{} for _ in real]
        + [
            {"is_padding": True, "copied_private_metadata": "ignored"}
            for _ in pad
        ],
    )
    return before, execution


def metadata(trajectory, turn=0):
    return {
        "task_id": "aime-1",
        "uid": "prompt-1",
        "task_name": "aime",
        "trajectory_id": trajectory,
        "turn_index": turn,
        "terminal_reward": 1.0,
        "process_score": 0.5,
        "valid_for_training": True,
        "rollout_policy_revision": "checkpoint-3",
        "rollout_policy_update_id": "3",
    }


def test_real_view_filters_authoritative_padding_tag_only() -> None:
    before, execution = execution_batch()
    execution.tags[0]["valid_for_training"] = False
    real = real_batch_view(execution)
    assert real.keys == before.keys
    assert padding_keys(execution) == ("pad-0",)
    assert "real-0" in real.keys


def test_padding_metadata_never_enters_reward_or_freshness_semantics() -> None:
    tags = [{}, {}, {"is_padding": True}]
    fields = [metadata("a"), metadata("b"), metadata("a")]
    fields[2]["rollout_policy_update_id"] = "stale"
    rows = build_turn_rewards_from_metadata(
        keys=["a", "b", "pad"], extra_fields=fields, padding_tags=tags
    )
    assert [row.key for row in rows] == ["a", "b"]
    assert policy_freshness_metrics(
        fields, expected_update_id="3", padding_tags=tags
    )["agentflow/rollout_policy_fresh"] == 1.0
    assert real_metadata_view(fields, tags) == fields[:2]


def test_padding_lifecycle_reclaims_tq_and_replay_and_is_idempotent() -> None:
    before, execution = execution_batch(real_count=33, padding_count=31)
    tq = Store(execution.keys)
    replay = Replay(execution.keys + ["other-owner"])
    lifecycle = PaddingLifecycle.register(
        stage="actor",
        before=before,
        execution=execution,
        transfer_queue=tq,
        replay_buffer=replay,
    )
    lifecycle.cleanup()
    lifecycle.cleanup()
    assert tq.keys == set(before.keys)
    assert replay.keys == set(before.keys) | {"other-owner"}
    assert len(tq.clear_calls) == len(replay.remove_calls) == 1
    assert lifecycle.metrics() == {
        "agentflow/actor_padding_created": 31.0,
        "agentflow/actor_padding_cleaned": 31.0,
        "agentflow/actor_padding_live": 0.0,
        "agentflow/actor_padding_cleanup_failed": 0.0,
    }


def test_repeated_padding_scopes_do_not_accumulate() -> None:
    tq = Store()
    replay = Replay()
    for index in range(50):
        before = Batch([f"real-{index}"], [{}])
        execution = Batch(
            before.keys + [f"pad-{index}"], [{}, {"is_padding": True}]
        )
        tq.keys.update(execution.keys)
        replay.keys.update(execution.keys)
        lifecycle = PaddingLifecycle.register(
            stage="old_log_prob",
            before=before,
            execution=execution,
            transfer_queue=tq,
            replay_buffer=replay,
        )
        lifecycle.cleanup()
    assert not any(key.startswith("pad-") for key in tq.keys | replay.keys)


def test_delayed_replay_poll_cannot_revive_reclaimed_padding() -> None:
    before, execution = execution_batch()
    tq = Store(execution.keys)
    replay = Replay(execution.keys)
    lifecycle = PaddingLifecycle.register(
        stage="actor",
        before=before,
        execution=execution,
        transfer_queue=tq,
        replay_buffer=replay,
    )
    lifecycle.cleanup()
    replay.add("train", {"pad-0": {"is_padding": True}})
    assert "pad-0" not in replay.keys


def test_cleanup_failure_is_counted_and_primary_error_is_preserved() -> None:
    before, execution = execution_batch()
    tq = Store(execution.keys)
    tq.fail = True
    lifecycle = PaddingLifecycle.register(
        stage="actor",
        before=before,
        execution=execution,
        transfer_queue=tq,
        replay_buffer=Replay(execution.keys),
    )
    primary = RuntimeError("actor failed")
    metrics = {}
    close_padding_lifecycle(lifecycle, metrics, active_error=primary)
    assert metrics["agentflow/actor_padding_cleanup_failed"] == 1.0
    assert any("padding cleanup also failed" in note for note in primary.__notes__)


def test_new_execution_key_requires_padding_identity() -> None:
    before, execution = execution_batch()
    execution.tags[-1] = {}
    with pytest.raises(ValueError, match="is_padding"):
        PaddingLifecycle.register(
            stage="actor",
            before=before,
            execution=execution,
            transfer_queue=Store(),
            replay_buffer=Replay(),
        )


def test_worker_specific_alignment_for_twenty_one_real_turns() -> None:
    assert aligned_execution_size(21, 2) == 22
    assert aligned_execution_size(21, 4) == 24
    assert aligned_execution_size(21, 32) == 32
