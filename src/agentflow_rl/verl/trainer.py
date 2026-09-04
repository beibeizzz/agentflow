from __future__ import annotations

# PPOTrainer override structure adapted from verl-project/verl v0.8.0
# (Apache License 2.0); see THIRD_PARTY_NOTICES.md.

from dataclasses import fields
from typing import Any, Sequence

from agentflow_rl.verl.padding import response_lengths
from agentflow_rl.verl.advantage import (
    AdvantageMetrics,
    TrajectoryTurn,
    normalize_trajectory_turns,
)


def build_trajectory_turns(
    *,
    keys: Sequence[str],
    rewards: Sequence[float],
    extra_fields: Sequence[dict[str, Any] | None],
) -> list[TrajectoryTurn]:
    if not (len(keys) == len(rewards) == len(extra_fields)):
        raise ValueError("veRL keys, rewards, and extra_fields must align")
    return [
        TrajectoryTurn(
            key=str(key),
            reward=float(reward),
            valid_for_training=bool((metadata or {}).get("valid_for_training", True)),
        )
        for key, reward, metadata in zip(keys, rewards, extra_fields, strict=True)
    ]


def advantage_metrics_dict(metrics: AdvantageMetrics) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in fields(metrics):
        value = getattr(metrics, field.name)
        if value is not None:
            result[f"agentflow/{field.name}"] = float(value)
    if metrics.trajectory_count:
        result["agentflow/valid_trajectory_fraction"] = (
            metrics.valid_trajectory_count / metrics.trajectory_count
        )
    if metrics.group_count:
        result["agentflow/zero_variance_group_fraction"] = (
            metrics.zero_variance_group_count / metrics.group_count
        )
        result["agentflow/skipped_group_fraction"] = (
            metrics.skipped_group_count / metrics.group_count
        )
    return result


def as_metadata_list(value: Any) -> list[Any]:
    """Normalize TensorDict non-tensor containers across releases."""
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def put_batch_fields(*, put: Any, batch: Any, fields: Any) -> Any:
    """Store fields and return TransferQueue's updated batch metadata."""
    return put(
        keys=batch.keys,
        partition_id=batch.partition_id,
        fields=fields,
    )


def valid_training_keys(
    *,
    keys: Sequence[str],
    extra_fields: Sequence[dict[str, Any] | None],
) -> tuple[str, ...]:
    if len(keys) != len(extra_fields):
        raise ValueError("veRL keys and extra_fields must align")
    return tuple(
        str(key)
        for key, metadata in zip(keys, extra_fields, strict=True)
        if bool((metadata or {}).get("valid_for_training", True))
    )


def build_unpadded_attention_mask(input_ids: Any) -> Any:
    """Build the full-sequence mask omitted by veRL 0.8 AgentLoopWorkerTQ."""
    import torch

    if not getattr(input_ids, "is_nested", False):
        return torch.ones_like(input_ids, dtype=torch.int64)
    sequence_lengths = input_ids.offsets().diff()
    max_length = int(sequence_lengths.max().item())
    positions = torch.arange(max_length, device=sequence_lengths.device).unsqueeze(0)
    return (positions < sequence_lengths.unsqueeze(1)).to(torch.int64)


def effective_turn_mini_batch_size(*, turn_count: int, requested_size: int) -> int:
    """Choose the largest memory-bounded mini-batch that divides all turns."""
    if turn_count <= 0:
        raise ValueError("turn_count must be positive")
    if requested_size <= 0:
        raise ValueError("requested_size must be positive")
    upper_bound = min(turn_count, requested_size)
    return next(
        size for size in range(upper_bound, 0, -1) if turn_count % size == 0
    )


def actor_update_metadata(
    *,
    turn_count: int,
    mini_batch_size: int,
    ppo_epochs: int,
    seed: int,
    shuffle: bool,
    temperature: float,
    calculate_entropy: bool,
    distillation_use_topk: bool,
) -> dict[str, Any]:
    if turn_count <= 0:
        raise ValueError("turn_count must be positive")
    if ppo_epochs <= 0:
        raise ValueError("ppo_epochs must be positive")
    if mini_batch_size <= 0:
        raise ValueError("mini_batch_size must be positive")
    return {
        "calculate_entropy": bool(calculate_entropy),
        "distillation_use_topk": bool(distillation_use_topk),
        "global_batch_size": int(mini_batch_size),
        "mini_batch_size": int(mini_batch_size),
        "epochs": int(ppo_epochs),
        "seed": int(seed),
        "dataloader_kwargs": {"shuffle": bool(shuffle)},
        "temperature": float(temperature),
    }


class _AgentFlowTrainerMixin:
    """Minimal semantic overrides on top of veRL's synchronous PPO trainer."""

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        if dp_size <= 0:
            raise ValueError("dp_size must be positive")
        # Turn count is trajectory-dependent. Only distributed sharding imposes a
        # fixed divisibility constraint; actor mini-batch size is set dynamically.
        return dp_size


try:  # pragma: no cover - exercised on the remote veRL training host.
    import numpy as np
    import torch
    from tensordict import TensorDict

    from verl.protocol import DataProto
    from verl.trainer.distillation import is_distillation_enabled
    from verl.trainer.main_ppo_sync import KVBatchMeta, PPOTrainer, tq
    from verl.trainer.ppo.metric_utils import (
        compute_data_metrics,
        compute_throughout_metrics,
        compute_timing_metrics,
        compute_variance_proxy_metrics,
    )
    from verl.trainer.ppo.ray_trainer import compute_spec_decode_metrics
    from verl.utils.metric import reduce_metrics
    from verl.utils.py_functional import rename_dict
    from verl.workers.utils.padding import response_to_nested

    _VERL_AVAILABLE = True
except ModuleNotFoundError:  # Local CPU tests intentionally do not install veRL.
    _VERL_AVAILABLE = False


if _VERL_AVAILABLE:  # pragma: no branch
    class AgentFlowPPOTrainer(_AgentFlowTrainerMixin, PPOTrainer):
        def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
            metadata = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            valid_keys = valid_training_keys(
                keys=batch.keys,
                extra_fields=as_metadata_list(metadata["extra_fields"]),
            )
            metrics["agentflow/old_log_prob_row_count"] = float(len(valid_keys))
            if not valid_keys:
                metrics["agentflow/old_log_prob_skipped"] = 1.0
                return batch
            super()._compute_old_log_prob(batch.select_keys(list(valid_keys)), metrics)
            metrics["agentflow/old_log_prob_skipped"] = 0.0
            return batch

        def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["response_mask", "rm_scores", "extra_fields"],
            )
            response_mask = data["response_mask"]
            padded = data.select("response_mask", "rm_scores").to_padded_tensor()
            rewards = padded["rm_scores"].sum(dim=-1).tolist()
            metadata = as_metadata_list(data["extra_fields"])
            rows = build_trajectory_turns(
                keys=batch.keys,
                rewards=rewards,
                extra_fields=metadata,
            )
            normalized = normalize_trajectory_turns(rows)
            scalar_advantages = torch.tensor(
                [normalized.advantages[key] for key in batch.keys],
                dtype=torch.float32,
                device=padded["response_mask"].device,
            )
            token_advantages = scalar_advantages.unsqueeze(-1) * padded["response_mask"]
            output = TensorDict(
                {
                    "advantages": response_to_nested(token_advantages, response_mask),
                    "returns": response_to_nested(token_advantages, response_mask),
                },
                batch_size=len(batch),
            )
            batch = put_batch_fields(put=tq.kv_batch_put, batch=batch, fields=output)
            self._agentflow_trainable_keys = normalized.trainable_keys
            metrics.update(advantage_metrics_dict(normalized.metrics))
            return batch

        def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
            trainable_keys = tuple(
                getattr(self, "_agentflow_trainable_keys", tuple(batch.keys))
            )
            metrics["agentflow/actor_trainable_turn_count"] = float(len(trainable_keys))
            metrics["agentflow/actor_trainable_turn_fraction"] = (
                len(trainable_keys) / len(batch) if len(batch) else 0.0
            )
            if not trainable_keys:
                metrics["agentflow/actor_update_skipped"] = 1.0
                return batch

            train_batch = batch.select_keys(list(trainable_keys))
            sequence_data = tq.kv_batch_get(
                keys=train_batch.keys,
                partition_id=train_batch.partition_id,
                select_fields=["input_ids"],
            )
            attention_mask = build_unpadded_attention_mask(sequence_data["input_ids"])
            train_batch = put_batch_fields(
                put=tq.kv_batch_put,
                batch=train_batch,
                fields=TensorDict(
                    {"attention_mask": attention_mask},
                    batch_size=len(train_batch),
                ),
            )
            actor_config = self.config.actor_rollout_ref.actor
            agentflow_config = self.config.get("agentflow", {})
            turn_mini_batch_size = int(
                agentflow_config.get("turn_mini_batch_size", actor_config.ppo_mini_batch_size)
            )
            turn_mini_batch_size = effective_turn_mini_batch_size(
                turn_count=len(train_batch),
                requested_size=turn_mini_batch_size,
            )
            calculate_entropy = actor_config.calculate_entropy or actor_config.entropy_coeff != 0.0
            distillation_use_topk = (
                self.distillation_config.distillation_loss.loss_settings.use_topk
                if is_distillation_enabled(self.config.get("distillation"))
                else False
            )
            train_batch.extra_info.update(actor_update_metadata(
                turn_count=len(train_batch),
                mini_batch_size=turn_mini_batch_size,
                ppo_epochs=actor_config.ppo_epochs,
                seed=actor_config.data_loader_seed,
                shuffle=actor_config.shuffle,
                temperature=self.config.actor_rollout_ref.rollout.temperature,
                calculate_entropy=calculate_entropy,
                distillation_use_topk=distillation_use_topk,
            ))
            output: TensorDict = self.actor_rollout_wg.update_actor(train_batch)
            output = rename_dict(output["metrics"], "actor/")
            output["perf/mfu/actor"] = output.pop("actor/mfu")
            metrics.update(reduce_metrics(output))
            metrics["agentflow/actor_turn_mini_batch_size"] = float(
                turn_mini_batch_size
            )
            metrics["agentflow/actor_update_skipped"] = 0.0
            return batch

        def _compute_metrics(
            self,
            batch: KVBatchMeta,
            metrics: dict,
            timing_raw: dict,
            global_steps: int,
            epoch: int,
        ) -> None:
            """Mirror veRL metrics with dense/nested AgentLoop length support."""
            non_padding_mask = np.array(
                [not tag.get("is_padding", False) for tag in batch.tags],
                dtype=bool,
            )
            fields = [
                "prompts",
                "responses",
                "response_mask",
                "values",
                "advantages",
                "returns",
                "rm_scores",
                "token_level_rewards",
                "num_turns",
            ]
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=fields,
            )
            num_turns = np.array(as_metadata_list(data.pop("num_turns")))
            prompt_lengths, generated_lengths = response_lengths(data)
            device = data["response_mask"].device
            prompt_length = torch.tensor(prompt_lengths, device=device)
            response_length = torch.tensor(generated_lengths, device=device)
            global_token_num = (prompt_length + response_length).tolist()

            spec_drafts = spec_accepts = spec_verifies = None
            mtp_config = getattr(self.config.actor_rollout_ref.model, "mtp", None)
            if mtp_config is not None and mtp_config.enable and mtp_config.enable_rollout:
                spec_data = tq.kv_batch_get(
                    keys=batch.keys,
                    partition_id=batch.partition_id,
                    select_fields=["extra_fields"],
                )
                extra_fields = as_metadata_list(spec_data["extra_fields"])
                spec_drafts = [item["spec_num_draft_tokens"] for item in extra_fields]
                spec_accepts = [item["spec_num_accepted_tokens"] for item in extra_fields]
                spec_verifies = [item["spec_num_verify_steps"] for item in extra_fields]

            data = data.to_padded_tensor()
            data["token_level_scores"] = data["rm_scores"]
            if "token_level_rewards" not in data:
                data["token_level_rewards"] = data["rm_scores"]
            data["prompt_length"] = prompt_length.float()
            data["response_length"] = response_length.float()
            proto = DataProto(batch=data, meta_info={"global_token_num": global_token_num})
            metrics_batch = proto.select_idxs(non_padding_mask) if non_padding_mask.any() else proto

            metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
            metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=proto, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(
                compute_throughout_metrics(
                    batch=proto,
                    timing_raw=timing_raw,
                    n_gpus=n_gpus,
                )
            )
            gradient_norm = metrics.get("actor/grad_norm")
            metrics.update(
                compute_variance_proxy_metrics(
                    batch=metrics_batch,
                    gradient_norm=gradient_norm,
                )
            )

            if non_padding_mask.any():
                num_turns = num_turns[non_padding_mask]
            metrics.update(
                {
                    "training/num_turns/mean": num_turns.mean(),
                    "training/num_turns/max": num_turns.max(),
                    "training/num_turns/min": num_turns.min(),
                }
            )
            metrics.update(
                compute_spec_decode_metrics(
                    spec_drafts,
                    spec_accepts,
                    spec_verifies,
                    non_padding_mask,
                )
            )
else:
    class AgentFlowPPOTrainer(_AgentFlowTrainerMixin):
        """Importable test stub; remote execution requires the pinned veRL stack."""


__all__ = [
    "AgentFlowPPOTrainer",
    "actor_update_metadata",
    "advantage_metrics_dict",
    "as_metadata_list",
    "build_unpadded_attention_mask",
    "build_trajectory_turns",
    "effective_turn_mini_batch_size",
    "put_batch_fields",
    "valid_training_keys",
]
