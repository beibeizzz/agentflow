from __future__ import annotations

from dataclasses import dataclass, fields
import inspect
import json
from pathlib import Path
from typing import Any, Sequence
import uuid

import numpy as np

from agentflow_rl.rewards import (
    AdvantageResult,
    TurnAdvantageMetrics,
    TurnReward,
    compute_turn_advantages,
)
from agentflow_rl.runtime.contracts import TaskName
from agentflow_rl.sampling import select_dapo_groups

from .policy_identity import write_policy_update_id


def build_turn_rewards_from_metadata(
    *, keys: Sequence[str], extra_fields: Sequence[dict[str, Any] | None]
) -> list[TurnReward]:
    if len(keys) != len(extra_fields):
        raise ValueError("veRL keys and extra fields must align")
    rows = []
    for key, metadata in zip(keys, extra_fields, strict=True):
        item = metadata or {}
        rows.append(
            TurnReward(
                key=str(key),
                task_id=str(item["task_id"]),
                task_name=item["task_name"],
                trajectory_id=str(item["trajectory_id"]),
                turn_index=int(item["turn_index"]),
                terminal_reward=float(item["terminal_reward"]),
                process_score=(
                    None if item.get("process_score") is None else float(item["process_score"])
                ),
                valid_for_training=bool(item.get("valid_for_training", True)),
                prompt_group_id=str(item.get("uid") or item["task_id"]),
            )
        )
    return rows


def advantage_metrics_dict(metrics: TurnAdvantageMetrics) -> dict[str, float]:
    return {
        f"agentflow/{field.name}": float(getattr(metrics, field.name))
        for field in fields(metrics)
    }


def as_metadata_list(value: Any) -> list[Any]:
    return list(value.tolist()) if hasattr(value, "tolist") else list(value)


@dataclass(frozen=True)
class TurnMiniBatchLayout:
    """Fixed-size outer PPO batches plus minimal masked padding."""

    real_turn_count: int
    padded_turn_count: int
    padding_turn_count: int
    mini_batch_size: int
    optimizer_step_count: int
    minimum_real_turns_per_step: int
    maximum_real_turns_per_step: int


def fixed_turn_mini_batch_layout(
    *, turn_count: int, mini_batch_size: int
) -> TurnMiniBatchLayout:
    """Round a variable turn batch up to fixed-size PPO mini-batches.

    Real Planner turns are retained exactly once. The runtime appends synthetic
    rows whose response/loss masks are zero, matching current veRL V1 padding
    semantics. This keeps the optimizer batch bounded without the prime-number
    collapse caused by searching for a divisor of ``turn_count``.
    """
    if turn_count <= 0 or mini_batch_size <= 0:
        raise ValueError("turn count and mini-batch size must be positive")
    optimizer_step_count = (turn_count + mini_batch_size - 1) // mini_batch_size
    padded_turn_count = optimizer_step_count * mini_batch_size
    return TurnMiniBatchLayout(
        real_turn_count=turn_count,
        padded_turn_count=padded_turn_count,
        padding_turn_count=padded_turn_count - turn_count,
        mini_batch_size=mini_batch_size,
        optimizer_step_count=optimizer_step_count,
        minimum_real_turns_per_step=turn_count // optimizer_step_count,
        maximum_real_turns_per_step=(
            turn_count + optimizer_step_count - 1
        )
        // optimizer_step_count,
    )


def balanced_padded_key_order(
    *, real_keys: Sequence[str], padding_keys: Sequence[str], mini_batch_size: int
) -> tuple[str, ...]:
    """Distribute real rows evenly across fixed-size padded mini-batches."""
    if not real_keys or mini_batch_size <= 0:
        raise ValueError("real keys and mini-batch size must be positive")
    total = len(real_keys) + len(padding_keys)
    if total % mini_batch_size != 0:
        raise ValueError("real and padding keys must form complete mini-batches")
    step_count = total // mini_batch_size
    if step_count <= 0 or len(padding_keys) >= total:
        raise ValueError("each padded batch set must contain real rows")

    base, extra = divmod(len(real_keys), step_count)
    real_offset = 0
    padding_offset = 0
    ordered: list[str] = []
    for step_index in range(step_count):
        real_count = base + int(step_index < extra)
        padding_count = mini_batch_size - real_count
        ordered.extend(real_keys[real_offset : real_offset + real_count])
        ordered.extend(
            padding_keys[padding_offset : padding_offset + padding_count]
        )
        real_offset += real_count
        padding_offset += padding_count
    if real_offset != len(real_keys) or padding_offset != len(padding_keys):
        raise RuntimeError("balanced padded ordering did not consume every key")
    return tuple(ordered)


def zero_response_aligned_training_fields(
    response_mask: Any, *, include_reference: bool
) -> dict[str, Any]:
    """Build shape-safe no-op PPO fields for synthetic actor rows."""
    import torch

    zeros = torch.zeros_like(response_mask, dtype=torch.float32)
    fields_out = {
        "old_log_probs": zeros.clone(),
        "advantages": zeros.clone(),
        "returns": zeros.clone(),
    }
    if include_reference:
        fields_out["ref_log_prob"] = zeros.clone()
    return fields_out


def policy_freshness_metrics(
    metadata: Sequence[dict[str, Any] | None], *, expected_update_id: str
) -> dict[str, float]:
    revisions = {
        str(item["rollout_policy_revision"])
        for item in metadata
        if item and item.get("rollout_policy_revision")
    }
    update_ids = {
        str(item["rollout_policy_update_id"])
        for item in metadata
        if item and item.get("rollout_policy_update_id") is not None
    }
    return {
        "agentflow/rollout_policy_revision_count": float(len(revisions)),
        "agentflow/rollout_policy_update_id_count": float(len(update_ids)),
        "agentflow/rollout_policy_fresh": float(
            len(revisions) == 1
            and update_ids == {str(expected_update_id)}
        ),
    }


def bind_policy_update_identity(
    manager: Any, update_id_provider, *, identity_path: str | Path | None = None
) -> None:
    if getattr(manager, "_agentflow_identity_bound", False):
        return
    original = manager.update_weights

    def update_weights(global_steps=None):
        update_id = int(update_id_provider())
        if global_steps is not None and int(global_steps) != update_id:
            raise RuntimeError("veRL requested a conflicting policy update ID")
        result = original(global_steps=update_id)
        if inspect.isawaitable(result):
            async def wait_and_record():
                value = await result
                if identity_path is not None:
                    write_policy_update_id(identity_path, update_id)
                return value

            return wait_and_record()
        if identity_path is not None:
            write_policy_update_id(identity_path, update_id)
        return result

    manager.update_weights = update_weights
    manager._agentflow_identity_bound = True


def build_unpadded_attention_mask(input_ids: Any) -> Any:
    import torch

    if not getattr(input_ids, "is_nested", False):
        return torch.ones_like(input_ids, dtype=torch.int64)
    lengths = input_ids.offsets().diff()
    positions = torch.arange(int(lengths.max().item()), device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).to(torch.int64)


def actor_update_metadata(
    *, turn_count: int, mini_batch_size: int, ppo_epochs: int, seed: int,
    shuffle: bool, temperature: float, calculate_entropy: bool,
    distillation_use_topk: bool,
) -> dict[str, Any]:
    if min(turn_count, mini_batch_size, ppo_epochs) <= 0:
        raise ValueError("actor update sizes must be positive")
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


def dynamic_quota_for_step(step: int) -> dict[TaskName, int]:
    schedules = (
        {TaskName.AIME: 2, TaskName.TWOWIKI: 1, TaskName.TACO: 1},
        {TaskName.AIME: 1, TaskName.TWOWIKI: 2, TaskName.TACO: 1},
        {TaskName.AIME: 1, TaskName.TWOWIKI: 1, TaskName.TACO: 2},
    )
    return schedules[(step - 1) % len(schedules)]


@dataclass(frozen=True)
class ProcessAvailability:
    available_turns: int
    missing_turns: int
    missing_rate: float
    per_task_available: dict[TaskName, int]
    passed: bool


@dataclass(frozen=True)
class TrainingSelectionPlan:
    advantages: AdvantageResult
    trainable_keys: tuple[str, ...]
    metrics: dict[str, float]


def process_availability_gate(
    rows: Sequence[TurnReward],
    *,
    max_missing_rate: float,
    require_each_task: bool,
) -> ProcessAvailability:
    if not 0.0 <= max_missing_rate <= 1.0:
        raise ValueError("maximum process-score missing rate must be within [0, 1]")
    valid = [row for row in rows if row.valid_for_training]
    available = [row for row in valid if row.process_score is not None]
    missing = len(valid) - len(available)
    per_task_total = {
        task: sum(row.task_name is task for row in valid)
        for task in (TaskName.AIME, TaskName.TWOWIKI, TaskName.TACO)
    }
    per_task_available = {
        task: sum(row.task_name is task for row in available)
        for task in (TaskName.AIME, TaskName.TWOWIKI, TaskName.TACO)
    }
    missing_rate = missing / len(valid) if valid else 1.0
    task_support = all(
        per_task_total[task] == 0 or per_task_available[task] > 0
        for task in per_task_total
    )
    return ProcessAvailability(
        available_turns=len(available),
        missing_turns=missing,
        missing_rate=missing_rate,
        per_task_available=per_task_available,
        passed=bool(valid)
        and missing_rate <= max_missing_rate
        and (task_support or not require_each_task),
    )


def build_training_selection(
    rows: Sequence[TurnReward],
    metadata: Sequence[dict[str, Any] | None],
    *,
    expected_update_id: str,
    lambda_process: float,
    max_advantage: float,
    process_required: bool,
    max_missing_rate: float,
    require_each_task: bool,
    dynamic_enabled: bool,
    per_task_quota: dict[TaskName, int],
    train_prompt_group_count: int,
    expected_trajectories_per_group: int,
) -> TrainingSelectionPlan:
    result = compute_turn_advantages(
        rows,
        lambda_process=lambda_process,
        max_advantage=max_advantage,
    )
    metrics = advantage_metrics_dict(result.metrics)
    freshness = policy_freshness_metrics(
        metadata, expected_update_id=expected_update_id
    )
    metrics.update(freshness)
    trainable_keys = result.trainable_keys
    if freshness["agentflow/rollout_policy_fresh"] != 1.0:
        trainable_keys = ()

    if process_required:
        availability = process_availability_gate(
            rows,
            max_missing_rate=max_missing_rate,
            require_each_task=require_each_task,
        )
        metrics.update(
            {
                "agentflow/process_available_turn_count": float(
                    availability.available_turns
                ),
                "agentflow/process_missing_turn_count_gate": float(
                    availability.missing_turns
                ),
                "agentflow/process_missing_rate": availability.missing_rate,
                "agentflow/process_availability_gate_passed": float(
                    availability.passed
                ),
            }
        )
        for task, count in availability.per_task_available.items():
            metrics[f"agentflow/process_available_{task.value}"] = float(count)
        if not availability.passed:
            trainable_keys = ()

    if dynamic_enabled:
        selection = select_dapo_groups(
            rows,
            trainable_keys=trainable_keys,
            per_task_quota=per_task_quota,
            target_group_count=train_prompt_group_count,
            expected_trajectories_per_group=expected_trajectories_per_group,
        )
        trainable_keys = selection.trainable_keys
        metrics.update(
            {
                "agentflow/dynamic_sampling_complete": float(selection.complete),
                "agentflow/dynamic_candidate_group_count": float(
                    selection.candidate_group_count
                ),
                "agentflow/dynamic_informative_group_count": float(
                    selection.informative_group_count
                ),
                "agentflow/dynamic_filtered_group_count": float(
                    selection.filtered_group_count
                ),
                "agentflow/dynamic_discarded_qualified_group_count": float(
                    selection.discarded_informative_group_count
                ),
                "agentflow/dynamic_kept_group_count": float(
                    len(selection.selected_group_ids)
                ),
                "agentflow/dynamic_dropped_group_count": float(
                    selection.dropped_group_count
                ),
                "agentflow/dynamic_target_group_count": float(
                    selection.target_group_count
                ),
            }
        )
        for task in (TaskName.AIME, TaskName.TWOWIKI, TaskName.TACO):
            metrics[f"agentflow/dynamic_kept_{task.value}"] = float(
                selection.per_task_selected[task] if selection.complete else 0
            )
    return TrainingSelectionPlan(
        advantages=result,
        trainable_keys=tuple(trainable_keys),
        metrics=metrics,
    )


def training_state_path(root: str | Path, global_step: int) -> Path:
    return Path(root) / f"global_step_{global_step}" / "agentflow_state.json"


def write_step_metrics(
    root: str | Path, global_step: int, metrics: dict[str, Any]
) -> Path:
    target = Path(root) / "agentflow_metrics" / f"step_{global_step}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(key): value
        for key, value in metrics.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


class _AgentFlowTrainerMixin:
    def fit(self):
        from .training_loop import fit_data_epochs

        return fit_data_epochs(self)

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        if dp_size <= 0:
            raise ValueError("data-parallel size must be positive")
        return dp_size


try:  # pragma: no cover - exercised by the pinned veRL runtime.
    import torch
    from tensordict import TensorDict

    from verl.trainer.distillation import is_distillation_enabled
    from verl.trainer.main_ppo_sync import KVBatchMeta, PPOTrainer, tq, tu
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
    from verl.utils.debug import marked_timer
    from verl.utils.metric import reduce_metrics
    from verl.utils.py_functional import rename_dict
    from verl.workers.utils.padding import response_to_nested

    _VERL_AVAILABLE = True
except ModuleNotFoundError:
    _VERL_AVAILABLE = False


if _VERL_AVAILABLE:
    class AgentFlowPPOTrainer(_AgentFlowTrainerMixin, PPOTrainer):
        def init_workers(self):
            result = super().init_workers()
            bind_policy_update_identity(
                self.checkpoint_manager,
                self._successful_updates,
                identity_path=str(
                    self.config.agentflow.get(
                        "policy_identity_path",
                        Path(self.config.trainer.default_local_dir)
                        / "policy_identity.json",
                    )
                ),
            )
            return result

        def _successful_updates(self) -> int:
            return int(getattr(self, "_agentflow_successful_update_count", 0))

        def _collection_batches(self) -> int:
            return int(getattr(self, "_agentflow_collection_batch_count", 0))

        def _load_checkpoint(self):
            super()._load_checkpoint()
            self._agentflow_successful_update_count = 0
            self._agentflow_collection_batch_count = 0
            if self.global_steps <= 0:
                return
            path = training_state_path(
                self.config.trainer.default_local_dir, self.global_steps
            )
            if not path.exists():
                raise RuntimeError(f"missing AgentFlow training state: {path}")
            state = json.loads(path.read_text(encoding="utf-8"))
            self._agentflow_successful_update_count = int(state["successful_updates"])
            self._agentflow_collection_batch_count = int(
                state.get("collection_batches", state.get("collection_step", self.global_steps))
            )
            marker = Path(self.config.trainer.default_local_dir) / "agentflow_reload.json"
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "loaded_collection_step": int(self.global_steps),
                        "loaded_successful_updates": self._successful_updates(),
                        "state_sha256": __import__("hashlib").sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(marker)

        def _save_checkpoint(self):
            super()._save_checkpoint()
            path = training_state_path(
                self.config.trainer.default_local_dir, self.global_steps
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "collection_step": int(self.global_steps),
                        "collection_batches": self._collection_batches(),
                        "successful_updates": self._successful_updates(),
                        "configured_data_epochs": int(self.config.trainer.total_epochs),
                        "expected_collection_batches": int(
                            len(self.train_dataloader) * self.config.trainer.total_epochs
                        ),
                        "data_epoch_complete": bool(
                            self._collection_batches()
                            >= len(self.train_dataloader) * self.config.trainer.total_epochs
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)

        def _prepare_training_selection(
            self,
            *,
            keys: Sequence[str],
            metadata: Sequence[dict[str, Any] | None],
        ) -> TrainingSelectionPlan:
            agentflow = self.config.get("agentflow", {})
            process_config = agentflow.get("process_reward", {})
            lambda_process = float(agentflow.get("lambda_process", 0.0))
            plan = build_training_selection(
                build_turn_rewards_from_metadata(keys=keys, extra_fields=metadata),
                metadata,
                expected_update_id=str(self._successful_updates()),
                lambda_process=lambda_process,
                max_advantage=float(agentflow.get("max_advantage", 5.0)),
                process_required=(
                    lambda_process > 0.0
                    and str(process_config.get("mode", "none")) != "none"
                ),
                max_missing_rate=float(
                    process_config.get("max_missing_rate", 0.05)
                ),
                require_each_task=bool(
                    process_config.get("require_each_task", True)
                ),
                dynamic_enabled=bool(
                    agentflow.get("dynamic_sampling", {}).get("enabled", False)
                ),
                per_task_quota=dynamic_quota_for_step(
                    self._successful_updates() + 1
                ),
                train_prompt_group_count=int(self.config.data.train_batch_size),
                expected_trajectories_per_group=int(
                    self.config.actor_rollout_ref.rollout.n
                ),
            )
            self._agentflow_selection_batch_keys = tuple(str(key) for key in keys)
            self._agentflow_selection_plan = plan
            self._agentflow_trainable_keys = plan.trainable_keys
            return plan

        def _generate_prompt_batch(self, batch_dict: dict) -> None:
            limit = int(self.config.get("agentflow", {}).get("max_collection_steps", 1000))
            if self._collection_batches() >= limit:
                raise RuntimeError(
                    f"collection limit {limit} exhausted with "
                    f"{self._collection_batches()} prompt batches collected and "
                    f"{self._successful_updates()} successful actor updates"
                )
            prompt_batch = dict(batch_dict)
            prompt_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(prompt_batch["raw_prompt"]))], dtype=object
            )
            batch = tu.get_tensordict(prompt_batch)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            self.async_rollout_manager.generate_sequences(batch)
            self._agentflow_collection_batch_count = self._collection_batches() + 1

        def step(self, batch_dict: dict, metrics: dict, timing_raw: dict) -> KVBatchMeta:
            """Collect DAPO groups to quota before old log-prob and actor work."""
            if self.config.algorithm.adv_estimator == core_algos.AdvantageEstimator.REMAX:
                raise ValueError("AgentFlow DAPO collection does not support REMAX")
            dynamic = self.config.get("agentflow", {}).get("dynamic_sampling", {})
            enabled = bool(dynamic.get("enabled", False))
            max_batches = int(dynamic.get("max_num_gen_batches", 1)) if enabled else 1
            if max_batches <= 0:
                raise ValueError("dynamic_sampling.max_num_gen_batches must be positive")
            current_prompts = batch_dict
            batch = None
            plan = None
            epoch_tail_incomplete = False
            with marked_timer("gen", timing_raw, color="red"):
                for round_index in range(max_batches):
                    self._generate_prompt_batch(current_prompts)
                    batch = self.replay_buffer.sample(
                        partition_id="train", global_steps=self.global_steps
                    )
                    metadata_data = tq.kv_batch_get(
                        keys=batch.keys,
                        partition_id=batch.partition_id,
                        select_fields=["extra_fields"],
                    )
                    plan = self._prepare_training_selection(
                        keys=batch.keys,
                        metadata=as_metadata_list(metadata_data["extra_fields"]),
                    )
                    metrics["agentflow/dynamic_generation_batches"] = float(round_index + 1)
                    metrics["agentflow/dynamic_accumulated_turn_rows"] = float(len(batch.keys))
                    if not enabled or plan.metrics.get("agentflow/dynamic_sampling_complete") == 1.0:
                        break
                    provider = getattr(self, "_agentflow_replenishment_provider", None)
                    if provider is None:
                        raise RuntimeError("DAPO replenishment requires a fresh prompt-batch provider")
                    current_prompts = provider()
                    if current_prompts is None:
                        epoch_tail_incomplete = True
                        break
                else:
                    raise ValueError(
                        f"DAPO dynamic sampling failed to fill task quotas after {max_batches} generation batches"
                    )
            assert batch is not None and plan is not None
            metrics["agentflow/dynamic_epoch_tail_incomplete"] = float(
                epoch_tail_incomplete
            )
            self.checkpoint_manager.sleep_replicas()

            if self.reward_loop_manager.reward_loop_worker_handles is None:
                with marked_timer("reward", timing_raw, color="yellow"):
                    batch = self._compute_reward_colocate(batch)
            batch = self._balance_batch(batch, metrics=metrics)
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                batch = self._compute_old_log_prob(batch, metrics=metrics)
            if self.use_reference_policy:
                with marked_timer("ref", timing_raw, color="olive"):
                    batch = self._compute_ref_log_prob(batch, metrics=metrics)
            if self.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    batch = self._compute_values(batch, metrics=metrics)
            with marked_timer("adv", timing_raw, color="brown"):
                batch = self._compute_advantage(batch, metrics=metrics)
            if self.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    batch = self._update_critic(batch, metrics=metrics)
            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch = self._update_actor(batch, metrics=metrics)
            return batch

        def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["response_mask", "extra_fields"],
            )
            metadata = as_metadata_list(data["extra_fields"])
            batch_keys = tuple(str(key) for key in batch.keys)
            if getattr(self, "_agentflow_selection_batch_keys", ()) == batch_keys:
                plan = self._agentflow_selection_plan
            else:
                plan = self._prepare_training_selection(
                    keys=batch.keys, metadata=metadata
                )
            result = plan.advantages
            trainable_keys = plan.trainable_keys
            padded_mask = data.select("response_mask").to_padded_tensor()["response_mask"]
            scalar = torch.tensor(
                [result.combined[str(key)] for key in batch.keys],
                dtype=torch.float32,
                device=padded_mask.device,
            )
            token_advantages = scalar.unsqueeze(-1) * padded_mask
            fields_out = TensorDict(
                {
                    "advantages": response_to_nested(token_advantages, data["response_mask"]),
                    "returns": response_to_nested(token_advantages, data["response_mask"]),
                },
                batch_size=len(batch),
            )
            batch = tq.kv_batch_put(
                keys=batch.keys,
                partition_id=batch.partition_id,
                fields=fields_out,
            )
            self._agentflow_trainable_keys = trainable_keys
            metrics["agentflow/successful_update_count"] = float(
                self._successful_updates()
            )
            metrics.update(plan.metrics)
            return batch

        def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            metadata = as_metadata_list(data["extra_fields"])
            plan = self._prepare_training_selection(keys=batch.keys, metadata=metadata)
            metrics.update(plan.metrics)
            if plan.trainable_keys:
                super()._compute_old_log_prob(
                    batch.select_keys(list(plan.trainable_keys)), metrics
                )
            metrics["agentflow/old_log_prob_row_count"] = float(
                len(plan.trainable_keys)
            )
            metrics["agentflow/candidate_turn_row_count"] = float(len(batch.keys))
            return batch

        def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
            keys = tuple(getattr(self, "_agentflow_trainable_keys", tuple(batch.keys)))
            if not keys:
                metrics["agentflow/actor_update_skipped"] = 1.0
                write_step_metrics(
                    self.config.trainer.default_local_dir,
                    int(self.global_steps),
                    metrics,
                )
                return batch
            train_batch = batch.select_keys(list(keys))
            actor = self.config.actor_rollout_ref.actor
            agentflow = self.config.get("agentflow", {})
            mini_batch_size = int(
                agentflow.get("turn_mini_batch_size", actor.ppo_mini_batch_size)
            )
            layout = fixed_turn_mini_batch_layout(
                turn_count=len(train_batch), mini_batch_size=mini_batch_size
            )
            train_batch = upsample_batch_to_divisible_size(
                train_batch,
                batch_multiple=layout.mini_batch_size,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            if len(train_batch) != layout.padded_turn_count:
                raise RuntimeError(
                    "veRL padding produced an unexpected actor batch size: "
                    f"expected {layout.padded_turn_count}, got {len(train_batch)}"
                )
            if layout.padding_turn_count:
                # AgentFlow pads after old-log-prob and advantage construction.
                # veRL's helper copies the source row before replacing its
                # response mask, so rebuild every response-aligned training
                # field to match the synthetic one-token, fully masked row.
                padding_keys = train_batch.keys[layout.real_turn_count :]
                padding_data = tq.kv_batch_get(
                    keys=padding_keys,
                    partition_id=train_batch.partition_id,
                    select_fields=["response_mask"],
                )
                zero_fields = zero_response_aligned_training_fields(
                    padding_data["response_mask"],
                    include_reference=self.use_reference_policy,
                )
                tq.kv_batch_put(
                    keys=padding_keys,
                    partition_id=train_batch.partition_id,
                    fields=TensorDict(
                        zero_fields, batch_size=layout.padding_turn_count
                    ),
                )
                train_batch = train_batch.select_keys(
                    list(
                        balanced_padded_key_order(
                            real_keys=train_batch.keys[: layout.real_turn_count],
                            padding_keys=padding_keys,
                            mini_batch_size=layout.mini_batch_size,
                        )
                    )
                )
            data = tq.kv_batch_get(
                keys=train_batch.keys,
                partition_id=train_batch.partition_id,
                select_fields=["input_ids"],
            )
            attention_mask = build_unpadded_attention_mask(data["input_ids"])
            train_batch = tq.kv_batch_put(
                keys=train_batch.keys,
                partition_id=train_batch.partition_id,
                fields=TensorDict({"attention_mask": attention_mask}, batch_size=len(train_batch)),
            )
            distillation_topk = (
                self.distillation_config.distillation_loss.loss_settings.use_topk
                if is_distillation_enabled(self.config.get("distillation"))
                else False
            )
            train_batch.extra_info.update(
                actor_update_metadata(
                    turn_count=len(train_batch),
                    mini_batch_size=layout.mini_batch_size,
                    ppo_epochs=actor.ppo_epochs,
                    seed=actor.data_loader_seed,
                    shuffle=actor.shuffle,
                    temperature=self.config.actor_rollout_ref.rollout.temperature,
                    calculate_entropy=actor.calculate_entropy or actor.entropy_coeff != 0.0,
                    distillation_use_topk=distillation_topk,
                )
            )
            output = self.actor_rollout_wg.update_actor(train_batch)
            self._agentflow_successful_update_count = self._successful_updates() + 1
            output = rename_dict(output["metrics"], "actor/")
            if "actor/mfu" in output:
                output["perf/mfu/actor"] = output.pop("actor/mfu")
            metrics.update(reduce_metrics(output))
            metrics["agentflow/actor_turn_mini_batch_size"] = float(
                layout.mini_batch_size
            )
            metrics["agentflow/actor_real_turn_count"] = float(
                layout.real_turn_count
            )
            metrics["agentflow/actor_padded_turn_count"] = float(
                layout.padded_turn_count
            )
            metrics["agentflow/actor_padding_turn_count"] = float(
                layout.padding_turn_count
            )
            metrics["agentflow/actor_padding_fraction"] = (
                layout.padding_turn_count / layout.padded_turn_count
            )
            metrics["agentflow/actor_optimizer_step_count"] = float(
                layout.optimizer_step_count * int(actor.ppo_epochs)
            )
            metrics["agentflow/actor_min_real_turns_per_step"] = float(
                layout.minimum_real_turns_per_step
            )
            metrics["agentflow/actor_max_real_turns_per_step"] = float(
                layout.maximum_real_turns_per_step
            )
            metrics["agentflow/actor_update_skipped"] = 0.0
            metrics["agentflow/successful_update_count"] = float(
                self._successful_updates()
            )
            write_step_metrics(
                self.config.trainer.default_local_dir,
                int(self.global_steps),
                metrics,
            )
            return batch
else:
    class AgentFlowPPOTrainer(_AgentFlowTrainerMixin):
        """Importable CPU stub; formal execution loads the pinned veRL trainer."""


__all__ = [
    "AgentFlowPPOTrainer",
    "actor_update_metadata",
    "advantage_metrics_dict",
    "balanced_padded_key_order",
    "build_turn_rewards_from_metadata",
    "build_training_selection",
    "bind_policy_update_identity",
    "build_unpadded_attention_mask",
    "fixed_turn_mini_batch_layout",
    "dynamic_quota_for_step",
    "policy_freshness_metrics",
    "process_availability_gate",
    "training_state_path",
    "TrainingSelectionPlan",
    "TurnMiniBatchLayout",
    "write_step_metrics",
    "zero_response_aligned_training_fields",
]
