from __future__ import annotations

# TaskRunner structure follows verl-project/verl v0.8.0 (Apache-2.0).

import argparse
from pathlib import Path
from pprint import pprint
from typing import Any

from .verl_trainer import AgentFlowPPOTrainer


def load_config(path: str | Path, overrides: list[str] | None = None):
    import verl.trainer.main_ppo_sync as main_ppo_sync
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(main_ppo_sync.__file__).resolve().parent / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        base = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(base, False)
    config = OmegaConf.merge(base, OmegaConf.load(Path(path).resolve()))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config


def build_task_runner():  # pragma: no cover - requires Ray/CUDA.
    import ray
    import transfer_queue as tq
    from omegaconf import OmegaConf
    from verl.single_controller.ray import ResourcePoolManager
    from verl.trainer.distillation import is_distillation_enabled
    from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

    class AgentFlowActorRolloutRefWorker(ActorRolloutRefWorker):
        def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
            from agentflow_rl.integrations.verl_padding import no_padding_to_padding
            from verl.workers.utils import losses

            losses.no_padding_2_padding = no_padding_to_padding
            rollout_custom = config.rollout.get("custom") or {}
            if rollout_custom.get("force_shm_weight_transfer", False):
                from verl.workers.rollout.vllm_rollout import vllm_rollout

                vllm_rollout.is_support_ipc = lambda: False
            super().__init__(config, *args, **kwargs)

    @ray.remote
    class AgentFlowTaskRunner:
        def __init__(self) -> None:
            self.role_worker_mapping = {}
            self.mapping = {}

        def add_actor_rollout_worker(self, config: Any) -> None:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or (
                config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            )
            role = (
                Role.ActorRolloutRef
                if need_reference_policy(config) and not ref_in_actor
                else Role.ActorRollout
            )
            self.role_worker_mapping[role] = ray.remote(AgentFlowActorRolloutRefWorker)
            self.mapping[role] = "global_pool"

        def add_critic_worker(self, config: Any) -> None:
            if need_critic(config):
                self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
                self.mapping[Role.Critic] = "global_pool"

        def init_resource_pool(self, config: Any) -> None:
            resources = {
                "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes
            }
            if config.reward.reward_model.enable_resource_pool:
                resources["reward_pool"] = [
                    config.reward.reward_model.n_gpus_per_node
                ] * config.reward.reward_model.nnodes
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"
            distillation = config.get("distillation")
            if is_distillation_enabled(distillation):
                resources["teacher_pool"] = [
                    distillation.n_gpus_per_node
                ] * distillation.nnodes
                self.mapping[Role.TeacherModel] = "teacher_pool"
            self.resource_pool_manager = ResourcePoolManager(
                resource_pool_spec=resources,
                mapping=self.mapping,
            )

        def run(self, config: Any) -> None:
            pprint(OmegaConf.to_container(config, resolve=True))
            OmegaConf.resolve(config)
            tq.init(config.transfer_queue)
            trainer = None
            try:
                self.add_actor_rollout_worker(config)
                self.add_critic_worker(config)
                self.init_resource_pool(config)
                trainer = AgentFlowPPOTrainer(
                    config=config,
                    role_worker_mapping=self.role_worker_mapping,
                    resource_pool_manager=self.resource_pool_manager,
                )
                trainer.init_workers()
                trainer.fit()
            finally:
                if trainer is not None:
                    trainer.replay_buffer.close()
                tq.close()

    return AgentFlowTaskRunner


def validate(config) -> None:
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    config.transfer_queue.enable = True
    dynamic = config.get("agentflow", {}).get("dynamic_sampling", {})
    if dynamic.get("enabled", False):
        if str(dynamic.get("metric", "")) != "terminal_reward":
            raise ValueError("DAPO dynamic sampling metric must be terminal_reward")
        if int(dynamic.get("max_num_gen_batches", 0)) <= 0:
            raise ValueError("DAPO max_num_gen_batches must be positive")
        if int(config.data.train_batch_size) <= 0:
            raise ValueError("DAPO data.train_batch_size must be positive")
        if int(config.data.gen_batch_size) <= 0:
            raise ValueError("DAPO data.gen_batch_size must be positive")
    model = config.actor_rollout_ref.model
    if int(model.get("lora_rank", 0)) <= 0:
        raise ValueError("formal Planner training requires a positive LoRA rank")
    if int(model.get("lora_alpha", 0)) <= 0:
        raise ValueError("formal Planner training requires a positive LoRA alpha")
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )


def run(config) -> None:  # pragma: no cover - requires Ray/CUDA.
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    validate(config)
    run_ppo(config, task_runner_class=build_task_runner())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AgentFlow Beta through veRL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    if args.dry_run:
        from omegaconf import OmegaConf

        validate(config)
        print(OmegaConf.to_yaml(config, resolve=True))
        return 0
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
