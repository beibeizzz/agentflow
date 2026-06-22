from __future__ import annotations

import argparse
import random
from pathlib import Path
import json
import sys
import time
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
GSM_DIR = PROJECT_DIR / "try_gsm8k_0522"
if str(GSM_DIR) not in sys.path:
    sys.path.insert(0, str(GSM_DIR))

from flowgrpo_light.grpo_objective import build_loss_items, train_step_grpo
from flowgrpo_light.agentflow_rollout import AgentFlowBatchRolloutRunner
from flowgrpo_light.frozen_client import FrozenClient
from flowgrpo_light.rollout import PlannerSample, RolloutResult
from flowgrpo_light.train_light_grpo import (
    append_jsonl,
    config_value,
    flatten_rollout_groups,
    iter_batches,
    load_rows,
    load_yaml_config,
)
from agentflow.models.memory import Memory
from try_ticket_agent.ticket_env.episode_io import parse_episode
from try_ticket_agent.ticket_env.solver_factory import construct_ticket_runtime


DEFAULT_CLIP_RANGE_LOW = 0.001
DEFAULT_CLIP_RANGE_HIGH = 0.003


def bind_ticket_runtime(runtime: Any) -> Any:
    runtime.solver._ticket_backend = runtime.backend
    return runtime.solver


def build_ticket_rollout_runner(
    *,
    policy: Any,
    frozen_model: str,
    frozen_base_url: str,
    max_steps: int,
    max_time: int,
    max_tokens: int,
    rollout_concurrency: int,
    planner_batch_size: int,
    planner_batch_timeout_s: float,
    think_mode: str,
    query_analysis_think_mode: str,
) -> AgentFlowBatchRolloutRunner:
    engine_name = frozen_model if frozen_model.startswith("vllm-") else f"vllm-{frozen_model}"

    def solver_factory() -> Any:
        runtime = construct_ticket_runtime(
            llm_engine_name=engine_name,
            base_url=frozen_base_url,
            max_steps=max_steps,
            max_time=max_time,
            max_tokens=max_tokens,
            temperature=0.0,
            think_mode=think_mode,
            query_analysis_think_mode=query_analysis_think_mode,
        )
        return bind_ticket_runtime(runtime)

    return AgentFlowBatchRolloutRunner(
        policy=policy,
        solver_factory=solver_factory,
        reset_solver=reset_ticket_solver,
        question_getter=lambda row: str(row["user_request"]),
        result_adapter=ticket_result_adapter,
        rollout_concurrency=rollout_concurrency,
        planner_batch_size=planner_batch_size,
        planner_batch_timeout_s=planner_batch_timeout_s,
        think_mode=think_mode,
    )


def reset_ticket_solver(solver: Any, row: dict[str, Any]) -> None:
    episode = parse_episode(row)
    backend = solver._ticket_backend
    backend.reset(episode.initial_state, episode.goal_spec)
    solver.memory = Memory()
    solver.max_steps = episode.max_steps
    solver.verifier.max_steps = episode.max_steps


def ticket_result_adapter(
    solver: Any,
    row: dict[str, Any],
    result: dict[str, Any],
    samples: list[PlannerSample],
) -> RolloutResult:
    step_count = int(result.get("step_count", 0))
    verification = solver.verifier.verify_final(solver.memory, step_count)
    return RolloutResult(
        reward=1.0 if verification.success else 0.0,
        answer=json.dumps(verification.to_dict(), ensure_ascii=False, sort_keys=True),
        samples=list(samples),
        memory=result.get("memory") or solver.memory.get_actions(),
        query_analysis=str(result.get("query_analysis") or ""),
        errors=[],
        valid_for_training=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ticket AgentFlow binary turn-level GSPO training.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    model_path = str(config_value(config, "model_path", "Qwen/Qwen3-0.6B"))
    train_file = Path(config_value(config, "train_file", "try_ticket_agent/data/generated/train.jsonl"))
    output_dir = Path(config_value(config, "output_dir", "try_ticket_agent/flowgrpo_general_2x40g/outputs"))
    if not train_file.is_file():
        raise SystemExit(f"Training file does not exist: {train_file}")

    import torch
    from flowgrpo_light.policy import PlannerPolicy

    seed = int(config_value(config, "seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    frozen_model = str(config_value(config, "frozen_model", "Qwen3-0.6B"))
    frozen_base_url = str(config_value(config, "frozen_base_url", "http://127.0.0.1:8000/v1"))
    FrozenClient(
        base_url=frozen_base_url,
        model=frozen_model,
        timeout=int(config_value(config, "frozen_timeout", 60)),
    ).check_model()
    policy = PlannerPolicy(
        model_path,
        lora_rank=int(config_value(config, "lora_rank", 64)),
        lora_alpha=int(config_value(config, "lora_alpha", 128)),
        lora_dropout=float(config_value(config, "lora_dropout", 0.0)),
        max_new_tokens=int(config_value(config, "planner_max_new_tokens", 256)),
        temperature=float(config_value(config, "planner_temperature", 1.2)),
        top_p=float(config_value(config, "planner_top_p", 0.95)),
        dtype=str(config_value(config, "dtype", "bfloat16")),
        gradient_checkpointing=bool(config_value(config, "gradient_checkpointing", True)),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.model.parameters() if parameter.requires_grad],
        lr=float(config_value(config, "learning_rate", 2e-6)),
        weight_decay=float(config_value(config, "weight_decay", 0.0)),
    )
    group_size = int(config_value(config, "group_size", 8))
    question_batch_size = int(config_value(config, "question_batch_size", 4))
    clip_low = float(config_value(config, "clip_range_low", DEFAULT_CLIP_RANGE_LOW))
    clip_high = float(config_value(config, "clip_range_high", DEFAULT_CLIP_RANGE_HIGH))
    if clip_low <= 0 or clip_high <= 0:
        raise SystemExit("clip ranges must be positive")
    runner = build_ticket_rollout_runner(
        policy=policy,
        frozen_model=frozen_model,
        frozen_base_url=frozen_base_url,
        max_steps=int(config_value(config, "max_steps", 3)),
        max_time=int(config_value(config, "max_time", 120)),
        max_tokens=int(config_value(config, "max_tokens", 512)),
        rollout_concurrency=int(config_value(config, "rollout_concurrency", 32)),
        planner_batch_size=int(config_value(config, "planner_batch_size", 32)),
        planner_batch_timeout_s=float(config_value(config, "planner_batch_timeout_s", 0.02)),
        think_mode=str(config_value(config, "think_mode", "off")),
        query_analysis_think_mode=str(config_value(config, "query_analysis_think_mode", "on")),
    )
    rows = load_rows(train_file)[: int(config_value(config, "max_train_items", 2500))]
    random.shuffle(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    step = 0
    started_at = time.time()
    try:
        for epoch in range(int(config_value(config, "epochs", 1))):
            for row_index, batch in iter_batches(rows, question_batch_size):
                step_started = time.time()
                groups = runner.run_batch(batch, group_size=group_size)
                rollouts, advantages, reward_groups, advantage_groups = flatten_rollout_groups(groups)
                stats = train_step_grpo(
                    policy=policy,
                    optimizer=optimizer,
                    rollouts=rollouts,
                    advantages=advantages,
                    clip_range_low=clip_low,
                    clip_range_high=clip_high,
                    max_grad_norm=float(config_value(config, "max_grad_norm", 1.0)),
                    logprob_micro_batch_size=int(config_value(config, "logprob_micro_batch_size", 8)),
                    policy_epochs=int(config_value(config, "policy_epochs", 2)),
                )
                step += 1
                flat_group = [item for group in groups for item in group]
                record = {
                    "step": step,
                    "epoch": epoch,
                    "row_index": row_index,
                    "episode_ids": [row["episode_id"] for row in batch],
                    "reward_groups": reward_groups,
                    "advantage_groups": advantage_groups,
                    "valid_rollout_count": len(rollouts),
                    "invalid_rollout_count": len(flat_group) - len(rollouts),
                    "nonzero_trajectory_count": sum(abs(value) >= 1e-8 for value in advantages),
                    "nonzero_turn_count": sum(
                        len(rollout.samples)
                        for rollout, advantage in zip(rollouts, advantages, strict=True)
                        if abs(advantage) >= 1e-8
                    ),
                    "token_count": sum(
                        len(sample.response_token_ids or []) for rollout in rollouts for sample in rollout.samples
                    ),
                    "zero_variance_group_count": sum(
                        all(value is None or abs(value) < 1e-8 for value in group)
                        for group in advantage_groups
                    ),
                    "status": "update" if stats else "skip_no_advantage",
                    "train_stats": stats,
                    "clip_range_low": clip_low,
                    "clip_range_high": clip_high,
                    "elapsed_s": round(time.time() - step_started, 3),
                }
                append_jsonl(metrics_path, record)
                if step % int(config_value(config, "save_every", 25)) == 0:
                    policy.save_adapter(str(output_dir / f"checkpoint_step_{step}"))
    finally:
        runner.close()
    final_adapter = output_dir / "final_adapter"
    policy.save_adapter(str(final_adapter))
    (output_dir / "train_summary.json").write_text(
        json.dumps(
            {"steps": step, "rows": len(rows), "elapsed_s": round(time.time() - started_at, 3)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "DEFAULT_CLIP_RANGE_HIGH",
    "DEFAULT_CLIP_RANGE_LOW",
    "build_loss_items",
    "bind_ticket_runtime",
    "flatten_rollout_groups",
    "reset_ticket_solver",
    "ticket_result_adapter",
    "train_step_grpo",
]


if __name__ == "__main__":
    main()
