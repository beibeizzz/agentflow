from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowgrpo_light.frozen_client import FrozenClient
from flowgrpo_light.train_light_grpo import (
    append_jsonl,
    config_value,
    flatten_rollout_groups,
    iter_batches,
    load_rows,
    load_yaml_config,
)
from flowgrpo_light.grpo_objective import train_step_grpo


def summarize_rewards(rewards: list[float]) -> dict[str, float | int]:
    if not rewards:
        return {
            "reward_count": 0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "reward_min": 0.0,
            "reward_max": 0.0,
        }
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / len(rewards))
    return {
        "reward_count": len(rewards),
        "reward_mean": mean,
        "reward_std": std,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
    }


def count_nonzero_advantages(advantages: Iterable[float | None], eps: float = 1e-8) -> int:
    return sum(1 for advantage in advantages if advantage is not None and abs(float(advantage)) >= eps)


def count_effective_samples(rollouts: list[Any], advantages: list[float], eps: float = 1e-8) -> int:
    count = 0
    for rollout, advantage in zip(rollouts, advantages, strict=True):
        if abs(float(advantage)) >= eps:
            count += len(getattr(rollout, "samples", []))
    return count


def gpu_memory_snapshot(torch_module: Any) -> dict[str, int | str | bool]:
    if not torch_module.cuda.is_available():
        return {"cuda": False}
    device = torch_module.cuda.current_device()
    return {
        "cuda": True,
        "device": str(device),
        "allocated_mib": int(torch_module.cuda.memory_allocated(device) / 1024 / 1024),
        "reserved_mib": int(torch_module.cuda.memory_reserved(device) / 1024 / 1024),
        "max_allocated_mib": int(torch_module.cuda.max_memory_allocated(device) / 1024 / 1024),
        "max_reserved_mib": int(torch_module.cuda.max_memory_reserved(device) / 1024 / 1024),
    }


def collect_planner_raw_outputs(row_batch: list[dict[str, Any]], rollout_groups: list[list[Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_batch_index, (row, group) in enumerate(zip(row_batch, rollout_groups, strict=True)):
        row_id = row.get("id") or row.get("pid")
        for rollout_index, rollout in enumerate(group):
            for planner_step, sample in enumerate(getattr(rollout, "samples", []), start=1):
                records.append(
                    {
                        "id": row_id,
                        "row_batch_index": row_batch_index,
                        "rollout_index": rollout_index,
                        "planner_step": planner_step,
                        "response": getattr(sample, "response", ""),
                    }
                )
    return records


def print_planner_raw_outputs(step: int, records: list[dict[str, Any]]) -> None:
    print(f"[planner_next_step_raw_outputs] step={step} count={len(records)}")
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    print("[/planner_next_step_raw_outputs]")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planner-only GRPO training with old-logprob ratio clipping for GSM8K.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frozen-base-url", default=None)
    parser.add_argument("--frozen-model", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--question-batch-size", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--rollout-concurrency", type=int, default=None)
    parser.add_argument("--planner-batch-size", type=int, default=None)
    parser.add_argument("--planner-batch-timeout-s", type=float, default=None)
    parser.add_argument("--logprob-micro-batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--policy-epochs", type=int, default=None)
    parser.add_argument("--rollout-backend", choices=["agentflow", "light"], default=None)
    parser.add_argument("--think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--query-analysis-think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--final-output-think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--verifier-think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--quiet-rollout", action="store_true")
    parser.add_argument("--planner-output-every", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_yaml_config(args.config)

    model_path = Path(args.model_path or config_value(config, "model_path", "model/Qwen/Qwen3-0.6B"))
    train_file = Path(args.train_file or config_value(config, "train_file", "try_gsm8k_0522/data/train/gsm8k_smoke_train.parquet"))
    output_dir = Path(args.output_dir or config_value(config, "output_dir", "try_gsm8k_0522/flowgrpo_light/outputs/smoke_12g"))
    frozen_base_url = args.frozen_base_url or config_value(config, "frozen_base_url", "http://localhost:8000/v1")
    frozen_model = args.frozen_model or config_value(config, "frozen_model", "Qwen3-0.6B")
    seed = int(args.seed if args.seed is not None else config_value(config, "seed", 0))
    epochs = int(args.epochs if args.epochs is not None else config_value(config, "epochs", 1))
    max_train_items = int(
        args.max_train_items if args.max_train_items is not None else config_value(config, "max_train_items", 8)
    )
    group_size = int(args.group_size if args.group_size is not None else config_value(config, "group_size", 2))
    question_batch_size = int(
        args.question_batch_size
        if args.question_batch_size is not None
        else config_value(config, "question_batch_size", 1)
    )
    if question_batch_size < 1:
        raise SystemExit("question_batch_size must be >= 1")
    if group_size < 1:
        raise SystemExit("group_size must be >= 1")
    rollout_concurrency = int(
        args.rollout_concurrency
        if args.rollout_concurrency is not None
        else config_value(config, "rollout_concurrency", question_batch_size * group_size)
    )
    planner_batch_size = int(
        args.planner_batch_size
        if args.planner_batch_size is not None
        else config_value(config, "planner_batch_size", question_batch_size * group_size)
    )
    if rollout_concurrency < 1:
        raise SystemExit("rollout_concurrency must be >= 1")
    if planner_batch_size < 1:
        raise SystemExit("planner_batch_size must be >= 1")
    planner_batch_timeout_s = float(
        args.planner_batch_timeout_s
        if args.planner_batch_timeout_s is not None
        else config_value(config, "planner_batch_timeout_s", 0.02)
    )
    if planner_batch_timeout_s < 0:
        raise SystemExit("planner_batch_timeout_s must be >= 0")
    logprob_micro_batch_size = int(
        args.logprob_micro_batch_size
        if args.logprob_micro_batch_size is not None
        else config_value(config, "logprob_micro_batch_size", 8)
    )
    if logprob_micro_batch_size < 1:
        raise SystemExit("logprob_micro_batch_size must be >= 1")
    max_steps = int(args.max_steps if args.max_steps is not None else config_value(config, "max_steps", 3))
    lora_rank = int(args.lora_rank if args.lora_rank is not None else config_value(config, "lora_rank", 8))
    lora_alpha = int(args.lora_alpha if args.lora_alpha is not None else config_value(config, "lora_alpha", 16))
    learning_rate = float(
        args.learning_rate if args.learning_rate is not None else config_value(config, "learning_rate", 5e-5)
    )
    clip_range = float(args.clip_range if args.clip_range is not None else config_value(config, "clip_range", 0.2))
    if clip_range <= 0:
        raise SystemExit("clip_range must be > 0")
    policy_epochs = int(
        args.policy_epochs if args.policy_epochs is not None else config_value(config, "policy_epochs", 1)
    )
    if policy_epochs < 1:
        raise SystemExit("policy_epochs must be >= 1")
    rollout_backend = str(
        args.rollout_backend if args.rollout_backend is not None else config_value(config, "rollout_backend", "agentflow")
    )
    if rollout_backend not in {"agentflow", "light"}:
        raise SystemExit(f"Unsupported rollout_backend: {rollout_backend}")
    think_mode = str(args.think_mode if args.think_mode is not None else config_value(config, "think_mode", "default"))
    if think_mode not in {"default", "on", "off"}:
        raise SystemExit(f"Unsupported think_mode: {think_mode}")
    query_analysis_think_mode = str(
        args.query_analysis_think_mode
        if args.query_analysis_think_mode is not None
        else config_value(config, "query_analysis_think_mode", think_mode)
    )
    final_output_think_mode = str(
        args.final_output_think_mode
        if args.final_output_think_mode is not None
        else config_value(config, "final_output_think_mode", think_mode)
    )
    verifier_think_mode = str(
        args.verifier_think_mode
        if args.verifier_think_mode is not None
        else config_value(config, "verifier_think_mode", think_mode)
    )
    for name, mode in {
        "query_analysis_think_mode": query_analysis_think_mode,
        "final_output_think_mode": final_output_think_mode,
        "verifier_think_mode": verifier_think_mode,
    }.items():
        if mode not in {"default", "on", "off"}:
            raise SystemExit(f"Unsupported {name}: {mode}")

    quiet_rollout = bool(config_value(config, "quiet_rollout", False)) or bool(args.quiet_rollout)
    planner_output_every = int(
        args.planner_output_every
        if args.planner_output_every is not None
        else config_value(config, "planner_output_every", 50)
    )
    if planner_output_every < 0:
        raise SystemExit("planner_output_every must be >= 0")

    if not model_path.exists() and len(model_path.parts) > 2:
        raise SystemExit(
            f"Model path does not exist: {model_path}. "
            "Pass an existing local path with --model-path or MODEL_PATH=..., "
            "or use a HuggingFace repo id like Qwen/Qwen3-0.6B."
        )
    if not train_file.exists():
        raise SystemExit(f"Training file does not exist: {train_file}")

    import torch

    from flowgrpo_light.policy import PlannerPolicy
    from flowgrpo_light.rollout import run_rollout

    random.seed(seed)
    torch.manual_seed(seed)

    frozen_client = FrozenClient(
        base_url=frozen_base_url,
        model=frozen_model,
        timeout=int(config_value(config, "frozen_timeout", 60)),
        think_mode=think_mode,
    )
    frozen_client.check_model()

    policy = PlannerPolicy(
        str(model_path),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=float(config_value(config, "lora_dropout", 0.0)),
        max_new_tokens=int(config_value(config, "planner_max_new_tokens", 128)),
        temperature=float(config_value(config, "planner_temperature", 0.8)),
        top_p=float(config_value(config, "planner_top_p", 0.95)),
        dtype=str(config_value(config, "dtype", "bfloat16")),
        gradient_checkpointing=bool(config_value(config, "gradient_checkpointing", True)),
    )
    policy.train()
    optimizer = torch.optim.AdamW(
        [param for param in policy.model.parameters() if param.requires_grad],
        lr=learning_rate,
        weight_decay=float(config_value(config, "weight_decay", 0.0)),
    )

    if rollout_backend == "agentflow":
        from flowgrpo_light.agentflow_rollout import build_agentflow_rollout_runner

        subagent_config = config_value(config, "subagent_config", None)
        rollout_runner = build_agentflow_rollout_runner(
            policy=policy,
            frozen_model=frozen_model,
            frozen_base_url=frozen_base_url,
            frozen_temperature=float(config_value(config, "frozen_temperature", 0.0)),
            output_types=str(config_value(config, "output_types", "direct")),
            max_steps=max_steps,
            max_time=int(config_value(config, "max_time", 120)),
            max_tokens=int(config_value(config, "max_tokens", 2048)),
            subagent_config_path=Path(subagent_config) if subagent_config else None,
            think_mode=think_mode,
            query_analysis_think_mode=query_analysis_think_mode,
            final_output_think_mode=final_output_think_mode,
            verifier_think_mode=verifier_think_mode,
            rollout_concurrency=rollout_concurrency,
            planner_batch_size=planner_batch_size,
            planner_batch_timeout_s=planner_batch_timeout_s,
        )

        def collect_rollout_batch(batch_rows: list[dict[str, Any]]) -> list[list[Any]]:
            return rollout_runner.run_batch(batch_rows, group_size=group_size)

    else:

        def collect_rollout(row: dict[str, Any]) -> Any:
            return run_rollout(
                row,
                policy=policy,
                frozen_client=frozen_client,
                max_steps=max_steps,
                frozen_temperature=float(config_value(config, "frozen_temperature", 0.0)),
                frozen_max_tokens=int(config_value(config, "frozen_max_tokens", 512)),
                query_analysis_think_mode=query_analysis_think_mode,
                final_output_think_mode=final_output_think_mode,
            )

        def collect_rollout_batch(batch_rows: list[dict[str, Any]]) -> list[list[Any]]:
            return [
                [collect_rollout(row) for _ in range(group_size)]
                for row in batch_rows
            ]

    rows = load_rows(train_file)[:max_train_items]
    random.shuffle(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_metrics_path = output_dir / "summary_metrics.jsonl"

    print(f"Training rows={len(rows)} output_dir={output_dir}")
    print(f"quiet_rollout={quiet_rollout} planner_output_every={planner_output_every}")

    global_step = 0
    train_update_count = 0
    skipped_no_advantage_count = 0
    started_at = time.time()
    try:
        for epoch in range(epochs):
            for row_index, row_batch in iter_batches(rows, question_batch_size):
                step_started_at = time.time()
                next_step = global_step + 1
                rollout_started_at = time.time()
                if quiet_rollout:
                    with open(os.devnull, "w", encoding="utf-8") as devnull:
                        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            rollout_groups = collect_rollout_batch(row_batch)
                else:
                    rollout_groups = collect_rollout_batch(row_batch)
                rollout_elapsed_s = round(time.time() - rollout_started_at, 3)

                rollouts, advantages, reward_groups, advantage_groups = flatten_rollout_groups(rollout_groups)
                rewards = [reward for group in reward_groups for reward in group]
                reward_summary = summarize_rewards(rewards)
                ids = [row.get("id") or row.get("pid") for row in row_batch]
                answer_groups = [[rollout.answer for rollout in group] for group in rollout_groups]
                error_groups = [[rollout.errors for rollout in group] for group in rollout_groups]
                valid_for_training_groups = [
                    [rollout.valid_for_training for rollout in group]
                    for group in rollout_groups
                ]
                total_rollout_count = sum(len(group) for group in rollout_groups)
                nonzero_advantage_count = count_nonzero_advantages(advantages)
                effective_sample_count = count_effective_samples(rollouts, advantages)
                emit_planner_outputs = planner_output_every > 0 and next_step % planner_output_every == 0
                planner_raw_outputs = (
                    collect_planner_raw_outputs(row_batch, rollout_groups)
                    if emit_planner_outputs
                    else []
                )
                if emit_planner_outputs:
                    print_planner_raw_outputs(next_step, planner_raw_outputs)

                train_started_at = time.time()
                train_stats = train_step_grpo(
                    policy=policy,
                    optimizer=optimizer,
                    rollouts=rollouts,
                    advantages=advantages,
                    clip_range=clip_range,
                    max_grad_norm=float(config_value(config, "max_grad_norm", 1.0)),
                    logprob_micro_batch_size=logprob_micro_batch_size,
                    policy_epochs=policy_epochs,
                )
                train_elapsed_s = round(time.time() - train_started_at, 3)
                global_step += 1
                updated = train_stats is not None
                if updated:
                    train_update_count += 1
                else:
                    skipped_no_advantage_count += 1
                loss = train_stats["loss"] if train_stats else None

                gpu_memory = gpu_memory_snapshot(torch)
                step_elapsed_s = round(time.time() - step_started_at, 3)
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "row_index": row_index,
                    "row_indices": list(range(row_index, row_index + len(row_batch))),
                    "id": ids[0] if len(ids) == 1 else ids,
                    "ids": ids,
                    "question_batch_size": len(row_batch),
                    "configured_question_batch_size": question_batch_size,
                    "group_size": group_size,
                    "rollout_concurrency": rollout_concurrency,
                    "planner_batch_size": planner_batch_size,
                    "planner_batch_timeout_s": planner_batch_timeout_s,
                    "logprob_micro_batch_size": logprob_micro_batch_size,
                    "valid_rollout_count": len(rollouts),
                    "invalid_rollout_count": total_rollout_count - len(rollouts),
                    "valid_for_training_groups": valid_for_training_groups,
                    "rewards": rewards,
                    "reward_groups": reward_groups,
                    "reward_summary": reward_summary,
                    "advantages": advantages,
                    "advantage_groups": advantage_groups,
                    "nonzero_advantage_count": nonzero_advantage_count,
                    "effective_sample_count": effective_sample_count,
                    "skipped_no_advantage": not updated,
                    "loss": loss,
                    "train_stats": train_stats,
                    "clip_range": clip_range,
                    "policy_epochs": policy_epochs,
                    "answers": [answer for group in answer_groups for answer in group],
                    "answer_groups": answer_groups,
                    "errors": [errors for group in error_groups for errors in group],
                    "error_groups": error_groups,
                    "rollout_error_count": sum(1 for errors in error_groups for item in errors if item),
                    "rollout_backend": rollout_backend,
                    "think_mode": think_mode,
                    "query_analysis_think_mode": query_analysis_think_mode,
                    "final_output_think_mode": final_output_think_mode,
                    "verifier_think_mode": verifier_think_mode,
                    "planner_output_every": planner_output_every,
                    "planner_raw_outputs": planner_raw_outputs if emit_planner_outputs else None,
                    "rollout_elapsed_s": rollout_elapsed_s,
                    "train_elapsed_s": train_elapsed_s,
                    "step_elapsed_s": step_elapsed_s,
                    "gpu_memory": gpu_memory,
                    "train_update_count": train_update_count,
                    "skipped_no_advantage_count": skipped_no_advantage_count,
                }
                append_jsonl(metrics_path, record)

                summary_record = {
                    "step": global_step,
                    "epoch": epoch,
                    "ids": ids,
                    "status": "update" if updated else "skip_no_advantage",
                    "loss": loss,
                    "policy_loss": train_stats["policy_loss"] if train_stats else None,
                    "ratio_mean": train_stats["ratio_mean"] if train_stats else None,
                    "ratio_min": train_stats["ratio_min"] if train_stats else None,
                    "ratio_max": train_stats["ratio_max"] if train_stats else None,
                    "clip_fraction": train_stats["clip_fraction"] if train_stats else None,
                    "approx_kl": train_stats["approx_kl"] if train_stats else None,
                    "clip_range": clip_range,
                    "policy_epochs": policy_epochs,
                    "policy_update_count": train_stats["policy_update_count"] if train_stats else 0,
                    "epoch_losses": train_stats["epoch_losses"] if train_stats else [],
                    "epoch_ratio_means": train_stats["epoch_ratio_means"] if train_stats else [],
                    "epoch_clip_fractions": train_stats["epoch_clip_fractions"] if train_stats else [],
                    "epoch_approx_kls": train_stats["epoch_approx_kls"] if train_stats else [],
                    "valid_rollout_count": len(rollouts),
                    "invalid_rollout_count": total_rollout_count - len(rollouts),
                    "nonzero_advantage_count": nonzero_advantage_count,
                    "effective_sample_count": effective_sample_count,
                    "rollout_error_count": record["rollout_error_count"],
                    "rollout_elapsed_s": rollout_elapsed_s,
                    "train_elapsed_s": train_elapsed_s,
                    "step_elapsed_s": step_elapsed_s,
                    "gpu_memory": gpu_memory,
                    **reward_summary,
                }
                append_jsonl(summary_metrics_path, summary_record)

                print(
                    f"step={global_step} status={summary_record['status']} "
                    f"reward_mean={reward_summary['reward_mean']:.3f} "
                    f"reward_std={reward_summary['reward_std']:.3f} "
                    f"valid={len(rollouts)}/{total_rollout_count} "
                    f"nonzero_adv={nonzero_advantage_count} "
                    f"effective_samples={effective_sample_count} "
                    f"policy_epochs={policy_epochs} "
                    f"loss={loss if loss is not None else 'skip'} "
                    f"ratio={summary_record['ratio_mean'] if summary_record['ratio_mean'] is not None else 'na'} "
                    f"clip_frac={summary_record['clip_fraction'] if summary_record['clip_fraction'] is not None else 'na'} "
                    f"gpu_alloc={gpu_memory.get('allocated_mib', 'na')}MiB "
                    f"gpu_reserved={gpu_memory.get('reserved_mib', 'na')}MiB "
                    f"rollout_s={rollout_elapsed_s:.1f} train_s={train_elapsed_s:.1f} "
                    f"ids={ids}"
                )

                if global_step % int(config_value(config, "save_every", 4)) == 0:
                    policy.save_adapter(str(output_dir / f"checkpoint_step_{global_step}"))
    finally:
        close_rollout_runner = getattr(locals().get("rollout_runner", None), "close", None)
        if callable(close_rollout_runner):
            close_rollout_runner()

    final_adapter_path = output_dir / "final_adapter"
    policy.save_adapter(str(final_adapter_path))
    train_summary = {
        "final_adapter": str(final_adapter_path),
        "metrics_path": str(metrics_path),
        "summary_metrics_path": str(summary_metrics_path),
        "rows": len(rows),
        "global_step": global_step,
        "train_update_count": train_update_count,
        "skipped_no_advantage_count": skipped_no_advantage_count,
        "policy_epochs": policy_epochs,
        "clip_range": clip_range,
        "elapsed_s": round(time.time() - started_at, 3),
    }
    write_json(output_dir / "train_summary.json", train_summary)
    print(f"Saved final adapter to {final_adapter_path}")
    print(json.dumps(train_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
