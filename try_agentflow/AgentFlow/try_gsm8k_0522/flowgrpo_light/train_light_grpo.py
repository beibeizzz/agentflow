from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowgrpo_light.frozen_client import FrozenClient


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyyaml is required when --config is used.") from exc
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return loaded


def config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Training file does not exist: {path}")
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit(f"Expected a JSON list in {path}")
        return data
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("Missing dependency: pyarrow is required to read parquet data.") from exc
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    raise SystemExit(f"Unsupported train file suffix: {path}")


def train_step(
    *,
    policy: PlannerPolicy,
    optimizer: torch.optim.Optimizer,
    rollouts: list[RolloutResult],
    advantages: list[float],
    kl_coef: float,
    max_grad_norm: float,
    logprob_micro_batch_size: int = 8,
) -> float | None:
    import torch

    policy.train()
    loss_items: list[tuple[str, str, float, int]] = []
    for rollout, advantage in zip(rollouts, advantages, strict=True):
        if abs(advantage) < 1e-8:
            continue
        for sample in rollout.samples:
            response_len = max(1, len(policy._tokenize(sample.response, add_special_tokens=False)))
            loss_items.append((sample.prompt, sample.response, float(advantage), response_len))

    if not loss_items:
        return None

    micro_batch_size = max(1, int(logprob_micro_batch_size))
    optimizer.zero_grad(set_to_none=True)
    total_loss_numerator = 0.0
    total_items = len(loss_items)
    for start in range(0, total_items, micro_batch_size):
        batch = loss_items[start : start + micro_batch_size]
        prompts = [item[0] for item in batch]
        responses = [item[1] for item in batch]
        logprobs = policy.sequence_logprob_many(prompts, responses, use_adapter=True)
        advantages_tensor = torch.tensor([item[2] for item in batch], dtype=logprobs.dtype, device=logprobs.device)
        response_lengths_tensor = torch.tensor(
            [item[3] for item in batch],
            dtype=logprobs.dtype,
            device=logprobs.device,
        )
        normalized_logprobs = logprobs / response_lengths_tensor
        losses = -advantages_tensor * normalized_logprobs
        if kl_coef > 0:
            with torch.no_grad():
                ref_logprobs = policy.sequence_logprob_many(prompts, responses, use_adapter=False)
                normalized_ref_logprobs = ref_logprobs / response_lengths_tensor
            losses = losses + kl_coef * (normalized_logprobs - normalized_ref_logprobs)
        micro_loss_sum = losses.sum()
        (micro_loss_sum / total_items).backward()
        total_loss_numerator += float(micro_loss_sum.detach().cpu())
    torch.nn.utils.clip_grad_norm_(policy.model.parameters(), max_grad_norm)
    optimizer.step()
    return total_loss_numerator / total_items


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_batches(rows: list[dict[str, Any]], batch_size: int) -> list[tuple[int, list[dict[str, Any]]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [(start, rows[start : start + batch_size]) for start in range(0, len(rows), batch_size)]


def flatten_rollout_groups(
    rollout_groups: list[list[RolloutResult]],
) -> tuple[list[RolloutResult], list[float], list[list[float]], list[list[float | None]]]:
    from flowgrpo_light.policy import normalize_advantages

    flat_rollouts: list[RolloutResult] = []
    flat_advantages: list[float] = []
    reward_groups: list[list[float]] = []
    advantage_groups: list[list[float | None]] = []
    for rollouts in rollout_groups:
        rewards = [rollout.reward for rollout in rollouts]
        valid_rollouts = [rollout for rollout in rollouts if rollout.valid_for_training]
        advantages = normalize_advantages([rollout.reward for rollout in valid_rollouts])
        valid_advantages = iter(advantages)
        group_advantages = [
            next(valid_advantages) if rollout.valid_for_training else None
            for rollout in rollouts
        ]
        reward_groups.append(rewards)
        advantage_groups.append(group_advantages)
        flat_rollouts.extend(valid_rollouts)
        flat_advantages.extend(advantages)
    return flat_rollouts, flat_advantages, reward_groups, advantage_groups


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight planner-only GRPO training for GSM8K.")
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
    parser.add_argument("--kl-coef", type=float, default=None)
    parser.add_argument("--rollout-backend", choices=["agentflow", "light"], default=None)
    parser.add_argument("--think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--query-analysis-think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--final-output-think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--verifier-think-mode", choices=["default", "on", "off"], default=None)
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
    kl_coef = float(args.kl_coef if args.kl_coef is not None else config_value(config, "kl_coef", 0.0))
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

    if not model_path.exists() and len(model_path.parts) > 2:
        raise SystemExit(
            f"Model path does not exist: {model_path}. "
            "Pass an existing local path with --model-path or MODEL_PATH=..., "
            "or use a HuggingFace repo id like Qwen/Qwen3-0.6B."
        )
    if not train_file.exists():
        raise SystemExit(f"Training file does not exist: {train_file}")

    import torch

    from flowgrpo_light.policy import PlannerPolicy, normalize_advantages
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

        def collect_rollout(row: dict[str, Any]) -> Any:
            return rollout_runner.run(row)

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

    global_step = 0
    try:
        for epoch in range(epochs):
            for row_index, row_batch in iter_batches(rows, question_batch_size):
                rollout_groups = collect_rollout_batch(row_batch)
                rollouts, advantages, reward_groups, advantage_groups = flatten_rollout_groups(rollout_groups)
                rewards = [reward for group in reward_groups for reward in group]
                ids = [row.get("id") or row.get("pid") for row in row_batch]
                answer_groups = [[rollout.answer for rollout in group] for group in rollout_groups]
                error_groups = [[rollout.errors for rollout in group] for group in rollout_groups]
                valid_for_training_groups = [
                    [rollout.valid_for_training for rollout in group]
                    for group in rollout_groups
                ]
                total_rollout_count = sum(len(group) for group in rollout_groups)
                loss = train_step(
                    policy=policy,
                    optimizer=optimizer,
                    rollouts=rollouts,
                    advantages=advantages,
                    kl_coef=kl_coef,
                    max_grad_norm=float(config_value(config, "max_grad_norm", 1.0)),
                    logprob_micro_batch_size=logprob_micro_batch_size,
                )
                global_step += 1
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
                    "advantages": advantages,
                    "advantage_groups": advantage_groups,
                    "loss": loss,
                    "answers": [answer for group in answer_groups for answer in group],
                    "answer_groups": answer_groups,
                    "errors": [errors for group in error_groups for errors in group],
                    "error_groups": error_groups,
                    "rollout_backend": rollout_backend,
                    "think_mode": think_mode,
                    "query_analysis_think_mode": query_analysis_think_mode,
                    "final_output_think_mode": final_output_think_mode,
                    "verifier_think_mode": verifier_think_mode,
                }
                append_jsonl(metrics_path, record)
                print(
                    f"step={global_step} question_batch={len(row_batch)} group_size={group_size} "
                    f"reward_mean={sum(rewards)/len(rewards):.3f} "
                    f"loss={loss if loss is not None else 'skip'} ids={ids}"
                )

                if global_step % int(config_value(config, "save_every", 4)) == 0:
                    policy.save_adapter(str(output_dir / f"checkpoint_step_{global_step}"))
    finally:
        close_rollout_runner = getattr(locals().get("rollout_runner", None), "close", None)
        if callable(close_rollout_runner):
            close_rollout_runner()

    policy.save_adapter(str(output_dir / "final_adapter"))
    print(f"Saved final adapter to {output_dir / 'final_adapter'}")


if __name__ == "__main__":
    main()
