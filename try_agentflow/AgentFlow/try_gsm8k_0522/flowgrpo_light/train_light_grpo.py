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
) -> float | None:
    import torch

    policy.train()
    loss_terms: list[torch.Tensor] = []
    for rollout, advantage in zip(rollouts, advantages, strict=True):
        if abs(advantage) < 1e-8:
            continue
        for sample in rollout.samples:
            response_len = max(1, len(policy._tokenize(sample.response, add_special_tokens=False)))
            logprob = policy.sequence_logprob(sample.prompt, sample.response, use_adapter=True) / response_len
            loss = -float(advantage) * logprob
            if kl_coef > 0:
                with torch.no_grad():
                    ref_logprob = policy.sequence_logprob(sample.prompt, sample.response, use_adapter=False) / response_len
                loss = loss + kl_coef * (logprob - ref_logprob)
            loss_terms.append(loss)

    if not loss_terms:
        return None

    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.stack(loss_terms).mean()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.model.parameters(), max_grad_norm)
    optimizer.step()
    return float(total_loss.detach().cpu())


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
    parser.add_argument("--group-size", type=int, default=None)
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
        )

        def collect_rollout(row: dict[str, Any]) -> Any:
            return rollout_runner.run(row)

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

    rows = load_rows(train_file)[:max_train_items]
    random.shuffle(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    global_step = 0
    for epoch in range(epochs):
        for row_index, row in enumerate(rows):
            rollouts = [
                collect_rollout(row)
                for _ in range(group_size)
            ]
            rewards = [rollout.reward for rollout in rollouts]
            advantages = normalize_advantages(rewards)
            loss = train_step(
                policy=policy,
                optimizer=optimizer,
                rollouts=rollouts,
                advantages=advantages,
                kl_coef=kl_coef,
                max_grad_norm=float(config_value(config, "max_grad_norm", 1.0)),
            )
            global_step += 1
            record = {
                "step": global_step,
                "epoch": epoch,
                "row_index": row_index,
                "id": row.get("id") or row.get("pid"),
                "rewards": rewards,
                "advantages": advantages,
                "loss": loss,
                "answers": [rollout.answer for rollout in rollouts],
                "errors": [rollout.errors for rollout in rollouts],
                "rollout_backend": rollout_backend,
                "think_mode": think_mode,
                "query_analysis_think_mode": query_analysis_think_mode,
                "final_output_think_mode": final_output_think_mode,
                "verifier_think_mode": verifier_think_mode,
            }
            append_jsonl(metrics_path, record)
            print(
                f"step={global_step} reward_mean={sum(rewards)/len(rewards):.3f} "
                f"loss={loss if loss is not None else 'skip'} id={record['id']}"
            )

            if global_step % int(config_value(config, "save_every", 4)) == 0:
                policy.save_adapter(str(output_dir / f"checkpoint_step_{global_step}"))

    policy.save_adapter(str(output_dir / "final_adapter"))
    print(f"Saved final adapter to {output_dir / 'final_adapter'}")


if __name__ == "__main__":
    main()
