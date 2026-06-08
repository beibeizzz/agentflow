from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from flowgrpo_light.train_light_grpo import config_value, load_rows, load_yaml_config
from gsm8k_utils import answers_match, extract_predicted_answer


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def adapter_disabled(value: Any) -> bool:
    return str(value).strip().lower() in {"", "0", "false", "none", "no", "off"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate general GRPO planner on learnable GSM8K rows.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--eval-mode", choices=["adapter", "baseline"], default="adapter")
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frozen-base-url", default=None)
    parser.add_argument("--frozen-model", default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--rollout-backend", choices=["agentflow", "light"], default=None)
    parser.add_argument("--think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--query-analysis-think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--final-output-think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--verifier-think-mode", choices=["default", "on", "off"], default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_yaml_config(args.config)

    model_path = Path(args.model_path or config_value(config, "model_path", "/home/north/vllm_test/models/Qwen/Qwen3-0.6B"))
    adapter_source = (
        args.adapter_path
        if args.adapter_path is not None
        else config_value(
            config,
            "adapter_path",
            "try_gsm8k_0522/flowgrpo_general_2x40g/outputs/train_general_2x40g/final_adapter",
        )
    )
    effective_eval_mode = "baseline" if args.eval_mode == "baseline" or adapter_disabled(adapter_source) else "adapter"
    adapter_path = None if effective_eval_mode == "baseline" else Path(str(adapter_source))
    eval_file = Path(args.eval_file or config_value(config, "eval_file", "try_gsm8k_0522/data/gsm8k_train_learnable.json"))
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif effective_eval_mode == "baseline":
        output_dir = Path("try_gsm8k_0522/flowgrpo_general_2x40g/outputs/eval_learnable_baseline")
    else:
        output_dir = Path(
            config_value(
                config,
                "output_dir",
                f"try_gsm8k_0522/flowgrpo_general_2x40g/outputs/eval_learnable_{effective_eval_mode}",
            )
        )
    frozen_base_url = args.frozen_base_url or config_value(config, "frozen_base_url", "http://127.0.0.1:8000/v1")
    frozen_model = args.frozen_model or config_value(config, "frozen_model", "Qwen3-0.6B")
    max_eval_items = int(
        args.max_eval_items if args.max_eval_items is not None else config_value(config, "max_eval_items", 1000000)
    )
    max_steps = int(args.max_steps if args.max_steps is not None else config_value(config, "max_steps", 3))
    rollout_backend = str(
        args.rollout_backend if args.rollout_backend is not None else config_value(config, "rollout_backend", "agentflow")
    )
    think_mode = str(args.think_mode if args.think_mode is not None else config_value(config, "think_mode", "off"))
    query_analysis_think_mode = str(
        args.query_analysis_think_mode
        if args.query_analysis_think_mode is not None
        else config_value(config, "query_analysis_think_mode", "on")
    )
    final_output_think_mode = str(
        args.final_output_think_mode
        if args.final_output_think_mode is not None
        else config_value(config, "final_output_think_mode", "off")
    )
    verifier_think_mode = str(
        args.verifier_think_mode
        if args.verifier_think_mode is not None
        else config_value(config, "verifier_think_mode", "on")
    )

    if not model_path.exists() and len(model_path.parts) > 2:
        raise SystemExit(f"Model path does not exist: {model_path}")
    if effective_eval_mode == "adapter" and adapter_path is not None and not adapter_path.exists():
        raise SystemExit(f"Adapter path does not exist: {adapter_path}")
    if not eval_file.exists():
        raise SystemExit(f"Eval file does not exist: {eval_file}")

    from flowgrpo_light.frozen_client import FrozenClient
    from flowgrpo_light.policy import PlannerPolicy
    from flowgrpo_light.rollout import run_rollout

    frozen_client = FrozenClient(
        base_url=frozen_base_url,
        model=frozen_model,
        timeout=int(config_value(config, "frozen_timeout", 120)),
        think_mode=think_mode,
    )
    frozen_client.check_model()

    policy = PlannerPolicy(
        str(model_path),
        adapter_path=str(adapter_path) if effective_eval_mode == "adapter" else "False",
        lora_rank=int(config_value(config, "lora_rank", 8)),
        max_new_tokens=int(config_value(config, "planner_max_new_tokens", 256)),
        temperature=float(config_value(config, "planner_temperature", 0.0)),
        top_p=float(config_value(config, "planner_top_p", 1.0)),
        dtype=str(config_value(config, "dtype", "bfloat16")),
        gradient_checkpointing=False,
    )
    policy.eval()

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
            max_tokens=int(config_value(config, "max_tokens", 512)),
            subagent_config_path=Path(subagent_config) if subagent_config else None,
            think_mode=think_mode,
            query_analysis_think_mode=query_analysis_think_mode,
            final_output_think_mode=final_output_think_mode,
            verifier_think_mode=verifier_think_mode,
        )

        def collect_rollout(row: dict[str, Any]):
            return rollout_runner.run(row)

    else:

        def collect_rollout(row: dict[str, Any]):
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

    rows = load_rows(eval_file)[:max_eval_items]
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "eval_details.jsonl"
    summary_path = output_dir / "eval_summary.json"
    if details_path.exists():
        details_path.unlink()

    final_correct = 0
    try:
        for index, row in enumerate(rows, start=1):
            rollout = collect_rollout(row)
            gold_answer = str(row.get("result") or row.get("gold_answer") or row.get("extra_info", {}).get("gold_answer"))
            final_prediction = extract_predicted_answer(rollout.answer)
            final_ok = answers_match(final_prediction, gold_answer)
            final_correct += int(final_ok)
            record = {
                "index": index,
                "id": row.get("id") or row.get("pid"),
                "question": row.get("question"),
                "gold_answer": gold_answer,
                "eval_mode": effective_eval_mode,
                "final_answer": rollout.answer,
                "final_prediction": final_prediction,
                "final_correct": final_ok,
                "planner_sample_count": len(rollout.samples),
                "planner_responses": [sample.response for sample in rollout.samples],
                "memory": rollout.memory,
                "query_analysis": rollout.query_analysis,
                "errors": rollout.errors,
            }
            append_jsonl(details_path, record)
            print(f"eval {index}/{len(rows)} id={record['id']} final={int(final_ok)}")
    finally:
        close_rollout_runner = getattr(locals().get("rollout_runner", None), "close", None)
        if callable(close_rollout_runner):
            close_rollout_runner()

    total = len(rows)
    summary = {
        "eval_mode": effective_eval_mode,
        "eval_file": str(eval_file),
        "model_path": str(model_path),
        "adapter_path": str(adapter_path) if effective_eval_mode == "adapter" else None,
        "output_dir": str(output_dir),
        "details_path": str(details_path),
        "max_steps": max_steps,
        "rollout_backend": rollout_backend,
        "think_mode": think_mode,
        "query_analysis_think_mode": query_analysis_think_mode,
        "final_output_think_mode": final_output_think_mode,
        "verifier_think_mode": verifier_think_mode,
        "total": total,
        "final_correct": final_correct,
        "final_accuracy": final_correct / total if total else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
