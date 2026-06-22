from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types
from typing import Any, Callable

PROJECT_DIR = Path(__file__).resolve().parents[2]
GSM_DIR = PROJECT_DIR / "try_gsm8k_0522"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(GSM_DIR) not in sys.path:
    sys.path.insert(0, str(GSM_DIR))
if "agentflow" not in sys.modules:
    agentflow_core = PROJECT_DIR / "agentflow" / "agentflow"
    agentflow_package = types.ModuleType("agentflow")
    agentflow_package.__path__ = [str(agentflow_core)]
    agentflow_package.__file__ = str(agentflow_core / "__init__.py")
    sys.modules["agentflow"] = agentflow_package

from flowgrpo_light.train_light_grpo import config_value, load_rows, load_yaml_config
from try_ticket_agent.run_ticket_agentflow import append_jsonl, atomic_write_json, check_model
from try_ticket_agent.ticket_env.solver_factory import construct_ticket_runtime
from try_ticket_agent.flowgrpo_general_2x40g.train_ticket_gspo import build_ticket_rollout_runner


def summarize_results(results: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(results)
    successes = sum((item.get("verification") or {}).get("success") is True for item in results)
    infrastructure = sum(item.get("valid_for_training") is False for item in results)
    invalid_actions = sum(
        int((item.get("verification") or {}).get("invalid_action_count", 0)) > 0
        for item in results
    )
    summary: dict[str, float | int] = {
        "total": total,
        "success_count": successes,
        "episode_success_rate": successes / total if total else 0.0,
        "invalid_action_rate": invalid_actions / total if total else 0.0,
        "infrastructure_failure_rate": infrastructure / total if total else 0.0,
    }
    for mode in ("direct", "indirect"):
        subset = [item for item in results if item.get("curriculum_mode") == mode]
        count = sum((item.get("verification") or {}).get("success") is True for item in subset)
        summary[f"{mode}_count"] = len(subset)
        summary[f"{mode}_success_rate"] = count / len(subset) if subset else 0.0
    return summary


def _adapter_record(row: dict[str, Any], rollout: Any) -> dict[str, Any]:
    verification = None
    if rollout.valid_for_training:
        try:
            verification = json.loads(rollout.answer)
        except (TypeError, json.JSONDecodeError):
            verification = {"success": False, "failure_codes": ["INVALID_ADAPTER_RESULT"]}
    return {
        "episode_id": row["episode_id"],
        "curriculum_mode": row["curriculum_mode"],
        "reward": float(rollout.reward),
        "verification": verification,
        "valid_for_training": bool(rollout.valid_for_training),
        "errors": list(rollout.errors),
        "planner_responses": [sample.response for sample in rollout.samples],
        "planner_response_token_ids": [sample.response_token_ids for sample in rollout.samples],
        "memory": rollout.memory,
    }


def evaluate(rows: list[dict[str, Any]], collect: Callable[[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        try:
            record = collect(row)
            record.setdefault("valid_for_training", True)
            record.setdefault("curriculum_mode", row["curriculum_mode"])
        except Exception as exc:
            record = {
                "episode_id": row["episode_id"],
                "curriculum_mode": row["curriculum_mode"],
                "reward": 0.0,
                "verification": None,
                "valid_for_training": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        results.append(record)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline or adapter Ticket AgentFlow.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval-mode", choices=["baseline", "adapter"], default=None)
    parser.add_argument("--adapter-path", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    mode = str(args.eval_mode or config_value(config, "eval_mode", "adapter"))
    model_path = str(config_value(config, "model_path", "Qwen/Qwen3-0.6B"))
    frozen_model = str(config_value(config, "frozen_model", "Qwen3-0.6B"))
    base_url = str(config_value(config, "frozen_base_url", "http://127.0.0.1:8000/v1"))
    eval_file = Path(config_value(config, "eval_file", "try_ticket_agent/data/generated/test.jsonl"))
    output_dir = Path(config_value(config, "output_dir", f"try_ticket_agent/outputs/eval_{mode}"))
    rows = load_rows(eval_file)[: int(config_value(config, "max_eval_items", 1000000))]
    check_model(base_url, frozen_model)
    closer = None
    if mode == "baseline":
        runtime = construct_ticket_runtime(
            llm_engine_name=f"vllm-{frozen_model}",
            base_url=base_url,
            max_steps=int(config_value(config, "max_steps", 3)),
            max_time=int(config_value(config, "max_time", 120)),
            max_tokens=int(config_value(config, "max_tokens", 512)),
            temperature=0.0,
            think_mode=str(config_value(config, "think_mode", "off")),
            query_analysis_think_mode=str(config_value(config, "query_analysis_think_mode", "on")),
        )
        collect = runtime.run_episode
    else:
        from flowgrpo_light.policy import PlannerPolicy

        adapter = args.adapter_path or config_value(config, "adapter_path", None)
        if not adapter or not Path(str(adapter)).is_dir():
            raise SystemExit(f"Adapter path does not exist: {adapter}")
        policy = PlannerPolicy(
            model_path,
            adapter_path=str(adapter),
            max_new_tokens=int(config_value(config, "planner_max_new_tokens", 256)),
            temperature=0.0,
            top_p=1.0,
            dtype=str(config_value(config, "dtype", "bfloat16")),
            gradient_checkpointing=False,
        )
        policy.eval()
        runner = build_ticket_rollout_runner(
            policy=policy,
            frozen_model=frozen_model,
            frozen_base_url=base_url,
            max_steps=int(config_value(config, "max_steps", 3)),
            max_time=int(config_value(config, "max_time", 120)),
            max_tokens=int(config_value(config, "max_tokens", 512)),
            rollout_concurrency=1,
            planner_batch_size=1,
            planner_batch_timeout_s=0.0,
            think_mode=str(config_value(config, "think_mode", "off")),
            query_analysis_think_mode=str(config_value(config, "query_analysis_think_mode", "on")),
        )
        closer = runner.close
        collect = lambda row: _adapter_record(row, runner.run(row))
    try:
        results = evaluate(rows, collect)
    finally:
        if closer is not None:
            closer()
    output_dir.mkdir(parents=True, exist_ok=True)
    details = output_dir / "eval_details.jsonl"
    if details.exists():
        details.unlink()
    for result in results:
        append_jsonl(details, result)
    summary = {"eval_mode": mode, "eval_file": str(eval_file), **summarize_results(results)}
    atomic_write_json(output_dir / "eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
