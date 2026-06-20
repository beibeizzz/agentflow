from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from subagent_model_config import apply_subagent_config, load_subagent_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def bootstrap_agentflow_runtime() -> None:
    agentflow_core = PROJECT_ROOT / "agentflow" / "agentflow"
    agentflow_pkg = types.ModuleType("agentflow")
    agentflow_pkg.__path__ = [str(agentflow_core)]
    agentflow_pkg.__file__ = str(agentflow_core / "__init__.py")
    sys.modules["agentflow"] = agentflow_pkg


def served_model_name(llm_engine_name: str) -> str:
    return llm_engine_name.removeprefix("vllm-")


def check_vllm_server(base_url: str, model_name: str) -> None:
    url = base_url.rstrip("/") + "/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not reach vLLM models endpoint at {url}: {exc}") from exc

    ids = {item.get("id") for item in payload.get("data", [])}
    if model_name not in ids:
        raise SystemExit(f"vLLM is reachable, but model {model_name!r} is not in /models. Found: {sorted(ids)}")
    print(f"Verified vLLM model {model_name!r} at {url}")


def load_data(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list in {path}")
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def construct_solver(
    llm_engine_name: str,
    base_url: str,
    output_types: str,
    max_steps: int,
    max_time: int,
    max_tokens: int,
    temperature: float,
    subagent_config_path: Path | None,
    think_mode: str = "default",
    query_analysis_think_mode: str | None = None,
    final_output_think_mode: str | None = None,
    verifier_think_mode: str | None = None,
):
    bootstrap_agentflow_runtime()
    from agentflow.solver import construct_solver as build_solver

    solver = build_solver(
        llm_engine_name=llm_engine_name,
        base_url=base_url,
        enabled_tools=["Calculator_Tool"],
        tool_engine=["Default"],
        model_engine=["trainable", "trainable", "trainable", "trainable"],
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        temperature=temperature,
        think_mode=think_mode,
        query_analysis_think_mode=query_analysis_think_mode or think_mode,
        final_output_think_mode=final_output_think_mode or think_mode,
        verifier_think_mode=verifier_think_mode or think_mode,
        verbose=False,
    )
    if subagent_config_path is not None and not subagent_config_path.is_absolute():
        cwd_path = subagent_config_path
        script_path = SCRIPT_DIR / subagent_config_path
        if not cwd_path.exists() and script_path.exists():
            subagent_config_path = script_path
    subagent_config = load_subagent_config(subagent_config_path)
    apply_subagent_config(solver, subagent_config)
    return solver


def reset_solver_memory(solver: Any) -> None:
    from agentflow.models.memory import Memory

    solver.memory = Memory()


def run_examples(args: argparse.Namespace) -> None:
    data = load_data(args.data_file)
    if args.start:
        data = data[args.start :]
    if args.limit is not None:
        data = data[: args.limit]

    check_vllm_server(args.base_url, served_model_name(args.llm_engine_name))
    query_analysis_think_mode = args.query_analysis_think_mode or args.think_mode
    final_output_think_mode = args.final_output_think_mode or args.think_mode
    verifier_think_mode = args.verifier_think_mode or args.think_mode
    solver = construct_solver(
        llm_engine_name=args.llm_engine_name,
        base_url=args.base_url,
        output_types=args.output_types,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        subagent_config_path=args.subagent_config,
        think_mode=args.think_mode,
        query_analysis_think_mode=query_analysis_think_mode,
        final_output_think_mode=final_output_think_mode,
        verifier_think_mode=verifier_think_mode,
    )
    available_tools = list(getattr(solver.planner, "available_tools", []))
    print(f"AgentFlow available tools: {available_tools}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    failed = 0
    started_at = time.time()

    for position, row in enumerate(data, start=1):
        pid = row["pid"]
        output_path = args.output_dir / f"output_{pid}.json"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{position}/{len(data)}] skip pid={pid}: {output_path} exists")
            continue

        print(f"[{position}/{len(data)}] run pid={pid}")
        reset_solver_memory(solver)
        item_started = time.time()
        payload: dict[str, Any] = {
            "pid": pid,
            "question": row["question"],
            "query": row["question"],
            "original_query": row.get("query"),
            "answer": row["answer"],
            "gold_answer": row["gold_answer"],
            "llm_engine_name": args.llm_engine_name,
            "base_url": args.base_url,
            "output_types": args.output_types,
            "max_steps": args.max_steps,
            "max_time": args.max_time,
            "max_tokens": args.max_tokens,
            "think_mode": args.think_mode,
            "query_analysis_think_mode": query_analysis_think_mode,
            "final_output_think_mode": final_output_think_mode,
            "verifier_think_mode": verifier_think_mode,
            "available_tools": available_tools,
        }
        try:
            if args.solver_log_dir:
                args.solver_log_dir.mkdir(parents=True, exist_ok=True)
                solver_log_path = args.solver_log_dir / f"output_{pid}.log"
                with solver_log_path.open("w", encoding="utf-8") as log_file:
                    with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                        result = solver.solve(row["question"])
                payload["solver_log"] = str(solver_log_path)
            else:
                result = solver.solve(row["question"])
            payload.update(result)
            payload["ok"] = True
            completed += 1
        except Exception as exc:
            payload["ok"] = False
            payload["error_type"] = type(exc).__name__
            payload["error"] = str(exc)
            failed += 1
            print(f"[{position}/{len(data)}] error pid={pid}: {type(exc).__name__}: {exc}")
            if args.stop_on_error:
                atomic_write_json(output_path, payload)
                raise
        finally:
            payload["wall_time"] = round(time.time() - item_started, 3)
            atomic_write_json(output_path, payload)

    elapsed = round(time.time() - started_at, 3)
    print(
        f"Finished run: completed={completed}, skipped={skipped}, failed={failed}, "
        f"elapsed={elapsed}s, output_dir={args.output_dir}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local vLLM-backed AgentFlow on GSM8K JSON data.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solver-log-dir", type=Path, default=None)
    parser.add_argument("--llm-engine-name", default="vllm-Qwen3-0.6B")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--output-types", default="direct", choices=["base", "direct", "final", "final,direct"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-time", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think-mode", choices=["default", "on", "off"], default="default")
    parser.add_argument("--query-analysis-think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--final-output-think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--verifier-think-mode", choices=["default", "on", "off"], default=None)
    parser.add_argument("--subagent-config", type=Path, default=SCRIPT_DIR / "subagent_model_config.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    run_examples(parse_args())


if __name__ == "__main__":
    main()
