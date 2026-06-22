from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import types
import urllib.request
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if "agentflow" not in sys.modules:
    agentflow_core = PROJECT_DIR / "agentflow" / "agentflow"
    agentflow_package = types.ModuleType("agentflow")
    agentflow_package.__path__ = [str(agentflow_core)]
    agentflow_package.__file__ = str(agentflow_core / "__init__.py")
    sys.modules["agentflow"] = agentflow_package

from try_ticket_agent.ticket_env.solver_factory import construct_ticket_runtime


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a list of episode objects in {path}")
    return rows


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def summarize_results(results: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(results)
    success_count = sum(1 for item in results if item.get("verification", {}).get("success") is True)
    return {
        "total": total,
        "success_count": success_count,
        "episode_success_rate": success_count / total if total else 0.0,
        "average_steps": (
            sum(int(item.get("step_count", 0)) for item in results) / total
            if total else 0.0
        ),
    }


def check_model(base_url: str, model_name: str) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base_url.rstrip("/") + "/models", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    available = {item.get("id") for item in payload.get("data", [])}
    if model_name not in available:
        raise RuntimeError(f"Model {model_name!r} is unavailable; found {sorted(available)}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.data_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    served_name = args.llm_engine_name.removeprefix("vllm-")
    check_model(args.base_url, served_name)
    runtime = construct_ticket_runtime(
        llm_engine_name=args.llm_engine_name,
        base_url=args.base_url,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        think_mode=args.think_mode,
        query_analysis_think_mode=args.query_analysis_think_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "baseline_details.jsonl"
    if details_path.exists() and args.overwrite:
        details_path.unlink()
    completed: set[str] = set()
    results: list[dict[str, Any]] = []
    if details_path.exists():
        for item in load_rows(details_path):
            completed.add(str(item["episode_id"]))
            results.append(item)
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in completed:
            continue
        result = runtime.run_episode(row)
        append_jsonl(details_path, result)
        results.append(result)
    summary = {
        **summarize_results(results),
        "model": served_name,
        "data_file": str(args.data_file),
        "details_file": str(details_path),
    }
    atomic_write_json(args.output_dir / "baseline_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic Ticket task through core AgentFlow.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--llm-engine-name")
    parser.add_argument("--base-url")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-time", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--think-mode", choices=["default", "on", "off"])
    parser.add_argument("--query-analysis-think-mode", choices=["default", "on", "off"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    args = parser.parse_args(argv)
    config: dict[str, Any] = {}
    if args.config is not None:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read baseline config") from exc
        loaded = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            parser.error("baseline config must be a YAML object")
        config = loaded
    defaults = {
        "data_file": config.get("eval_file"),
        "output_dir": config.get("output_dir"),
        "llm_engine_name": config.get("llm_engine_name", "vllm-Qwen3-0.6B"),
        "base_url": config.get("frozen_base_url", "http://127.0.0.1:8000/v1"),
        "max_steps": int(config.get("max_steps", 3)),
        "max_time": int(config.get("max_time", 120)),
        "max_tokens": int(config.get("max_tokens", 512)),
        "temperature": float(config.get("temperature", 0.0)),
        "think_mode": config.get("think_mode", "off"),
        "query_analysis_think_mode": config.get("query_analysis_think_mode", "on"),
        "overwrite": bool(config.get("overwrite", False)),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.data_file is None or args.output_dir is None:
        parser.error("data file and output directory are required via CLI or config")
    args.data_file = Path(args.data_file)
    args.output_dir = Path(args.output_dir)
    return args


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
