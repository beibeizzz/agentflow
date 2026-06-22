from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request
from typing import Any

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
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-engine-name", default="vllm-Qwen3-0.6B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-time", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think-mode", choices=["default", "on", "off"], default="off")
    parser.add_argument("--query-analysis-think-mode", choices=["default", "on", "off"], default="on")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
