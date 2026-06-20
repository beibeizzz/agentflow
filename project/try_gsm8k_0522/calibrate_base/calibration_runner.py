from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

CALIBRATE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CALIBRATE_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_gsm8k_agentflow


_thread_local = threading.local()
_manifest_lock = threading.Lock()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected JSON list in {path}")
    return rows


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def append_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def get_solver(args: argparse.Namespace) -> Any:
    solver = getattr(_thread_local, "solver", None)
    if solver is None:
        solver = run_gsm8k_agentflow.construct_solver(
            llm_engine_name=args.llm_engine_name,
            base_url=args.base_url,
            output_types=args.output_types,
            max_steps=args.max_steps,
            max_time=args.max_time,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            subagent_config_path=args.subagent_config,
            think_mode=args.think_mode,
            query_analysis_think_mode=args.query_analysis_think_mode,
            final_output_think_mode=args.final_output_think_mode,
            verifier_think_mode=args.verifier_think_mode,
        )
        _thread_local.solver = solver
    return solver


def run_one(row: dict[str, Any], repeat_index: int, args: argparse.Namespace) -> dict[str, Any]:
    pid = row["pid"]
    output_path = args.output_dir / "raw" / f"repeat_{repeat_index:02d}" / f"output_{pid}.json"
    if output_path.exists() and not args.overwrite:
        return {"pid": pid, "repeat": repeat_index, "status": "skipped", "output_path": str(output_path)}

    solver = get_solver(args)
    run_gsm8k_agentflow.reset_solver_memory(solver)
    started_at = time.time()
    payload: dict[str, Any] = {
        "pid": pid,
        "repeat": repeat_index,
        "question": row["question"],
        "query": row.get("query") or row["question"],
        "answer": row["answer"],
        "gold_answer": row["gold_answer"],
        "llm_engine_name": args.llm_engine_name,
        "base_url": args.base_url,
        "output_types": args.output_types,
        "max_steps": args.max_steps,
        "max_time": args.max_time,
        "max_tokens": args.max_tokens,
        "think_mode": args.think_mode,
        "query_analysis_think_mode": args.query_analysis_think_mode,
        "final_output_think_mode": args.final_output_think_mode,
        "verifier_think_mode": args.verifier_think_mode,
        "subagent_config": str(args.subagent_config),
    }
    try:
        result = solver.solve(row["question"])
        payload.update(result)
        payload["ok"] = True
        status = "completed"
    except Exception as exc:
        payload["ok"] = False
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        status = "error"
    finally:
        payload["wall_time"] = round(time.time() - started_at, 3)
        atomic_write_json(output_path, payload)

    return {
        "pid": pid,
        "repeat": repeat_index,
        "status": status,
        "output_path": str(output_path),
        "wall_time": payload["wall_time"],
        "error_type": payload.get("error_type"),
    }


def iter_batches(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel base-model AgentFlow calibration rollout for GSM8K.")
    parser.add_argument("--data-file", type=Path, default=SCRIPT_DIR / "data" / "gsm8k_train.json")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "calibrate_base" / "outputs" / "base_calibration")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-engine-name", default="vllm-Qwen3-0.6B")
    parser.add_argument("--subagent-config", type=Path, default=CALIBRATE_DIR / "calibration_subagent_config.json")
    parser.add_argument("--output-types", default="direct", choices=["base", "direct", "final", "final,direct"])
    parser.add_argument("--response-field", default="direct_output")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-batch-size", type=int, default=4)
    parser.add_argument("--rollouts-per-question", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-time", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think-mode", choices=["default", "on", "off"], default="off")
    parser.add_argument("--query-analysis-think-mode", choices=["default", "on", "off"], default="on")
    parser.add_argument("--final-output-think-mode", choices=["default", "on", "off"], default="on")
    parser.add_argument("--verifier-think-mode", choices=["default", "on", "off"], default="on")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.question_batch_size < 1:
        raise SystemExit("--question-batch-size must be >= 1")
    if args.rollouts_per_question < 1:
        raise SystemExit("--rollouts-per-question must be >= 1")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")

    rows = load_rows(args.data_file)
    rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]

    run_gsm8k_agentflow.check_vllm_server(args.base_url, run_gsm8k_agentflow.served_model_name(args.llm_engine_name))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "calibration_manifest.jsonl"
    if manifest_path.exists() and args.overwrite:
        manifest_path.unlink()

    print(f"Calibration rows: {len(rows)}")
    print(f"Question batch size: {args.question_batch_size}")
    print(f"Rollouts per question: {args.rollouts_per_question}")
    print(f"Max workers: {args.max_workers}")
    print(f"Output dir: {args.output_dir}")

    completed = skipped = errored = 0
    started_at = time.time()
    batches = iter_batches(rows, args.question_batch_size)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for batch_index, row_batch in enumerate(batches, start=1):
            futures = [
                executor.submit(run_one, row, repeat_index, args)
                for row in row_batch
                for repeat_index in range(args.rollouts_per_question)
            ]
            for future in as_completed(futures):
                record = future.result()
                append_manifest(manifest_path, record)
                status = record["status"]
                if status == "completed":
                    completed += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errored += 1
            done = min(batch_index * args.question_batch_size, len(rows))
            print(
                f"batch={batch_index}/{len(batches)} questions={done}/{len(rows)} "
                f"completed={completed} skipped={skipped} errored={errored}"
            )

    summary = {
        "data_file": str(args.data_file),
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "question_batch_size": args.question_batch_size,
        "rollouts_per_question": args.rollouts_per_question,
        "max_workers": args.max_workers,
        "completed": completed,
        "skipped": skipped,
        "errored": errored,
        "elapsed_s": round(time.time() - started_at, 3),
    }
    atomic_write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
