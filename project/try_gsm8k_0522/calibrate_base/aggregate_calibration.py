from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

CALIBRATE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CALIBRATE_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gsm8k_utils import answers_match, extract_predicted_answer


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected JSON list in {path}")
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def choose_bucket(*, scored_count: int, rollout_count: int, reward_mean: float, reward_std: float) -> str:
    if scored_count == 0 or scored_count < max(1, rollout_count // 2):
        return "bad"
    if reward_std > 1e-8:
        return "learnable"
    if reward_mean >= 0.875:
        return "easy"
    if reward_mean <= 0.125:
        return "hard"
    return "learnable"


def score_one(result_path: Path, row: dict[str, Any], response_field: str) -> dict[str, Any]:
    if not result_path.exists():
        return {"status": "missing", "reward": None, "prediction": None, "result_path": str(result_path)}
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("ok") is False:
        return {
            "status": "error",
            "reward": None,
            "prediction": None,
            "result_path": str(result_path),
            "error_type": payload.get("error_type"),
            "error": payload.get("error"),
        }
    response = payload.get(response_field)
    prediction = extract_predicted_answer(response)
    correct = answers_match(prediction, row["gold_answer"])
    return {
        "status": "scored",
        "reward": 1.0 if correct else 0.0,
        "correct": bool(correct),
        "prediction": prediction,
        "response": response,
        "result_path": str(result_path),
        "wall_time": payload.get("wall_time"),
        "step_count": payload.get("step_count"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate base calibration rollouts into difficulty buckets.")
    parser.add_argument("--data-file", type=Path, default=SCRIPT_DIR / "data" / "gsm8k_train.json")
    parser.add_argument("--calibration-dir", type=Path, default=SCRIPT_DIR / "calibrate_base" / "outputs" / "base_calibration")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "calibrate_base" / "outputs" / "base_calibration")
    parser.add_argument("--rollouts-per-question", type=int, default=8)
    parser.add_argument("--response-field", default="direct_output")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.data_file)
    rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]

    records: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {"easy": [], "learnable": [], "hard": [], "bad": []}
    for row in rows:
        pid = row["pid"]
        rollout_scores = [
            score_one(
                args.calibration_dir / "raw" / f"repeat_{repeat_index:02d}" / f"output_{pid}.json",
                row,
                args.response_field,
            )
            for repeat_index in range(args.rollouts_per_question)
        ]
        rewards = [float(item["reward"]) for item in rollout_scores if item["reward"] is not None]
        scored_count = len(rewards)
        reward_mean = sum(rewards) / scored_count if scored_count else 0.0
        reward_std = population_std(rewards)
        bucket = choose_bucket(
            scored_count=scored_count,
            rollout_count=args.rollouts_per_question,
            reward_mean=reward_mean,
            reward_std=reward_std,
        )
        records.append(
            {
                "pid": pid,
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "rollout_count": args.rollouts_per_question,
                "scored_count": scored_count,
                "missing_count": sum(1 for item in rollout_scores if item["status"] == "missing"),
                "error_count": sum(1 for item in rollout_scores if item["status"] == "error"),
                "correct_count": sum(1 for item in rollout_scores if item.get("correct")),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "bucket": bucket,
                "rollouts": rollout_scores,
            }
        )
        buckets[bucket].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "calibration_records.jsonl", records)
    bucket_dir = args.output_dir / "buckets"
    for bucket, bucket_rows in buckets.items():
        write_json(bucket_dir / f"gsm8k_train_{bucket}.json", bucket_rows)

    summary = {
        "data_file": str(args.data_file),
        "calibration_dir": str(args.calibration_dir),
        "rollouts_per_question": args.rollouts_per_question,
        "total_questions": len(rows),
        "bucket_counts": {bucket: len(bucket_rows) for bucket, bucket_rows in buckets.items()},
        "avg_reward_mean": sum(item["reward_mean"] for item in records) / len(records) if records else 0.0,
        "learnable_rate": len(buckets["learnable"]) / len(rows) if rows else 0.0,
        "records_path": str(args.output_dir / "calibration_records.jsonl"),
        "bucket_dir": str(bucket_dir),
    }
    write_json(args.output_dir / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
