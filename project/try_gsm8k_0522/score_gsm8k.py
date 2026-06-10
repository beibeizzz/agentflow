from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gsm8k_utils import answers_match, extract_predicted_answer


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score(data_file: Path, result_dir: Path, response_field: str) -> dict[str, Any]:
    rows = load_json(data_file)
    details = []
    correct = 0
    missing = 0
    errored = 0
    unparseable = 0

    for row in rows:
        pid = row["pid"]
        result_path = result_dir / f"output_{pid}.json"
        detail: dict[str, Any] = {
            "pid": pid,
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "result_file": str(result_path),
        }

        if not result_path.exists():
            detail.update({"status": "missing", "correct": False, "prediction": None})
            missing += 1
            details.append(detail)
            continue

        result = load_json(result_path)
        if result.get("ok") is False:
            detail.update(
                {
                    "status": "error",
                    "correct": False,
                    "prediction": None,
                    "error_type": result.get("error_type"),
                    "error": result.get("error"),
                }
            )
            errored += 1
            details.append(detail)
            continue

        response = result.get(response_field)
        prediction = extract_predicted_answer(response)
        is_correct = answers_match(prediction, row["gold_answer"])
        if prediction is None:
            unparseable += 1
        if is_correct:
            correct += 1

        detail.update(
            {
                "status": "scored",
                "correct": is_correct,
                "prediction": prediction,
                "response": response,
                "wall_time": result.get("wall_time"),
                "step_count": result.get("step_count"),
            }
        )
        details.append(detail)

    total = len(rows)
    scored = total - missing - errored
    return {
        "data_file": str(data_file),
        "result_dir": str(result_dir),
        "response_field": response_field,
        "total": total,
        "scored": scored,
        "correct": correct,
        "missing": missing,
        "errored": errored,
        "unparseable": unparseable,
        "accuracy": round(correct / total * 100, 4) if total else 0.0,
        "accuracy_on_scored": round(correct / scored * 100, 4) if scored else 0.0,
        "wrong_pids": [item["pid"] for item in details if not item["correct"]],
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score GSM8K AgentFlow outputs with deterministic numeric matching.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--response-field", default="direct_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = score(args.data_file, args.result_dir, args.response_field)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Accuracy: {summary['accuracy']}% ({summary['correct']}/{summary['total']}), "
        f"missing={summary['missing']}, errored={summary['errored']}, unparseable={summary['unparseable']}"
    )
    print(f"Summary saved to {args.output_file}")


if __name__ == "__main__":
    main()
