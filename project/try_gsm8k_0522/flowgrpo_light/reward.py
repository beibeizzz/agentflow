from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gsm8k_utils import answers_match, extract_predicted_answer  # noqa: E402


def compute_reward(model_output: object, gold_answer: str) -> float:
    predicted = extract_predicted_answer(model_output)
    return 1.0 if answers_match(predicted, gold_answer) else 0.0
