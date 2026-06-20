from __future__ import annotations

from typing import Any

from gsm8k_utils import answers_match, extract_predicted_answer


def extract_rollout_answer(result: dict[str, Any]) -> str | None:
    for key in ("direct_output", "final_output", "base_response"):
        value = result.get(key)
        if value:
            answer = extract_predicted_answer(value)
            if answer is not None:
                return answer
    return None


def compute_gsm8k_reward(model_output: object, gold_answer: str) -> float:
    predicted = extract_predicted_answer(model_output)
    return 1.0 if answers_match(predicted, gold_answer) else 0.0


def compute_result_reward(result: dict[str, Any], gold_answer: str) -> tuple[float, str | None]:
    answer = extract_rollout_answer(result)
    return (1.0 if answers_match(answer, gold_answer) else 0.0), answer

