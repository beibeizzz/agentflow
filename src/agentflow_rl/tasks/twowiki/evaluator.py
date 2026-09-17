from __future__ import annotations

import re
import string
from collections.abc import Iterable

from agentflow_rl.runtime.contracts import (
    FinalAnswerEnvelope,
    PrivateEvaluationRecordRef,
    TaskName,
)
from agentflow_rl.tasks.contracts import PrivateEvaluationStore, VerificationResult


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    lowered = "".join(character for character in lowered if character not in string.punctuation)
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    return " ".join(lowered.split())


def answer_exact_match(prediction: str, aliases: Iterable[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(normalized_prediction == normalize_answer(alias) for alias in aliases)
    )


class TwoWikiEvaluator:
    task_name = TaskName.TWOWIKI

    def __init__(self, store: PrivateEvaluationStore) -> None:
        self.store = store

    async def evaluate(
        self,
        answer: FinalAnswerEnvelope,
        reference: PrivateEvaluationRecordRef,
    ) -> VerificationResult:
        record = self.store.get(reference)
        answer_em = answer_exact_match(
            answer.answer,
            tuple(str(value) for value in record.payload["answer_aliases"]),
        )
        return VerificationResult(
            success=answer_em == 1.0,
            reward=answer_em,
            failure_codes=() if answer_em == 1.0 else ("ANSWER_MISMATCH",),
            metrics={"answer_em": answer_em},
        )


__all__ = [
    "TwoWikiEvaluator",
    "answer_exact_match",
    "normalize_answer",
]
