from __future__ import annotations

import re

from agentflow_rl.runtime.contracts import (
    FinalAnswerEnvelope,
    PrivateEvaluationRecordRef,
    TaskName,
)
from agentflow_rl.tasks.contracts import PrivateEvaluationStore, VerificationResult


INTEGER_RE = re.compile(r"^[+-]?\d+$")


def extract_aime_answer(value: str) -> int | None:
    candidate = value.strip()
    if not INTEGER_RE.fullmatch(candidate):
        return None
    return int(candidate)


class AIMEEvaluator:
    task_name = TaskName.AIME

    def __init__(self, store: PrivateEvaluationStore) -> None:
        self.store = store

    async def evaluate(
        self,
        answer: FinalAnswerEnvelope,
        reference: PrivateEvaluationRecordRef,
    ) -> VerificationResult:
        record = self.store.get(reference)
        expected = int(record.payload["canonical_answer"])
        predicted = extract_aime_answer(answer.answer)
        success = predicted == expected
        failures = () if success else (("ANSWER_PARSE_ERROR",) if predicted is None else ("ANSWER_MISMATCH",))
        return VerificationResult(
            success=success,
            reward=float(success),
            failure_codes=failures,
            metrics={"answer_parsed": float(predicted is not None)},
        )


__all__ = ["AIMEEvaluator", "extract_aime_answer"]
