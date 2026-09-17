from __future__ import annotations

from agentflow_rl.runtime.contracts import (
    FinalAnswerEnvelope,
    PrivateEvaluationRecordRef,
    TaskName,
)
from agentflow_rl.tasks.contracts import PrivateEvaluationStore, VerificationResult


class GPQAEvaluator:
    task_name = TaskName.GPQA

    def __init__(self, store: PrivateEvaluationStore) -> None:
        self.store = store

    async def evaluate(
        self,
        answer: FinalAnswerEnvelope,
        reference: PrivateEvaluationRecordRef,
    ) -> VerificationResult:
        record = self.store.get(reference)
        success = answer.answer == str(record.payload["correct_option"]).upper()
        return VerificationResult(
            success=success,
            reward=float(success),
            failure_codes=() if success else ("ANSWER_MISMATCH",),
        )


__all__ = ["GPQAEvaluator"]
