from __future__ import annotations

from agentflow_rl.backends.bigcodebench import BigCodeBenchBackend
from agentflow_rl.runtime.contracts import TaskName
from agentflow_rl.tasks.contracts import PrivateEvaluationStore
from agentflow_rl.runtime.contracts import FinalAnswerEnvelope, PrivateEvaluationRecordRef
from agentflow_rl.tasks.contracts import PrivateEvaluationStore, VerificationResult


class BigCodeBenchEvaluator:
    task_name = TaskName.BIGCODEBENCH

    def __init__(
        self,
        store: PrivateEvaluationStore,
        backend: BigCodeBenchBackend,
    ) -> None:
        self.store = store
        self.backend = backend

    async def evaluate(
        self,
        answer: FinalAnswerEnvelope,
        reference: PrivateEvaluationRecordRef,
    ) -> VerificationResult:
        record = self.store.get(reference)
        result = await self.backend.evaluate(
            code=answer.answer,
            test_code=str(record.payload["test_code"]),
            entry_point=str(record.payload["entry_point"]),
            gt_time_limit=float(record.payload.get("gt_time_limit", 60.0)),
        )
        failure_codes = () if result.passed else (
            "OFFICIAL_HARNESS_TIMEOUT" if result.timed_out else "OFFICIAL_TEST_FAILURE",
        )
        return VerificationResult(
            success=result.passed,
            reward=float(result.passed),
            failure_codes=failure_codes,
            metrics={
                "pass_at_1": float(result.passed),
                "official_harness_timeout": float(result.timed_out),
            },
        )


__all__ = ["BigCodeBenchEvaluator"]
