from __future__ import annotations

from agentflow_rl.backends.contracts import SandboxBackend, SandboxRequest
from agentflow_rl.runtime.contracts import (
    FinalAnswerEnvelope,
    PrivateEvaluationRecordRef,
    TaskName,
)
from agentflow_rl.tasks.contracts import PrivateEvaluationStore, VerificationResult


class TACOEvaluator:
    task_name = TaskName.TACO

    def __init__(
        self,
        store: PrivateEvaluationStore,
        sandbox: SandboxBackend,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        self.store = store
        self.sandbox = sandbox
        self.timeout_s = timeout_s

    async def evaluate(
        self,
        answer: FinalAnswerEnvelope,
        reference: PrivateEvaluationRecordRef,
    ) -> VerificationResult:
        record = self.store.get(reference)
        tests = tuple(record.payload["hidden_tests"])
        result = await self.sandbox.execute(
            SandboxRequest(
                code=answer.answer,
                mode="test",
                tests=tests,
                timeout_s=self.timeout_s,
            )
        )
        passed = int(result.passed or 0)
        total = int(result.total or len(tests))
        pass_rate = passed / total if total else 0.0
        success = result.ok and total > 0 and passed == total
        error_types = {
            str(item.get("error_type", "UNKNOWN")) for item in result.failures
        }
        failures = []
        if result.timed_out:
            failures.append("HIDDEN_TEST_TIMEOUT")
        if "OUTPUT_LIMIT" in error_types:
            failures.append("HIDDEN_TEST_OUTPUT_LIMIT")
        if "RUNTIME_ERROR" in error_types:
            failures.append("HIDDEN_TEST_RUNTIME_ERROR")
        if "HARNESS_ERROR" in error_types:
            failures.append("HIDDEN_TEST_HARNESS_ERROR")
        if total == 0:
            failures.append("HIDDEN_TEST_SET_EMPTY")
        if not success:
            failures.append("HIDDEN_TEST_FAILURE")
        return VerificationResult(
            success=success,
            reward=pass_rate,
            failure_codes=tuple(failures),
            metrics={
                "hidden_pass_rate": pass_rate,
                "hidden_passed": float(passed),
                "hidden_total": float(total),
                "hidden_timeout": float(result.timed_out),
                "hidden_output_limit": float("OUTPUT_LIMIT" in error_types),
                "hidden_runtime_error": float("RUNTIME_ERROR" in error_types),
                "hidden_harness_error": float("HARNESS_ERROR" in error_types),
            },
        )


__all__ = ["TACOEvaluator"]
