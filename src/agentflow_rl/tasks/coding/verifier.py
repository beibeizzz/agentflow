from __future__ import annotations

from agentflow_rl.tasks.base import VerificationResult

from .sandbox import CodeSandbox, TestRunResult
from .schemas import CodeExample


def evaluate_code(code: str, example: CodeExample, sandbox: CodeSandbox, *, timeout_s: float = 10.0) -> tuple[VerificationResult, TestRunResult]:
    result = sandbox.run(code, example.hidden_tests, timeout_s=timeout_s)
    failures = []
    if result.timed_out:
        failures.append("HIDDEN_TEST_TIMEOUT")
    if result.passed < result.total:
        failures.append("HIDDEN_TEST_FAILURE")
    verification = VerificationResult(
        success=result.total > 0 and result.passed == result.total,
        reward=result.pass_rate,
        failure_codes=tuple(failures),
        metrics={
            "hidden_pass_rate": result.pass_rate,
            "hidden_passed": float(result.passed),
            "hidden_total": float(result.total),
        },
    )
    return verification, result


__all__ = ["evaluate_code"]
