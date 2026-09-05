from __future__ import annotations

import hashlib
from typing import Any

from .sandbox import CodeSandbox, TestRunResult
from .schemas import CodeAction, CodeExample


class CodingEnvironment:
    def __init__(self, example: CodeExample, sandbox: CodeSandbox, *, test_timeout_s: float = 10.0) -> None:
        self.example = example
        self.sandbox = sandbox
        self.test_timeout_s = test_timeout_s
        self.code = example.starter_code
        self.code_revision = 0
        self.last_result: TestRunResult | None = None
        self.last_tested_revision: int | None = None

    @property
    def code_sha256(self) -> str:
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()

    def execute(self, action: CodeAction) -> dict[str, Any]:
        if action.tool_name == "Code_Write_Tool":
            code = str(action.arguments.get("code", ""))
            if not code.strip():
                return {"ok": False, "code": "EMPTY_CODE"}
            self.code = code
            self.code_revision += 1
            self.last_result = None
            self.last_tested_revision = None
            return {
                "ok": True,
                "data": {
                    "characters": len(code),
                    "code_revision": self.code_revision,
                    "code_sha256": self.code_sha256,
                },
            }
        if action.tool_name == "Code_Run_Tests_Tool":
            if not self.code.strip():
                return {"ok": False, "code": "NO_CODE"}
            self.last_result = self.sandbox.run(
                self.code, self.example.public_tests, timeout_s=self.test_timeout_s
            )
            self.last_tested_revision = self.code_revision
            return {
                "ok": True,
                "data": {
                    "passed": self.last_result.passed,
                    "total": self.last_result.total,
                    "pass_rate": self.last_result.pass_rate,
                    "tests_passed": (
                        self.last_result.total > 0
                        and self.last_result.passed == self.last_result.total
                    ),
                    "failures": list(self.last_result.failures),
                    "timed_out": self.last_result.timed_out,
                    "code_revision": self.last_tested_revision,
                    "code_sha256": self.code_sha256,
                },
            }
        return {"ok": False, "code": "EXTERNAL_GENERATOR_REQUIRED"}


__all__ = ["CodingEnvironment"]
