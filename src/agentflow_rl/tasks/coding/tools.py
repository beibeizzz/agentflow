from __future__ import annotations

from typing import Any

from .sandbox import CodeSandbox, TestRunResult
from .schemas import CodeAction, CodeExample


class CodingEnvironment:
    def __init__(self, example: CodeExample, sandbox: CodeSandbox, *, test_timeout_s: float = 10.0) -> None:
        self.example = example
        self.sandbox = sandbox
        self.test_timeout_s = test_timeout_s
        self.code = example.starter_code
        self.last_result: TestRunResult | None = None

    def execute(self, action: CodeAction) -> dict[str, Any]:
        if action.tool_name == "Code_Write_Tool":
            code = str(action.arguments.get("code", ""))
            if not code.strip():
                return {"ok": False, "code": "EMPTY_CODE"}
            self.code = code
            return {"ok": True, "data": {"characters": len(code)}}
        if action.tool_name == "Code_Run_Tests_Tool":
            if not self.code.strip():
                return {"ok": False, "code": "NO_CODE"}
            self.last_result = self.sandbox.run(
                self.code, self.example.public_tests, timeout_s=self.test_timeout_s
            )
            return {
                "ok": True,
                "data": {
                    "passed": self.last_result.passed,
                    "total": self.last_result.total,
                    "failures": self.last_result.failures,
                    "timed_out": self.last_result.timed_out,
                },
            }
        if action.tool_name == "Code_Inspect_Error_Tool":
            if self.last_result is None:
                return {"ok": False, "code": "NO_TEST_RESULT"}
            return {"ok": True, "data": {"failures": self.last_result.failures}}
        if action.tool_name == "Code_Finish_Tool":
            return {"ok": True, "data": {"finish": True, "has_code": bool(self.code.strip())}}
        return {"ok": False, "code": "EXTERNAL_GENERATOR_REQUIRED"}


__all__ = ["CodingEnvironment"]
