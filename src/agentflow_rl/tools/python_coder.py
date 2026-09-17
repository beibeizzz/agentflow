from __future__ import annotations

import hashlib
from time import monotonic
from agentflow_rl.backends.contracts import SandboxBackend, SandboxRequest
from agentflow_rl.runtime.contracts import ToolName
from .catalog import tool_spec
from .contracts import BackendCall, ToolFailureCode, ToolRequest, ToolResult, ToolStatus


class PythonCoderTool:
    name = ToolName.PYTHON_CODER

    def __init__(self, backend: SandboxBackend, *, default_timeout_s: float = 10.0) -> None:
        if default_timeout_s <= 0:
            raise ValueError("Python timeout must be positive")
        self.backend = backend
        self.default_timeout_s = default_timeout_s
        self.revision = f"python-coder-tool-v2:{backend.revision}"
        self._arguments = tool_spec(self.name, timeout_s=default_timeout_s).execution_arguments

    async def execute(self, request: ToolRequest) -> ToolResult:
        args = self._arguments.model_validate(request.arguments)
        started = monotonic()
        mode = "test" if args.tests else "run"
        sandbox_request = SandboxRequest(code=args.code, mode=mode, stdin=args.stdin,
                                         tests=tuple(test.model_dump() for test in args.tests),
                                         timeout_s=min(self.default_timeout_s, request.timeout_s))
        result = await self.backend.execute(sandbox_request)
        elapsed = (monotonic() - started) * 1000
        status = ToolStatus.SUCCESS if result.ok else ToolStatus.MODEL_ERROR
        failure = None if result.ok else ToolFailureCode.EXECUTION_FAILED
        return ToolResult(
            request_id=request.request_id, tool_name=self.name, status=status, failure_code=failure,
            data={"code": args.code, "code_revision": request.turn_index + 1,
                  "code_sha256": hashlib.sha256(args.code.encode()).hexdigest(),
                  "execution": result.model_dump(mode="json")},
            message="" if result.ok else "sandbox execution failed", backend_revision=self.revision,
            latency_ms=elapsed, backend_calls=(BackendCall(stage=mode, status=status,
                arguments=sandbox_request.model_dump(mode="json"), failure_code=failure, latency_ms=elapsed),),
        )
