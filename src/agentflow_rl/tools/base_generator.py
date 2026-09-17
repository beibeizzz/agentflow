from __future__ import annotations

from time import monotonic
from agentflow_rl.backends.contracts import BaseGeneratorBackend
from agentflow_rl.runtime.contracts import ToolName
from agentflow_rl.runtime.prompt_budget import role_token_count
from .catalog import tool_spec
from .contracts import BackendCall, ToolRequest, ToolResult, ToolStatus


class BaseGeneratorTool:
    name = ToolName.BASE_GENERATOR

    def __init__(self, backend: BaseGeneratorBackend, *, default_max_tokens: int = 2048) -> None:
        if default_max_tokens <= 0:
            raise ValueError("Generator token budget must be positive")
        self.backend = backend
        self.default_max_tokens = default_max_tokens
        self.revision = f"base-generator-tool-v2:{backend.revision}"
        self._arguments = tool_spec(self.name, max_tokens=default_max_tokens).execution_arguments

    async def execute(self, request: ToolRequest) -> ToolResult:
        args = self._arguments.model_validate(request.arguments)
        started = monotonic()
        text = await self.backend.generate(prompt=args.query, max_tokens=args.max_tokens)
        elapsed = (monotonic() - started) * 1000
        tokenizer = getattr(getattr(self.backend, "gateway", None), "tokenizer", None)
        input_tokens = role_token_count(tokenizer, "Provide concise general reasoning for the requested sub-goal.", args.query) if tokenizer else None
        output_tokens = len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer else None
        return ToolResult(request_id=request.request_id, tool_name=self.name, status=ToolStatus.SUCCESS,
                          data={"text": text}, backend_revision=self.revision, latency_ms=elapsed,
                          backend_calls=(BackendCall(stage="generate", input_tokens=input_tokens, output_tokens=output_tokens, arguments=args.model_dump(), latency_ms=elapsed),))
