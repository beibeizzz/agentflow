from __future__ import annotations

import asyncio
from time import monotonic
from typing import Protocol
from agentflow_rl.runtime.contracts import ToolName, PlannerAction
from agentflow_rl.runtime.errors import ActionParseError, InfrastructureError, RateLimitError, ToolDispatchError, PromptBudgetError
from .contracts import BackendCall, ToolFailureCode, ToolRequest, ToolResult, ToolStatus
from .catalog import ToolSpec, tool_spec


class ToolHandler(Protocol):
    name: ToolName
    revision: str
    async def execute(self, request: ToolRequest) -> ToolResult: ...


class ToolRegistry:
    def __init__(self, handlers: tuple[ToolHandler, ...]) -> None:
        self._handlers = {handler.name: handler for handler in handlers}
        if len(self._handlers) != len(handlers):
            raise ValueError("duplicate tool handler")
        self._specs = {name: tool_spec(name, max_tokens=getattr(handler, "default_max_tokens", 2048),
            timeout_s=getattr(handler, "default_timeout_s", 10.0), read_top_k=getattr(handler, "read_top_k", 2),
            max_passages=getattr(handler, "max_passages", 3), max_source_chars=getattr(handler, "max_source_chars", 4000))
            for name, handler in self._handlers.items()}

    def spec(self, name: ToolName) -> ToolSpec:
        return self._specs[name]

    def catalog(self) -> list[dict]:
        return [self._specs[name].view() for name in ToolName if name in self._specs]

    def validate_action(self, action: PlannerAction) -> None:
        try:
            self.spec(action.tool_name).intent_arguments.model_validate(action.arguments)
        except ValueError as exc:
            raise ActionParseError("Planner intent parameters violate the tool schema") from exc

    def normalize_request(self, request: ToolRequest) -> ToolRequest:
        arguments = self.spec(request.tool_name).execution_arguments.model_validate(request.arguments).model_dump()
        return request.model_copy(update={"arguments": arguments})

    @property
    def names(self) -> tuple[ToolName, ...]:
        return tuple(self._handlers)

    def assert_complete(self) -> None:
        if set(self._handlers) != set(ToolName):
            raise ValueError("tool registry must contain exactly the four shared tools")

    async def dispatch(self, request: ToolRequest) -> ToolResult:
        started = monotonic()
        handler = self._handlers.get(request.tool_name)
        if handler is None:
            raise ToolDispatchError(f"unregistered tool: {request.tool_name}")
        try:
            request = self.normalize_request(request)
            return await asyncio.wait_for(handler.execute(request), timeout=request.timeout_s)
        except (TypeError, ValueError) as exc:
            status, failure, message = ToolStatus.MODEL_ERROR, ToolFailureCode.INVALID_ARGUMENTS, str(exc)
        except PromptBudgetError:
            status, failure, message = ToolStatus.MODEL_ERROR, ToolFailureCode.INVALID_ARGUMENTS, "Supplied tool context exceeds its input token budget; shorten the prompt."
        except (TimeoutError, asyncio.TimeoutError):
            status, failure, message = ToolStatus.INFRASTRUCTURE_ERROR, ToolFailureCode.TIMEOUT, "tool backend timed out"
        except RateLimitError as exc:
            status, failure, message = ToolStatus.INFRASTRUCTURE_ERROR, ToolFailureCode.RATE_LIMITED, str(exc)
        except (InfrastructureError, OSError) as exc:
            status, failure, message = ToolStatus.INFRASTRUCTURE_ERROR, ToolFailureCode.BACKEND_UNAVAILABLE, str(exc)
        elapsed = (monotonic() - started) * 1000
        return ToolResult(request_id=request.request_id, tool_name=request.tool_name, status=status,
                          failure_code=failure, message=message, backend_revision=handler.revision, latency_ms=elapsed,
                          backend_calls=(BackendCall(stage="dispatch_failure", status=status, failure_code=failure,
                                                     latency_ms=elapsed),))
