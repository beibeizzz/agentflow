"""Bounded search-and-read policy shared by web and Wikipedia tools."""
from __future__ import annotations

import asyncio
from time import monotonic
from agentflow_rl.runtime.errors import InfrastructureError, RateLimitError
from .contracts import BackendCall, ToolFailureCode, ToolStatus


def bounded_document(document: dict, *, max_passages: int, max_chars: int) -> dict:
    source = document.get("passages", [])
    passages, ids, remaining = [], [], max_chars
    for i, passage in enumerate(source[:max_passages]):
        if remaining <= 0:
            break
        passages.append(passage[:remaining])
        remaining -= len(passages[-1])
        if "passage_ids" in document:
            ids.append(document["passage_ids"][i])
    truncated = passages != list(source)
    result = {**document, "passages": passages, "truncated": truncated, "read_status": "success"}
    if "passage_ids" in document:
        result["passage_ids"] = ids
    return result


async def read_source(read, arguments: dict, identity: dict, *, timeout_s: float,
                      max_passages: int, max_chars: int):
    started = monotonic()
    status, failure = ToolStatus.SUCCESS, None
    try:
        document = await asyncio.wait_for(read(**arguments), timeout=timeout_s)
        data = bounded_document(document.model_dump(mode="json"), max_passages=max_passages, max_chars=max_chars)
    except ValueError:
        status, failure = ToolStatus.MODEL_ERROR, ToolFailureCode.INVALID_ARGUMENTS
        data = {**identity, "read_status": "unavailable", "failure_code": failure.value,
                "message": "This source cannot be read under the content and URL policy."}
    except (InfrastructureError, TimeoutError, OSError) as exc:
        status = ToolStatus.INFRASTRUCTURE_ERROR
        failure = (ToolFailureCode.TIMEOUT if isinstance(exc, TimeoutError) else
                   ToolFailureCode.RATE_LIMITED if isinstance(exc, RateLimitError) else ToolFailureCode.BACKEND_UNAVAILABLE)
        data = {**identity, "read_status": "infrastructure_error", "failure_code": failure.value,
                "message": "Source reading failed at the service boundary."}
    return data, BackendCall(stage="read", status=status, failure_code=failure, arguments=arguments,
                             latency_ms=(monotonic() - started) * 1000)


class RetrievalPolicy:
    def __init__(self, *, read_top_k=2, max_passages=3, max_source_chars=4000, read_timeout_s=8.0):
        if not 1 <= read_top_k <= 20 or not 1 <= max_passages <= 40 or max_source_chars <= 0 or read_timeout_s <= 0:
            raise ValueError("invalid search-and-read budget")
        self.read_top_k = read_top_k
        self.max_passages = max_passages
        self.max_source_chars = max_source_chars
        self.read_timeout_s = read_timeout_s

    @property
    def policy_revision(self):
        return f"read{self.read_top_k}-passages{self.max_passages}-chars{self.max_source_chars}-timeout{self.read_timeout_s:g}"

    def read_kwargs(self, request):
        return dict(timeout_s=min(self.read_timeout_s, request.timeout_s),
                    max_passages=self.max_passages, max_chars=self.max_source_chars)
