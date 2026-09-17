from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentflow_rl.runtime.contracts import ToolName
from agentflow_rl.runtime.privacy import assert_public_payload


TOOL_CONTRACT_REVISION = "shared-tools-v3"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    MODEL_ERROR = "model_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class ToolFailureCode(StrEnum):
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    EMPTY_RESULT = "EMPTY_RESULT"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    tool_name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def enforce_public_arguments(self) -> "ToolRequest":
        assert_public_payload(self.arguments, path="tool_request.arguments")
        return self


class BackendCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stage: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    status: ToolStatus = ToolStatus.SUCCESS
    arguments: dict[str, Any] = Field(default_factory=dict)
    failure_code: ToolFailureCode | None = None
    latency_ms: float = Field(default=0, ge=0)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    tool_name: ToolName
    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    failure_code: ToolFailureCode | None = None
    message: str = ""
    backend_revision: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    truncated: bool = False
    backend_calls: tuple[BackendCall, ...] = ()
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_result(self) -> "ToolResult":
        assert_public_payload(self.data, path="tool_result.data")
        assert_public_payload([call.arguments for call in self.backend_calls], path="tool_result.backend_calls")
        if self.status is ToolStatus.SUCCESS and self.failure_code is not None:
            raise ValueError("successful tool result cannot contain failure_code")
        if self.status is not ToolStatus.SUCCESS and self.failure_code is None:
            raise ValueError("failed tool result requires failure_code")
        expected = hashlib.sha256(
            json.dumps(
                self.data,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if self.content_sha256 is not None and self.content_sha256 != expected:
            raise ValueError("content_sha256 does not match tool result data")
        object.__setattr__(self, "content_sha256", expected)
        return self

    @property
    def valid_for_training(self) -> bool:
        return self.status is not ToolStatus.INFRASTRUCTURE_ERROR


__all__ = [
    "TOOL_CONTRACT_REVISION",
    "BackendCall",
    "ToolFailureCode",
    "ToolRequest",
    "ToolResult",
    "ToolStatus",
]
