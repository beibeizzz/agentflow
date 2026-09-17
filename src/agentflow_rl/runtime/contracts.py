from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ActionParseError
from .parsing import strict_json_object
from .privacy import assert_public_payload


class TaskName(StrEnum):
    AIME = "aime"
    TWOWIKI = "twowiki"
    TACO = "taco"
    GPQA = "gpqa"
    BIGCODEBENCH = "bigcodebench"


class ToolName(StrEnum):
    BASE_GENERATOR = "Base_Generator_Tool"
    PYTHON_CODER = "Python_Coder_Tool"
    GOOGLE_SEARCH = "Google_Search_Tool"
    WIKIPEDIA_SEARCH = "Wikipedia_Search_Tool"


class VerifierOutcome(StrEnum):
    CONTINUE = "continue"
    FINISH = "finish"


class MemoryVisibility(StrEnum):
    MODEL = "model"
    SCORER = "scorer"
    AUDIT = "audit"


class MemoryAudience(StrEnum):
    MODEL = "model"
    SCORER = "scorer"
    AUDIT = "audit"


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrajectoryIdentity(StrictFrozenModel):
    task_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    session_id: int = Field(ge=0)


class PublicTaskRecord(StrictFrozenModel):
    task_id: str = Field(min_length=1)
    task_name: TaskName
    prompt: str = Field(min_length=1)
    public_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_public_boundary(self) -> "PublicTaskRecord":
        assert_public_payload(self.public_payload, path="public_payload")
        assert_public_payload(self.metadata, path="metadata")
        return self


class PrivateEvaluationRecordRef(StrictFrozenModel):
    key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TaskEnvelope(StrictFrozenModel):
    public: PublicTaskRecord
    private_ref: PrivateEvaluationRecordRef


class PlannerAction(StrictFrozenModel):
    sub_goal: str = Field(min_length=1)
    tool_name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_public_arguments(self) -> "PlannerAction":
        assert_public_payload(self.arguments, path="planner_action.arguments")
        return self

    @classmethod
    def parse(cls, text: str) -> "PlannerAction":
        try:
            return cls.model_validate(strict_json_object(text))
        except Exception as exc:
            if isinstance(exc, ActionParseError):
                raise
            raise ActionParseError("Planner action violates the strict schema") from exc


class ExecutedToolCall(StrictFrozenModel):
    tool_name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_public_arguments(self) -> "ExecutedToolCall":
        assert_public_payload(self.arguments, path="executed_tool_call.arguments")
        return self

    @classmethod
    def parse(cls, text: str) -> "ExecutedToolCall":
        try:
            return cls.model_validate(strict_json_object(text))
        except Exception as exc:
            raise ActionParseError("Executor output violates ExecutedToolCall schema") from exc


class VerifierDecision(StrictFrozenModel):
    outcome: VerifierOutcome
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    failure_codes: tuple[str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> "VerifierDecision":
        try:
            return cls.model_validate(strict_json_object(text))
        except Exception as exc:
            if isinstance(exc, ActionParseError):
                raise
            raise ActionParseError("Verifier decision violates the strict schema") from exc


class FinalAnswerEnvelope(StrictFrozenModel):
    task_name: TaskName
    answer: str = Field(min_length=1)
    public_artifacts: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_public_artifacts(self) -> "FinalAnswerEnvelope":
        assert_public_payload(self.public_artifacts, path="public_artifacts")
        return self


class MemoryEvent(StrictFrozenModel):
    event_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    turn_index: int = Field(ge=-1)
    role: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    content: Any
    tags: tuple[str, ...] = ()
    visibility: MemoryVisibility = MemoryVisibility.MODEL
    model_revision: str | None = None
    prompt_revision: str | None = None
    tool_revision: str | None = None
    backend_revision: str | None = None
    environment_revision: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    response_tokens: int | None = Field(default=None, ge=0)
    truncated: bool = False
    status: str = "ok"
    failure_code: str | None = None
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_content(self) -> "MemoryEvent":
        assert_public_payload(self.content, path=f"memory_event[{self.event_id}].content")
        expected = hashlib.sha256(
            json.dumps(
                self.content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if self.content_sha256 is not None and self.content_sha256 != expected:
            raise ValueError("content_sha256 does not match Memory event content")
        object.__setattr__(self, "content_sha256", expected)
        return self

    def render(self) -> str:
        body = (
            self.content
            if isinstance(self.content, str)
            else json.dumps(self.content, ensure_ascii=False, sort_keys=True)
        )
        return (
            f"[event={self.event_id} turn={self.turn_index} "
            f"role={self.role} kind={self.kind}]\n{body}"
        )


__all__ = [
    "FinalAnswerEnvelope",
    "MemoryAudience",
    "MemoryEvent",
    "MemoryVisibility",
    "PlannerAction",
    "ExecutedToolCall",
    "PrivateEvaluationRecordRef",
    "PublicTaskRecord",
    "TaskEnvelope",
    "TaskName",
    "ToolName",
    "TrajectoryIdentity",
    "VerifierDecision",
    "VerifierOutcome",
]
