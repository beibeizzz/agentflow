from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentflow_rl.runtime.contracts import (
    PlannerAction,
    PublicTaskRecord,
    VerifierDecision,
)
from agentflow_rl.runtime.privacy import assert_public_payload
from agentflow_rl.tools.contracts import ToolRequest, ToolResult


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProcessScore(StrictFrozenModel):
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    failure_code: str | None = None
    scorer_revision: str = Field(min_length=1)
    rubric_revision: str = "legacy"


class ProcessTransition(StrictFrozenModel):
    transition_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    task: PublicTaskRecord
    turn_index: int = Field(ge=0)
    memory_before_action: tuple[dict[str, Any], ...]
    planner_response: str
    planner_action: PlannerAction | None
    tool_request: ToolRequest | None
    tool_result: ToolResult | None
    verifier_decision: VerifierDecision
    planner_memory_view: str | None = None
    planner_memory_event_ids: tuple[str, ...] = ()
    planner_memory_compacted_event_ids: tuple[str, ...] = ()
    planner_memory_core: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def enforce_public_boundary(self) -> "ProcessTransition":
        assert_public_payload(
            self.memory_before_action,
            path=f"process_transition[{self.transition_id}].memory",
        )
        assert_public_payload(self.planner_memory_view, path="process_transition.planner_memory_view")
        assert_public_payload(self.planner_memory_core, path="process_transition.planner_memory_core")
        available_ids = {
            str(event.get("event_id"))
            for event in self.memory_before_action
            if isinstance(event, dict) and event.get("event_id") is not None
        }
        included = set(self.planner_memory_event_ids)
        compacted = set(self.planner_memory_compacted_event_ids)
        if len(included) != len(self.planner_memory_event_ids):
            raise ValueError("Planner-visible Memory event IDs must be unique")
        if not included <= available_ids:
            raise ValueError("Planner-visible Memory event IDs must reference stored pre-action events")
        if not compacted <= included:
            raise ValueError("Compacted Planner Memory events must be included in the Planner view")
        core_ids = {
            str(event.get("event_id"))
            for event in self.planner_memory_core
            if event.get("event_id") is not None
        }
        if not core_ids <= included:
            raise ValueError("PRM core Memory must be a subset of the Planner-visible events")
        return self


__all__ = ["ProcessScore", "ProcessTransition"]
