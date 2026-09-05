from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentflow_rl.runtime.actions import ToolAction, strict_json_object
from agentflow_rl.runtime.errors import ActionParseError


STATUSES = ("open", "pending_customer", "pending_finance", "resolved", "closed")
TEAMS = ("support", "billing", "finance", "logistics", "fraud")
PRIORITIES = ("low", "normal", "high", "urgent")
MUTABLE_FIELDS = ("status", "assigned_team", "priority")
FINISH_OUTCOMES = ("completed",)


class TicketAction(ToolAction):
    sub_goal: str = Field(default="execute the selected ticket operation", min_length=1)

    @classmethod
    def parse(cls, text: str) -> "TicketAction":
        try:
            return cls.model_validate(strict_json_object(text))
        except (ActionParseError, ValueError) as exc:
            raise ActionParseError("ticket action must be one strict JSON object") from exc


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ticket(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ticket_id: str
    customer_id: str
    order_id: str
    subject: str
    status: Literal["open", "pending_customer", "pending_finance", "resolved", "closed"]
    assigned_team: Literal["support", "billing", "finance", "logistics", "fraud"]
    priority: Literal["low", "normal", "high", "urgent"]


class GoalSpec(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_ticket_id: str
    field: Literal["status", "assigned_team", "priority"]
    value: str
    finish_outcome: Literal["completed"] = "completed"


class FinishSubmission(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticket_id: str
    outcome: Literal["completed"]


class ActionEvent(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: str
    arguments: dict[str, str]
    ok: bool
    code: str


class InitialState(StrictModel):
    tickets: tuple[Ticket, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ticket_ids(self) -> "InitialState":
        ids = [ticket.ticket_id for ticket in self.tickets]
        if len(ids) != len(set(ids)):
            raise ValueError("ticket_id values must be unique")
        return self


class TicketEpisode(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    user_request: str
    lookup_mode: Literal["ticket_id", "customer_id", "order_id"]
    max_steps: Literal[2, 3]
    initial_state: InitialState
    goal_spec: GoalSpec
    curriculum_mode: Literal["direct", "indirect"]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_workflow_length(self) -> "TicketEpisode":
        expected_mode = "direct" if self.lookup_mode == "ticket_id" else "indirect"
        expected_steps = 2 if expected_mode == "direct" else 3
        if self.curriculum_mode != expected_mode or self.max_steps != expected_steps:
            raise ValueError("lookup_mode, curriculum_mode, and max_steps are inconsistent")
        return self

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TicketEpisode":
        required = {
            "episode_id",
            "user_request",
            "lookup_mode",
            "max_steps",
            "initial_state",
            "goal_spec",
        }
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Episode is missing fields: {sorted(missing)}")
        known = required | {"curriculum_mode"}
        lookup_mode = str(row["lookup_mode"])
        curriculum = row.get(
            "curriculum_mode", "direct" if lookup_mode == "ticket_id" else "indirect"
        )
        return cls(
            episode_id=str(row["episode_id"]),
            user_request=str(row["user_request"]),
            lookup_mode=lookup_mode,
            max_steps=int(row["max_steps"]),
            initial_state=row["initial_state"],
            goal_spec=row["goal_spec"],
            curriculum_mode=curriculum,
            metadata={key: value for key, value in row.items() if key not in known},
        )
