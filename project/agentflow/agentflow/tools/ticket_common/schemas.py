from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


STATUSES = ("open", "pending_customer", "pending_finance", "resolved", "closed")
TEAMS = ("support", "billing", "finance", "logistics", "fraud")
PRIORITIES = ("low", "normal", "high", "urgent")
MUTABLE_FIELDS = ("status", "assigned_team", "priority")
FINISH_OUTCOMES = ("completed",)


@dataclass
class Ticket:
    ticket_id: str
    customer_id: str
    order_id: str
    subject: str
    status: str
    assigned_team: str
    priority: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Ticket":
        required = {
            "ticket_id", "customer_id", "order_id", "subject",
            "status", "assigned_team", "priority",
        }
        if set(payload) != required:
            raise ValueError(f"Ticket fields must be exactly {sorted(required)}")
        values = {key: str(payload[key]) for key in required}
        if values["status"] not in STATUSES:
            raise ValueError(f"Unknown status: {values['status']}")
        if values["assigned_team"] not in TEAMS:
            raise ValueError(f"Unknown assigned_team: {values['assigned_team']}")
        if values["priority"] not in PRIORITIES:
            raise ValueError(f"Unknown priority: {values['priority']}")
        return cls(**values)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class GoalSpec:
    target_ticket_id: str
    field: str
    value: str
    finish_outcome: str = "completed"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GoalSpec":
        required = {"target_ticket_id", "field", "value", "finish_outcome"}
        if set(payload) != required:
            raise ValueError(f"Goal fields must be exactly {sorted(required)}")
        goal = cls(**{key: str(payload[key]) for key in required})
        if goal.field not in MUTABLE_FIELDS:
            raise ValueError(f"Unknown goal field: {goal.field}")
        if goal.finish_outcome not in FINISH_OUTCOMES:
            raise ValueError(f"Unknown finish outcome: {goal.finish_outcome}")
        return goal


@dataclass(frozen=True)
class FinishSubmission:
    ticket_id: str
    outcome: str


@dataclass(frozen=True)
class ActionEvent:
    operation: str
    arguments: dict[str, str]
    ok: bool
    code: str
