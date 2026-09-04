from __future__ import annotations

from copy import deepcopy
from typing import Any

from .schemas import (
    FINISH_OUTCOMES,
    MUTABLE_FIELDS,
    PRIORITIES,
    STATUSES,
    TEAMS,
    ActionEvent,
    FinishSubmission,
    GoalSpec,
    InitialState,
    Ticket,
)


LEGAL_STATUS_TRANSITIONS = {
    "open": {"pending_customer", "pending_finance", "resolved"},
    "pending_customer": {"open", "resolved"},
    "pending_finance": {"open", "resolved"},
    "resolved": {"closed"},
    "closed": set(),
}


def result(ok: bool, code: str, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": ok, "code": code, "message": message, "data": data or {}}


class TicketEnvironment:
    """A fresh, in-memory sandbox for exactly one trajectory."""

    def __init__(self, *, initial_state: dict[str, Any], goal_spec: dict[str, Any] | GoalSpec) -> None:
        state = InitialState.model_validate(initial_state)
        self._initial_tickets = {
            ticket.ticket_id: ticket.model_copy(deep=True) for ticket in state.tickets
        }
        self.tickets = deepcopy(self._initial_tickets)
        self.goal_spec = (
            goal_spec if isinstance(goal_spec, GoalSpec) else GoalSpec.model_validate(goal_spec)
        )
        self.action_log: list[ActionEvent] = []
        self.finish_submission: FinishSubmission | None = None

    @property
    def step_count(self) -> int:
        return len(self.action_log)

    def _record(self, operation: str, arguments: dict[str, str], response: dict[str, Any]):
        self.action_log.append(
            ActionEvent(
                operation=operation,
                arguments=dict(arguments),
                ok=response["ok"] is True,
                code=str(response["code"]),
            )
        )
        return response

    def query(self, lookup_by: str, value: str) -> dict[str, Any]:
        arguments = {"lookup_by": str(lookup_by), "value": str(value)}
        if lookup_by not in {"ticket_id", "customer_id", "order_id"}:
            return self._record(
                "query", arguments, result(False, "INVALID_LOOKUP", "Unsupported lookup field.")
            )
        matches = [
            ticket for ticket in self.tickets.values() if getattr(ticket, lookup_by) == str(value)
        ]
        if not matches:
            response = result(False, "NOT_FOUND", "No ticket matched the lookup.")
        elif len(matches) > 1:
            response = result(False, "NON_UNIQUE_MATCH", "Lookup matched more than one ticket.")
        else:
            response = result(True, "OK", "Ticket found.", matches[0].model_dump())
        return self._record("query", arguments, response)

    def update(self, ticket_id: str, field: str, value: str) -> dict[str, Any]:
        arguments = {"ticket_id": str(ticket_id), "field": str(field), "value": str(value)}
        ticket = self.tickets.get(str(ticket_id))
        if ticket is None:
            return self._record("update", arguments, result(False, "NOT_FOUND", "Ticket does not exist."))
        if ticket.status == "closed":
            return self._record(
                "update", arguments, result(False, "CLOSED_TICKET", "Closed tickets cannot be changed.")
            )
        if field not in MUTABLE_FIELDS:
            return self._record("update", arguments, result(False, "INVALID_FIELD", "Field is not mutable."))
        if field == "status":
            if value not in STATUSES:
                return self._record("update", arguments, result(False, "INVALID_VALUE", "Unknown status."))
            if value not in LEGAL_STATUS_TRANSITIONS[ticket.status]:
                return self._record(
                    "update", arguments, result(False, "ILLEGAL_TRANSITION", "Status transition is not legal.")
                )
        elif field == "assigned_team" and value not in TEAMS:
            return self._record("update", arguments, result(False, "INVALID_VALUE", "Unknown team."))
        elif field == "priority" and value not in PRIORITIES:
            return self._record("update", arguments, result(False, "INVALID_VALUE", "Unknown priority."))
        setattr(ticket, field, str(value))
        return self._record("update", arguments, result(True, "OK", "Ticket updated.", ticket.model_dump()))

    def finish(self, ticket_id: str, outcome: str) -> dict[str, Any]:
        arguments = {"ticket_id": str(ticket_id), "outcome": str(outcome)}
        if ticket_id not in self.tickets:
            return self._record("finish", arguments, result(False, "NOT_FOUND", "Ticket does not exist."))
        if outcome not in FINISH_OUTCOMES:
            return self._record("finish", arguments, result(False, "INVALID_OUTCOME", "Unknown finish outcome."))
        self.finish_submission = FinishSubmission(ticket_id=str(ticket_id), outcome=outcome)
        return self._record(
            "finish",
            arguments,
            result(True, "OK", "Workflow submitted.", self.finish_submission.model_dump()),
        )

    def state_dict(self) -> dict[str, Any]:
        return {"tickets": [self.tickets[key].model_dump() for key in sorted(self.tickets)]}

    def state_diff(self) -> dict[str, dict[str, dict[str, str]]]:
        changes: dict[str, dict[str, dict[str, str]]] = {}
        for ticket_id, ticket in self.tickets.items():
            before = self._initial_tickets[ticket_id]
            fields: dict[str, dict[str, str]] = {}
            for field in MUTABLE_FIELDS:
                old, new = str(getattr(before, field)), str(getattr(ticket, field))
                if old != new:
                    fields[field] = {"before": old, "after": new}
            if fields:
                changes[ticket_id] = fields
        return changes
