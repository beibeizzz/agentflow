from __future__ import annotations

import pytest

from agentflow_rl.runtime.actions import ToolAction


def ticket(ticket_id: str, customer_id: str, order_id: str, *, status: str = "open") -> dict:
    return {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "order_id": order_id,
        "subject": "Payment review",
        "status": status,
        "assigned_team": "support",
        "priority": "normal",
    }


def environment(*, duplicate_customer: bool = False):
    from agentflow_rl.tasks.ticket.environment import TicketEnvironment

    return TicketEnvironment(
        initial_state={
            "tickets": [
                ticket("T-1", "C-1", "O-1"),
                ticket("T-2", "C-1" if duplicate_customer else "C-2", "O-2"),
            ]
        },
        goal_spec={
            "target_ticket_id": "T-1",
            "field": "priority",
            "value": "high",
            "finish_outcome": "completed",
        },
    )


@pytest.mark.parametrize(
    ("lookup_by", "value"),
    [("ticket_id", "T-1"), ("customer_id", "C-1"), ("order_id", "O-1")],
)
def test_query_supports_all_three_lookup_keys(lookup_by: str, value: str) -> None:
    response = environment().query(lookup_by, value)

    assert set(response) == {"ok", "code", "message", "data"}
    assert response == {
        "ok": True,
        "code": "OK",
        "message": "Ticket found.",
        "data": ticket("T-1", "C-1", "O-1"),
    }


def test_query_reports_invalid_not_found_and_non_unique() -> None:
    env = environment(duplicate_customer=True)

    assert env.query("subject", "x")["code"] == "INVALID_LOOKUP"
    assert env.query("ticket_id", "missing")["code"] == "NOT_FOUND"
    assert env.query("customer_id", "C-1")["code"] == "NON_UNIQUE_MATCH"


def test_update_enforces_fields_values_transitions_and_closed_ticket() -> None:
    env = environment()

    assert env.update("missing", "priority", "high")["code"] == "NOT_FOUND"
    assert env.update("T-1", "subject", "x")["code"] == "INVALID_FIELD"
    assert env.update("T-1", "priority", "extreme")["code"] == "INVALID_VALUE"
    assert env.update("T-1", "assigned_team", "sales")["code"] == "INVALID_VALUE"
    assert env.update("T-1", "status", "closed")["code"] == "ILLEGAL_TRANSITION"
    assert env.update("T-1", "status", "resolved")["code"] == "OK"
    assert env.update("T-1", "status", "closed")["code"] == "OK"
    assert env.update("T-1", "priority", "high")["code"] == "CLOSED_TICKET"


def test_finish_and_state_diff_match_original_contract() -> None:
    env = environment()

    assert env.update("T-1", "priority", "high")["ok"] is True
    response = env.finish("T-1", "completed")

    assert response == {
        "ok": True,
        "code": "OK",
        "message": "Workflow submitted.",
        "data": {"ticket_id": "T-1", "outcome": "completed"},
    }
    assert env.state_diff() == {
        "T-1": {"priority": {"before": "normal", "after": "high"}}
    }
    assert env.finish("missing", "completed")["code"] == "NOT_FOUND"
    assert env.finish("T-1", "cancelled")["code"] == "INVALID_OUTCOME"


def test_tool_dispatch_validates_exact_argument_schema() -> None:
    from agentflow_rl.tasks.ticket.tools import ToolDispatchError, execute_tool

    env = environment()
    response = execute_tool(
        env,
        ToolAction(
            tool_name="Ticket_Update_Tool",
            arguments={"ticket_id": "T-1", "field": "priority", "value": "high"},
        ),
    )
    assert response["code"] == "OK"

    with pytest.raises(ToolDispatchError, match="strict schema"):
        execute_tool(
            env,
            ToolAction(
                tool_name="Ticket_Update_Tool",
                arguments={"ticket_id": "T-1", "field": "priority", "value": "high", "extra": 1},
            ),
        )
    with pytest.raises(ToolDispatchError, match="unknown ticket tool"):
        execute_tool(env, ToolAction(tool_name="Other", arguments={}))
