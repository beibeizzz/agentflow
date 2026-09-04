from __future__ import annotations

import json


def ticket(ticket_id: str, customer_id: str, order_id: str) -> dict:
    return {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "order_id": order_id,
        "subject": "Payment review",
        "status": "open",
        "assigned_team": "support",
        "priority": "normal",
    }


def environment():
    from agentflow_rl.tasks.ticket.environment import TicketEnvironment

    return TicketEnvironment(
        initial_state={
            "tickets": [ticket("T-1", "C-1", "O-1"), ticket("T-2", "C-2", "O-2")]
        },
        goal_spec={
            "target_ticket_id": "T-1",
            "field": "priority",
            "value": "high",
            "finish_outcome": "completed",
        },
    )


def test_domain_constants_equal_original_agentflow_ticket_sandbox() -> None:
    from agentflow_rl.tasks.ticket.environment import LEGAL_STATUS_TRANSITIONS
    from agentflow_rl.tasks.ticket.schemas import MUTABLE_FIELDS, PRIORITIES, STATUSES, TEAMS

    assert STATUSES == ("open", "pending_customer", "pending_finance", "resolved", "closed")
    assert TEAMS == ("support", "billing", "finance", "logistics", "fraud")
    assert PRIORITIES == ("low", "normal", "high", "urgent")
    assert MUTABLE_FIELDS == ("status", "assigned_team", "priority")
    assert LEGAL_STATUS_TRANSITIONS["open"] == {"pending_customer", "pending_finance", "resolved"}
    assert LEGAL_STATUS_TRANSITIONS["closed"] == set()


def test_golden_environment_transcript_is_json_stable() -> None:
    env = environment()
    transcript = [
        env.query("order_id", "O-1"),
        env.update("T-1", "priority", "high"),
        env.finish("T-1", "completed"),
    ]

    assert json.loads(json.dumps(transcript, sort_keys=True)) == [
        {"ok": True, "code": "OK", "message": "Ticket found.", "data": ticket("T-1", "C-1", "O-1")},
        {
            "ok": True,
            "code": "OK",
            "message": "Ticket updated.",
            "data": {**ticket("T-1", "C-1", "O-1"), "priority": "high"},
        },
        {
            "ok": True,
            "code": "OK",
            "message": "Workflow submitted.",
            "data": {"ticket_id": "T-1", "outcome": "completed"},
        },
    ]


def test_episode_parser_and_balancer_are_task_local() -> None:
    from agentflow_rl.tasks.ticket.dataset import balance_ticket_rows
    from agentflow_rl.tasks.ticket.schemas import TicketEpisode

    def row(index: int, lookup_mode: str) -> dict:
        return {
            "episode_id": f"e-{index}",
            "user_request": "request",
            "lookup_mode": lookup_mode,
            "max_steps": 2 if lookup_mode == "ticket_id" else 3,
            "initial_state": {"tickets": [ticket(f"T-{index}", f"C-{index}", f"O-{index}")]},
            "goal_spec": {
                "target_ticket_id": f"T-{index}",
                "field": "priority",
                "value": "high",
                "finish_outcome": "completed",
            },
            "curriculum_mode": "direct" if lookup_mode == "ticket_id" else "indirect",
            "generator_version": "ignored metadata",
        }

    rows = [row(i, "ticket_id") for i in range(4)] + [row(i + 4, "customer_id") for i in range(2)]
    parsed = TicketEpisode.from_row(rows[0])
    balanced = balance_ticket_rows(rows, direct_fraction=0.5, seed=7)

    assert parsed.episode_id == "e-0"
    assert len(balanced) == 4
    assert sum(item.lookup_mode == "ticket_id" for item in balanced) == 2
    assert sum(item.lookup_mode != "ticket_id" for item in balanced) == 2
