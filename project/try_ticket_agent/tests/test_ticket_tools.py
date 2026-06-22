from __future__ import annotations

from pathlib import Path
import json
import unittest

from agentflow.tools.ticket_common.backend import TicketBackend
from agentflow.tools.ticket_finish.tool import Ticket_Finish_Tool
from agentflow.tools.ticket_query.tool import Ticket_Query_Tool
from agentflow.tools.ticket_update.tool import Ticket_Update_Tool
from agentflow.models.initializer import Initializer


PROJECT_DIR = Path(__file__).resolve().parents[2]


class BaseToolDependencyTests(unittest.TestCase):
    def test_base_tool_has_no_llm_engine_import(self) -> None:
        source = (PROJECT_DIR / "agentflow" / "agentflow" / "tools" / "base.py").read_text(encoding="utf-8")
        self.assertNotIn("agentflow.engine.openai", source)


def make_state(*, status: str = "open") -> dict[str, object]:
    return {
        "tickets": [
            {
                "ticket_id": "T-1",
                "customer_id": "C-1",
                "order_id": "O-1",
                "subject": "Payment review",
                "status": status,
                "assigned_team": "support",
                "priority": "normal",
            },
            {
                "ticket_id": "T-2",
                "customer_id": "C-2",
                "order_id": "O-2",
                "subject": "Delivery review",
                "status": "open",
                "assigned_team": "logistics",
                "priority": "low",
            },
        ]
    }


def make_goal() -> dict[str, str]:
    return {
        "target_ticket_id": "T-1",
        "field": "priority",
        "value": "urgent",
        "finish_outcome": "completed",
    }


class TicketToolTests(unittest.TestCase):
    def test_initializer_discovers_all_ticket_tools(self) -> None:
        initializer = Initializer(
            enabled_tools=["Ticket_Query_Tool", "Ticket_Update_Tool", "Ticket_Finish_Tool"],
            tool_engine=["Default", "Default", "Default"],
            verbose=False,
            parallel_loading=False,
        )
        self.assertEqual(
            set(initializer.available_tools),
            {"Ticket_Query_Tool", "Ticket_Update_Tool", "Ticket_Finish_Tool"},
        )
        self.assertEqual(set(initializer.tool_instances_cache), set(initializer.available_tools))

    def test_unbound_tool_returns_structured_error(self) -> None:
        result = Ticket_Query_Tool().execute(lookup_by="ticket_id", value="T-1")
        self.assertEqual(result["code"], "BACKEND_NOT_BOUND")

    def test_three_tools_share_backend_and_complete_workflow(self) -> None:
        backend = TicketBackend()
        backend.reset(make_state(), make_goal())
        query = Ticket_Query_Tool()
        update = Ticket_Update_Tool()
        finish = Ticket_Finish_Tool()
        for tool in (query, update, finish):
            tool.bind_backend(backend)

        queried = query.execute(lookup_by="ticket_id", value="T-1")
        changed = update.execute(ticket_id="T-1", field="priority", value="urgent")
        submitted = finish.execute(ticket_id="T-1", outcome="completed")

        self.assertEqual(queried["data"]["ticket_id"], "T-1")
        self.assertTrue(changed["ok"])
        self.assertTrue(submitted["ok"])
        self.assertEqual(backend.tickets["T-1"].priority, "urgent")
        self.assertEqual(backend.finish_submission.ticket_id, "T-1")
        json.dumps([queried, changed, submitted])

    def test_rejected_transition_is_atomic(self) -> None:
        backend = TicketBackend()
        backend.reset(make_state(status="resolved"), make_goal())
        result = backend.update("T-1", "status", "open")
        self.assertEqual(result["code"], "ILLEGAL_TRANSITION")
        self.assertEqual(backend.tickets["T-1"].status, "resolved")

    def test_finish_does_not_mutate_ticket_state(self) -> None:
        backend = TicketBackend()
        backend.reset(make_state(), make_goal())
        before = backend.state_dict()
        result = backend.finish("T-1", "completed")
        self.assertTrue(result["ok"])
        self.assertEqual(backend.state_dict(), before)

    def test_reset_deep_copies_state_between_episodes(self) -> None:
        source = make_state()
        left = TicketBackend()
        right = TicketBackend()
        left.reset(source, make_goal())
        right.reset(source, make_goal())
        left.update("T-1", "priority", "urgent")
        self.assertEqual(right.tickets["T-1"].priority, "normal")
        self.assertEqual(source["tickets"][0]["priority"], "normal")

    def test_query_requires_unique_match(self) -> None:
        state = make_state()
        state["tickets"][1]["customer_id"] = "C-1"
        backend = TicketBackend()
        backend.reset(state, make_goal())
        result = backend.query("customer_id", "C-1")
        self.assertEqual(result["code"], "NON_UNIQUE_MATCH")


if __name__ == "__main__":
    unittest.main()
