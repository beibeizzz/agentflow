from __future__ import annotations

import inspect
import json
import unittest

from agentflow.models.formatters import StructuredToolAction
from agentflow.models.memory import Memory
from agentflow.models.planner import Planner


class RecordingEngine:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, prompt: object, **kwargs: object) -> object:
        self.calls.append((prompt, kwargs))
        return self.response


def make_planner(*, action_mode: str) -> Planner:
    planner = Planner.__new__(Planner)
    planner.action_mode = action_mode
    planner.available_tools = ["Ticket_Update_Tool", "Ticket_Finish_Tool"]
    planner.toolbox_metadata = {
        "Ticket_Update_Tool": {"input_types": {"ticket_id": "str", "field": "str", "value": "str"}},
        "Ticket_Finish_Tool": {"input_types": {"ticket_id": "str", "outcome": "str"}},
    }
    planner.is_multimodal = False
    planner.think_mode = "off"
    planner.generation_configs = {}
    planner.llm_engine = RecordingEngine(
        StructuredToolAction(
            tool_name="Ticket_Finish_Tool",
            arguments={"ticket_id": "T-1", "outcome": "completed"},
        )
    )
    return planner


class StructuredPlannerTests(unittest.TestCase):
    def test_structured_action_forbids_extra_top_level_fields(self) -> None:
        with self.assertRaises(Exception):
            StructuredToolAction(
                tool_name="Ticket_Finish_Tool",
                arguments={"ticket_id": "T-1", "outcome": "completed"},
                extra="forbidden",
            )

    def test_structured_extraction_returns_canonical_arguments(self) -> None:
        planner = make_planner(action_mode="structured")
        result = planner.extract_context_subgoal_and_tool(
            StructuredToolAction(
                tool_name="Ticket_Update_Tool",
                arguments={"value": "urgent", "ticket_id": "T-1", "field": "priority"},
            )
        )
        self.assertEqual(
            result,
            (
                '{"field":"priority","ticket_id":"T-1","value":"urgent"}',
                "Execute Ticket_Update_Tool",
                "Ticket_Update_Tool",
            ),
        )

    def test_structured_extraction_rejects_wrappers_and_extra_keys(self) -> None:
        planner = make_planner(action_mode="structured")
        invalid = [
            '```json\n{"tool_name":"Ticket_Finish_Tool","arguments":{}}\n```',
            '{"tool_name":"Ticket_Finish_Tool","arguments":{},"extra":1}',
            '[{"tool_name":"Ticket_Finish_Tool","arguments":{}}]',
        ]
        for response in invalid:
            with self.subTest(response=response):
                self.assertEqual(planner.extract_context_subgoal_and_tool(response), (None, None, None))

    def test_structured_generation_uses_action_schema(self) -> None:
        planner = make_planner(action_mode="structured")
        planner.generate_next_step("update T-1", None, "analysis", Memory(), 1, 3)
        prompt, kwargs = planner.llm_engine.calls[0]
        self.assertIn("Return exactly one JSON object", str(prompt))
        self.assertIs(kwargs["response_format"], StructuredToolAction)

    def test_planner_constructor_defaults_to_legacy_mode(self) -> None:
        signature = inspect.signature(Planner.__init__)
        self.assertEqual(signature.parameters["action_mode"].default, "legacy")


if __name__ == "__main__":
    unittest.main()
