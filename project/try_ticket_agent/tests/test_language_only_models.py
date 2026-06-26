from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import try_ticket_agent.tests  # Bootstrap inner AgentFlow when unittest uses discovery mode.

from agentflow.models.formatters import MemoryVerification, StructuredToolAction
from agentflow.models.memory import Memory
from agentflow.models.planner import Planner
from agentflow.models.verifier import Verifier


class RecordingEngine:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, prompt: object, **kwargs: object) -> object:
        self.calls.append((prompt, kwargs))
        return self.response


def assert_language_only_content(testcase: unittest.TestCase, content: object, image_path: str) -> str:
    if isinstance(content, list):
        testcase.assertEqual(len(content), 1)
        testcase.assertTrue(all(isinstance(item, str) for item in content))
        prompt = content[0]
    else:
        testcase.assertIsInstance(content, str)
        prompt = content
    testcase.assertNotIn("Image:", prompt)
    testcase.assertNotIn(image_path, prompt)
    return prompt


class LanguageOnlyModelTests(unittest.TestCase):
    def make_image_path(self) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        handle.write(b"not-a-real-image")
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_planner_ignores_image_input_even_when_multimodal_flag_is_set(self) -> None:
        image_path = self.make_image_path()
        planner = Planner.__new__(Planner)
        planner.is_multimodal = True
        planner.think_mode = "off"
        planner.query_analysis_think_mode = "off"
        planner.action_mode = "structured"
        planner.available_tools = ["Ticket_Update_Tool", "Ticket_Finish_Tool"]
        planner.toolbox_metadata = {"Ticket_Update_Tool": {}, "Ticket_Finish_Tool": {}}
        planner.generation_configs = {}
        planner.llm_engine_fixed = RecordingEngine("analysis")
        planner.llm_engine = RecordingEngine(
            StructuredToolAction(
                tool_name="Ticket_Finish_Tool",
                arguments={"ticket_id": "T-1", "outcome": "completed"},
            )
        )

        planner.analyze_query("Update ticket T-1", image_path)
        planner.generate_next_step("Update ticket T-1", image_path, "analysis", Memory(), 1, 3)

        query_prompt, _ = planner.llm_engine_fixed.calls[0]
        next_prompt, _ = planner.llm_engine.calls[0]

        assert_language_only_content(self, query_prompt, image_path)
        assert_language_only_content(self, next_prompt, image_path)

    def test_verifier_ignores_image_input_even_when_multimodal_flag_is_set(self) -> None:
        image_path = self.make_image_path()
        verifier = Verifier.__new__(Verifier)
        verifier.is_multimodal = True
        verifier.think_mode = "off"
        verifier.verifier_think_mode = "off"
        verifier.available_tools = ["Ticket_Update_Tool", "Ticket_Finish_Tool"]
        verifier.toolbox_metadata = {"Ticket_Update_Tool": {}, "Ticket_Finish_Tool": {}}
        verifier.generation_configs = {}
        verifier.llm_engine_fixed = RecordingEngine(
            MemoryVerification(analysis="complete", stop_signal=True)
        )

        verifier.verificate_context("Update ticket T-1", image_path, "analysis", Memory(), 1)

        prompt, _ = verifier.llm_engine_fixed.calls[0]

        assert_language_only_content(self, prompt, image_path)


if __name__ == "__main__":
    unittest.main()
