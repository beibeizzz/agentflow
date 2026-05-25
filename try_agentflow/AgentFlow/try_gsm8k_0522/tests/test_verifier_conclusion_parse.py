import sys
import types
import unittest
from pathlib import Path


def bootstrap_agentflow_runtime() -> None:
    agentflow_core = Path(__file__).resolve().parents[2] / "agentflow" / "agentflow"
    agentflow_pkg = types.ModuleType("agentflow")
    agentflow_pkg.__path__ = [str(agentflow_core)]
    agentflow_pkg.__file__ = str(agentflow_core / "__init__.py")
    sys.modules["agentflow"] = agentflow_pkg


class TestVerifierConclusionParse(unittest.TestCase):
    def setUp(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.verifier import Verifier

        self.verifier = Verifier.__new__(Verifier)
        self.verifier.available_tools = ["Calculator_Tool"]

    def test_uses_first_non_empty_conclusion_line(self):
        analysis, conclusion = self.verifier.extract_conclusion(
            """

Conclusion: CONTINUE
Expression can't computes the final answer. Missing logic: no calculator result.
Conclusion: STOP
"""
        )

        self.assertEqual(conclusion, "CONTINUE")
        self.assertIn("no calculator result", analysis)

    def test_parses_stop_from_first_non_empty_line(self):
        _, conclusion = self.verifier.extract_conclusion(
            """
Conclusion: STOP
"""
        )

        self.assertEqual(conclusion, "STOP")

    def test_missing_first_line_conclusion_defaults_to_continue(self):
        _, conclusion = self.verifier.extract_conclusion(
            """
Expression can't computes the final answer. Missing logic: should not stop.
Conclusion: STOP
"""
        )

        self.assertEqual(conclusion, "CONTINUE")

    def test_invalid_planner_output_short_circuits_verifier_llm(self):
        from agentflow.models.memory import Memory

        memory = Memory()
        memory.add_action(
            1,
            None,
            None,
            "Planner output was invalid; no tool command generated.",
            "No calculator execution was attempted.",
        )

        verifier = self.verifier
        verifier.is_multimodal = False

        def fail_if_called(*args, **kwargs):
            raise AssertionError("Verifier LLM should not be called for invalid planner output.")

        verifier.llm_engine_fixed = fail_if_called
        json_data = {}

        response = verifier.verificate_context(
            question="Problem text",
            image=None,
            query_analysis="",
            memory=memory,
            step_count=1,
            json_data=json_data,
        )

        self.assertEqual(
            response,
            "Conclusion: CONTINUE\nlast Planner output was invalid. check the Calculation first",
        )
        self.assertEqual(json_data["verifier_1_response"], response)
        self.assertNotIn("verifier_1_prompt", json_data)


if __name__ == "__main__":
    unittest.main()
