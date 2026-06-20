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


class DummyMemory:
    def get_actions(self):
        return {
            "Action Step 1": {
                "tool_name": "Calculator_Tool",
                "sub_goal": "Calculate 4 * 2",
                "command": 'execution = tool.execute(expression="4 * 2")',
                "result": ["8"],
            }
        }


class TestRolePrompts(unittest.TestCase):
    def test_planner_extracts_calculation_json_as_context(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.planner import Planner

        planner = Planner.__new__(Planner)
        planner.available_tools = ["Calculator_Tool"]

        response = """
        {
          "sub_goal": "Calculate the next arithmetic expression",
          "calculation": "Expression to calculate: 16 - 3 - 4",
          "tool_name": "Calculator_Tool"
        }
        """

        context, sub_goal, tool_name = planner.extract_context_subgoal_and_tool(response)

        self.assertEqual(context, "Expression to calculate: 16 - 3 - 4")
        self.assertEqual(sub_goal, "Calculate the next arithmetic expression")
        self.assertEqual(tool_name, "Calculator_Tool")

    def test_planner_extracts_calculation_only_json_for_calculator_tool(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.planner import Planner

        planner = Planner.__new__(Planner)
        planner.available_tools = ["Calculator_Tool"]

        context, sub_goal, tool_name = planner.extract_context_subgoal_and_tool(
            '{"sub_goal": "Calculate remaining eggs for sale", "calculation": "16 - 3 - 4"}'
        )

        self.assertEqual(context, "16 - 3 - 4")
        self.assertEqual(sub_goal, "Calculate remaining eggs for sale")
        self.assertEqual(tool_name, "Calculator_Tool")

    def test_planner_extracts_calculation_next_step_object(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.formatters import NextStep
        from agentflow.models.planner import Planner

        planner = Planner.__new__(Planner)
        planner.available_tools = ["Calculator_Tool"]
        response = NextStep(
            sub_goal="Calculate the next arithmetic expression",
            calculation="Expression to calculate: 8 * 12",
            tool_name="Calculator_Tool",
        )

        context, sub_goal, tool_name = planner.extract_context_subgoal_and_tool(response)

        self.assertEqual(context, "Expression to calculate: 8 * 12")
        self.assertEqual(sub_goal, "Calculate the next arithmetic expression")
        self.assertEqual(tool_name, "Calculator_Tool")

    def test_planner_extracts_calculation_only_next_step_object_for_calculator_tool(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.formatters import NextStep
        from agentflow.models.planner import Planner

        planner = Planner.__new__(Planner)
        planner.available_tools = ["Calculator_Tool"]
        response = NextStep(
            sub_goal="Convert dozens to cookies",
            calculation="8 * 12",
        )

        context, sub_goal, tool_name = planner.extract_context_subgoal_and_tool(response)

        self.assertEqual(context, "8 * 12")
        self.assertEqual(sub_goal, "Convert dozens to cookies")
        self.assertEqual(tool_name, "Calculator_Tool")

    def test_planner_analyze_query_prompt_sets_calculator_workflow(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.planner import Planner

        captured = {}
        planner = Planner.__new__(Planner)
        planner.is_multimodal = False
        planner.available_tools = ["Calculator_Tool"]
        planner.toolbox_metadata = {"Calculator_Tool": {"input_types": {"expression": "str"}}}

        def fake_llm(input_data, **kwargs):
            captured["input_data"] = input_data
            captured["kwargs"] = kwargs
            return "analysis"

        planner.llm_engine_fixed = fake_llm
        result = planner.analyze_query(
            question="Problem text",
            image=None,
        )

        self.assertEqual(result, "analysis")
        self.assertIn("Solve the following GSM8K math word problem.", captured["input_data"][0])
        self.assertIn("Problem:\nProblem text", captured["input_data"][0])
        self.assertNotIn("You should focus on your responsibility mentioned.", captured["input_data"][0])
        self.assertEqual(captured["kwargs"]["max_tokens"], 512)
        self.assertEqual(captured["kwargs"]["temperature"], 0.0)
        self.assertEqual(captured["kwargs"]["top_p"], 0.95)
        self.assertEqual(captured["kwargs"]["frequency_penalty"], 0)
        self.assertIn("careful grade-school math problem solver", captured["kwargs"]["system_prompt"])

    def test_planner_next_step_prompt_is_calculator_specific(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.formatters import NextStep
        from agentflow.models.planner import Planner

        captured = {}
        planner = Planner.__new__(Planner)
        planner.is_multimodal = False
        planner.available_tools = ["Calculator_Tool"]
        planner.toolbox_metadata = {"Calculator_Tool": {"input_types": {"expression": "str"}}}

        def fake_llm(prompt, response_format=None):
            captured["prompt"] = prompt
            return NextStep(
                justification="",
                context="Expression to calculate: 4 * 2",
                sub_goal="Calculate 4 * 2",
                tool_name="Calculator_Tool",
            )

        planner.llm_engine = fake_llm
        planner.generate_next_step(
            question="Problem text",
            image=None,
            query_analysis="Need arithmetic.",
            memory=DummyMemory(),
            step_count=1,
            max_step_count=3,
            json_data={},
        )

        self.assertIn("plan the next calculator step", captured["prompt"])
        self.assertIn('"Sub_goal": "Calculate reading time per night"', captured["prompt"])
        self.assertIn('"Calculation": "2 / 2"', captured["prompt"])
        self.assertIn('"Sub_goal": briefly say what this calculation computes', captured["prompt"])
        self.assertIn('"Calculation": write only the arithmetic expression', captured["prompt"])
        self.assertIn("Use only numbers from the problem or previous results", captured["prompt"])
        self.assertIn("digits, +, -, *, /, %, parentheses", captured["prompt"])
        self.assertIn('Do not include words, units, "=", or the result in calculation', captured["prompt"])
        self.assertIn("Judge", captured["prompt"])
        self.assertNotIn("Verifier_feedback", captured["prompt"])
        self.assertNotIn('"tool_name": "Calculator_Tool"', captured["prompt"])
        self.assertNotIn("<raw expression>", captured["prompt"])
        self.assertNotIn("<expression>", captured["prompt"])
        self.assertNotIn("<Calculator>", captured["prompt"])

    def test_generator_prompt_requires_generator_numeric_line(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.planner import Planner

        captured = {}
        planner = Planner.__new__(Planner)
        planner.is_multimodal = False
        planner.query_analysis = "Need arithmetic."
        planner.available_tools = ["Calculator_Tool"]

        def fake_llm(input_data):
            captured["prompt"] = input_data[0]
            return "Generator:\n8"

        planner.llm_engine_fixed = fake_llm
        planner.generate_direct_output(
            question="Problem text",
            image=None,
            memory=DummyMemory(),
        )

        self.assertIn("Return the final numeric answer only", captured["prompt"])
        self.assertIn("Memory Actions:", captured["prompt"])
        self.assertIn("command", captured["prompt"])
        self.assertIn("result", captured["prompt"])
        self.assertIn("Actions may be unreliable", captured["prompt"])
        self.assertIn("Check whether the commands cover every required quantity to solve the problem", captured["prompt"])
        self.assertIn("If Memory Actions are incomplete or inconsistent, compute the final answer from the problem and query_analysis only", captured["prompt"])
        self.assertIn("Output one number only", captured["prompt"])
        self.assertNotIn("<Calculator>", captured["prompt"])

    def test_verifier_prompt_is_calculator_specific(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.formatters import MemoryVerification
        from agentflow.models.verifier import Verifier

        captured = {}
        verifier = Verifier.__new__(Verifier)
        verifier.is_multimodal = False
        verifier.available_tools = ["Calculator_Tool"]
        verifier.toolbox_metadata = {"Calculator_Tool": {"input_types": {"expression": "str"}}}

        def fake_llm(input_data, response_format=None):
            captured["prompt"] = input_data[0]
            return MemoryVerification(analysis="Enough arithmetic.", stop_signal=True)

        verifier.llm_engine_fixed = fake_llm
        verifier.verificate_context(
            question="Problem text",
            image=None,
            query_analysis="Need arithmetic.",
            memory=DummyMemory(),
            step_count=1,
            json_data={},
        )

        self.assertIn("Decide whether memory has enough proof to solve the entire problem", captured["prompt"])
        self.assertIn("Memory:", captured["prompt"])
        self.assertIn("command", captured["prompt"])
        self.assertIn("result", captured["prompt"])
        self.assertIn("Conclusion: STOP", captured["prompt"])
        self.assertIn("Conclusion: CONTINUE", captured["prompt"])
        self.assertIn("Response Format", captured["prompt"])
        self.assertIn("When memory not solves the entire problem", captured["prompt"])
        self.assertIn("When memory solves the entire problem", captured["prompt"])
        self.assertIn("First line must be Conclusion", captured["prompt"])
        self.assertIn("Do not write another Conclusion later", captured["prompt"])
        self.assertNotIn("Planner output was invalid", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
