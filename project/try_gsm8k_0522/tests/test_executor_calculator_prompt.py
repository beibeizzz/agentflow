import sys
import types
import unittest
import tempfile
from pathlib import Path


def bootstrap_agentflow_runtime() -> None:
    agentflow_core = Path(__file__).resolve().parents[2] / "agentflow" / "agentflow"
    agentflow_pkg = types.ModuleType("agentflow")
    agentflow_pkg.__path__ = [str(agentflow_core)]
    agentflow_pkg.__file__ = str(agentflow_core / "__init__.py")
    sys.modules["agentflow"] = agentflow_pkg


class TestExecutorCalculatorPrompt(unittest.TestCase):
    def test_calculator_prompt_requires_expression_only_tool_call(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.executor import Executor
        from agentflow.models.formatters import ToolCommand

        captured = {}
        executor = Executor.__new__(Executor)

        def fake_llm(prompt, response_format=None):
            captured["prompt"] = prompt
            return ToolCommand(
                analysis="",
                explanation="",
                command='execution = tool.execute(expression="4 * 2")',
            )

        executor.llm_generate_tool_command = fake_llm
        executor.generate_tool_command(
            question="Problem text",
            image=None,
            context="Numbers: 4 and 2",
            sub_goal="Calculate 4 times 2.",
            tool_name="Calculator_Tool",
            tool_metadata={"input_types": {"expression": "str"}},
            step_count=1,
            json_data={},
        )

        self.assertIn('execution = tool.execute(expression="<raw arithmetic expression>")', captured["prompt"])
        self.assertIn("Extract and copy the original arithmetic expression from Context", captured["prompt"])
        self.assertIn("Do not combine Context with other calculations", captured["prompt"])
        self.assertIn("Return only one Python code block", captured["prompt"])
        self.assertNotIn("<Calculator>", captured["prompt"])

    def test_execute_calculator_command_returns_numeric_result_only(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.executor import Executor
        from agentflow.tools.calculator.tool import Calculator_Tool

        executor = Executor.__new__(Executor)
        executor.max_time = 10
        executor.tool_instances_cache = {"Calculator_Tool": Calculator_Tool()}
        with tempfile.TemporaryDirectory() as tmpdir:
            executor.query_cache_dir = tmpdir
            result = executor.execute_tool_command(
                "Calculator_Tool",
                'execution = tool.execute(expression="4 * 2")',
            )

        self.assertEqual(result, ["8"])


if __name__ == "__main__":
    unittest.main()
