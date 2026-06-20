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


class TestCalculatorTool(unittest.TestCase):
    def test_calculator_handles_basic_arithmetic_symbols(self):
        bootstrap_agentflow_runtime()
        from agentflow.tools.calculator.tool import Calculator_Tool

        tool = Calculator_Tool()

        self.assertEqual(tool.output_type, "str - The numeric result of the evaluated expression.")
        self.assertEqual(tool.execute(expression="4 × 2"), "8")
        self.assertEqual(tool.execute(expression="96 ÷ 16"), "6")
        self.assertEqual(tool.execute(query="3.5 + 2"), "5.5")
        self.assertEqual(tool.execute(expression="+1+1"), "2")
        self.assertEqual(tool.execute(expression="*12*2"), "24")
        self.assertEqual(tool.execute(expression="* (12 * 2)"), "24")
        self.assertEqual(
            tool.execute(expression="+ (2 * 3) * (3 + 15) / ((7 - 5) / (10 - 8)) + (-4)"),
            "104",
        )

    def test_calculator_rejects_non_arithmetic_expressions(self):
        bootstrap_agentflow_runtime()
        from agentflow.tools.calculator.tool import Calculator_Tool

        tool = Calculator_Tool()

        self.assertIn("Error:", tool.execute(expression="2 ** 3"))
        self.assertIn("Error:", tool.execute(expression="__import__('os').system('pwd')"))

    def test_calculator_reports_original_expression_for_syntax_errors(self):
        bootstrap_agentflow_runtime()
        from agentflow.tools.calculator.tool import Calculator_Tool

        tool = Calculator_Tool()

        self.assertEqual(
            tool.execute(expression="12*/2"),
            'invalid syntax: expression="12*/2"',
        )

    def test_calculator_handles_percentages(self):
        bootstrap_agentflow_runtime()
        from agentflow.tools.calculator.tool import Calculator_Tool

        tool = Calculator_Tool()

        self.assertIn("%", tool.input_types["expression"])
        self.assertNotIn('special symbols such as "%" are not allowed', tool.user_metadata["limitations"])
        self.assertEqual(tool.execute(expression="20%"), "0.2")
        self.assertEqual(tool.execute(expression="50 * 20%"), "10")
        self.assertEqual(tool.execute(expression="50 + 50 * 20%"), "60")
        self.assertEqual(tool.execute(expression="200 * (1 + 15%)"), "230")


if __name__ == "__main__":
    unittest.main()
