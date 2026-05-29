import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowgrpo_light.calculator import format_number, safe_eval_calculation
from flowgrpo_light.parsing import parse_planner_response
from flowgrpo_light.prompts import build_planner_prompt


class TestFlowGRPOLight(unittest.TestCase):
    def test_safe_eval_calculation_accepts_basic_arithmetic(self):
        self.assertEqual(format_number(safe_eval_calculation("(6 * 5) + 12")), "42")
        self.assertEqual(format_number(safe_eval_calculation("1 / 2")), "0.5")

    def test_safe_eval_rejects_non_arithmetic(self):
        with self.assertRaises(ValueError):
            safe_eval_calculation("__import__('os').system('echo bad')")

    def test_parse_planner_response_extracts_json(self):
        sub_goal, calculation = parse_planner_response(
            '```json\n{"Sub_goal": "Compute brother grapes", "Calculation": "6 * 5"}\n```'
        )

        self.assertEqual(sub_goal, "Compute brother grapes")
        self.assertEqual(calculation, "6 * 5")

    def test_planner_prompt_contains_training_contract(self):
        prompt = build_planner_prompt(
            question="What is 1+1?",
            query_analysis="Add the quantities.",
            memory={},
            step_count=1,
            max_steps=3,
        )

        self.assertIn("Return only one JSON object", prompt)
        self.assertIn('"Calculation"', prompt)
        self.assertIn("Memory: {}", prompt)


if __name__ == "__main__":
    unittest.main()
