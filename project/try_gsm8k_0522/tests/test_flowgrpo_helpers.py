import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowgrpo.data import build_parquet_rows
from flowgrpo.data import resolve_cli_path
from flowgrpo.reward import compute_gsm8k_reward, extract_rollout_answer
from flowgrpo.solver_factory import mark_planner_next_step_as_trainable


class TestFlowGRPOHelpers(unittest.TestCase):
    def test_resolve_cli_path_uses_current_working_directory_for_relative_paths(self):
        resolved = resolve_cli_path(Path("data/gsm8k_smoke_50.json"))

        self.assertEqual(resolved, (Path.cwd() / "data/gsm8k_smoke_50.json").resolve())

    def test_build_parquet_rows_uses_question_and_numeric_result(self):
        rows = [
            {
                "pid": 7,
                "question": "What is 1+1?",
                "answer": "Add them. #### 2",
                "gold_answer": "2",
            }
        ]

        converted = build_parquet_rows(rows)

        self.assertEqual(
            converted,
            [
                {
                    "id": "gsm8k-7",
                    "question": "What is 1+1?",
                    "result": "2",
                    "extra_info": {
                        "idx": 7,
                        "source": "gsm8k",
                        "answer": "Add them. #### 2",
                        "gold_answer": "2",
                    },
                }
            ],
        )

    def test_rule_reward_accepts_equivalent_numeric_answers(self):
        self.assertEqual(compute_gsm8k_reward("The answer is <answer>0.5</answer>", "1/2"), 1.0)
        self.assertEqual(compute_gsm8k_reward("Final: 2.00", "2"), 1.0)
        self.assertEqual(compute_gsm8k_reward("Final: 3", "2"), 0.0)

    def test_extract_rollout_answer_prefers_direct_output(self):
        result = {
            "final_output": "999",
            "direct_output": "After checking, <answer>42</answer>",
        }

        self.assertEqual(extract_rollout_answer(result), "42")

    def test_mark_planner_next_step_wraps_only_next_step_method(self):
        calls = []

        class FakePlanner:
            def generate_next_step(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "next"

            def analyze_query(self):
                return "analysis"

        solver = types.SimpleNamespace(planner=FakePlanner())

        mark_planner_next_step_as_trainable(solver)
        first_wrapper = solver.planner.generate_next_step
        self.assertTrue(getattr(first_wrapper, "_flowgrpo_planner_next_step", False))
        self.assertEqual(solver.planner.generate_next_step("q", step=1), "next")

        mark_planner_next_step_as_trainable(solver)
        self.assertIs(solver.planner.generate_next_step, first_wrapper)
        self.assertEqual(calls, [(("q",), {"step": 1})])


if __name__ == "__main__":
    unittest.main()
