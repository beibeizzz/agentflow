import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowgrpo.reward import compute_gsm8k_reward, extract_rollout_answer


class TestFlowGRPOReward(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
