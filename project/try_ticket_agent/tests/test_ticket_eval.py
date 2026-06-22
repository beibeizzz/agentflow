from __future__ import annotations

import unittest

from try_ticket_agent.flowgrpo_general_2x40g.eval_ticket_agent import summarize_results


class TicketEvalTests(unittest.TestCase):
    def test_summary_separates_model_and_infrastructure_failures(self) -> None:
        results = [
            {
                "reward": 1.0,
                "valid_for_training": True,
                "curriculum_mode": "direct",
                "verification": {"success": True, "failure_codes": [], "invalid_action_count": 0},
            },
            {
                "reward": 0.0,
                "valid_for_training": True,
                "curriculum_mode": "indirect",
                "verification": {
                    "success": False,
                    "failure_codes": ["INVALID_ACTION"],
                    "invalid_action_count": 1,
                },
            },
            {
                "reward": 0.0,
                "valid_for_training": False,
                "curriculum_mode": "direct",
                "verification": None,
                "errors": ["RuntimeError: timeout"],
            },
        ]
        summary = summarize_results(results)
        self.assertEqual(summary["episode_success_rate"], 1 / 3)
        self.assertEqual(summary["invalid_action_rate"], 1 / 3)
        self.assertEqual(summary["infrastructure_failure_rate"], 1 / 3)
        self.assertEqual(summary["direct_success_rate"], 1 / 2)
        self.assertEqual(summary["indirect_success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
