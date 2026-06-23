from __future__ import annotations

from pathlib import Path
import unittest

from try_ticket_agent.flowgrpo_general_2x40g.eval_ticket_agent import resolve_output_dir, summarize_results


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

    def test_resolve_output_dir_separates_baseline_and_adapter_defaults(self) -> None:
        config = {
            "output_dir": "try_ticket_agent/flowgrpo_general_2x40g/outputs/eval_adapter",
        }
        self.assertEqual(
            resolve_output_dir(config, mode="baseline", explicit_output_dir=None),
            Path("try_ticket_agent/flowgrpo_general_2x40g/outputs/eval_baseline"),
        )
        self.assertEqual(
            resolve_output_dir(config, mode="adapter", explicit_output_dir=None),
            Path("try_ticket_agent/flowgrpo_general_2x40g/outputs/eval_adapter"),
        )

    def test_resolve_output_dir_honors_explicit_override(self) -> None:
        config = {"output_dir": "configured/adapter"}
        self.assertEqual(
            resolve_output_dir(config, mode="baseline", explicit_output_dir=Path("manual/out")),
            Path("manual/out"),
        )


if __name__ == "__main__":
    unittest.main()
