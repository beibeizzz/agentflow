from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from try_gsm8k_0522.flowgrpo_light_40g.eval_light_grpo import parse_args, summarize_rewards


class FlowGrpoLight40GEvalTests(unittest.TestCase):
    def test_summarize_rewards_counts_accuracy(self) -> None:
        summary = summarize_rewards([1.0, 0.0, 1.0])

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["correct"], 2)
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)

    def test_summarize_rewards_handles_empty_input(self) -> None:
        summary = summarize_rewards([])

        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["correct"], 0)
        self.assertEqual(summary["accuracy"], 0.0)

    def test_parse_args_accepts_agentflow_rollout_backend(self) -> None:
        args = parse_args(["--rollout-backend", "agentflow", "--think-mode", "off"])

        self.assertEqual(args.rollout_backend, "agentflow")
        self.assertEqual(args.think_mode, "off")


if __name__ == "__main__":
    unittest.main()
