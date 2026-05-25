import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestDirectBaseline(unittest.TestCase):
    def test_build_prompt_requires_answer_tag_without_agentflow_terms(self):
        from direct_baseline.run_direct_vllm import build_prompt

        prompt = build_prompt("What is 1 + 1?")

        self.assertIn("Problem:", prompt)
        self.assertIn("<answer>", prompt)
        self.assertIn("</answer>", prompt)
        self.assertIn("No text after </answer>", prompt)
        self.assertNotIn("AgentFlow", prompt)
        self.assertNotIn("Calculator_Tool", prompt)

    def test_direct_runner_defaults_to_local_qwen3_model(self):
        from direct_baseline.run_direct_vllm import parse_args

        args = parse_args(
            [
                "--data-file",
                "data/gsm8k_smoke_50.json",
                "--output-dir",
                "direct_baseline/results/test",
            ]
        )

        self.assertEqual(args.model, "Qwen3-0.6B-Instruct")
        self.assertEqual(args.base_url, "http://localhost:8000/v1")
        self.assertEqual(args.max_tokens, 2048)
        self.assertEqual(args.temperature, 0.0)

    def test_direct_smoke_script_uses_parent_eval_flow(self):
        script = (ROOT / "direct_baseline" / "run_smoke_direct.sh").read_text(encoding="utf-8")

        self.assertIn("prepare_gsm8k_json.py", script)
        self.assertIn("direct_baseline/run_direct_vllm.py", script)
        self.assertIn("score_gsm8k.py", script)
        self.assertIn("--response-field direct_output", script)
        self.assertIn("direct_baseline/results/smoke_direct_vllm", script)
        self.assertIn("direct_baseline/summary/smoke_direct_vllm_summary.json", script)


if __name__ == "__main__":
    unittest.main()
