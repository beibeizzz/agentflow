import sys
import types
import unittest
import shutil
import json
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import prepare_gsm8k_json
import run_gsm8k_agentflow


class TestGSM8KScripts(unittest.TestCase):
    def test_problem_prompt_is_not_a_role_prompt_or_tool_result_template(self):
        prompt = prepare_gsm8k_json.PROMPT_TEMPLATE.format(question="What is 1+1?")

        self.assertIn("focus on your responsibility", prompt)
        self.assertIn("Problem:", prompt)
        self.assertNotIn("<Calculator> expression = result <Calculator>", prompt)
        self.assertNotIn("Executor:", prompt)
        self.assertNotIn("\\boxed{ANSWER}", prompt)

    def test_runner_exposes_output_types_and_max_tokens(self):
        args = run_gsm8k_agentflow.parse_args(
            [
                "--data-file",
                "data/gsm8k_smoke_20.json",
                "--output-dir",
                "results/test",
                "--output-types",
                "base",
                "--max-tokens",
                "2048",
                "--think-mode",
                "off",
                "--query-analysis-think-mode",
                "on",
                "--final-output-think-mode",
                "off",
                "--verifier-think-mode",
                "default",
            ]
        )

        self.assertEqual(args.output_types, "base")
        self.assertEqual(args.max_tokens, 2048)
        self.assertEqual(args.think_mode, "off")
        self.assertEqual(args.query_analysis_think_mode, "on")
        self.assertEqual(args.final_output_think_mode, "off")
        self.assertEqual(args.verifier_think_mode, "default")

    def test_smoke_script_runs_single_minimal_agentflow_variant(self):
        script = (ROOT / "run_smoke.sh").read_text(encoding="utf-8")

        self.assertIn('run_variant "smoke_calculator_steps4" "direct" "direct_output" 4', script)
        self.assertIn('"$PYTHON" run_gsm8k_agentflow.py', script)
        self.assertIn('"$PYTHON" score_gsm8k.py', script)
        self.assertIn("${MAX_TOKENS:-512}", script)
        self.assertNotIn('run_variant "smoke_calculator_steps1"', script)
        self.assertNotIn('run_variant "smoke_calculator_steps10"', script)
        self.assertNotIn("smoke_base_hash", script)

    def test_full_script_removed_in_favor_of_baseline_script(self):
        self.assertFalse((ROOT / "run_full.sh").exists())
        self.assertTrue((ROOT / "baseline.sh").exists())
        self.assertTrue((ROOT / "run_gsm8k_agentflow.py").exists())

    def test_runner_constructs_solver_with_only_calculator_tool(self):
        captured = {}

        def fake_build_solver(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                planner=types.SimpleNamespace(),
                executor=types.SimpleNamespace(),
                verifier=types.SimpleNamespace(),
            )

        fake_agentflow = types.ModuleType("agentflow")
        fake_solver_module = types.ModuleType("agentflow.solver")
        fake_solver_module.construct_solver = fake_build_solver

        original_agentflow = sys.modules.get("agentflow")
        original_solver = sys.modules.get("agentflow.solver")
        original_bootstrap = run_gsm8k_agentflow.bootstrap_agentflow_runtime
        try:
            sys.modules["agentflow"] = fake_agentflow
            sys.modules["agentflow.solver"] = fake_solver_module
            run_gsm8k_agentflow.bootstrap_agentflow_runtime = lambda: None

            run_gsm8k_agentflow.construct_solver(
                llm_engine_name="vllm-Qwen3-0.6B-Instruct",
                base_url="http://localhost:8000/v1",
                output_types="direct",
                max_steps=1,
                max_time=120,
                max_tokens=2048,
                temperature=0.0,
                subagent_config_path=None,
                think_mode="off",
                query_analysis_think_mode="on",
                final_output_think_mode="off",
                verifier_think_mode="default",
            )
        finally:
            run_gsm8k_agentflow.bootstrap_agentflow_runtime = original_bootstrap
            if original_agentflow is None:
                sys.modules.pop("agentflow", None)
            else:
                sys.modules["agentflow"] = original_agentflow
            if original_solver is None:
                sys.modules.pop("agentflow.solver", None)
            else:
                sys.modules["agentflow.solver"] = original_solver

        self.assertEqual(captured["enabled_tools"], ["Calculator_Tool"])
        self.assertEqual(captured["tool_engine"], ["Default"])
        self.assertEqual(captured["think_mode"], "off")
        self.assertEqual(captured["query_analysis_think_mode"], "on")
        self.assertEqual(captured["final_output_think_mode"], "off")
        self.assertEqual(captured["verifier_think_mode"], "default")

    def test_runner_passes_raw_question_to_solver_not_wrapped_query(self):
        captured = {}

        class FakeSolver:
            def __init__(self):
                self.planner = types.SimpleNamespace(available_tools=["Calculator_Tool"])

            def solve(self, question):
                captured["solved_question"] = question
                return {"direct_output": "2"}

        row = {
            "pid": 1,
            "question": "What is 1+1?",
            "query": "\nYou should focus on your responsibility mentioned.\n\nProblem:\nWhat is 1+1?",
            "answer": "#### 2",
            "gold_answer": "2",
        }

        original_load_data = run_gsm8k_agentflow.load_data
        original_check_vllm_server = run_gsm8k_agentflow.check_vllm_server
        original_construct_solver = run_gsm8k_agentflow.construct_solver
        original_reset_solver_memory = run_gsm8k_agentflow.reset_solver_memory
        try:
            run_gsm8k_agentflow.load_data = lambda path: [row]
            run_gsm8k_agentflow.check_vllm_server = lambda base_url, model_name: None
            run_gsm8k_agentflow.construct_solver = lambda **kwargs: FakeSolver()
            run_gsm8k_agentflow.reset_solver_memory = lambda solver: None
            output_dir = ROOT / ".tmp_test_outputs"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
            try:
                args = Namespace(
                    data_file=Path("unused.json"),
                    output_dir=output_dir,
                    solver_log_dir=None,
                    llm_engine_name="vllm-Qwen3-0.6B-Instruct",
                    base_url="http://localhost:8000/v1",
                    output_types="direct",
                    start=0,
                    limit=None,
                    max_steps=1,
                    max_time=120,
                    max_tokens=2048,
                    temperature=0.0,
                    subagent_config=Path("subagent_model_config.json"),
                    think_mode="off",
                    query_analysis_think_mode="on",
                    final_output_think_mode="off",
                    verifier_think_mode="default",
                    overwrite=True,
                    stop_on_error=False,
                )

                run_gsm8k_agentflow.run_examples(args)

                payload = json.loads((output_dir / "output_1.json").read_text(encoding="utf-8"))
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)

        finally:
            run_gsm8k_agentflow.load_data = original_load_data
            run_gsm8k_agentflow.check_vllm_server = original_check_vllm_server
            run_gsm8k_agentflow.construct_solver = original_construct_solver
            run_gsm8k_agentflow.reset_solver_memory = original_reset_solver_memory

        self.assertEqual(captured["solved_question"], "What is 1+1?")
        self.assertEqual(payload["query"], "What is 1+1?")
        self.assertEqual(payload["original_query"], row["query"])
        self.assertEqual(payload["think_mode"], "off")
        self.assertEqual(payload["query_analysis_think_mode"], "on")
        self.assertEqual(payload["final_output_think_mode"], "off")
        self.assertEqual(payload["verifier_think_mode"], "default")

    def test_vllm_server_check_uses_no_proxy_opener(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data":[{"id":"Qwen3-0.6B-Instruct"}]}'

        class FakeOpener:
            def open(self, url, timeout):
                captured["url"] = url
                captured["timeout"] = timeout
                return FakeResponse()

        def fake_build_opener(*handlers):
            captured["handlers"] = handlers
            return FakeOpener()

        def forbidden_urlopen(*args, **kwargs):
            raise AssertionError("check_vllm_server should bypass proxy env with build_opener")

        original_build_opener = run_gsm8k_agentflow.urllib.request.build_opener
        original_urlopen = run_gsm8k_agentflow.urllib.request.urlopen
        try:
            run_gsm8k_agentflow.urllib.request.build_opener = fake_build_opener
            run_gsm8k_agentflow.urllib.request.urlopen = forbidden_urlopen
            run_gsm8k_agentflow.check_vllm_server(
                "http://localhost:8000/v1",
                "Qwen3-0.6B-Instruct",
            )
        finally:
            run_gsm8k_agentflow.urllib.request.build_opener = original_build_opener
            run_gsm8k_agentflow.urllib.request.urlopen = original_urlopen

        self.assertEqual(captured["url"], "http://localhost:8000/v1/models")
        self.assertEqual(captured["timeout"], 10)
        self.assertTrue(any(isinstance(handler, run_gsm8k_agentflow.urllib.request.ProxyHandler) for handler in captured["handlers"]))


if __name__ == "__main__":
    unittest.main()
