import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import subagent_model_config


def bootstrap_agentflow_runtime() -> None:
    agentflow_core = Path(__file__).resolve().parents[2] / "agentflow" / "agentflow"
    agentflow_pkg = types.ModuleType("agentflow")
    agentflow_pkg.__path__ = [str(agentflow_core)]
    agentflow_pkg.__file__ = str(agentflow_core / "__init__.py")
    sys.modules["agentflow"] = agentflow_pkg


class TestSubagentModelConfig(unittest.TestCase):
    def test_default_config_uses_baseline_settings_for_all_subagents(self):
        config = subagent_model_config.default_subagent_config()
        expected = subagent_model_config.BASELINE_GENERATION_CONFIG

        self.assertEqual(config["query_analysis"], expected)
        self.assertEqual(config["planner_next_step"], expected)
        self.assertEqual(config["executor"], expected)
        self.assertEqual(config["verifier"], expected)
        self.assertEqual(config["generator"], expected)

    def test_planner_uses_subagent_configs_for_calculator_calls(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.formatters import NextStep
        from agentflow.models.planner import Planner

        captured = {}
        planner = Planner.__new__(Planner)
        planner.is_multimodal = False
        planner.available_tools = ["Calculator_Tool"]
        planner.toolbox_metadata = {"Calculator_Tool": {}}
        planner.generation_configs = subagent_model_config.default_subagent_config()

        def fake_fixed_llm(input_data, **kwargs):
            captured["query_analysis"] = kwargs
            return "analysis"

        def fake_main_llm(prompt, **kwargs):
            captured["planner_next_step"] = kwargs
            return NextStep(sub_goal="Calculate", calculation="1+1")

        planner.llm_engine_fixed = fake_fixed_llm
        planner.llm_engine = fake_main_llm
        planner.analyze_query("What is 1+1?", None)
        planner.generate_next_step(
            "What is 1+1?",
            None,
            "analysis",
            types.SimpleNamespace(get_actions=lambda: {}),
            1,
            1,
            {},
        )

        expected = subagent_model_config.BASELINE_GENERATION_CONFIG
        for key in ["system_prompt", "max_tokens", "temperature", "top_p", "frequency_penalty"]:
            self.assertEqual(captured["query_analysis"][key], expected[key])
            self.assertEqual(captured["planner_next_step"][key], expected[key])

    def test_apply_to_solver_sets_configs_on_all_components(self):
        solver = types.SimpleNamespace(
            planner=types.SimpleNamespace(),
            executor=types.SimpleNamespace(),
            verifier=types.SimpleNamespace(),
        )
        config = subagent_model_config.default_subagent_config()

        subagent_model_config.apply_subagent_config(solver, config)

        self.assertEqual(solver.planner.generation_configs["query_analysis"], config["query_analysis"])
        self.assertEqual(solver.planner.generation_configs["planner_next_step"], config["planner_next_step"])
        self.assertEqual(solver.planner.generation_configs["generator"], config["generator"])
        self.assertEqual(solver.executor.generation_configs["executor"], config["executor"])
        self.assertEqual(solver.verifier.generation_configs["verifier"], config["verifier"])


if __name__ == "__main__":
    unittest.main()
