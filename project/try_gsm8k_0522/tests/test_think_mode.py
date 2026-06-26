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


class TestThinkMode(unittest.TestCase):
    def test_vllm_disables_qwen_thinking_with_extra_body(self):
        bootstrap_agentflow_runtime()
        from agentflow.engine.vllm import ChatVLLM

        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content="ok"),
                        )
                    ]
                )

        engine = ChatVLLM.__new__(ChatVLLM)
        engine.model_string = "Qwen3-0.6B"
        engine.system_prompt = "system"
        engine.use_cache = False
        engine.think_mode = "off"
        engine.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )

        self.assertEqual(engine._generate_text("prompt"), "ok")
        self.assertEqual(
            captured["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_vllm_enables_qwen_thinking_with_extra_body(self):
        bootstrap_agentflow_runtime()
        from agentflow.engine.vllm import ChatVLLM

        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content="ok"),
                        )
                    ]
                )

        engine = ChatVLLM.__new__(ChatVLLM)
        engine.model_string = "Qwen3-0.6B"
        engine.system_prompt = "system"
        engine.use_cache = False
        engine.think_mode = "on"
        engine.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )

        self.assertEqual(engine._generate_text("prompt"), "ok")
        self.assertEqual(
            captured["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True}},
        )

    def test_vllm_call_time_think_mode_overrides_engine_default(self):
        bootstrap_agentflow_runtime()
        from agentflow.engine.vllm import ChatVLLM

        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content="ok"),
                        )
                    ]
                )

        engine = ChatVLLM.__new__(ChatVLLM)
        engine.model_string = "Qwen3-0.6B"
        engine.system_prompt = "system"
        engine.use_cache = False
        engine.think_mode = "off"
        engine.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )

        self.assertEqual(engine._generate_text("prompt", think_mode="on"), "ok")
        self.assertEqual(
            captured["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True}},
        )

    def test_stage_specific_think_modes_are_passed_to_llm_calls(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.formatters import MemoryVerification
        from agentflow.models.memory import Memory
        from agentflow.models.planner import Planner
        from agentflow.models.verifier import Verifier

        memory = Memory()
        memory.add_action(1, "Calculator_Tool", "Calculate", 'execution = tool.execute(expression="1+1")', "2")

        planner_calls = []
        planner = Planner.__new__(Planner)
        planner.is_multimodal = False
        planner.available_tools = ["Calculator_Tool"]
        planner.toolbox_metadata = {}
        planner.generation_configs = {}
        planner.query_analysis_think_mode = "off"
        planner.final_output_think_mode = "on"
        planner.query_analysis = "analysis"
        planner.llm_engine_fixed = lambda input_data, **kwargs: planner_calls.append(kwargs) or "2"

        verifier_calls = []
        verifier = Verifier.__new__(Verifier)
        verifier.is_multimodal = False
        verifier.available_tools = ["Calculator_Tool"]
        verifier.toolbox_metadata = {}
        verifier.generation_configs = {}
        verifier.verifier_think_mode = "off"
        verifier.llm_engine_fixed = (
            lambda input_data, **kwargs: verifier_calls.append(kwargs)
            or MemoryVerification(analysis="complete", stop_signal=True)
        )

        planner.analyze_query("What is 1+1?", None)
        planner.generate_direct_output("What is 1+1?", None, memory)
        verifier.verificate_context("What is 1+1?", None, "analysis", memory, 1)

        self.assertEqual(planner_calls[0]["think_mode"], "off")
        self.assertEqual(planner_calls[1]["think_mode"], "on")
        self.assertEqual(verifier_calls[0]["think_mode"], "off")

    def test_smoke_script_exposes_stage_specific_think_env_vars(self):
        script = (Path(__file__).resolve().parents[1] / "run_smoke.sh").read_text(encoding="utf-8")

        self.assertIn('QUERY_ANALYSIS_THINK_MODE="${QUERY_ANALYSIS_THINK_MODE:-on}"', script)
        self.assertIn('FINAL_OUTPUT_THINK_MODE="${FINAL_OUTPUT_THINK_MODE:-off}"', script)
        self.assertIn('VERIFIER_THINK_MODE="${VERIFIER_THINK_MODE:-on}"', script)
        self.assertIn('--query-analysis-think-mode "$QUERY_ANALYSIS_THINK_MODE"', script)
        self.assertIn('--final-output-think-mode "$FINAL_OUTPUT_THINK_MODE"', script)
        self.assertIn('--verifier-think-mode "$VERIFIER_THINK_MODE"', script)

    def test_calculator_prompts_omit_no_think_when_thinking_is_forced_on(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.formatters import MemoryVerification, NextStep, ToolCommand
        from agentflow.models.memory import Memory
        from agentflow.models.planner import Planner
        from agentflow.models.executor import Executor
        from agentflow.models.verifier import Verifier

        memory = Memory()
        memory.add_action(1, "Calculator_Tool", "Calculate", 'execution = tool.execute(expression="1+1")', "2")

        planner_captured = {}
        planner = Planner.__new__(Planner)
        planner.is_multimodal = False
        planner.available_tools = ["Calculator_Tool"]
        planner.toolbox_metadata = {}
        planner.generation_configs = {}
        planner.think_mode = "on"
        planner.llm_engine = lambda prompt, **kwargs: planner_captured.setdefault("prompt", prompt) or NextStep(
            sub_goal="Calculate", calculation="1+1"
        )

        executor_captured = {}
        executor = Executor.__new__(Executor)
        executor.generation_configs = {}
        executor.think_mode = "on"
        executor.llm_generate_tool_command = (
            lambda prompt, **kwargs: executor_captured.setdefault("prompt", prompt)
            or ToolCommand(analysis="", explanation="", command='execution = tool.execute(expression="1+1")')
        )

        verifier_captured = {}
        verifier = Verifier.__new__(Verifier)
        verifier.is_multimodal = False
        verifier.available_tools = ["Calculator_Tool"]
        verifier.toolbox_metadata = {}
        verifier.generation_configs = {}
        verifier.think_mode = "on"
        verifier.llm_engine_fixed = (
            lambda input_data, **kwargs: verifier_captured.setdefault("prompt", input_data[0])
            or MemoryVerification(analysis="complete", stop_signal=True)
        )

        planner.generate_next_step("What is 1+1?", None, "analysis", memory, 1, 2)
        executor.generate_tool_command("What is 1+1?", None, "1+1", "Calculate", "Calculator_Tool", {}, 1)
        verifier.verificate_context("What is 1+1?", None, "analysis", memory, 1)

        self.assertNotIn("/no_think", planner_captured["prompt"])
        self.assertNotIn("/no_think", executor_captured["prompt"])
        self.assertNotIn("/no_think", verifier_captured["prompt"])

    def test_calculator_prompts_include_no_think_when_thinking_is_forced_off(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.formatters import MemoryVerification, NextStep, ToolCommand
        from agentflow.models.memory import Memory
        from agentflow.models.planner import Planner
        from agentflow.models.executor import Executor
        from agentflow.models.verifier import Verifier

        memory = Memory()
        memory.add_action(1, "Calculator_Tool", "Calculate", 'execution = tool.execute(expression="1+1")', "2")

        planner_captured = {}
        planner = Planner.__new__(Planner)
        planner.is_multimodal = False
        planner.available_tools = ["Calculator_Tool"]
        planner.toolbox_metadata = {}
        planner.generation_configs = {}
        planner.think_mode = "off"
        planner.llm_engine = lambda prompt, **kwargs: planner_captured.setdefault("prompt", prompt) or NextStep(
            sub_goal="Calculate", calculation="1+1"
        )

        executor_captured = {}
        executor = Executor.__new__(Executor)
        executor.generation_configs = {}
        executor.think_mode = "off"
        executor.llm_generate_tool_command = (
            lambda prompt, **kwargs: executor_captured.setdefault("prompt", prompt)
            or ToolCommand(analysis="", explanation="", command='execution = tool.execute(expression="1+1")')
        )

        verifier_captured = {}
        verifier = Verifier.__new__(Verifier)
        verifier.is_multimodal = False
        verifier.available_tools = ["Calculator_Tool"]
        verifier.toolbox_metadata = {}
        verifier.generation_configs = {}
        verifier.think_mode = "off"
        verifier.llm_engine_fixed = (
            lambda input_data, **kwargs: verifier_captured.setdefault("prompt", input_data[0])
            or MemoryVerification(analysis="complete", stop_signal=True)
        )

        planner.generate_next_step("What is 1+1?", None, "analysis", memory, 1, 2)
        executor.generate_tool_command("What is 1+1?", None, "1+1", "Calculate", "Calculator_Tool", {}, 1)
        verifier.verificate_context("What is 1+1?", None, "analysis", memory, 1)

        self.assertIn("/no_think", planner_captured["prompt"])
        self.assertIn("/no_think", executor_captured["prompt"])
        self.assertIn("/no_think", verifier_captured["prompt"])


if __name__ == "__main__":
    unittest.main()
