from __future__ import annotations

import inspect
import json
import unittest

import try_ticket_agent.tests  # Bootstrap inner AgentFlow when unittest uses discovery mode.

from agentflow.models.formatters import StructuredToolAction
from agentflow.models.executor import Executor
from agentflow.models.memory import Memory
from agentflow.models.planner import Planner
from agentflow.solver import Solver, construct_solver


class RecordingEngine:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, prompt: object, **kwargs: object) -> object:
        self.calls.append((prompt, kwargs))
        return self.response


class RecordingTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(self, **kwargs: object) -> dict[str, bool]:
        self.calls.append(dict(kwargs))
        return {"ok": True}


class FailIfCalled:
    def __call__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("structured Executor must not call an LLM")


class ScriptedPlanner:
    available_tools = ["Ticket_Finish_Tool"]
    toolbox_metadata = {"Ticket_Finish_Tool": {}}

    def __init__(self) -> None:
        self.final_calls = 0

    def analyze_query(self, question: str, image: object, json_data: object = None) -> str:
        return "analysis"

    def generate_next_step(self, *args: object, **kwargs: object) -> str:
        return '{"tool_name":"Ticket_Finish_Tool","arguments":{"ticket_id":"T-1","outcome":"completed"}}'

    def extract_context_subgoal_and_tool(self, response: object) -> tuple[str, str, str]:
        return '{"ticket_id":"T-1","outcome":"completed"}', "finish", "Ticket_Finish_Tool"

    def generate_final_output(self, *args: object, **kwargs: object) -> str:
        self.final_calls += 1
        return "unexpected"

    def generate_direct_output(self, *args: object, **kwargs: object) -> str:
        self.final_calls += 1
        return "unexpected"


class RecordingExecutor:
    def set_query_cache_dir(self, root: str) -> None:
        self.root = root

    def generate_tool_command(self, *args: object, **kwargs: object) -> dict[str, str]:
        return {"command": "{}"}

    def extract_explanation_and_command(self, response: object) -> tuple[str, str, str]:
        return "analysis", "explanation", "{}"

    def execute_tool_command(self, tool_name: str, command: str) -> dict[str, bool]:
        return {"ok": True}


class StopVerifier:
    def verificate_context(self, *args: object, **kwargs: object) -> str:
        return "stop"

    def extract_conclusion(self, response: object) -> tuple[str, str]:
        return "verified", "STOP"


def make_planner(*, action_mode: str) -> Planner:
    planner = Planner.__new__(Planner)
    planner.action_mode = action_mode
    planner.available_tools = ["Ticket_Update_Tool", "Ticket_Finish_Tool"]
    planner.toolbox_metadata = {
        "Ticket_Update_Tool": {"input_types": {"ticket_id": "str", "field": "str", "value": "str"}},
        "Ticket_Finish_Tool": {"input_types": {"ticket_id": "str", "outcome": "str"}},
    }
    planner.is_multimodal = False
    planner.think_mode = "off"
    planner.generation_configs = {}
    planner.llm_engine = RecordingEngine(
        StructuredToolAction(
            tool_name="Ticket_Finish_Tool",
            arguments={"ticket_id": "T-1", "outcome": "completed"},
        )
    )
    return planner


class StructuredPlannerTests(unittest.TestCase):
    def test_structured_action_forbids_extra_top_level_fields(self) -> None:
        with self.assertRaises(Exception):
            StructuredToolAction(
                tool_name="Ticket_Finish_Tool",
                arguments={"ticket_id": "T-1", "outcome": "completed"},
                extra="forbidden",
            )

    def test_structured_extraction_returns_canonical_arguments(self) -> None:
        planner = make_planner(action_mode="structured")
        result = planner.extract_context_subgoal_and_tool(
            StructuredToolAction(
                tool_name="Ticket_Update_Tool",
                arguments={"value": "urgent", "ticket_id": "T-1", "field": "priority"},
            )
        )
        self.assertEqual(
            result,
            (
                '{"field":"priority","ticket_id":"T-1","value":"urgent"}',
                "Execute Ticket_Update_Tool",
                "Ticket_Update_Tool",
            ),
        )

    def test_structured_extraction_rejects_wrappers_and_extra_keys(self) -> None:
        planner = make_planner(action_mode="structured")
        invalid = [
            '{"tool_name":"Ticket_Finish_Tool","arguments":{},"extra":1}',
            '[{"tool_name":"Ticket_Finish_Tool","arguments":{}}]',
        ]
        for response in invalid:
            with self.subTest(response=response):
                self.assertEqual(planner.extract_context_subgoal_and_tool(response), (None, None, None))

    def test_structured_extraction_accepts_fenced_json_from_small_model(self) -> None:
        planner = make_planner(action_mode="structured")
        result = planner.extract_context_subgoal_and_tool(
            '```json\n'
            '{"tool_name":"Ticket_Finish_Tool","arguments":{"ticket_id":"T-1","outcome":"completed"}}\n'
            '```'
        )
        self.assertEqual(
            result,
            (
                '{"outcome":"completed","ticket_id":"T-1"}',
                "Execute Ticket_Finish_Tool",
                "Ticket_Finish_Tool",
            ),
        )

    def test_non_calculator_query_analysis_uses_prompt_template_override(self) -> None:
        planner = make_planner(action_mode="structured")
        planner.llm_engine_fixed = RecordingEngine("ticket plan")
        planner.query_analysis_think_mode = "on"
        planner.generation_configs = {
            "query_analysis": {
                "prompt_template": "Plan ticket request: {question}\nTools: {available_tools}",
                "max_tokens": 192,
                "temperature": 0.0,
            }
        }
        json_data: dict[str, object] = {}

        result = planner.analyze_query("Update ticket T-1", None, json_data=json_data)

        self.assertEqual(result, "ticket plan")
        prompt, kwargs = planner.llm_engine_fixed.calls[0]
        self.assertIn("Plan ticket request: Update ticket T-1", str(prompt))
        self.assertIn("Ticket_Update_Tool", str(prompt))
        self.assertNotIn("GSM8K", str(prompt))
        self.assertNotIn("<answer>", str(prompt))
        self.assertEqual(kwargs["max_tokens"], 192)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(json_data["query_analysis_think_mode"], "on")

    def test_structured_generation_uses_action_schema(self) -> None:
        planner = make_planner(action_mode="structured")
        planner.generate_next_step("update T-1", None, "analysis", Memory(), 1, 3)
        prompt, kwargs = planner.llm_engine.calls[0]
        self.assertIn("Return exactly one JSON object", str(prompt))
        self.assertIs(kwargs["response_format"], StructuredToolAction)

    def test_structured_generation_uses_next_step_prompt_template_and_strips_analysis_think(self) -> None:
        planner = make_planner(action_mode="structured")
        planner.generation_configs = {
            "planner_next_step": {
                "prompt_template": "Ticket next step\nRequest: {question}\nAnalysis: {query_analysis}\nMemory: {memory_actions}",
                "max_tokens": 128,
                "temperature": 0.0,
            }
        }
        json_data: dict[str, object] = {}

        planner.generate_next_step(
            "Update T-1",
            None,
            "<think>long private reasoning</think>Short plan: update then finish.",
            Memory(),
            1,
            3,
            json_data=json_data,
        )

        prompt, kwargs = planner.llm_engine.calls[0]
        self.assertIn("Ticket next step", str(prompt))
        self.assertIn("Short plan: update then finish.", str(prompt))
        self.assertNotIn("long private reasoning", str(prompt))
        self.assertEqual(kwargs["max_tokens"], 128)
        self.assertEqual(kwargs["temperature"], 0.0)


    def test_planner_constructor_defaults_to_legacy_mode(self) -> None:
        signature = inspect.signature(Planner.__init__)
        self.assertEqual(signature.parameters["action_mode"].default, "legacy")


class StructuredExecutorTests(unittest.TestCase):
    def make_executor(self) -> tuple[Executor, RecordingTool]:
        tool = RecordingTool()
        executor = Executor.__new__(Executor)
        executor.execution_mode = "structured"
        executor.tool_instances_cache = {"Ticket_Update_Tool": tool}
        executor.llm_generate_tool_command = FailIfCalled()
        return executor, tool

    def test_structured_executor_calls_cached_tool_directly(self) -> None:
        executor, tool = self.make_executor()
        result = executor.execute_tool_command(
            "Ticket_Update_Tool",
            '{"field":"priority","ticket_id":"T-1","value":"urgent"}',
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(
            tool.calls,
            [{"field": "priority", "ticket_id": "T-1", "value": "urgent"}],
        )

    def test_structured_generate_command_does_not_call_llm(self) -> None:
        executor, _ = self.make_executor()
        command = executor.generate_tool_command(
            "request",
            None,
            '{"ticket_id":"T-1","outcome":"completed"}',
            "finish",
            "Ticket_Finish_Tool",
            {},
            1,
        )
        self.assertEqual(command.command, '{"ticket_id":"T-1","outcome":"completed"}')

    def test_structured_executor_rejects_invalid_arguments_without_tool_call(self) -> None:
        executor, tool = self.make_executor()
        result = executor.execute_tool_command("Ticket_Update_Tool", "[]")
        self.assertEqual(result["code"], "INVALID_TOOL_ARGUMENTS")
        self.assertEqual(tool.calls, [])

    def test_executor_constructor_defaults_to_legacy_mode(self) -> None:
        signature = inspect.signature(Executor.__init__)
        self.assertEqual(signature.parameters["execution_mode"].default, "legacy")


class WorkflowSolverTests(unittest.TestCase):
    def test_workflow_runs_actions_without_final_generators(self) -> None:
        planner = ScriptedPlanner()
        solver = Solver(
            planner,
            StopVerifier(),
            Memory(),
            RecordingExecutor(),
            output_types="workflow",
            max_steps=1,
            verbose=False,
        )
        result = solver.solve("update ticket")
        self.assertEqual(result["step_count"], 1)
        self.assertNotIn("direct_output", result)
        self.assertNotIn("final_output", result)
        self.assertEqual(planner.final_calls, 0)

    def test_construct_solver_defaults_to_legacy_modes(self) -> None:
        signature = inspect.signature(construct_solver)
        self.assertEqual(signature.parameters["planner_action_mode"].default, "legacy")
        self.assertEqual(signature.parameters["executor_mode"].default, "legacy")


if __name__ == "__main__":
    unittest.main()
