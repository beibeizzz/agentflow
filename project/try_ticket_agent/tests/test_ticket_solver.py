from __future__ import annotations

import json
import unittest

from agentflow.models.executor import Executor
from agentflow.models.memory import Memory
from agentflow.models.planner import Planner
from agentflow.solver import Solver
from agentflow.tools.ticket_finish.tool import Ticket_Finish_Tool
from agentflow.tools.ticket_query.tool import Ticket_Query_Tool
from agentflow.tools.ticket_update.tool import Ticket_Update_Tool

from try_ticket_agent.ticket_env.solver_factory import construct_ticket_runtime


TOOLS = ["Ticket_Query_Tool", "Ticket_Update_Tool", "Ticket_Finish_Tool"]


def action(tool_name: str, **arguments: str) -> str:
    return json.dumps({"tool_name": tool_name, "arguments": arguments}, separators=(",", ":"))


def episode(*, indirect: bool = False) -> dict[str, object]:
    request = (
        "For customer C-1, set the matching ticket priority to urgent and complete the request."
        if indirect
        else "Set ticket T-1 priority to urgent and complete the request."
    )
    return {
        "episode_id": "E-1",
        "user_request": request,
        "lookup_mode": "customer_id" if indirect else "ticket_id",
        "max_steps": 3,
        "initial_state": {
            "tickets": [
                {
                    "ticket_id": "T-1", "customer_id": "C-1", "order_id": "O-1",
                    "subject": "Payment review", "status": "open",
                    "assigned_team": "support", "priority": "normal",
                },
                {
                    "ticket_id": "T-2", "customer_id": "C-2", "order_id": "O-2",
                    "subject": "SECRET-NONTARGET", "status": "open",
                    "assigned_team": "logistics", "priority": "low",
                },
            ]
        },
        "goal_spec": {
            "target_ticket_id": "T-1", "field": "priority",
            "value": "urgent", "finish_outcome": "completed",
        },
    }


class ScriptedPlanner:
    def __init__(self, actions: list[str]) -> None:
        self.actions = list(actions)
        self.action_mode = "structured"
        self.available_tools = list(TOOLS)
        self.toolbox_metadata = {name: {} for name in TOOLS}
        self.visible_inputs: list[str] = []

    def analyze_query(self, question: str, image: object, json_data: object = None) -> str:
        self.visible_inputs.append(question)
        return "Identify the requested ticket operation."

    def generate_next_step(
        self, question: str, image: object, query_analysis: str, memory: Memory,
        step_count: int, max_steps: int, json_data: object = None,
    ) -> str:
        self.visible_inputs.append(json.dumps(memory.get_actions(), ensure_ascii=False))
        return self.actions.pop(0) if self.actions else "not-json"

    def extract_context_subgoal_and_tool(self, response: object):
        return Planner._extract_structured_action(self, response)


class PlaceholderVerifier:
    pass


class CapturingSolverBuilder:
    def __init__(self, actions: list[str]) -> None:
        self.actions = actions
        self.kwargs: dict[str, object] = {}
        self.planner: ScriptedPlanner | None = None

    def __call__(self, **kwargs: object) -> Solver:
        self.kwargs = dict(kwargs)
        self.planner = ScriptedPlanner(self.actions)
        executor = Executor.__new__(Executor)
        executor.execution_mode = "structured"
        executor.tool_instances_cache = {
            "Ticket_Query_Tool": Ticket_Query_Tool(),
            "Ticket_Update_Tool": Ticket_Update_Tool(),
            "Ticket_Finish_Tool": Ticket_Finish_Tool(),
        }
        return Solver(
            planner=self.planner,
            verifier=PlaceholderVerifier(),
            memory=Memory(),
            executor=executor,
            output_types=str(kwargs["output_types"]),
            max_steps=int(kwargs["max_steps"]),
            verbose=False,
        )


class TicketSolverTests(unittest.TestCase):
    def build_runtime(self, actions: list[str]):
        builder = CapturingSolverBuilder(actions)
        runtime = construct_ticket_runtime(
            llm_engine_name="vllm-Qwen3-0.6B",
            base_url="http://127.0.0.1:8000/v1",
            max_steps=3,
            solver_builder=builder,
        )
        return runtime, builder

    def test_direct_episode_runs_core_solver_in_two_turns(self) -> None:
        runtime, builder = self.build_runtime([
            action("Ticket_Update_Tool", ticket_id="T-1", field="priority", value="urgent"),
            action("Ticket_Finish_Tool", ticket_id="T-1", outcome="completed"),
        ])
        result = runtime.run_episode(episode())
        self.assertTrue(result["verification"]["success"])
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["step_count"], 2)
        self.assertEqual(runtime.solver.__class__.__module__, "agentflow.solver")
        self.assertEqual(builder.kwargs["planner_action_mode"], "structured")
        self.assertEqual(builder.kwargs["executor_mode"], "structured")
        self.assertEqual(builder.kwargs["output_types"], "workflow")

    def test_indirect_episode_queries_then_updates_then_finishes(self) -> None:
        runtime, _ = self.build_runtime([
            action("Ticket_Query_Tool", lookup_by="customer_id", value="C-1"),
            action("Ticket_Update_Tool", ticket_id="T-1", field="priority", value="urgent"),
            action("Ticket_Finish_Tool", ticket_id="T-1", outcome="completed"),
        ])
        result = runtime.run_episode(episode(indirect=True))
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["step_count"], 3)

    def test_wrong_finish_receives_zero_reward(self) -> None:
        runtime, _ = self.build_runtime([
            action("Ticket_Update_Tool", ticket_id="T-1", field="priority", value="urgent"),
            action("Ticket_Finish_Tool", ticket_id="T-2", outcome="completed"),
        ])
        result = runtime.run_episode(episode())
        self.assertEqual(result["reward"], 0.0)
        self.assertIn("WRONG_FINISH", result["verification"]["failure_codes"])

    def test_invalid_planner_action_receives_zero_reward(self) -> None:
        runtime, _ = self.build_runtime(["not-json"])
        result = runtime.run_episode(episode())
        self.assertEqual(result["reward"], 0.0)
        self.assertIn("INVALID_ACTION", result["verification"]["failure_codes"])

    def test_hidden_state_does_not_enter_planner_visible_inputs(self) -> None:
        runtime, builder = self.build_runtime([
            action("Ticket_Update_Tool", ticket_id="T-1", field="priority", value="urgent"),
            action("Ticket_Finish_Tool", ticket_id="T-1", outcome="completed"),
        ])
        runtime.run_episode(episode())
        visible = "\n".join(builder.planner.visible_inputs)
        self.assertNotIn("SECRET-NONTARGET", visible)
        self.assertNotIn("goal_spec", visible)

    def test_runtime_reset_prevents_cross_episode_state_leak(self) -> None:
        runtime, builder = self.build_runtime([
            action("Ticket_Update_Tool", ticket_id="T-1", field="priority", value="urgent"),
            action("Ticket_Finish_Tool", ticket_id="T-1", outcome="completed"),
        ])
        self.assertEqual(runtime.run_episode(episode())["reward"], 1.0)
        builder.planner.actions = [
            action("Ticket_Update_Tool", ticket_id="T-1", field="priority", value="urgent"),
            action("Ticket_Finish_Tool", ticket_id="T-1", outcome="completed"),
        ]
        self.assertEqual(runtime.run_episode(episode())["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
