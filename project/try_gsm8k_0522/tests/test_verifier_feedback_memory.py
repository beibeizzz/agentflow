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


class FakePlanner:
    def __init__(self):
        self.available_tools = ["Calculator_Tool"]
        self.toolbox_metadata = {"Calculator_Tool": {"input_types": {"expression": "str"}}}
        self.memory_seen_on_second_step = None

    def analyze_query(self, question, image):
        return ""

    def generate_next_step(self, question, image, query_analysis, memory, step_count, max_step_count, json_data=None):
        if step_count == 2:
            self.memory_seen_on_second_step = memory.get_actions()
        return '{"sub_goal":"Calculate one expression","calculation":"1 + 1","tool_name":"Calculator_Tool"}'

    def extract_context_subgoal_and_tool(self, next_step):
        return "1 + 1", "Calculate one expression", "Calculator_Tool"

    def generate_direct_output(self, question, image, memory):
        return "2"


class InvalidPlanner(FakePlanner):
    def __init__(self):
        super().__init__()
        self.response = '{"sub_goal":"Calculate one expression","calculation":"2 + 2",}'

    def generate_next_step(self, question, image, query_analysis, memory, step_count, max_step_count, json_data=None):
        return self.response

    def extract_context_subgoal_and_tool(self, next_step):
        return None, None, None


class FakeExecutor:
    def __init__(self):
        self.execute_calls = 0

    def set_query_cache_dir(self, root_cache_dir):
        pass

    def generate_tool_command(self, question, image, context, sub_goal, tool_name, tool_metadata, step_count=0, json_data=None):
        return 'execution = tool.execute(expression="1 + 1")'

    def extract_explanation_and_command(self, tool_command):
        return "", "", tool_command

    def execute_tool_command(self, tool_name, command):
        self.execute_calls += 1
        return ["2"]


class FakeVerifier:
    def __init__(self):
        self.calls = 0

    def verificate_context(self, question, image, query_analysis, memory, step_count=0, json_data=None):
        self.calls += 1
        if self.calls == 1:
            return "Need one more calculation using the latest numeric result."
        return "The memory is sufficient."

    def extract_conclusion(self, response):
        if response.startswith("Need one more"):
            return response, "CONTINUE"
        return response, "STOP"


class TestVerifierFeedbackMemory(unittest.TestCase):
    def test_solver_writes_judge_into_memory_for_next_planner_step(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.memory import Memory
        from agentflow.solver import Solver

        planner = FakePlanner()
        solver = Solver(
            planner=planner,
            verifier=FakeVerifier(),
            memory=Memory(),
            executor=FakeExecutor(),
            output_types="direct",
            max_steps=2,
            verbose=False,
        )

        result = solver.solve("Problem text")

        first_action = result["memory"]["Action Step 1"]
        self.assertEqual(first_action["judge"], "Need one more calculation using the latest numeric result.")
        self.assertNotIn("verifier_feedback", first_action)
        self.assertNotIn("verifier_conclusion", first_action)
        self.assertEqual(
            planner.memory_seen_on_second_step["Action Step 1"]["judge"],
            "Need one more calculation using the latest numeric result.",
        )

    def test_solver_records_invalid_planner_output_for_verifier_memory(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.memory import Memory
        from agentflow.solver import Solver

        planner = InvalidPlanner()
        solver = Solver(
            planner=planner,
            verifier=FakeVerifier(),
            memory=Memory(),
            executor=FakeExecutor(),
            output_types="direct",
            max_steps=1,
            verbose=False,
        )

        result = solver.solve("Problem text")
        first_action = result["memory"]["Action Step 1"]

        self.assertEqual(first_action["command"], "Planner output was invalid; no tool command generated.")
        self.assertEqual(first_action["result"], "No calculator execution was attempted.")
        self.assertEqual(first_action["action_predictor_response"], planner.response)

    def test_solver_blocks_repeated_sub_goal_or_calculation_before_executor(self):
        bootstrap_agentflow_runtime()
        from agentflow.models.memory import Memory
        from agentflow.solver import Solver

        planner = FakePlanner()
        executor = FakeExecutor()
        solver = Solver(
            planner=planner,
            verifier=FakeVerifier(),
            memory=Memory(),
            executor=executor,
            output_types="direct",
            max_steps=2,
            verbose=False,
        )

        result = solver.solve("Problem text")

        self.assertEqual(executor.execute_calls, 1)
        second_action = result["memory"]["Action Step 2"]
        self.assertEqual(
            second_action["command"],
            "Planner repeated previous sub_goal or Calculation; no tool command generated.",
        )
        self.assertEqual(second_action["result"], [])
        self.assertNotIn("action_predictor_response", second_action)
        self.assertNotIn("repeat_reason", second_action)
        self.assertEqual(
            second_action["judge"],
            "Conclusion: CONTINUE\nPlanner sub_goal repeated and Calculation repeated. You must choose a different Sub_goal and a different Calculation that directly fix the missing logic; do not repeat any previous planner step.",
        )


if __name__ == "__main__":
    unittest.main()
