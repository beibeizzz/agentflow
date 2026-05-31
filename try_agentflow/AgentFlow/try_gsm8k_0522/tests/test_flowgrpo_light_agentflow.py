from __future__ import annotations

import types
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowgrpo_light.agentflow_rollout import AgentFlowPlannerEngine, AgentFlowRolloutRunner
from flowgrpo_light.policy import GeneratedResponse, PlannerPolicy


class FakePolicy:
    def __init__(self) -> None:
        self.calls = []

    def generate_for_agentflow(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        think_mode: str = "default",
    ) -> GeneratedResponse:
        self.calls.append((prompt, system_prompt, think_mode))
        return GeneratedResponse(prompt=f"rendered:{system_prompt}:{prompt}", response='{"Calculation": "1+1"}')


class AgentFlowLightRolloutTests(unittest.TestCase):
    def test_policy_renders_agentflow_system_and_user_prompt_with_chat_template(self) -> None:
        policy = PlannerPolicy.__new__(PlannerPolicy)

        class FakeTokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
                return {
                    "messages": messages,
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "kwargs": kwargs,
                }

        policy.tokenizer = FakeTokenizer()

        rendered = policy.render_agentflow_prompt("Planner prompt", system_prompt="Planner system")

        self.assertEqual(
            rendered,
            {
                "messages": [
                    {"role": "system", "content": "Planner system"},
                    {"role": "user", "content": "Planner prompt"},
                ],
                "tokenize": False,
                "add_generation_prompt": True,
                "kwargs": {},
            },
        )

    def test_policy_passes_qwen_thinking_flag_to_chat_template(self) -> None:
        policy = PlannerPolicy.__new__(PlannerPolicy)

        class FakeTokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
                return kwargs

        policy.tokenizer = FakeTokenizer()

        rendered = policy.render_agentflow_prompt("Planner prompt", think_mode="off")

        self.assertEqual(rendered, {"enable_thinking": False})

    def test_planner_engine_uses_policy_and_records_rendered_prompt_sample(self) -> None:
        policy = FakePolicy()
        engine = AgentFlowPlannerEngine(policy, think_mode="off")

        response = engine("Planner prompt", system_prompt="Planner system", max_tokens=512)

        self.assertEqual(response, '{"Calculation": "1+1"}')
        self.assertEqual(policy.calls, [("Planner prompt", "Planner system", "off")])
        self.assertEqual(len(engine.samples), 1)
        self.assertEqual(engine.samples[0].prompt, "rendered:Planner system:Planner prompt")
        self.assertEqual(engine.samples[0].response, '{"Calculation": "1+1"}')

    def test_rollout_runner_calls_solver_solve_and_scores_direct_output(self) -> None:
        policy = FakePolicy()
        solved_questions = []
        resets = []

        class FakeSolver:
            def __init__(self) -> None:
                self.planner = types.SimpleNamespace(llm_engine=None)

            def solve(self, question: str):
                solved_questions.append(question)
                planner_response = self.planner.llm_engine("Planner prompt", system_prompt="Planner system")
                return {
                    "query_analysis": "analysis",
                    "memory": {"Action Step 1": {"action_predictor_response": planner_response}},
                    "direct_output": "The answer is 2",
                }

        solver = FakeSolver()
        runner = AgentFlowRolloutRunner(
            solver=solver,
            policy=policy,
            reset_solver=lambda item: resets.append(item),
        )

        rollout = runner.run({"question": "What is 1+1?", "gold_answer": "2"})

        self.assertEqual(solved_questions, ["What is 1+1?"])
        self.assertEqual(resets, [solver])
        self.assertEqual(rollout.reward, 1.0)
        self.assertEqual(rollout.answer, "The answer is 2")
        self.assertEqual(rollout.query_analysis, "analysis")
        self.assertEqual(rollout.memory, {"Action Step 1": {"action_predictor_response": '{"Calculation": "1+1"}'}})
        self.assertEqual(len(rollout.samples), 1)
        self.assertEqual(rollout.samples[0].prompt, "rendered:Planner system:Planner prompt")


if __name__ == "__main__":
    unittest.main()
