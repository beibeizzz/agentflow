from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

PROJECT_DIR = Path(__file__).resolve().parents[2]
GSM_DIR = PROJECT_DIR / "try_gsm8k_0522"
if str(GSM_DIR) not in sys.path:
    sys.path.insert(0, str(GSM_DIR))

from flowgrpo_light.agentflow_rollout import AgentFlowBatchRolloutRunner
from flowgrpo_light.policy import GeneratedResponse
from flowgrpo_light.rollout import RolloutResult


class ExactIdPolicy:
    def generate_many_for_agentflow(
        self,
        prompts,
        *,
        system_prompts=None,
        think_mode="default",
        **generation_kwargs,
    ):
        return [
            GeneratedResponse(
                prompt=f"rendered:{prompt}",
                response='{"tool_name":"Ticket_Finish_Tool","arguments":{}}',
                prompt_token_ids=[101, 102],
                response_token_ids=[201, 202],
            )
            for prompt in prompts
        ]


class TicketRolloutHookTests(unittest.TestCase):
    def test_generic_adapter_receives_row_solver_result_and_exact_samples(self) -> None:
        resets = []
        adapter_calls = []

        class FakeSolver:
            def __init__(self):
                self.planner = types.SimpleNamespace(llm_engine=None)

            def solve(self, question):
                self.planner.llm_engine("ticket planner prompt", system_prompt="planner system")
                return {"workflow": True, "question": question}

        def reset_solver(solver, row):
            resets.append((solver, row["episode_id"]))

        def adapt(solver, row, result, samples):
            adapter_calls.append((solver, row["episode_id"], result, list(samples)))
            return RolloutResult(
                reward=float(row["expected_reward"]),
                answer=row["episode_id"],
                samples=list(samples),
                memory={},
            )

        runner = AgentFlowBatchRolloutRunner(
            policy=ExactIdPolicy(),
            solver_factory=FakeSolver,
            reset_solver=reset_solver,
            result_adapter=adapt,
            question_getter=lambda row: row["user_request"],
            rollout_concurrency=2,
            planner_batch_size=2,
        )
        row = {
            "episode_id": "ticket-tr-000001",
            "user_request": "Finish this ticket request.",
            "expected_reward": 1,
        }
        try:
            groups = runner.run_batch([row], group_size=2)
        finally:
            runner.close()

        self.assertEqual([item.reward for item in groups[0]], [1.0, 1.0])
        self.assertEqual(len(resets), 2)
        self.assertEqual(len(adapter_calls), 2)
        for rollout in groups[0]:
            self.assertEqual(rollout.samples[0].prompt_token_ids, [101, 102])
            self.assertEqual(rollout.samples[0].response_token_ids, [201, 202])
        for _, episode_id, result, samples in adapter_calls:
            self.assertEqual(episode_id, row["episode_id"])
            self.assertEqual(result["question"], row["user_request"])
            self.assertEqual(samples[0].response_token_ids, [201, 202])

    def test_question_getter_failure_returns_invalid_rollout_without_aborting_group(self) -> None:
        class FakeSolver:
            def __init__(self):
                self.planner = types.SimpleNamespace(llm_engine=None)

            def solve(self, question):
                raise AssertionError("solve must not run when question getter fails")

        runner = AgentFlowBatchRolloutRunner(
            policy=ExactIdPolicy(),
            solver_factory=FakeSolver,
            reset_solver=lambda solver, row: None,
            question_getter=lambda row: (_ for _ in ()).throw(KeyError("user_request")),
            rollout_concurrency=2,
            planner_batch_size=2,
        )
        try:
            groups = runner.run_batch([{"episode_id": "ticket-tr-000001"}], group_size=2)
        finally:
            runner.close()

        self.assertEqual(len(groups[0]), 2)
        self.assertTrue(all(item.reward == 0.0 for item in groups[0]))
        self.assertTrue(all(item.valid_for_training is False for item in groups[0]))
        self.assertTrue(all("KeyError" in item.errors[0] for item in groups[0]))

    def test_reset_failure_returns_invalid_rollout_without_aborting_other_rows(self) -> None:
        class FakeSolver:
            def __init__(self):
                self.planner = types.SimpleNamespace(llm_engine=None)

            def solve(self, question):
                self.planner.llm_engine("ticket planner prompt", system_prompt="planner system")
                return {"question": question}

        def reset_solver(solver, row):
            if row["episode_id"] == "bad":
                raise RuntimeError("reset failed")

        def adapt(solver, row, result, samples):
            return RolloutResult(
                reward=1.0,
                answer=row["episode_id"],
                samples=list(samples),
                memory={},
                valid_for_training=True,
            )

        runner = AgentFlowBatchRolloutRunner(
            policy=ExactIdPolicy(),
            solver_factory=FakeSolver,
            reset_solver=reset_solver,
            result_adapter=adapt,
            question_getter=lambda row: row["user_request"],
            rollout_concurrency=2,
            planner_batch_size=2,
        )
        rows = [
            {"episode_id": "bad", "user_request": "bad request"},
            {"episode_id": "good", "user_request": "good request"},
        ]
        try:
            groups = runner.run_batch(rows, group_size=1)
        finally:
            runner.close()

        self.assertEqual(groups[0][0].valid_for_training, False)
        self.assertIn("RuntimeError: reset failed", groups[0][0].errors)
        self.assertEqual(groups[1][0].valid_for_training, True)
        self.assertEqual(groups[1][0].reward, 1.0)
        self.assertEqual(groups[1][0].samples[0].response_token_ids, [201, 202])


if __name__ == "__main__":
    unittest.main()
