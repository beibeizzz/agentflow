from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

PROJECT_DIR = Path(__file__).resolve().parents[2]
GSM_DIR = PROJECT_DIR / "try_gsm8k_0522"
if str(GSM_DIR) not in sys.path:
    sys.path.insert(0, str(GSM_DIR))

from flowgrpo_light.rollout import PlannerSample, RolloutResult
from agentflow.models.memory import Memory
from agentflow.tools.ticket_common.backend import TicketBackend
from try_ticket_agent.data_synthesis.blueprints import generate_blueprint
from try_ticket_agent.ticket_env.verifier import TicketVerifier
from try_ticket_agent.flowgrpo_general_2x40g.train_ticket_gspo import (
    DEFAULT_CLIP_RANGE_HIGH,
    DEFAULT_CLIP_RANGE_LOW,
    build_loss_items,
    bind_ticket_runtime,
    flatten_rollout_groups,
    reset_ticket_solver,
    ticket_result_adapter,
)


def rollout(reward: float, turns: int, *, valid: bool = True) -> RolloutResult:
    return RolloutResult(
        reward=reward,
        answer="",
        samples=[
            PlannerSample(
                prompt=f"prompt-{index}",
                response=f"response-{index}",
                prompt_token_ids=[10, index],
                response_token_ids=[20, index],
            )
            for index in range(turns)
        ],
        valid_for_training=valid,
    )


class TicketGspoTests(unittest.TestCase):
    def test_binary_trajectory_advantage_is_broadcast_to_each_turn(self) -> None:
        success = rollout(1.0, 2)
        failure = rollout(0.0, 3)
        flat, advantages, rewards, grouped = flatten_rollout_groups([[success, failure]])
        self.assertEqual(rewards, [[1.0, 0.0]])
        self.assertAlmostEqual(advantages[0], 1.0, places=5)
        self.assertAlmostEqual(advantages[1], -1.0, places=5)
        self.assertEqual(grouped, [advantages])
        items = build_loss_items(object(), flat, advantages)
        self.assertEqual([item.advantage > 0 for item in items], [True, True, False, False, False])
        self.assertEqual([item.advantage for item in items[:2]], [advantages[0], advantages[0]])
        self.assertEqual([item.advantage for item in items[2:]], [advantages[1]] * 3)
        self.assertEqual(items[0].response_token_ids, [20, 0])

    def test_infrastructure_failure_is_excluded_before_group_normalization(self) -> None:
        success = rollout(1.0, 1)
        failure = rollout(0.0, 1)
        infrastructure = rollout(0.0, 1, valid=False)
        flat, advantages, rewards, grouped = flatten_rollout_groups(
            [[success, infrastructure, failure]]
        )
        self.assertEqual(rewards, [[1.0, 0.0, 0.0]])
        self.assertEqual(flat, [success, failure])
        self.assertAlmostEqual(advantages[0], 1.0, places=5)
        self.assertAlmostEqual(advantages[1], -1.0, places=5)
        self.assertEqual(grouped, [[advantages[0], None, advantages[1]]])

    def test_all_equal_binary_rewards_produce_no_loss_items(self) -> None:
        flat, advantages, _, _ = flatten_rollout_groups([[rollout(1.0, 2), rollout(1.0, 3)]])
        self.assertEqual(advantages, [0.0, 0.0])
        self.assertEqual(build_loss_items(object(), flat, advantages), [])

    def test_ticket_training_uses_required_asymmetric_clip_defaults(self) -> None:
        self.assertEqual((DEFAULT_CLIP_RANGE_LOW, DEFAULT_CLIP_RANGE_HIGH), (0.001, 0.003))

    def test_ticket_hooks_reset_hidden_state_and_emit_binary_reward(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=0)
        backend = TicketBackend()
        verifier = TicketVerifier(backend, max_steps=99)
        solver = types.SimpleNamespace(
            _ticket_backend=backend,
            memory=Memory(),
            verifier=verifier,
            max_steps=99,
        )
        old_memory = solver.memory
        row = blueprint.to_dict()
        reset_ticket_solver(solver, row)
        self.assertIsNot(solver.memory, old_memory)
        self.assertEqual(solver.max_steps, blueprint.max_steps)
        self.assertEqual(verifier.max_steps, blueprint.max_steps)

        update = backend.update(
            blueprint.goal_spec["target_ticket_id"],
            blueprint.goal_spec["field"],
            blueprint.goal_spec["value"],
        )
        finish = backend.finish(
            blueprint.goal_spec["target_ticket_id"],
            blueprint.goal_spec["finish_outcome"],
        )
        solver.memory.add_action(1, "Ticket_Update_Tool", "update", "{}", update)
        solver.memory.add_action(2, "Ticket_Finish_Tool", "finish", "{}", finish)
        sample = PlannerSample("prompt", "response", [1], [2])
        result = ticket_result_adapter(
            solver,
            row,
            {"step_count": 2, "memory": solver.memory.get_actions()},
            [sample],
        )
        self.assertEqual(result.reward, 1.0)
        self.assertTrue(result.valid_for_training)
        self.assertEqual(result.samples[0].response_token_ids, [2])

    def test_runtime_binding_exposes_only_its_isolated_backend_on_solver(self) -> None:
        first_solver = types.SimpleNamespace()
        second_solver = types.SimpleNamespace()
        first_backend = TicketBackend()
        second_backend = TicketBackend()
        first = types.SimpleNamespace(solver=first_solver, backend=first_backend)
        second = types.SimpleNamespace(solver=second_solver, backend=second_backend)
        self.assertIs(bind_ticket_runtime(first), first_solver)
        self.assertIs(bind_ticket_runtime(second), second_solver)
        self.assertIs(first_solver._ticket_backend, first_backend)
        self.assertIs(second_solver._ticket_backend, second_backend)
        self.assertIsNot(first_solver._ticket_backend, second_solver._ticket_backend)


if __name__ == "__main__":
    unittest.main()
