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
    build_training_record,
    build_loss_items,
    bind_ticket_runtime,
    flatten_rollout_groups,
    gpu_memory_snapshot,
    resolve_reward_mode,
    reset_ticket_solver,
    summarize_rewards,
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

    def test_reward_mode_must_be_binary(self) -> None:
        self.assertEqual(resolve_reward_mode({"reward_mode": "binary"}), "binary")
        with self.assertRaisesRegex(SystemExit, "reward_mode must be binary"):
            resolve_reward_mode({"reward_mode": "dense"})

    def test_summarize_rewards_uses_population_statistics(self) -> None:
        summary = summarize_rewards([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 0.5)
        self.assertEqual(summary["min"], 0.0)
        self.assertEqual(summary["max"], 1.0)
        self.assertEqual(summary["std"], 0.5)

    def test_gpu_memory_snapshot_handles_cpu_only_torch(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        fake_torch = types.SimpleNamespace(cuda=FakeCuda())
        self.assertEqual(gpu_memory_snapshot(fake_torch), {"cuda": False})

    def test_training_record_exposes_remote_diagnostics_at_top_level(self) -> None:
        groups = [[rollout(1.0, 2), rollout(0.0, 1), rollout(0.0, 1, valid=False)]]
        flat, advantages, reward_groups, advantage_groups = flatten_rollout_groups(groups)
        record = build_training_record(
            step=1,
            epoch=0,
            row_index=0,
            batch=[{"episode_id": "ticket-tr-000001"}],
            groups=groups,
            rollouts=flat,
            advantages=advantages,
            reward_groups=reward_groups,
            advantage_groups=advantage_groups,
            stats={
                "loss": 0.25,
                "ratio_mean": 1.001,
                "ratio_min": 0.999,
                "ratio_max": 1.004,
                "clip_fraction": 0.125,
                "approx_kl": 0.002,
            },
            clip_low=0.001,
            clip_high=0.003,
            policy_epochs=2,
            step_elapsed_s=1.25,
            rollout_elapsed_s=0.75,
            train_elapsed_s=0.5,
            gpu_memory={"cuda": False},
        )
        self.assertEqual(record["status"], "update")
        self.assertEqual(record["reward_count"], 3)
        self.assertEqual(record["reward_mean"], 1 / 3)
        self.assertIn("reward_std", record)
        self.assertEqual(record["valid_rollout_count"], 2)
        self.assertEqual(record["infrastructure_failure_count"], 1)
        self.assertEqual(record["response_token_count"], 6)
        self.assertEqual(record["policy_epochs"], 2)
        self.assertEqual(record["loss"], 0.25)
        self.assertEqual(record["ratio_mean"], 1.001)
        self.assertEqual(record["clip_fraction"], 0.125)
        self.assertEqual(record["approx_kl"], 0.002)
        self.assertEqual(record["gpu_memory"], {"cuda": False})
        self.assertIn("train_stats", record)

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
