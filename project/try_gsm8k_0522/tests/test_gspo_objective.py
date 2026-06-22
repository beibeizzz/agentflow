from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowgrpo_light.grpo_objective import clipped_grpo_loss, train_step_grpo
from flowgrpo_light.rollout import PlannerSample, RolloutResult
from flowgrpo_light.train_light_grpo_general import parse_args, resolve_clip_ranges


class TestGSPOObjective(unittest.TestCase):
    def test_general_parser_and_config_resolve_asymmetric_clip_ranges(self) -> None:
        args = parse_args(
            [
                "--clip-range-low",
                "0.001",
                "--clip-range-high",
                "0.003",
            ]
        )
        self.assertEqual(resolve_clip_ranges(args, {}), (0.001, 0.003))

        defaults = parse_args([])
        self.assertEqual(resolve_clip_ranges(defaults, {}), (0.001, 0.003))

    def test_general_clip_ranges_must_be_positive(self) -> None:
        args = parse_args(["--clip-range-low", "0"])
        with self.assertRaises(SystemExit):
            resolve_clip_ranges(args, {})

    def test_asymmetric_clip_uses_distinct_upper_and_lower_bounds(self) -> None:
        upper_loss, _ = clipped_grpo_loss(
            torch.tensor([math.log(1.01)]),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            clip_range_low=0.001,
            clip_range_high=0.003,
        )
        lower_loss, _ = clipped_grpo_loss(
            torch.tensor([math.log(0.99)]),
            torch.tensor([0.0]),
            torch.tensor([-1.0]),
            clip_range_low=0.001,
            clip_range_high=0.003,
        )

        self.assertAlmostEqual(float(upper_loss), -1.003, places=6)
        self.assertAlmostEqual(float(lower_loss), 0.999, places=6)

    def test_ratio_math_is_float32_for_bfloat16_logprobs(self) -> None:
        loss, stats = clipped_grpo_loss(
            torch.tensor([0.0], dtype=torch.bfloat16),
            torch.tensor([0.0], dtype=torch.bfloat16),
            torch.tensor([1.0], dtype=torch.bfloat16),
            clip_range_low=0.001,
            clip_range_high=0.003,
        )

        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(stats["ratio_mean"], 1.0)

    def test_train_step_prefers_exact_token_ids_and_falls_back_for_legacy_samples(self) -> None:
        class FakePolicy:
            def __init__(self) -> None:
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.model = torch.nn.Module()
                self.model.register_parameter("weight", self.weight)
                self.token_calls = []
                self.text_calls = []

            @property
            def device(self):
                return self.weight.device

            def train(self):
                self.model.train()

            def _tokenize(self, text, *, add_special_tokens):
                return text.split()

            def sequence_logprob_token_ids_many(
                self,
                prompt_token_ids,
                response_token_ids,
                *,
                use_adapter=True,
            ):
                self.token_calls.append(
                    (prompt_token_ids, response_token_ids, use_adapter)
                )
                lengths = torch.tensor(
                    [float(len(item)) for item in response_token_ids]
                )
                return self.weight * lengths

            def sequence_logprob_many(self, prompts, responses, *, use_adapter=True):
                self.text_calls.append((prompts, responses, use_adapter))
                lengths = torch.tensor(
                    [float(len(item.split())) for item in responses]
                )
                return self.weight * lengths

        policy = FakePolicy()
        optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.01)
        rollouts = [
            RolloutResult(
                reward=1.0,
                answer="",
                samples=[
                    PlannerSample(
                        prompt="exact prompt",
                        response="decoded response",
                        prompt_token_ids=[4, 5],
                        response_token_ids=[7, 2],
                    )
                ],
            ),
            RolloutResult(
                reward=0.0,
                answer="",
                samples=[PlannerSample(prompt="legacy prompt", response="a b")],
            ),
        ]

        stats = train_step_grpo(
            policy=policy,
            optimizer=optimizer,
            rollouts=rollouts,
            advantages=[1.0, -1.0],
            clip_range_low=0.001,
            clip_range_high=0.003,
            max_grad_norm=1.0,
            logprob_micro_batch_size=2,
            policy_epochs=1,
        )

        self.assertIsNotNone(stats)
        self.assertEqual(len(policy.token_calls), 2)
        self.assertEqual(len(policy.text_calls), 2)
        self.assertEqual(policy.token_calls[0][0], [[4, 5]])
        self.assertEqual(policy.token_calls[0][1], [[7, 2]])


if __name__ == "__main__":
    unittest.main()
