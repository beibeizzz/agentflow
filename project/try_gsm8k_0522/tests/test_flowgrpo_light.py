import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowgrpo_light.calculator import format_number, safe_eval_calculation
from flowgrpo_light.parsing import parse_planner_response
from flowgrpo_light.prompts import build_planner_prompt
from flowgrpo_light.policy import PlannerPolicy
from flowgrpo_light.rollout import PlannerSample, RolloutResult
from flowgrpo_light.train_light_grpo import flatten_rollout_groups, parse_args, train_step


class TestFlowGRPOLight(unittest.TestCase):
    def test_safe_eval_calculation_accepts_basic_arithmetic(self):
        self.assertEqual(format_number(safe_eval_calculation("(6 * 5) + 12")), "42")
        self.assertEqual(format_number(safe_eval_calculation("1 / 2")), "0.5")

    def test_safe_eval_rejects_non_arithmetic(self):
        with self.assertRaises(ValueError):
            safe_eval_calculation("__import__('os').system('echo bad')")

    def test_parse_planner_response_extracts_json(self):
        sub_goal, calculation = parse_planner_response(
            '```json\n{"Sub_goal": "Compute brother grapes", "Calculation": "6 * 5"}\n```'
        )

        self.assertEqual(sub_goal, "Compute brother grapes")
        self.assertEqual(calculation, "6 * 5")

    def test_parse_planner_response_matches_agentflow_calculator_only_style(self):
        sub_goal, calculation = parse_planner_response(
            '{"Sub_goal": "List quantities", "Calculation": "Adam: 50 + Betty: 65"}'
        )

        self.assertEqual(sub_goal, "List quantities")
        self.assertEqual(calculation, "Adam: 50 + Betty: 65")

    def test_parse_planner_response_strips_leading_think_block_like_agentflow(self):
        sub_goal, calculation = parse_planner_response(
            '<think>hidden reasoning</think>\n{"Sub_goal": "Add", "Calculation": "1 + 1"}'
        )

        self.assertEqual(sub_goal, "Add")
        self.assertEqual(calculation, "1 + 1")

    def test_planner_prompt_contains_training_contract(self):
        prompt = build_planner_prompt(
            question="What is 1+1?",
            query_analysis="Add the quantities.",
            memory={},
            step_count=1,
            max_steps=3,
        )

        self.assertIn("Return only one JSON object", prompt)
        self.assertIn('"Calculation"', prompt)
        self.assertIn("Memory: {}", prompt)

    def test_train_args_accept_think_mode(self):
        args = parse_args(["--rollout-backend", "agentflow", "--think-mode", "off"])

        self.assertEqual(args.rollout_backend, "agentflow")
        self.assertEqual(args.think_mode, "off")

    def test_train_args_accept_question_batch_size(self):
        args = parse_args(["--question-batch-size", "4", "--group-size", "4", "--logprob-micro-batch-size", "8"])

        self.assertEqual(args.question_batch_size, 4)
        self.assertEqual(args.group_size, 4)
        self.assertEqual(args.logprob_micro_batch_size, 8)

    def test_flatten_rollout_groups_excludes_invalid_rollouts_from_training_advantages(self):
        valid_high = RolloutResult(reward=1.0, answer="2")
        invalid = RolloutResult(reward=0.0, answer="", errors=["RuntimeError: vLLM timeout"], valid_for_training=False)
        valid_low = RolloutResult(reward=0.0, answer="3")

        rollouts, advantages, reward_groups, advantage_groups = flatten_rollout_groups(
            [[valid_high, invalid, valid_low]]
        )

        self.assertEqual(rollouts, [valid_high, valid_low])
        self.assertEqual(reward_groups, [[1.0, 0.0, 0.0]])
        self.assertEqual(len(advantages), 2)
        self.assertAlmostEqual(advantages[0], 1.0, places=4)
        self.assertAlmostEqual(advantages[1], -1.0, places=4)
        self.assertEqual(len(advantage_groups), 1)
        self.assertAlmostEqual(advantage_groups[0][0], 1.0, places=4)
        self.assertIsNone(advantage_groups[0][1])
        self.assertAlmostEqual(advantage_groups[0][2], -1.0, places=4)

    def test_policy_sequence_logprob_many_masks_prompt_and_padding_tokens(self):
        policy = PlannerPolicy.__new__(PlannerPolicy)

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def __call__(self, text, *, add_special_tokens=True):
                ids = [ord(char) % 31 + 3 for char in text]
                if add_special_tokens:
                    ids = ids + [self.eos_token_id]
                return {"input_ids": ids}

        class FakeModel:
            def __init__(self):
                self.param = torch.nn.Parameter(torch.zeros(1))

            def parameters(self):
                yield self.param

            def __call__(self, *, input_ids, attention_mask):
                vocab_scores = torch.arange(64, dtype=torch.float32, device=input_ids.device)
                logits = vocab_scores.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
                return type("Output", (), {"logits": logits})

        policy.tokenizer = FakeTokenizer()
        policy.model = FakeModel()

        prompts = ["A", "Long"]
        responses = ["BC", "D"]
        logprobs = policy.sequence_logprob_many(prompts, responses)

        token_logprobs = torch.log_softmax(torch.arange(64, dtype=torch.float32), dim=-1)
        expected = []
        for response in responses:
            response_ids = policy._tokenize(response, add_special_tokens=False)
            expected.append(token_logprobs[response_ids].sum())

        self.assertTrue(torch.allclose(logprobs.cpu(), torch.stack(expected)))

    def test_train_step_uses_batched_logprob_micro_batches_for_one_optimizer_step(self):
        class FakePolicy:
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.model = torch.nn.Module()
                self.model.register_parameter("weight", self.weight)
                self.calls = []

            def train(self):
                self.model.train()

            def _tokenize(self, text, *, add_special_tokens):
                return text.split()

            def sequence_logprob_many(self, prompts, responses, *, use_adapter=True):
                self.calls.append((list(prompts), list(responses), use_adapter))
                values = torch.tensor([float(len(response.split())) for response in responses])
                return self.weight * values

        policy = FakePolicy()
        optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.1)
        rollouts = [
            RolloutResult(
                reward=1.0,
                answer="",
                samples=[
                    PlannerSample(prompt="p1", response="a b"),
                    PlannerSample(prompt="p2", response="c"),
                ],
            ),
            RolloutResult(
                reward=0.0,
                answer="",
                samples=[
                    PlannerSample(prompt="p3", response="d e"),
                    PlannerSample(prompt="p4", response="f"),
                    PlannerSample(prompt="p5", response="g h"),
                ],
            ),
        ]

        loss = train_step(
            policy=policy,
            optimizer=optimizer,
            rollouts=rollouts,
            advantages=[1.0, -1.0],
            kl_coef=0.0,
            max_grad_norm=10.0,
            logprob_micro_batch_size=2,
        )

        self.assertIsNotNone(loss)
        self.assertEqual([len(call[0]) for call in policy.calls], [2, 2, 1])
        self.assertEqual(sum(1 for call in policy.calls if call[2]), 3)
        self.assertNotEqual(float(policy.weight.detach()), 1.0)


if __name__ == "__main__":
    unittest.main()
