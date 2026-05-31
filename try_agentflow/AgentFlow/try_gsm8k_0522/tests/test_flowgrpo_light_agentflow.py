from __future__ import annotations

import types
import unittest
from pathlib import Path
import sys
import threading
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowgrpo_light.agentflow_rollout import (
    AgentFlowBatchRolloutRunner,
    AgentFlowPlannerEngine,
    AgentFlowRolloutRunner,
    BatchedAgentFlowPlannerEngine,
)
from flowgrpo_light.policy import GeneratedResponse, PlannerPolicy


class FakePolicy:
    def __init__(self) -> None:
        self.calls = []
        self.batch_calls = []

    def generate_for_agentflow(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        think_mode: str = "default",
    ) -> GeneratedResponse:
        self.calls.append((prompt, system_prompt, think_mode))
        return GeneratedResponse(prompt=f"rendered:{system_prompt}:{prompt}", response='{"Calculation": "1+1"}')

    def generate_many_for_agentflow(
        self,
        prompts: list[str],
        *,
        system_prompts: list[str | None] | None = None,
        think_mode: str = "default",
    ) -> list[GeneratedResponse]:
        effective_system_prompts = system_prompts or [None for _ in prompts]
        self.batch_calls.append((list(prompts), list(effective_system_prompts), think_mode))
        return [
            GeneratedResponse(
                prompt=f"rendered:{system_prompt}:{prompt}",
                response=f"response:{prompt}",
            )
            for prompt, system_prompt in zip(prompts, effective_system_prompts, strict=True)
        ]


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

    def test_policy_generate_many_batches_prompts_in_one_model_call(self) -> None:
        policy = PlannerPolicy.__new__(PlannerPolicy)

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 99
            padding_side = "right"

            def __call__(self, texts, *, return_tensors=None, add_special_tokens=True, padding=False):
                if isinstance(texts, str):
                    texts = [texts]
                tokenized = []
                for text in texts:
                    ids = [ord(char) % 50 + 1 for char in text]
                    if add_special_tokens:
                        ids = ids + [self.eos_token_id]
                    tokenized.append(ids)
                width = max(len(ids) for ids in tokenized)
                padded = []
                masks = []
                for ids in tokenized:
                    pad_count = width - len(ids)
                    if self.padding_side == "left":
                        padded.append([self.pad_token_id] * pad_count + ids)
                        masks.append([0] * pad_count + [1] * len(ids))
                    else:
                        padded.append(ids + [self.pad_token_id] * pad_count)
                        masks.append([1] * len(ids) + [0] * pad_count)
                return {
                    "input_ids": torch.tensor(padded),
                    "attention_mask": torch.tensor(masks),
                }

            def decode(self, ids, *, skip_special_tokens=True):
                return " ".join(str(int(item)) for item in ids)

        class FakeModel:
            def __init__(self) -> None:
                self.generate_calls = []
                self.param = torch.nn.Parameter(torch.zeros(1))

            def eval(self) -> None:
                pass

            def parameters(self):
                yield self.param

            def generate(self, **kwargs):
                input_ids = kwargs["input_ids"]
                self.generate_calls.append(input_ids.clone())
                additions = torch.tensor([[201, 202], [203, 204]])
                return torch.cat([input_ids, additions], dim=1)

        policy.tokenizer = FakeTokenizer()
        policy.model = FakeModel()
        policy.max_new_tokens = 2
        policy.temperature = 0.8
        policy.top_p = 0.95

        responses = policy.generate_many(["short", "longer"])

        self.assertEqual([item.response for item in responses], ["201 202", "203 204"])
        self.assertEqual([item.prompt for item in responses], ["short", "longer"])
        self.assertEqual(len(policy.model.generate_calls), 1)
        self.assertEqual(policy.tokenizer.padding_side, "left")

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
        self.assertTrue(rollout.valid_for_training)
        self.assertEqual(rollout.reward, 1.0)
        self.assertEqual(rollout.answer, "The answer is 2")
        self.assertEqual(rollout.query_analysis, "analysis")
        self.assertEqual(rollout.memory, {"Action Step 1": {"action_predictor_response": '{"Calculation": "1+1"}'}})
        self.assertEqual(len(rollout.samples), 1)
        self.assertEqual(rollout.samples[0].prompt, "rendered:Planner system:Planner prompt")

    def test_batched_planner_engine_batches_concurrent_requests_and_records_per_rollout_samples(self) -> None:
        policy = FakePolicy()
        engine = BatchedAgentFlowPlannerEngine(
            policy,
            think_mode="off",
            max_batch_size=4,
            batch_timeout_s=0.05,
        )
        proxies = [engine.create_proxy() for _ in range(4)]
        barrier = threading.Barrier(4)
        responses = [None for _ in range(4)]

        def call_proxy(index: int) -> None:
            barrier.wait()
            responses[index] = proxies[index](f"prompt-{index}", system_prompt=f"system-{index}")

        threads = [threading.Thread(target=call_proxy, args=(index,)) for index in range(4)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            engine.close()

        self.assertEqual(sorted(responses), [f"response:prompt-{index}" for index in range(4)])
        self.assertEqual(len(policy.batch_calls), 1)
        prompts, system_prompts, think_mode = policy.batch_calls[0]
        self.assertEqual(sorted(prompts), [f"prompt-{index}" for index in range(4)])
        self.assertEqual(sorted(system_prompts), [f"system-{index}" for index in range(4)])
        self.assertEqual(think_mode, "off")
        for index, proxy in enumerate(proxies):
            self.assertEqual(len(proxy.samples), 1)
            self.assertEqual(proxy.samples[0].response, f"response:prompt-{index}")

    def test_batch_rollout_runner_runs_question_batch_with_independent_solver_workers(self) -> None:
        policy = FakePolicy()
        solver_ids = []
        reset_ids = []

        class FakeSolver:
            def __init__(self, solver_id: int) -> None:
                self.solver_id = solver_id
                self.planner = types.SimpleNamespace(llm_engine=None)

            def solve(self, question: str):
                planner_response = self.planner.llm_engine(
                    f"planner prompt {self.solver_id} {question}",
                    system_prompt="Planner system",
                )
                return {
                    "query_analysis": f"analysis {question}",
                    "memory": {"Action Step 1": {"action_predictor_response": planner_response}},
                    "direct_output": "The answer is 2",
                }

        def solver_factory():
            solver_id = len(solver_ids)
            solver_ids.append(solver_id)
            return FakeSolver(solver_id)

        runner = AgentFlowBatchRolloutRunner(
            policy=policy,
            solver_factory=solver_factory,
            reset_solver=lambda solver: reset_ids.append(solver.solver_id),
            rollout_concurrency=4,
            planner_batch_size=4,
            planner_batch_timeout_s=0.05,
            think_mode="off",
        )
        try:
            groups = runner.run_batch(
                [
                    {"question": "What is 1+1?", "gold_answer": "2"},
                    {"question": "What is 1+1 again?", "gold_answer": "2"},
                ],
                group_size=2,
            )
        finally:
            runner.close()

        self.assertEqual([len(group) for group in groups], [2, 2])
        self.assertEqual(len(solver_ids), 4)
        self.assertEqual(sorted(reset_ids), [0, 1, 2, 3])
        self.assertEqual(len(policy.batch_calls), 1)
        self.assertEqual(sum(len(rollout.samples) for group in groups for rollout in group), 4)

    def test_rollout_runner_marks_solver_exception_invalid_for_training(self) -> None:
        policy = FakePolicy()

        class FakeSolver:
            def __init__(self) -> None:
                self.planner = types.SimpleNamespace(llm_engine=None)

            def solve(self, question: str):
                self.planner.llm_engine("Planner prompt", system_prompt="Planner system")
                raise RuntimeError("frozen vLLM timeout")

        runner = AgentFlowRolloutRunner(
            solver=FakeSolver(),
            policy=policy,
            reset_solver=lambda item: None,
        )

        rollout = runner.run({"question": "What is 1+1?", "gold_answer": "2"})

        self.assertFalse(rollout.valid_for_training)
        self.assertEqual(rollout.reward, 0.0)
        self.assertEqual(rollout.errors, ["RuntimeError: frozen vLLM timeout"])
        self.assertEqual(len(rollout.samples), 1)


if __name__ == "__main__":
    unittest.main()
