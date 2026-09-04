from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agentflow_rl.verl.ports import AsyncFrozenModel, generate_planner_turn, normalize_prompt_ids


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [10, 11, 12]

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return '{"tool_name":"X","arguments":{}}'


class FakeServer:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            token_ids=[21, 22, 2],
            log_probs=[-0.2, -0.3, -0.1],
            num_preempted=0,
            extra_fields={},
        )


def test_planner_port_preserves_exact_rollout_token_ids_and_logprobs() -> None:
    tokenizer = FakeTokenizer()
    server = FakeServer()
    turn = asyncio.run(generate_planner_turn(
        server_manager=server,
        tokenizer=tokenizer,
        request_id="trajectory-turn-0",
        system_prompt="system",
        prompt="prompt",
        sampling_params={"temperature": 1.2, "top_p": 1.0, "top_k": -1, "logprobs": True},
    ))

    assert turn.prompt_ids == (10, 11, 12)
    assert turn.response_ids == (21, 22, 2)
    assert turn.response_logprobs == (-0.2, -0.3, -0.1)
    assert turn.response.endswith("}")
    assert server.calls[0]["prompt_ids"] == [10, 11, 12]
    assert server.calls[0]["sampling_params"]["temperature"] == 1.2
    assert tokenizer.kwargs["enable_thinking"] is False


def test_normalize_prompt_ids_accepts_transformers_batch_encoding_shape() -> None:
    assert normalize_prompt_ids({
        "input_ids": [[10, 11, 12]],
        "attention_mask": [[1, 1, 1]],
    }) == [10, 11, 12]


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="frozen output")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_frozen_port_keeps_role_prompt_and_think_mode_outside_actor_tokens() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model = AsyncFrozenModel(client, model="Qwen3-0.6B")

    output = asyncio.run(model.generate(
        prompt="analyze",
        system_prompt="math system",
        think_mode="on",
        max_tokens=512,
    ))

    assert output == "frozen output"
    assert completions.kwargs["messages"] == [
        {"role": "system", "content": "math system"},
        {"role": "user", "content": "analyze"},
    ]
    assert completions.kwargs["temperature"] == 0.0
    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
