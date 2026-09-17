import asyncio
from types import SimpleNamespace

import pytest

from agentflow_rl.integrations.planner import (
    OpenAICompatiblePlannerGateway,
    VerlPlannerGateway,
    normalize_token_ids,
)
from agentflow_rl.runtime.errors import InfrastructureError


@pytest.mark.parametrize('explicit_budget, expected', [(8192, 8192), (None, 4096)])
def test_frozen_input_budget_can_be_independent_of_planner(monkeypatch, explicit_budget, expected):
    from agentflow_rl.integrations import bootstrap
    observed = {}
    class StopAfterGateway(Exception):
        pass
    def gateway(*args, **kwargs):
        observed.update(kwargs)
        raise StopAfterGateway()
    monkeypatch.setattr(bootstrap, '_openai_client', lambda **kwargs: object())
    monkeypatch.setattr(bootstrap, '_frozen_tokenizer', lambda *args: object())
    monkeypatch.setattr(bootstrap, 'OpenAICompatibleFrozenGateway', gateway)
    config = {'data': {'max_prompt_length': 4096}, 'agentflow': {
        'frozen_model': 'frozen', 'frozen_base_url': 'http://fixture.invalid'}}
    if explicit_budget is not None:
        config['agentflow']['frozen_max_input_tokens'] = explicit_budget
    with pytest.raises(StopAfterGateway):
        bootstrap.build_runtime_bundle(config, planner=object())
    assert observed['max_input_tokens'] == expected


def test_bootstrap_passes_role_specific_frozen_input_budgets(monkeypatch):
    from agentflow_rl.integrations import bootstrap

    observed = {}

    class StopAfterGateway(Exception):
        pass

    def gateway(*args, **kwargs):
        observed.update(kwargs)
        raise StopAfterGateway()

    monkeypatch.setattr(bootstrap, '_openai_client', lambda **kwargs: object())
    monkeypatch.setattr(bootstrap, '_frozen_tokenizer', lambda *args: object())
    monkeypatch.setattr(bootstrap, 'OpenAICompatibleFrozenGateway', gateway)
    config = {
        'data': {'max_prompt_length': 4096},
        'agentflow': {
            'frozen_model': 'frozen',
            'frozen_base_url': 'http://fixture.invalid',
            'frozen_max_input_tokens': 4096,
            'role_max_input_tokens': {
                'query_analyzer': 4096,
                'executor': 4096,
                'verifier': 4096,
                'generator': 8192,
            },
        },
    }
    with pytest.raises(StopAfterGateway):
        bootstrap.build_runtime_bundle(config, planner=object())
    assert observed['role_max_input_tokens']['generator'] == 8192


def test_formal_bootstrap_passes_sandbox_service_identity(monkeypatch):
    from agentflow_rl.integrations import bootstrap

    observed = {}

    class StopAfterSandbox(Exception):
        pass

    def sandbox(url, **kwargs):
        observed["url"] = url
        observed.update(kwargs)
        raise StopAfterSandbox()

    monkeypatch.setattr(bootstrap, '_openai_client', lambda **kwargs: object())
    monkeypatch.setattr(bootstrap, '_frozen_tokenizer', lambda *args: object())
    monkeypatch.setattr(
        bootstrap, 'OpenAICompatibleFrozenGateway', lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(bootstrap, 'HttpSandboxBackend', sandbox)
    config = {
        'data': {'max_prompt_length': 4096},
        'agentflow': {
            'environment_mode': 'formal',
            'frozen_model': 'frozen',
            'frozen_base_url': 'http://fixture.invalid',
            'tools': {
                'sandbox_image': 'sandbox@sha256:1',
                'sandbox_service_url': 'http://sandbox:8005',
                'sandbox_service_revision': 'sandbox-r7',
                'sandbox_service_timeout_s': 17.5,
            },
        },
    }
    with pytest.raises(StopAfterSandbox):
        bootstrap.build_runtime_bundle(config, planner=object())
    assert observed == {
        'url': 'http://sandbox:8005',
        'expected_revision': 'sandbox-r7',
        'timeout_s': 17.5,
    }


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [[1, 2, 3]]

    def decode(self, token_ids, **kwargs):
        return '{"sub_goal":"solve","tool_name":"Base_Generator_Tool",' \
               '"arguments":{"query":"solve"}}'

    def encode(self, text, **kwargs):
        return [4, 5]


class Server:
    async def generate(self, **kwargs):
        return SimpleNamespace(
            token_ids=[4, 5], log_probs=[-0.2, -0.3],
            extra_fields={"global_steps": 7},
        )


def test_verl_planner_captures_tokens_and_logprobs() -> None:
    gateway = VerlPlannerGateway(Server(), Tokenizer(), revision="checkpoint-7")
    result = asyncio.run(
        gateway.generate(
            request_id="r1", system_prompt="system", prompt="task", sampling_params={}
        )
    )
    assert result.prompt_ids == (1, 2, 3)
    assert result.response_ids == (4, 5)
    assert result.response_logprobs == (-0.2, -0.3)
    assert result.model_revision == "checkpoint-7:update-7"
    assert result.policy_update_id == "7"


def test_verl_planner_validates_server_reported_policy_update_id() -> None:
    gateway = VerlPlannerGateway(Server(), Tokenizer(), revision="checkpoint-7")
    result = asyncio.run(
        gateway.generate(
            request_id="r1",
            system_prompt="system",
            prompt="task",
            sampling_params={"agentflow_expected_policy_update_id": 7},
        )
    )
    assert result.model_revision == "checkpoint-7:update-7"


def test_verl_planner_rejects_stale_server_policy() -> None:
    with pytest.raises(InfrastructureError, match="stale"):
        asyncio.run(
            VerlPlannerGateway(Server(), Tokenizer(), revision="checkpoint-7").generate(
                request_id="r1", system_prompt="s", prompt="p",
                sampling_params={"agentflow_expected_policy_update_id": 8},
            )
        )


def test_verl_planner_requires_server_policy_identity() -> None:
    class UnidentifiedServer:
        async def generate(self, **kwargs):
            return SimpleNamespace(token_ids=[4, 5], log_probs=[-0.2, -0.3])

    with pytest.raises(InfrastructureError, match="update ID"):
        asyncio.run(
            VerlPlannerGateway(
                UnidentifiedServer(), Tokenizer(), revision="checkpoint"
            ).generate(
                request_id="r1", system_prompt="s", prompt="p", sampling_params={}
            )
        )


def test_verl_planner_rejects_misaligned_logprobs() -> None:
    class BrokenServer:
        async def generate(self, **kwargs):
            return SimpleNamespace(
                token_ids=[4, 5], log_probs=[-0.2], extra_fields={"global_steps": 1}
            )

    with pytest.raises(InfrastructureError):
        asyncio.run(
            VerlPlannerGateway(BrokenServer(), Tokenizer(), revision="r").generate(
                request_id="r1", system_prompt="s", prompt="p", sampling_params={}
            )
        )


def test_normalize_token_ids_accepts_batch_encoding_shape() -> None:
    assert normalize_token_ids({"input_ids": [[1, 2]]}) == (1, 2)


def test_verl_planner_enforces_prompt_token_budget() -> None:
    gateway = VerlPlannerGateway(
        Server(), Tokenizer(), revision="checkpoint-7", max_prompt_tokens=2
    )
    with pytest.raises(InfrastructureError, match="budget"):
        asyncio.run(
            gateway.generate(
                request_id="r1", system_prompt="system", prompt="task", sampling_params={}
            )
        )


class ChatCompletions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"sub_goal":"solve"}')
                )
            ]
        )


def test_openai_planner_gateway_builds_evaluation_generation() -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=ChatCompletions()))
    gateway = OpenAICompatiblePlannerGateway(
        client, Tokenizer(), model="planner", revision="checkpoint-9"
    )
    result = asyncio.run(
        gateway.generate(
            request_id="r1",
            system_prompt="system",
            prompt="task",
            sampling_params={"max_tokens": 32, "seed": 3},
        )
    )
    assert result.prompt_ids == (1, 2, 3)
    assert result.response_ids == (4, 5)
    assert result.response_logprobs == (0.0, 0.0)
