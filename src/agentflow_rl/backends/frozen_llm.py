from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentflow_rl.roles.schemas import RoleName
from agentflow_rl.runtime.errors import InfrastructureError, PromptBudgetError
from agentflow_rl.runtime.prompt_budget import role_token_count, chat_token_count


class OpenAICompatibleFrozenGateway:
    """Frozen Qwen role and Base Generator gateway over a vLLM HTTP server."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        revision: str,
        temperature: float = 0.0,
        max_input_tokens: int = 8192,
        role_max_input_tokens: Mapping[str | RoleName, int] | None = None,
        tokenizer: Any = None,
    ) -> None:
        if max_input_tokens <= 0:
            raise ValueError("frozen role input budget must be positive")
        self.client = client
        self.model = model
        self.revision = revision
        self.temperature = temperature
        self.max_input_tokens = max_input_tokens
        self.role_max_input_tokens = {
            role: int(
                (role_max_input_tokens or {}).get(
                    role,
                    (role_max_input_tokens or {}).get(role.value, max_input_tokens),
                )
            )
            for role in RoleName
            if role is not RoleName.PLANNER
        }
        if any(value <= 0 for value in self.role_max_input_tokens.values()):
            raise ValueError("frozen role input budgets must be positive")
        self.tokenizer = tokenizer

    def input_limit(self, role: RoleName) -> int:
        return self.role_max_input_tokens.get(role, self.max_input_tokens)

    async def generate(
        self,
        *,
        role: RoleName,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        return await self._generate(
            system_prompt=system_prompt,
            prompt=prompt,
            max_tokens=max_tokens,
            max_input_tokens=self.input_limit(role),
        )

    async def generate_base(self, *, prompt: str, max_tokens: int) -> str:
        return await self._generate(
            system_prompt="Provide concise general reasoning for the requested sub-goal.",
            prompt=prompt,
            max_tokens=max_tokens,
            max_input_tokens=self.max_input_tokens,
        )

    async def _generate(
        self,
        *,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        max_input_tokens: int,
    ) -> str:
        if role_token_count(self.tokenizer, system_prompt, prompt) > max_input_tokens:
            raise PromptBudgetError("prompt_budget_exceeded: frozen role input does not fit")
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=self.temperature,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise InfrastructureError("frozen vLLM generation failed") from exc
        if not isinstance(content, str) or not content.strip():
            raise InfrastructureError("frozen vLLM generation returned empty content")
        return content


class BaseGeneratorGatewayAdapter:
    def __init__(self, gateway: OpenAICompatibleFrozenGateway) -> None:
        self.gateway = gateway
        self.revision = gateway.revision

    async def generate(self, *, prompt: str, max_tokens: int) -> str:
        return await self.gateway.generate_base(prompt=prompt, max_tokens=max_tokens)


class OpenAICompatibleDirectGateway:
    """Direct Qwen benchmark generation through a dedicated Planner vLLM server."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        revision: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 20,
        max_input_tokens: int = 8192,
        enable_thinking: bool = True,
        tokenizer: Any = None,
    ) -> None:
        self.client = client
        self.model = model
        self.revision = revision
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_input_tokens = max_input_tokens
        self.enable_thinking = enable_thinking
        self.tokenizer = tokenizer

    async def generate(self, *, prompt: str, max_tokens: int, seed: int) -> str:
        if chat_token_count(
            self.tokenizer,
            [{"role": "user", "content": prompt}],
            enable_thinking=self.enable_thinking,
        ) > self.max_input_tokens:
            raise PromptBudgetError("prompt_budget_exceeded: direct input does not fit")
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=seed,
                extra_body={
                    "top_k": self.top_k,
                    "chat_template_kwargs": {
                        "enable_thinking": self.enable_thinking
                    },
                },
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise InfrastructureError("direct vLLM generation failed") from exc
        if not isinstance(content, str) or not content.strip():
            raise InfrastructureError("direct vLLM generation returned empty content")
        return content


__all__ = [
    "BaseGeneratorGatewayAdapter",
    "OpenAICompatibleDirectGateway",
    "OpenAICompatibleFrozenGateway",
]
