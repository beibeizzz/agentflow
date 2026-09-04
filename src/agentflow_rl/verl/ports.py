from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class GeneratedPlannerTurn:
    prompt: str
    response: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_logprobs: tuple[float, ...]


def normalize_prompt_ids(value: Any) -> list[int]:
    """Normalize tokenizer list, tensor, and BatchEncoding outputs."""
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(
        value[0], (list, tuple)
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("Planner chat template returned unsupported input_ids")
    return [int(token_id) for token_id in value]


async def generate_planner_turn(
    *,
    server_manager: Any,
    tokenizer: Any,
    request_id: str,
    system_prompt: str,
    prompt: str,
    sampling_params: dict[str, Any],
) -> GeneratedPlannerTurn:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    prompt_ids = normalize_prompt_ids(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))
    if not prompt_ids:
        raise ValueError("Planner chat template produced an empty prompt")
    output = await server_manager.generate(
        request_id=request_id,
        prompt_ids=prompt_ids,
        sampling_params=dict(sampling_params),
    )
    response_ids = tuple(int(value) for value in output.token_ids)
    if not response_ids:
        raise ValueError("Planner rollout generated an empty response")
    logprobs = tuple(float(value) for value in (output.log_probs or ()))
    if len(logprobs) != len(response_ids):
        raise ValueError("Planner rollout log-probabilities do not align with token IDs")
    return GeneratedPlannerTurn(
        prompt=prompt,
        response=tokenizer.decode(response_ids, skip_special_tokens=True),
        prompt_ids=tuple(prompt_ids),
        response_ids=response_ids,
        response_logprobs=logprobs,
    )


class AsyncFrozenModel:
    """OpenAI-compatible async client for non-trainable AgentFlow roles."""

    def __init__(self, client: Any, *, model: str) -> None:
        self.client = client
        self.model = model

    async def generate(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        think_mode: Literal["on", "off"] = "off",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        if max_tokens <= 0 or temperature < 0:
            raise ValueError("invalid frozen generation parameters")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": think_mode == "on"}
            },
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("frozen model returned no text content")
        return content
