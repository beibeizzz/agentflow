from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentflow_rl.roles.schemas import PlannerGeneration
from agentflow_rl.runtime.errors import InfrastructureError


def normalize_token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    while (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("Planner chat template returned unsupported token IDs")
    return tuple(int(token_id) for token_id in value)


class VerlPlannerGateway:
    """Route trainable Planner generations through veRL's vLLM server manager."""

    def __init__(
        self,
        server_manager: Any,
        tokenizer: Any,
        *,
        revision: str,
        max_prompt_tokens: int = 8192,
    ) -> None:
        if max_prompt_tokens <= 0:
            raise ValueError("Planner prompt budget must be positive")
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.revision = revision
        self.max_prompt_tokens = max_prompt_tokens

    async def generate(
        self,
        *,
        request_id: str,
        system_prompt: str,
        prompt: str,
        sampling_params: dict[str, Any],
    ) -> PlannerGeneration:
        request_sampling_params = dict(sampling_params)
        expected_update_id = request_sampling_params.pop(
            "agentflow_expected_policy_update_id", None
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_ids = normalize_token_ids(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        if not prompt_ids:
            raise InfrastructureError("Planner prompt tokenization returned no tokens")
        if len(prompt_ids) > self.max_prompt_tokens:
            raise InfrastructureError(
                f"Planner prompt has {len(prompt_ids)} tokens; budget is {self.max_prompt_tokens}"
            )
        try:
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=list(prompt_ids),
                sampling_params=request_sampling_params,
            )
            response_ids = tuple(int(value) for value in output.token_ids)
            logprobs = tuple(float(value) for value in (output.log_probs or ()))
        except Exception as exc:
            raise InfrastructureError("veRL Planner rollout failed") from exc
        if not response_ids or len(response_ids) != len(logprobs):
            raise InfrastructureError("Planner tokens and log-probabilities do not align")
        extra_fields = getattr(output, "extra_fields", None)
        if not isinstance(extra_fields, Mapping) or extra_fields.get("global_steps") is None:
            raise InfrastructureError("veRL rollout server returned no loaded policy update ID")
        policy_update_id = str(extra_fields["global_steps"])
        if expected_update_id is not None and policy_update_id != str(expected_update_id):
            raise InfrastructureError("veRL rollout server policy update ID is stale")
        return PlannerGeneration(
            prompt=prompt,
            response=self.tokenizer.decode(response_ids, skip_special_tokens=True),
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=logprobs,
            model_revision=f"{self.revision}:update-{policy_update_id}",
            policy_update_id=policy_update_id,
        )


class OpenAICompatiblePlannerGateway:
    """Evaluation-only Planner gateway for a standalone vLLM server."""

    def __init__(
        self,
        client: Any,
        tokenizer: Any,
        *,
        model: str,
        revision: str,
        max_prompt_tokens: int = 8192,
    ) -> None:
        self.client = client
        self.tokenizer = tokenizer
        self.model = model
        self.revision = revision
        self.max_prompt_tokens = max_prompt_tokens

    async def generate(
        self,
        *,
        request_id: str,
        system_prompt: str,
        prompt: str,
        sampling_params: dict[str, Any],
    ) -> PlannerGeneration:
        request_sampling_params = dict(sampling_params)
        request_sampling_params.pop("agentflow_expected_policy_update_id", None)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_ids = normalize_token_ids(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        if len(prompt_ids) > self.max_prompt_tokens:
            raise InfrastructureError(
                f"Planner prompt has {len(prompt_ids)} tokens; budget is {self.max_prompt_tokens}"
            )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=int(request_sampling_params.get("max_tokens", 2048)),
                temperature=float(request_sampling_params.get("temperature", 1.0)),
                top_p=float(request_sampling_params.get("top_p", 1.0)),
                seed=request_sampling_params.get("seed"),
                extra_body={
                    "top_k": int(request_sampling_params.get("top_k", 20)),
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise InfrastructureError("evaluation Planner vLLM generation failed") from exc
        if not isinstance(content, str) or not content.strip():
            raise InfrastructureError("evaluation Planner returned empty content")
        response_ids = tuple(
            int(value)
            for value in self.tokenizer.encode(content, add_special_tokens=False)
        )
        if not response_ids:
            raise InfrastructureError("evaluation Planner response tokenization was empty")
        return PlannerGeneration(
            prompt=prompt,
            response=content,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=(0.0,) * len(response_ids),
            model_revision=self.revision,
            policy_update_id=self.revision,
        )


__all__ = [
    "OpenAICompatiblePlannerGateway",
    "VerlPlannerGateway",
    "normalize_token_ids",
]
