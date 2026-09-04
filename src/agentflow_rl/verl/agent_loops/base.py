from __future__ import annotations

import asyncio
from types import SimpleNamespace
from time import monotonic
from typing import Any

from agentflow_rl.runtime.memory import MemoryStore
from agentflow_rl.verl.compat import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from agentflow_rl.verl.ports import AsyncFrozenModel, GeneratedPlannerTurn, generate_planner_turn


def config_value(root: Any, path: str, default: Any = None) -> Any:
    value = root
    for name in path.split("."):
        if isinstance(value, dict):
            if name not in value:
                return default
            value = value[name]
        else:
            value = getattr(value, name, default)
        if value is default:
            return default
    return value


def bounded_memory_text(
    memory: MemoryStore,
    *,
    token_counter: Any,
    max_prompt_tokens: int,
    max_memory_tokens: int,
    reserve_tokens: int,
    reserved_texts: tuple[str, ...],
    required_tags: tuple[str, ...] = ("identity",),
) -> str:
    """Project memory into the space left by role-specific prompt content."""
    available = max_prompt_tokens - reserve_tokens - sum(
        token_counter(text) for text in reserved_texts if text
    )
    if available <= 0:
        return ""
    return memory.project(
        max_tokens=min(max_memory_tokens, available),
        token_counter=token_counter,
        required_tags=required_tags,
    ).text


class AgentFlowLoopBase(AgentLoopBase):
    def __init__(self, *args: Any, frozen_model: Any | None = None, **kwargs: Any) -> None:
        if args:
            positional = list(args)
            if not hasattr(positional[0], "config"):
                positional[0] = SimpleNamespace(config=positional[0])
            args = tuple(positional)
        elif "trainer_config" in kwargs and not hasattr(kwargs["trainer_config"], "config"):
            kwargs["trainer_config"] = SimpleNamespace(config=kwargs["trainer_config"])
        super().__init__(*args, **kwargs)
        if frozen_model is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                base_url=str(config_value(self.config, "agentflow.frozen_base_url")),
                api_key="not-required",
            )
            frozen_model = AsyncFrozenModel(
                client,
                model=str(config_value(self.config, "agentflow.frozen_model")),
            )
        self.frozen_model = frozen_model

    async def frozen_generate(
        self,
        *,
        deadline: float,
        prompt: str,
        system_prompt: str,
        think_mode: str,
        max_tokens: int,
    ) -> str:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("AgentFlow trajectory deadline exceeded")
        try:
            return await asyncio.wait_for(
                self.frozen_model.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    think_mode=think_mode,
                    max_tokens=max_tokens,
                    temperature=0.0,
                ),
                timeout=remaining,
            )
        except TimeoutError:
            raise
        except Exception as exc:
            raise RuntimeError("frozen role generation failed") from exc

    async def planner_generate(
        self,
        *,
        uid: str,
        session_id: int,
        turn_index: int,
        system_prompt: str,
        prompt: str,
        sampling_params: dict[str, Any],
        deadline: float | None = None,
    ) -> GeneratedPlannerTurn:
        generation = generate_planner_turn(
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            request_id=f"{uid}-{session_id}-{turn_index}",
            system_prompt=system_prompt,
            prompt=prompt,
            sampling_params=sampling_params,
        )
        try:
            if deadline is None:
                return await generation
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("AgentFlow trajectory deadline exceeded")
            return await asyncio.wait_for(generation, timeout=remaining)
        except TimeoutError:
            raise
        except Exception as exc:
            raise RuntimeError("Planner rollout generation failed") from exc

    @staticmethod
    async def run_blocking(*, deadline: float, operation: Any, args: tuple[Any, ...]) -> Any:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("AgentFlow trajectory deadline exceeded")
        return await asyncio.wait_for(
            asyncio.to_thread(operation, *args),
            timeout=remaining,
        )

    @staticmethod
    def planner_output(
        turn: GeneratedPlannerTurn,
        *,
        uid: str,
        session_id: int,
        turn_index: int,
        extra_fields: dict[str, Any] | None = None,
    ) -> AgentLoopOutput:
        metadata = {
            "uid": str(uid),
            "session_id": int(session_id),
            "turn_index": int(turn_index),
            "valid_for_training": True,
            **(extra_fields or {}),
        }
        return AgentLoopOutput(
            prompt_ids=list(turn.prompt_ids),
            response_ids=list(turn.response_ids),
            response_mask=[1] * len(turn.response_ids),
            response_logprobs=list(turn.response_logprobs),
            num_turns=1,
            metrics=AgentLoopMetrics(generate_sequences=1.0),
            extra_fields=metadata,
        )

    def finalize_outputs(
        self,
        outputs: list[AgentLoopOutput],
        *,
        reward: float,
        valid_for_training: bool,
        terminal_reason: str,
        final_fields: dict[str, Any] | None = None,
    ) -> list[AgentLoopOutput]:
        if not outputs:
            token_id = int(
                getattr(self.tokenizer, "eos_token_id", None)
                or getattr(self.tokenizer, "pad_token_id", None)
                or 0
            )
            outputs.append(AgentLoopOutput(
                prompt_ids=[token_id],
                response_ids=[token_id],
                response_mask=[0],
                response_logprobs=[0.0],
                num_turns=0,
                metrics=AgentLoopMetrics(),
                extra_fields={"synthetic_diagnostic": True},
            ))
            valid_for_training = False
        for output in outputs:
            output.extra_fields["valid_for_training"] = bool(valid_for_training)
            output.extra_fields["terminal_reason"] = terminal_reason
        if outputs:
            outputs[-1].reward_score = float(reward)
            outputs[-1].extra_fields.update(final_fields or {})
            outputs[-1].extra_fields.setdefault("reward_extra_info", {})
        return outputs
