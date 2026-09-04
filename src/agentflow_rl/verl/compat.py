from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


try:  # pragma: no cover - exercised in the remote veRL environment.
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
except ModuleNotFoundError:  # Lightweight local unit-test compatibility.
    class AgentLoopMetrics(BaseModel):
        generate_sequences: float = 0.0
        tool_calls: float = 0.0
        compute_score: float = 0.0
        num_preempted: int = -1


    class AgentLoopOutput(BaseModel):
        prompt_ids: list[int]
        response_ids: list[int]
        response_mask: list[int]
        response_logprobs: list[float] | None = None
        reward_score: float | None = None
        num_turns: int = 0
        metrics: AgentLoopMetrics
        extra_fields: dict[str, Any] = Field(default_factory=dict)


    class AgentLoopBase:
        def __init__(
            self,
            trainer_config: Any,
            server_manager: Any,
            tokenizer: Any,
            processor: Any,
            dataset_cls: Any,
            data_config: Any,
            **_: Any,
        ) -> None:
            self.config = getattr(trainer_config, "config", trainer_config)
            self.rollout_config = self.config.actor_rollout_ref.rollout
            self.server_manager = server_manager
            self.tokenizer = tokenizer
            self.processor = processor
            self.dataset_cls = dataset_cls
            self.data_config = getattr(data_config, "config", data_config)


    def register(_: str):
        def decorator(cls):
            return cls

        return decorator


__all__ = ["AgentLoopBase", "AgentLoopMetrics", "AgentLoopOutput", "register"]
