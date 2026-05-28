from __future__ import annotations

from functools import wraps
from typing import Any


def _vllm_engine_name(model_name: str) -> str:
    return model_name if model_name.startswith("vllm-") else f"vllm-{model_name}"


def mark_planner_next_step_as_trainable(solver: Any) -> Any:
    original = solver.planner.generate_next_step
    if getattr(original, "_flowgrpo_planner_next_step", False):
        return solver

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)

    try:
        from agentops.sdk.decorators import agent

        wrapped = agent(name="planner_next_step")(wrapped)
    except Exception:
        pass

    setattr(wrapped, "_flowgrpo_planner_next_step", True)
    solver.planner.generate_next_step = wrapped
    return solver


def route_frozen_subagents(
    solver: Any,
    frozen_model_name: str,
    frozen_base_url: str,
    temperature: float = 0.0,
) -> Any:
    from agentflow.engine.factory import create_llm_engine

    frozen_engine_name = _vllm_engine_name(frozen_model_name)
    frozen_engine = create_llm_engine(
        model_string=frozen_engine_name,
        is_multimodal=False,
        base_url=frozen_base_url,
        temperature=temperature,
    )
    solver.planner.llm_engine_fixed = frozen_engine
    solver.verifier.llm_engine_fixed = frozen_engine
    solver.executor.llm_generate_tool_command = frozen_engine
    return solver


def construct_flowgrpo_solver(
    *,
    trainable_model_name: str,
    trainable_base_url: str,
    frozen_model_name: str,
    frozen_base_url: str,
    output_types: str,
    max_steps: int,
    max_time: int,
    max_tokens: int,
    train_temperature: float,
    frozen_temperature: float,
) -> Any:
    from agentflow.solver import construct_solver

    trainable_engine = _vllm_engine_name(trainable_model_name)
    frozen_engine = _vllm_engine_name(frozen_model_name)
    solver = construct_solver(
        llm_engine_name=trainable_engine,
        base_url=trainable_base_url,
        enabled_tools=["Calculator_Tool"],
        tool_engine=["Default"],
        model_engine=["trainable", frozen_engine, frozen_engine, frozen_engine],
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        temperature=train_temperature,
        verbose=False,
    )
    route_frozen_subagents(
        solver,
        frozen_model_name=frozen_model_name,
        frozen_base_url=frozen_base_url,
        temperature=frozen_temperature,
    )
    mark_planner_next_step_as_trainable(solver)
    return solver
