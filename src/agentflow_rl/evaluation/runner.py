from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Any, Iterable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agentflow_rl.runtime.contracts import TaskEnvelope, TaskName
from agentflow_rl.runtime.loop import UnifiedAgentFlowLoop
from agentflow_rl.tasks.contracts import TaskRegistry
from agentflow_rl.integrations.artifacts import execution_metrics


class DirectGenerationGateway(Protocol):
    revision: str

    async def generate(self, *, prompt: str, max_tokens: int, seed: int) -> str: ...


class EvaluationCondition(StrEnum):
    DIRECT = "E0_direct"
    INITIAL_AGENTFLOW = "E1_initial_agentflow"
    TERMINAL_RL = "E2_terminal_rl"
    TERMINAL_PRM_RL = "E3_terminal_prm_rl"
    TERMINAL_JUDGE_RL = "E4_terminal_judge_rl"


ACTIVE_EVALUATION_CONDITIONS = (
    EvaluationCondition.DIRECT,
    EvaluationCondition.INITIAL_AGENTFLOW,
    EvaluationCondition.TERMINAL_RL,
    EvaluationCondition.TERMINAL_PRM_RL,
)


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: EvaluationCondition
    task_id: str = Field(min_length=1)
    task_name: TaskName
    seed: int
    model_revision: str = Field(min_length=1)
    success: bool
    reward: float = Field(ge=0.0, le=1.0)
    valid: bool
    failure_codes: tuple[str, ...] = ()
    terminal_reason: str
    planner_turns: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    planner_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    task_metrics: dict[str, float] = Field(default_factory=dict)
    execution_metrics: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None


async def evaluate_agentflow(
    *,
    loop: UnifiedAgentFlowLoop,
    envelopes: Iterable[TaskEnvelope],
    condition: EvaluationCondition,
    model_revision: str,
    seeds: Iterable[int],
    sampling_params: dict[str, Any],
    max_concurrency: int = 1,
    trajectory_timeout_s: float = 600.0,
) -> list[EvaluationRecord]:
    if max_concurrency <= 0 or trajectory_timeout_s <= 0:
        raise ValueError("evaluation concurrency and timeout must be positive")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def evaluate_one(seed: int, envelope: TaskEnvelope) -> EvaluationRecord:
        async with semaphore:
            started = monotonic()
            try:
                result = await asyncio.wait_for(
                    loop.run(
                        envelope=envelope,
                        trajectory_id=(
                            f"eval:{condition.value}:{envelope.public.task_id}:{seed}"
                        ),
                        session_id=seed,
                        sampling_params={**sampling_params, "seed": seed},
                    ),
                    timeout=trajectory_timeout_s,
                )
            except (TimeoutError, asyncio.TimeoutError):
                return EvaluationRecord(
                    condition=condition,
                    task_id=envelope.public.task_id,
                    task_name=envelope.public.task_name,
                    seed=seed,
                    model_revision=model_revision,
                    success=False,
                    reward=0.0,
                    valid=False,
                    failure_codes=("TRAJECTORY_TIMEOUT",),
                    terminal_reason="trajectory_timeout",
                    planner_turns=0,
                    tool_calls=0,
                    planner_tokens=0,
                    latency_ms=(monotonic() - started) * 1000.0,
                )
            except Exception:
                return EvaluationRecord(
                    condition=condition,
                    task_id=envelope.public.task_id,
                    task_name=envelope.public.task_name,
                    seed=seed,
                    model_revision=model_revision,
                    success=False,
                    reward=0.0,
                    valid=False,
                    failure_codes=("TRAJECTORY_INFRASTRUCTURE_ERROR",),
                    terminal_reason="infrastructure_error",
                    planner_turns=0,
                    tool_calls=0,
                    planner_tokens=0,
                    latency_ms=(monotonic() - started) * 1000.0,
                )
            return EvaluationRecord(
                condition=condition,
                task_id=envelope.public.task_id,
                task_name=envelope.public.task_name,
                seed=seed,
                model_revision=model_revision,
                success=result.verification.success,
                reward=result.verification.reward,
                valid=result.valid_for_training,
                failure_codes=result.verification.failure_codes,
                terminal_reason=result.terminal_reason,
                planner_turns=len(result.turns),
                tool_calls=sum(turn.tool_result is not None for turn in result.turns),
                planner_tokens=sum(
                    len(turn.generation.response_ids) for turn in result.turns
                ),
                latency_ms=(monotonic() - started) * 1000.0,
                task_metrics=result.verification.metrics,
                execution_metrics=execution_metrics(result),
                final_answer=(result.final_answer.answer if result.final_answer else None),
            )
    envelope_values = tuple(envelopes)
    work = [
        evaluate_one(seed, envelope)
        for seed in seeds
        for envelope in envelope_values
    ]
    return list(await asyncio.gather(*work))


async def evaluate_direct(
    *,
    generator: DirectGenerationGateway,
    tasks: TaskRegistry,
    envelopes: Iterable[TaskEnvelope],
    seeds: Iterable[int],
    max_tokens: int,
    max_concurrency: int = 1,
    trajectory_timeout_s: float = 600.0,
) -> list[EvaluationRecord]:
    if max_concurrency <= 0 or trajectory_timeout_s <= 0:
        raise ValueError("evaluation concurrency and timeout must be positive")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def evaluate_one(seed: int, envelope: TaskEnvelope) -> EvaluationRecord:
        async with semaphore:
            adapter = tasks.adapter(envelope.public.task_name)
            evaluator = tasks.evaluator(envelope.public.task_name)
            prompt = (
                f"Task instructions: {adapter.task_instructions}\n\n"
                f"Question:\n{adapter.render_task(envelope.public)}\n\n"
                "Return exactly one JSON object matching this schema:\n"
                f"{adapter.final_answer_instructions}"
            )
            started = monotonic()
            try:
                raw_answer = await asyncio.wait_for(
                    generator.generate(
                        prompt=prompt, max_tokens=max_tokens, seed=seed
                    ),
                    timeout=trajectory_timeout_s,
                )
            except Exception:
                return EvaluationRecord(
                        condition=EvaluationCondition.DIRECT,
                        task_id=envelope.public.task_id,
                        task_name=envelope.public.task_name,
                        seed=seed,
                        model_revision=generator.revision,
                        success=False,
                        reward=0.0,
                        valid=False,
                        failure_codes=("direct_generation_infrastructure_error",),
                        terminal_reason="infrastructure_error",
                        planner_turns=0,
                        tool_calls=0,
                        planner_tokens=0,
                        latency_ms=(monotonic() - started) * 1000.0,
                    )
            try:
                answer = adapter.parse_final_answer(raw_answer)
            except Exception:
                return EvaluationRecord(
                        condition=EvaluationCondition.DIRECT,
                        task_id=envelope.public.task_id,
                        task_name=envelope.public.task_name,
                        seed=seed,
                        model_revision=generator.revision,
                        success=False,
                        reward=0.0,
                        valid=True,
                        failure_codes=("final_answer_parse_error",),
                        terminal_reason="model_parse_error",
                        planner_turns=0,
                        tool_calls=0,
                        planner_tokens=0,
                        latency_ms=(monotonic() - started) * 1000.0,
                        final_answer=raw_answer,
                    )
            try:
                verification = await evaluator.evaluate(answer, envelope.private_ref)
            except Exception:
                return EvaluationRecord(
                        condition=EvaluationCondition.DIRECT,
                        task_id=envelope.public.task_id,
                        task_name=envelope.public.task_name,
                        seed=seed,
                        model_revision=generator.revision,
                        success=False,
                        reward=0.0,
                        valid=False,
                        failure_codes=("terminal_evaluator_infrastructure_error",),
                        terminal_reason="infrastructure_error",
                        planner_turns=0,
                        tool_calls=0,
                        planner_tokens=0,
                        latency_ms=(monotonic() - started) * 1000.0,
                        final_answer=answer.answer,
                    )
            return EvaluationRecord(
                    condition=EvaluationCondition.DIRECT,
                    task_id=envelope.public.task_id,
                    task_name=envelope.public.task_name,
                    seed=seed,
                    model_revision=generator.revision,
                    success=verification.success,
                    reward=verification.reward,
                    valid=True,
                    failure_codes=verification.failure_codes,
                    terminal_reason="direct_generation_complete",
                    planner_turns=0,
                    tool_calls=0,
                    planner_tokens=0,
                    latency_ms=(monotonic() - started) * 1000.0,
                    task_metrics=verification.metrics,
                    final_answer=answer.answer,
                )
    envelope_values = tuple(envelopes)
    work = [
        evaluate_one(seed, envelope)
        for seed in seeds
        for envelope in envelope_values
    ]
    return list(await asyncio.gather(*work))


def write_evaluation(
    records: Iterable[EvaluationRecord],
    *,
    samples_path: str | Path,
    metrics_path: str | Path,
) -> None:
    from .metrics import aggregate_records

    values = list(records)
    sample_target = Path(samples_path)
    sample_target.parent.mkdir(parents=True, exist_ok=True)
    with sample_target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in values:
            handle.write(record.model_dump_json() + "\n")
    metric_target = Path(metrics_path)
    metric_target.parent.mkdir(parents=True, exist_ok=True)
    metric_target.write_text(
        json.dumps(aggregate_records(values), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ACTIVE_EVALUATION_CONDITIONS",
    "EvaluationCondition",
    "EvaluationRecord",
    "evaluate_direct",
    "evaluate_agentflow",
    "write_evaluation",
]
