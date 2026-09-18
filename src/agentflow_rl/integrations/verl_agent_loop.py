from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from agentflow_rl.rewards.process import ProcessScorer, build_process_transition
from agentflow_rl.runtime.contracts import (
    PrivateEvaluationRecordRef,
    PublicTaskRecord,
    TaskEnvelope,
    TaskName,
)
from agentflow_rl.runtime.loop import UnifiedAgentFlowLoop

from .planner import VerlPlannerGateway
from .policy_identity import read_policy_update_id
from .artifacts import TrajectoryArtifactWriter
from .verl_compat import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register


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


@dataclass(frozen=True)
class RuntimeBundle:
    loop: UnifiedAgentFlowLoop
    process_scorer: ProcessScorer | None = None
    trajectory_writer: TrajectoryArtifactWriter | None = None


async def score_process_transitions(
    scorer: ProcessScorer,
    transitions: tuple[Any, ...],
) -> tuple[dict[int, float], dict[int, str]]:
    """Score turns independently while retaining successful process rewards."""
    if not transitions:
        return {}, {}
    values: list[Any]
    batch_scorer = getattr(scorer, "score_many", None)
    if batch_scorer is not None:
        try:
            batch_values = await batch_scorer(transitions)
            if len(batch_values) != len(transitions):
                raise ValueError("process score count does not match transitions")
            values = list(batch_values)
        except Exception:
            values = list(
                await asyncio.gather(
                    *(scorer.score(transition) for transition in transitions),
                    return_exceptions=True,
                )
            )
    else:
        values = list(
            await asyncio.gather(
                *(scorer.score(transition) for transition in transitions),
                return_exceptions=True,
            )
        )
    scores: dict[int, float] = {}
    failures: dict[int, str] = {}
    for transition, value in zip(transitions, values, strict=True):
        if isinstance(value, BaseException):
            failures[transition.turn_index] = type(value).__name__
        else:
            scores[transition.turn_index] = float(value.score)
    return scores, failures


def task_envelope_from_extra_info(extra_info: dict[str, Any]) -> TaskEnvelope:
    public_value = extra_info["public_task"]
    reference_value = extra_info["private_ref"]
    public = (
        PublicTaskRecord.model_validate_json(public_value)
        if isinstance(public_value, str)
        else PublicTaskRecord.model_validate(public_value)
    )
    private_ref = (
        PrivateEvaluationRecordRef.model_validate_json(reference_value)
        if isinstance(reference_value, str)
        else PrivateEvaluationRecordRef.model_validate(reference_value)
    )
    return TaskEnvelope(public=public, private_ref=private_ref)


@register("agentflow_unified")
class UnifiedVerlAgentLoop(AgentLoopBase):
    def __init__(
        self,
        *args: Any,
        runtime_bundle: RuntimeBundle | None = None,
        **kwargs: Any,
    ) -> None:
        if args:
            positional = list(args)
            if not hasattr(positional[0], "config"):
                positional[0] = SimpleNamespace(config=positional[0])
            args = tuple(positional)
        elif "trainer_config" in kwargs and not hasattr(kwargs["trainer_config"], "config"):
            kwargs["trainer_config"] = SimpleNamespace(config=kwargs["trainer_config"])
        super().__init__(*args, **kwargs)
        if runtime_bundle is None:
            from .bootstrap import build_runtime_bundle

            planner = VerlPlannerGateway(
                self.server_manager,
                self.tokenizer,
                revision=str(
                    config_value(self.config, "agentflow.planner_revision", "rollout-policy")
                ),
                max_prompt_tokens=int(
                    config_value(self.config, "data.max_prompt_length", 8192)
                ),
            )
            runtime_bundle = build_runtime_bundle(self.config, planner=planner)
        self.runtime_bundle = runtime_bundle

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> list[AgentLoopOutput]:
        uid = str(kwargs["uid"])
        session_id = int(kwargs.get("session_id", 0))
        envelope = task_envelope_from_extra_info(dict(kwargs["extra_info"]))
        trajectory_id = f"{uid}:{session_id}"
        rollout_sampling_params = dict(sampling_params)
        rollout_step = kwargs.get("global_steps")
        policy_identity_path = config_value(
            self.config, "agentflow.policy_identity_path", None
        )
        if policy_identity_path:
            rollout_sampling_params["agentflow_expected_policy_update_id"] = (
                read_policy_update_id(str(policy_identity_path))
            )
        timeout_s = float(config_value(self.config, "agentflow.trajectory_timeout_s", 600.0))
        try:
            result = await asyncio.wait_for(
                self.runtime_bundle.loop.run(
                    envelope=envelope,
                    trajectory_id=trajectory_id,
                    session_id=session_id,
                    sampling_params=rollout_sampling_params,
                ),
                timeout=timeout_s,
            )
        except (TimeoutError, asyncio.TimeoutError):
            if self.runtime_bundle.trajectory_writer is not None:
                await asyncio.to_thread(
                    self.runtime_bundle.trajectory_writer.write_timeout,
                    task=envelope.public,
                    trajectory_id=trajectory_id,
                    session_id=session_id,
                )
            token_id = int(
                getattr(self.tokenizer, "eos_token_id", None)
                or getattr(self.tokenizer, "pad_token_id", None)
                or 0
            )
            return [
                AgentLoopOutput(
                    prompt_ids=[token_id],
                    response_ids=[token_id],
                    response_mask=[0],
                    response_logprobs=[0.0],
                    reward_score=0.0,
                    metrics=AgentLoopMetrics(),
                    extra_fields={
                        "uid": uid,
                        "task_id": envelope.public.task_id,
                        "task_name": envelope.public.task_name.value,
                        "trajectory_id": trajectory_id,
                        "session_id": session_id,
                        "turn_index": 0,
                        "terminal_reward": 0.0,
                        "process_score": None,
                        "process_score_failure": None,
                        "valid_for_training": False,
                        "terminal_reason": "trajectory_timeout",
                        "synthetic_diagnostic": True,
                        "failure_code": "TRAJECTORY_TIMEOUT",
                        "reward_extra_info": {
                            "success": 0.0,
                            "valid_for_training": 0.0,
                            "turns": 0.0,
                            "process_score_available": 0.0,
                            "process_score_failed": 0.0,
                        },
                    },
                )
            ]

        process_scores: dict[int, float] = {}
        process_failures: dict[int, str] = {}
        transitions = tuple(
            build_process_transition(
                task=envelope.public,
                trajectory_id=trajectory_id,
                turn=turn,
                max_turns=self.runtime_bundle.loop.max_turns,
            )
            for turn in result.turns
        )
        if self.runtime_bundle.trajectory_writer is not None:
            await asyncio.to_thread(
                self.runtime_bundle.trajectory_writer.write,
                task=envelope.public,
                result=result,
                transitions=transitions,
            )
        if result.valid_for_training and self.runtime_bundle.process_scorer is not None:
            process_scores, process_failures = await score_process_transitions(
                self.runtime_bundle.process_scorer,
                transitions,
            )

        outputs = []
        for turn in result.turns:
            generation = turn.generation
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=list(generation.prompt_ids),
                    response_ids=list(generation.response_ids),
                    response_mask=[1] * len(generation.response_ids),
                    response_logprobs=list(generation.response_logprobs),
                    reward_score=(
                        result.verification.reward
                        if turn.turn_index == result.turns[-1].turn_index
                        else None
                    ),
                    num_turns=1,
                    metrics=AgentLoopMetrics(
                        generate_sequences=1.0,
                        tool_calls=float(turn.tool_result is not None),
                    ),
                    extra_fields={
                        "uid": uid,
                        "task_id": envelope.public.task_id,
                        "task_name": envelope.public.task_name.value,
                        "trajectory_id": trajectory_id,
                        "session_id": session_id,
                        "turn_index": turn.turn_index,
                        "terminal_reward": result.verification.reward,
                        "process_score": process_scores.get(turn.turn_index),
                        "process_score_failure": process_failures.get(turn.turn_index),
                        "valid_for_training": result.valid_for_training,
                        "terminal_reason": result.terminal_reason,
                        "rollout_policy_revision": generation.model_revision,
                        "rollout_policy_update_id": generation.policy_update_id,
                        "rollout_global_step": (
                            None if rollout_step is None else int(rollout_step)
                        ),
                        "verification": result.verification.model_dump(mode="json"),
                        "reward_extra_info": {
                            "success": float(result.verification.success),
                            "valid_for_training": float(result.valid_for_training),
                            "turns": float(len(result.turns)),
                            "process_score_available": float(
                                turn.turn_index in process_scores
                            ),
                            "process_score_failed": float(
                                turn.turn_index in process_failures
                            ),
                        },
                    },
                )
            )
        if outputs:
            return outputs

        token_id = int(
            getattr(self.tokenizer, "eos_token_id", None)
            or getattr(self.tokenizer, "pad_token_id", None)
            or 0
        )
        return [
            AgentLoopOutput(
                prompt_ids=[token_id],
                response_ids=[token_id],
                response_mask=[0],
                response_logprobs=[0.0],
                reward_score=0.0,
                metrics=AgentLoopMetrics(),
                extra_fields={
                    "uid": uid,
                    "task_id": envelope.public.task_id,
                    "task_name": envelope.public.task_name.value,
                    "trajectory_id": trajectory_id,
                    "session_id": session_id,
                    "turn_index": 0,
                    "terminal_reward": 0.0,
                    "process_score": None,
                    "valid_for_training": False,
                    "terminal_reason": result.terminal_reason,
                    "synthetic_diagnostic": True,
                    "reward_extra_info": {
                        "success": 0.0,
                        "valid_for_training": 0.0,
                        "turns": 0.0,
                        "process_score_available": 0.0,
                    },
                },
            )
        ]


__all__ = [
    "RuntimeBundle",
    "UnifiedVerlAgentLoop",
    "config_value",
    "score_process_transitions",
    "task_envelope_from_extra_info",
]
