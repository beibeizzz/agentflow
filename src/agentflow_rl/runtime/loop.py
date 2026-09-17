from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from collections.abc import Mapping
from typing import Any

from agentflow_rl.backends.query_analyzer_cache import QueryAnalyzerCache
from agentflow_rl.roles.gateway import FrozenRoleGateway, PlannerGateway
from agentflow_rl.roles.prompts import RolePromptRenderer
from agentflow_rl.roles.schemas import PlannerGeneration, RoleName
from agentflow_rl.tasks.contracts import TaskRegistry, VerificationResult
from agentflow_rl.tools.contracts import ToolRequest, ToolResult, ToolStatus
from agentflow_rl.tools.registry import ToolRegistry

from .contracts import (
    FinalAnswerEnvelope,
    ExecutedToolCall,
    MemoryAudience,
    MemoryEvent,
    MemoryVisibility,
    PlannerAction,
    TaskEnvelope,
    TrajectoryIdentity,
    VerifierDecision,
    VerifierOutcome,
)
from .errors import ActionParseError, InfrastructureError, PromptBudgetError
from .memory import MemoryStore
from .observations import planner_core_history
from .projections import RoleMemoryProjector
from .prompt_budget import role_token_count


@dataclass(frozen=True)
class PlannerTurnRecord:
    turn_index: int
    generation: PlannerGeneration
    memory_before_action: tuple[dict[str, Any], ...]
    action: PlannerAction | None
    tool_request: ToolRequest | None
    tool_result: ToolResult | None
    verifier_decision: VerifierDecision
    planner_memory_view: str | None = None
    planner_memory_event_ids: tuple[str, ...] = ()
    planner_memory_compacted_event_ids: tuple[str, ...] = ()
    planner_memory_core: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TrajectoryResult:
    identity: TrajectoryIdentity
    turns: tuple[PlannerTurnRecord, ...]
    final_answer: FinalAnswerEnvelope | None
    verification: VerificationResult
    terminal_reason: str
    valid_for_training: bool
    memory: tuple[dict[str, Any], ...]


class UnifiedAgentFlowLoop:
    def __init__(
        self,
        *,
        planner: PlannerGateway,
        frozen_roles: FrozenRoleGateway,
        tools: ToolRegistry,
        tasks: TaskRegistry,
        prompts: RolePromptRenderer | None = None,
        projector: RoleMemoryProjector | None = None,
        max_turns: int = 5,
        max_role_tokens: int | Mapping[str | RoleName, int] = 2048,
        run_query_analyzer: bool = True,
        query_analyzer_cache: QueryAnalyzerCache | None = None,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("AgentFlow budgets must be positive")
        tools.assert_complete()
        self.planner = planner
        self.frozen_roles = frozen_roles
        self.tools = tools
        self.tasks = tasks
        self.prompts = prompts or RolePromptRenderer(tools)
        self.projector = projector or RoleMemoryProjector()
        self.max_turns = max_turns
        self.max_role_tokens = self._normalize_role_tokens(max_role_tokens)
        self.run_query_analyzer = run_query_analyzer
        self.query_analyzer_cache = query_analyzer_cache

    @staticmethod
    def _normalize_role_tokens(
        value: int | Mapping[str | RoleName, int],
    ) -> dict[RoleName, int]:
        roles = (
            RoleName.QUERY_ANALYZER,
            RoleName.EXECUTOR,
            RoleName.VERIFIER,
            RoleName.GENERATOR,
        )
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("AgentFlow role output budgets must be positive")
            return {role: value for role in roles}
        normalized: dict[RoleName, int] = {}
        for role in roles:
            raw = value.get(role, value.get(role.value))
            if raw is None or int(raw) <= 0:
                raise ValueError(f"missing positive output budget for {role.value}")
            normalized[role] = int(raw)
        return normalized

    async def run(
        self,
        *,
        envelope: TaskEnvelope,
        trajectory_id: str,
        session_id: int,
        sampling_params: dict[str, Any],
    ) -> TrajectoryResult:
        record = envelope.public
        identity = TrajectoryIdentity(
            task_id=record.task_id,
            trajectory_id=trajectory_id,
            session_id=session_id,
        )
        adapter = self.tasks.adapter(record.task_name)
        evaluator = self.tasks.evaluator(record.task_name)
        memory = MemoryStore()
        turns: list[PlannerTurnRecord] = []
        valid_for_training = True
        terminal_reason = "turn_limit"
        final_answer: FinalAnswerEnvelope | None = None

        memory.append(self._event(
            identity,
            event_id="task",
            turn_index=-1,
            role="user",
            kind="task",
            content={
                "task_name": record.task_name.value,
                "prompt": record.prompt,
                "public_payload": record.public_payload,
            },
            tags=("identity",),
        ))

        try:
            if self.run_query_analyzer:
                analysis = await self._generate_query_analysis(
                    memory,
                    prompt=self.prompts.analyzer(record, adapter.task_instructions),
                )
                memory.append(self._event(
                    identity,
                    event_id="analysis",
                    turn_index=-1,
                    role=RoleName.QUERY_ANALYZER.value,
                    kind="analysis",
                    content=analysis,
                    tags=("analysis",),
                    model_revision=self.frozen_roles.revision,
                ))

            for turn_index in range(self.max_turns):
                memory_before = tuple(memory.snapshot(audience=MemoryAudience.SCORER))
                planner_prompt, planner_view = self._make_prompt(
                    memory, RoleName.PLANNER, self.prompts.planner_system,
                    lambda view: self.prompts.planner(record, adapter.task_instructions, view,
                                                     turn_index=turn_index, max_turns=self.max_turns),
                )
                generation = await self.planner.generate(
                    request_id=f"{trajectory_id}-planner-{turn_index}",
                    system_prompt=self.prompts.planner_system,
                    prompt=planner_prompt,
                    sampling_params=sampling_params,
                )
                action: PlannerAction | None = None
                request: ToolRequest | None = None
                tool_result: ToolResult | None = None
                try:
                    action = PlannerAction.parse(generation.response)
                    self.tools.validate_action(action)
                    memory.append(self._event(
                        identity,
                        event_id=f"turn-{turn_index}-planner",
                        turn_index=turn_index,
                        role=RoleName.PLANNER.value,
                        kind="action",
                        content=action.model_dump(mode="json"),
                        tags=("planner_action",),
                        model_revision=generation.model_revision,
                    ))
                    executor_prompt, _ = self._make_prompt(
                        memory, RoleName.EXECUTOR, self.prompts.executor_system,
                        lambda view: self.prompts.executor(record, adapter.task_instructions, action, view),
                    )
                    executor_text = await self._generate_frozen(memory,
                        role=RoleName.EXECUTOR,
                        system_prompt=self.prompts.executor_system,
                        prompt=executor_prompt,
                        max_tokens=self.max_role_tokens[RoleName.EXECUTOR],
                    )
                    executed_action = ExecutedToolCall.parse(executor_text)
                    if executed_action.tool_name != action.tool_name:
                        raise ActionParseError("Executor changed the selected tool")
                    request = ToolRequest(
                        request_id=f"{trajectory_id}-tool-{turn_index}",
                        trajectory_id=trajectory_id,
                        task_id=record.task_id,
                        turn_index=turn_index,
                        tool_name=executed_action.tool_name,
                        arguments=executed_action.arguments,
                    )
                    try:
                        request = self.tools.normalize_request(request)
                    except ValueError:
                        pass  # dispatch returns the common structured parameter failure
                    memory.append(self._event(
                        identity, event_id=f"turn-{turn_index}-executed-action", turn_index=turn_index,
                        role=RoleName.EXECUTOR.value, kind="executed_action",
                        content={**executed_action.model_dump(mode="json"), "arguments": request.arguments,
                                 "sub_goal": action.sub_goal},
                        tags=("executed_action",), model_revision=self.frozen_roles.revision,
                    ))
                    tool_result = await self.tools.dispatch(request)
                    memory.append(self._event(
                        identity,
                        event_id=f"turn-{turn_index}-tool",
                        turn_index=turn_index,
                        role=RoleName.EXECUTOR.value,
                        kind="tool_result",
                        content=tool_result.model_dump(mode="json"),
                        tags=("tool_result", "latest_result"),
                        tool_revision=tool_result.backend_revision,
                        latency_ms=tool_result.latency_ms,
                        status=tool_result.status.value,
                        failure_code=(
                            tool_result.failure_code.value
                            if tool_result.failure_code is not None
                            else None
                        ),
                    ))
                    if tool_result.status is ToolStatus.INFRASTRUCTURE_ERROR:
                        valid_for_training = False
                        terminal_reason = "infrastructure_failure"
                except ActionParseError as exc:
                    memory.append(self._event(
                        identity,
                        event_id=f"turn-{turn_index}-action-error",
                        turn_index=turn_index,
                        role=RoleName.EXECUTOR.value,
                        kind="action_error",
                        content={"failure_code": "INVALID_ACTION", "message": str(exc)},
                        tags=("tool_result", "latest_result"),
                        status="model_error",
                        failure_code="INVALID_ACTION",
                    ))

                verifier_prompt, verifier_view = self._make_prompt(
                    memory, RoleName.VERIFIER, self.prompts.verifier_system,
                    lambda view: self.prompts.verifier(record, adapter.task_instructions, view),
                )
                verifier_text = await self._generate_frozen(memory,
                    role=RoleName.VERIFIER,
                    system_prompt=self.prompts.verifier_system,
                    prompt=verifier_prompt,
                    max_tokens=self.max_role_tokens[RoleName.VERIFIER],
                )
                try:
                    verifier = VerifierDecision.parse(verifier_text)
                    visible_evidence = {event.event_id for event in memory.events
                                        if event.kind == "tool_result" and event.turn_index <= turn_index
                                        and event.event_id in verifier_view.included_event_ids}
                    if not set(verifier.evidence_ids) <= visible_evidence:
                        raise ActionParseError("Verifier cited unavailable evidence")
                except ActionParseError as exc:
                    verifier = VerifierDecision(
                        outcome=VerifierOutcome.CONTINUE,
                        rationale="Verifier output failed schema validation.",
                        failure_codes=("INVALID_VERIFIER_OUTPUT",),
                    )
                    verifier_text = f"{verifier_text}\n\nParser error: {exc}"
                memory.append(self._event(
                    identity,
                    event_id=f"turn-{turn_index}-verifier",
                    turn_index=turn_index,
                    role=RoleName.VERIFIER.value,
                    kind="verifier_decision",
                    content=verifier.model_dump(mode="json"),
                    tags=("verifier_decision", "latest_judgement"),
                    model_revision=self.frozen_roles.revision,
                ))
                turns.append(PlannerTurnRecord(
                    turn_index=turn_index,
                    generation=generation,
                    memory_before_action=memory_before,
                    action=action,
                    tool_request=request,
                    tool_result=tool_result,
                    verifier_decision=verifier,
                    planner_memory_view=planner_view.text,
                    planner_memory_event_ids=planner_view.included_event_ids,
                    planner_memory_compacted_event_ids=planner_view.compacted_event_ids,
                    planner_memory_core=planner_core_history(
                        memory_before,
                        included_event_ids=planner_view.included_event_ids,
                        compacted_event_ids=planner_view.compacted_event_ids,
                    ),
                ))
                if not valid_for_training:
                    break
                if verifier.outcome is VerifierOutcome.FINISH:
                    terminal_reason = "verifier_finish"
                    break

            if valid_for_training:
                generator_prompt, _ = self._make_prompt(
                    memory, RoleName.GENERATOR, self.prompts.generator_system,
                    lambda view: self.prompts.generator(record, adapter.task_instructions,
                                                       adapter.final_answer_instructions, view),
                )
                final_text = await self._generate_frozen(memory,
                    role=RoleName.GENERATOR,
                    system_prompt=self.prompts.generator_system,
                    prompt=generator_prompt,
                    max_tokens=self.max_role_tokens[RoleName.GENERATOR],
                )
                try:
                    final_answer = adapter.parse_final_answer(final_text)
                    verification = await evaluator.evaluate(final_answer, envelope.private_ref)
                except ActionParseError:
                    verification = VerificationResult(
                        success=False,
                        reward=0.0,
                        failure_codes=("INVALID_FINAL_OUTPUT",),
                    )
                memory.append(self._event(
                    identity,
                    event_id="final-answer",
                    turn_index=max(0, len(turns) - 1),
                    role=RoleName.GENERATOR.value,
                    kind="final_answer",
                    content={"text": final_text},
                    tags=("final_answer",),
                    model_revision=self.frozen_roles.revision,
                ))
            else:
                verification = VerificationResult(
                    success=False,
                    reward=0.0,
                    failure_codes=("INFRASTRUCTURE_INVALID",),
                )
        except (InfrastructureError, TimeoutError, OSError, RuntimeError) as exc:
            valid_for_training = False
            terminal_reason = "infrastructure_failure"
            if isinstance(exc, PromptBudgetError):
                terminal_reason = "prompt_budget_exceeded"
            verification = VerificationResult(
                success=False,
                reward=0.0,
                failure_codes=("PROMPT_BUDGET_EXCEEDED" if isinstance(exc, PromptBudgetError) else "INFRASTRUCTURE_INVALID",),
            )

        return TrajectoryResult(
            identity=identity,
            turns=tuple(turns),
            final_answer=final_answer,
            verification=verification,
            terminal_reason=terminal_reason,
            valid_for_training=valid_for_training,
            memory=tuple(memory.snapshot(audience=MemoryAudience.AUDIT)),
        )

    async def _generate_query_analysis(self, memory, *, prompt: str) -> str:
        role = RoleName.QUERY_ANALYZER
        system_prompt = self.prompts.analyzer_system
        max_tokens = self.max_role_tokens[role]
        if self.query_analyzer_cache is None:
            return await self._generate_frozen(
                memory,
                role=role,
                system_prompt=system_prompt,
                prompt=prompt,
                max_tokens=max_tokens,
            )

        started = monotonic()
        response = None
        status = "failed"
        cache_key = self.query_analyzer_cache.key_for(
            model_revision=self.frozen_roles.revision,
            prompt_revision=self.prompts.revision,
            system_prompt=system_prompt,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        lookup = None
        try:
            lookup = await self.query_analyzer_cache.get_or_compute(
                cache_key=cache_key,
                compute=lambda: self.frozen_roles.generate(
                    role=role,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    max_tokens=max_tokens,
                ),
            )
            response = lookup.text
            status = "cache_hit" if lookup.cache_hit else "cache_miss"
            return response
        finally:
            tokenizer = getattr(self.frozen_roles, "tokenizer", None)
            events = memory.events
            memory.append(MemoryEvent(
                event_id=f"model-call-{len(events)}",
                trajectory_id=events[0].trajectory_id,
                task_id=events[0].task_id,
                turn_index=events[-1].turn_index,
                role=role.value,
                kind="model_call",
                visibility=MemoryVisibility.AUDIT,
                content={
                    "status": status,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "response": response,
                    "cache_key": cache_key,
                    "cache_hit": None if lookup is None else lookup.cache_hit,
                    "cache_queue_wait_ms": (
                        None if lookup is None else lookup.queue_wait_ms
                    ),
                    "token_count_method": "local_tokenizer" if tokenizer else "unavailable",
                    "input_tokens": role_token_count(tokenizer, system_prompt, prompt) if tokenizer else None,
                    "output_tokens": len(tokenizer.encode(response, add_special_tokens=False)) if tokenizer and response is not None else None,
                },
                latency_ms=(monotonic() - started) * 1000,
                model_revision=self.frozen_roles.revision,
                prompt_revision=self.prompts.revision,
            ))

    async def _generate_frozen(self, memory, **kwargs):
        started = monotonic()
        response = None
        status = "failed"
        tokenizer = getattr(self.frozen_roles, "tokenizer", None)
        try:
            response = await self.frozen_roles.generate(**kwargs)
            status = "success"
            return response
        finally:
            events = memory.events
            memory.append(MemoryEvent(
                event_id=f"model-call-{len(events)}", trajectory_id=events[0].trajectory_id,
                task_id=events[0].task_id, turn_index=events[-1].turn_index,
                role=kwargs["role"].value, kind="model_call", visibility=MemoryVisibility.AUDIT,
                content={"status": status, "system_prompt": kwargs["system_prompt"],
                         "prompt": kwargs["prompt"], "response": response,
                         "token_count_method": "local_tokenizer" if tokenizer else "unavailable",
                         "input_tokens": role_token_count(tokenizer, kwargs["system_prompt"], kwargs["prompt"]) if tokenizer else None,
                         "output_tokens": len(tokenizer.encode(response, add_special_tokens=False)) if tokenizer and response is not None else None},
                latency_ms=(monotonic() - started) * 1000,
                model_revision=self.frozen_roles.revision, prompt_revision=self.prompts.revision,
            ))

    def _make_prompt(self, memory, role, system, render):
        gateway = self.planner if role is RoleName.PLANNER else self.frozen_roles
        tokenizer = getattr(gateway, "tokenizer", None)
        role_input_limit = getattr(gateway, "input_limit", None)
        limit = (
            role_input_limit(role)
            if callable(role_input_limit)
            else getattr(
                gateway,
                "max_prompt_tokens",
                getattr(gateway, "max_input_tokens", 8192),
            )
        )
        count = lambda text: role_token_count(tokenizer, system, render(text))
        try:
            view = self.projector.project(
                memory, role, token_counter=count, omit_task=True,
                max_tokens=min(limit, count("") + self.projector.specs[role].max_tokens),
            )
        except ValueError as exc:
            raise PromptBudgetError("prompt_budget_exceeded: required role context does not fit") from exc
        prompt = render(view.text)
        if role_token_count(tokenizer, system, prompt) > limit:
            raise PromptBudgetError("prompt_budget_exceeded: complete role prompt does not fit")
        events = memory.events
        memory.append(MemoryEvent(
            event_id=f"prompt-{role.value}-{len(events)}", trajectory_id=events[0].trajectory_id,
            task_id=events[0].task_id, turn_index=events[-1].turn_index,
            role=role.value, kind="role_prompt", visibility=MemoryVisibility.AUDIT,
            content={"system_prompt": system, "prompt": prompt,
                     "included_event_ids": view.included_event_ids,
                     "compacted_event_ids": view.compacted_event_ids,
                     "omitted_events": view.omitted_events},
            prompt_revision=self.prompts.revision,
            prompt_tokens=role_token_count(tokenizer, system, prompt),
        ))
        return prompt, view

    def _event(
        self,
        identity: TrajectoryIdentity,
        *,
        event_id: str,
        turn_index: int,
        role: str,
        kind: str,
        content: Any,
        tags: tuple[str, ...] = (),
        model_revision: str | None = None,
        tool_revision: str | None = None,
        latency_ms: float | None = None,
        status: str = "ok",
        failure_code: str | None = None,
    ) -> MemoryEvent:
        return MemoryEvent(
            event_id=event_id,
            trajectory_id=identity.trajectory_id,
            task_id=identity.task_id,
            turn_index=turn_index,
            role=role,
            kind=kind,
            content=content,
            tags=tags,
            model_revision=model_revision,
            prompt_revision=self.prompts.revision,
            tool_revision=tool_revision,
            latency_ms=latency_ms,
            status=status,
            failure_code=failure_code,
        )


__all__ = ["PlannerTurnRecord", "TrajectoryResult", "UnifiedAgentFlowLoop"]
