from __future__ import annotations

import asyncio
import re
from time import monotonic
from typing import Any

from pydantic import ValidationError

from agentflow_rl.runtime.actions import ToolEvent
from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.runtime.memory import MemoryStore
from agentflow_rl.tasks.deepresearch.prompts import (
    BASE_GENERATOR_SYSTEM,
    EXECUTOR_SYSTEM,
    GENERATOR_SYSTEM,
    PLANNER_SYSTEM,
    QUERY_SYSTEM,
    VERIFIER_SYSTEM,
    executor_prompt,
    generator_prompt,
    planner_prompt,
    query_prompt,
    verifier_prompt,
)
from agentflow_rl.tasks.deepresearch.retrieval import (
    InMemoryBM25Index,
    PyseriniResearchIndex,
    ResearchDocument,
    ResearchIndex,
)
from agentflow_rl.tasks.deepresearch.schemas import (
    Citation,
    DeepResearchExample,
    ResearchAction,
    ResearchFinalAnswer,
)
from agentflow_rl.tasks.deepresearch.tools import DeepResearchEnvironment
from agentflow_rl.tasks.deepresearch.verifier import evaluate_research_answer
from agentflow_rl.verl.compat import AgentLoopOutput, register

from .base import AgentFlowLoopBase, config_value


CONCLUSION_RE = re.compile(r"Conclusion:\s*(STOP|CONTINUE)", re.IGNORECASE)


def extract_conclusion(text: str) -> str:
    matches = CONCLUSION_RE.findall(text)
    return matches[-1].upper() if matches else "CONTINUE"


@register("agentflow_deepresearch")
class DeepResearchAgentLoop(AgentFlowLoopBase):
    def __init__(self, *args: Any, research_index: ResearchIndex | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        retrieval_mode = str(config_value(self.config, "agentflow.deepresearch.retrieval_mode", "global"))
        if research_index is None and retrieval_mode == "global":
            index_path = config_value(self.config, "agentflow.deepresearch.index_path")
            if not index_path:
                raise ValueError("agentflow.deepresearch.index_path is required")
            research_index = PyseriniResearchIndex(str(index_path))
        self.research_index = research_index
        self.retrieval_semaphore = asyncio.Semaphore(int(config_value(
            self.config,
            "agentflow.deepresearch.max_concurrent_retrievals",
            4,
        )))

    def _research_index(self, example: DeepResearchExample) -> ResearchIndex:
        mode = str(config_value(self.config, "agentflow.deepresearch.retrieval_mode", "global"))
        if mode == "global":
            if self.research_index is None:
                raise RuntimeError("global research index is unavailable")
            return self.research_index
        if mode == "local_context":
            rows = example.metadata.get("retrieval_documents", ())
            documents = [ResearchDocument(
                doc_id=str(row["doc_id"]),
                title=str(row["title"]),
                sentences=tuple(str(sentence) for sentence in row["sentences"]),
            ) for row in rows]
            if not documents:
                raise ValueError("local_context mode requires retrieval_documents")
            return InMemoryBM25Index(documents)
        raise ValueError(f"unsupported retrieval mode: {mode}")

    def _view(self, memory: MemoryStore, role: str, *reserved_texts: str) -> str:
        default_limits = {
            "planner": 6144,
            "executor": 4096,
            "verifier": 6144,
            "generator": 6144,
            "base_generator": 4096,
        }
        default_recent = {"executor": 4, "base_generator": 4}
        return self.role_memory_text(
            memory,
            role,
            *reserved_texts,
            default_max_tokens=default_limits[role],
            default_max_recent_entries=default_recent.get(role, 1000),
            required_tags=("identity",),
            required_latest_tags=(
                "evidence",
                "latest_search",
                "latest_generated_note",
                "latest_judgement",
            ),
            include_roles=("query_analyzer", "executor", "verifier"),
            include_kinds=("analysis", "tool_event", "judgement"),
        )

    @staticmethod
    def _observed_citations(events: list[ToolEvent]) -> tuple[Citation, ...]:
        citations = []
        for event in events:
            if event.tool_name != "Research_Read_Tool" or not event.ok:
                continue
            data = event.result.get("data", {}) if isinstance(event.result, dict) else {}
            title = data.get("title") if isinstance(data, dict) else None
            sentences = data.get("sentences", ()) if isinstance(data, dict) else ()
            if not title or not isinstance(sentences, list):
                continue
            for sentence in sentences:
                if isinstance(sentence, dict) and isinstance(sentence.get("sentence_id"), int):
                    citations.append(Citation(title=str(title), sentence_id=sentence["sentence_id"]))
        return tuple(citations)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> list[AgentLoopOutput]:
        uid = str(kwargs["uid"])
        session_id = int(kwargs.get("session_id", 0))
        example = DeepResearchExample.from_row(dict(kwargs["extra_info"]))
        max_steps = int(config_value(self.config, "agentflow.max_steps", 5))
        role_tokens = int(config_value(self.config, "agentflow.role_max_tokens", 2048))
        deadline = monotonic() + float(config_value(self.config, "agentflow.max_time_s", 300.0))
        environment = DeepResearchEnvironment(
            self._research_index(example),
            top_k=int(config_value(self.config, "agentflow.deepresearch.top_k", 10)),
        )
        memory = MemoryStore()
        memory.add(turn_index=-1, role="user", kind="question", content=example.question)
        outputs: list[AgentLoopOutput] = []
        events: list[ToolEvent] = []
        terminal_reason = ""
        valid = True
        error: str | None = None
        final_text = ""
        verification_data: dict[str, Any]

        try:
            analysis = await self.frozen_generate(
                deadline=deadline,
                prompt=query_prompt(example.question),
                system_prompt=QUERY_SYSTEM,
                think_mode="off",
                max_tokens=role_tokens,
            )
            memory.add(turn_index=-1, role="query_analyzer", kind="analysis", content=analysis, tags=("identity",))
            for turn_index in range(max_steps):
                view = self._view(memory, "planner", example.question, PLANNER_SYSTEM)
                prompt = planner_prompt(example.question, view)
                turn = await self.planner_generate(
                    uid=uid,
                    session_id=session_id,
                    turn_index=turn_index,
                    system_prompt=PLANNER_SYSTEM,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    deadline=deadline,
                )
                outputs.append(self.planner_output(
                    turn,
                    uid=uid,
                    session_id=session_id,
                    turn_index=turn_index,
                    extra_fields={"planner_prompt": prompt, "planner_response": turn.response},
                ))
                try:
                    proposed = ResearchAction.parse(turn.response)
                    executor_text = await self.frozen_generate(
                        deadline=deadline,
                        prompt=executor_prompt(
                            turn.response,
                            self._view(memory, "executor", turn.response, EXECUTOR_SYSTEM),
                        ),
                        system_prompt=EXECUTOR_SYSTEM,
                        think_mode="off",
                        max_tokens=role_tokens,
                    )
                    action = ResearchAction.parse(executor_text)
                    if action.tool_name != proposed.tool_name:
                        raise ActionParseError("executor changed the Planner tool selection")
                    if action.tool_name == "Base_Generator_Tool":
                        generator_view = self._view(
                            memory, "base_generator", action.sub_goal, BASE_GENERATOR_SYSTEM
                        )
                        result = {
                            "ok": True,
                            "data": await self.frozen_generate(
                                deadline=deadline,
                                prompt=f"Sub-goal: {action.sub_goal}\n\nMemory:\n{generator_view}",
                                system_prompt=BASE_GENERATOR_SYSTEM,
                                think_mode="off",
                                max_tokens=role_tokens,
                            ),
                        }
                    else:
                        async with self.retrieval_semaphore:
                            result = await self.run_blocking(
                                deadline=deadline,
                                operation=environment.execute,
                                args=(action,),
                            )
                except (ActionParseError, ValidationError, ValueError) as exc:
                    action = None
                    result = {"ok": False, "code": "INVALID_ACTION", "message": str(exc)}
                event = ToolEvent(
                    turn_index=turn_index,
                    tool_name=action.tool_name if action else "__INVALID_ACTION__",
                    arguments=action.arguments if action else {},
                    result=result,
                    ok=result.get("ok") is True,
                    metadata={"sub_goal": action.sub_goal} if action else {},
                )
                events.append(event)
                outputs[-1].metrics.tool_calls = float(action is not None)
                event_tags = ["tool_result"]
                if action and result.get("ok") is True:
                    if action.tool_name == "Research_Search_Tool":
                        event_tags.append("latest_search")
                    elif action.tool_name == "Research_Read_Tool":
                        event_tags.append("evidence")
                    elif action.tool_name == "Base_Generator_Tool":
                        event_tags.append("latest_generated_note")
                memory.add(
                    turn_index=turn_index,
                    role="executor",
                    kind="tool_event",
                    content=event.model_dump(mode="json"),
                    tags=event_tags,
                )
                verifier_text = await self.frozen_generate(
                    deadline=deadline,
                    prompt=verifier_prompt(
                        example.question,
                        self._view(memory, "verifier", example.question, VERIFIER_SYSTEM),
                    ),
                    system_prompt=VERIFIER_SYSTEM,
                    think_mode="off",
                    max_tokens=role_tokens,
                )
                memory.add(
                    turn_index=turn_index,
                    role="verifier",
                    kind="judgement",
                    content=verifier_text,
                    tags=("latest_judgement",),
                )
                outputs[-1].extra_fields.update({
                    "tool_event": event.model_dump(mode="json"),
                    "verifier_conclusion": extract_conclusion(verifier_text),
                    "memory": memory.snapshot(),
                })
                if extract_conclusion(verifier_text) == "STOP":
                    terminal_reason = "verifier_stop"
                    break
                if turn_index + 1 == max_steps:
                    terminal_reason = "step_limit"

            final_text = await self.frozen_generate(
                deadline=deadline,
                prompt=generator_prompt(
                    example.question,
                    self._view(memory, "generator", example.question, GENERATOR_SYSTEM),
                ),
                system_prompt=GENERATOR_SYSTEM,
                think_mode="off",
                max_tokens=role_tokens,
            )
            try:
                prediction = ResearchFinalAnswer.parse(final_text)
                verification = evaluate_research_answer(
                    prediction,
                    example,
                    observed_citations=self._observed_citations(events),
                )
            except (ActionParseError, ValidationError, ValueError):
                verification = None
            if verification is None:
                reward = 0.0
                verification_data = {"success": False, "reward": 0.0, "failure_codes": ["INVALID_FINAL_OUTPUT"]}
            else:
                reward = verification.reward
                verification_data = verification.model_dump(mode="json")
        except TimeoutError:
            reward = 0.0
            terminal_reason = "time_limit"
            verification_data = {"success": False, "reward": 0.0, "failure_codes": ["TIME_LIMIT"]}
        except (OSError, RuntimeError) as exc:
            valid = False
            reward = 0.0
            terminal_reason = "infrastructure_failure"
            error = f"{type(exc).__name__}: {exc}"
            verification_data = {"success": False, "reward": 0.0, "failure_codes": ["INFRASTRUCTURE_INVALID"]}

        success = float(verification_data.get("success", False))
        metrics = dict(verification_data.get("metrics", {}))
        return self.finalize_outputs(
            outputs,
            reward=reward,
            valid_for_training=valid,
            terminal_reason=terminal_reason or "completed",
            final_fields={
                "episode_id": example.episode_id,
                "dataset": example.dataset,
                "tool_events": [event.model_dump(mode="json") for event in events],
                "memory": memory.snapshot(),
                "final_answer": final_text,
                "verification": verification_data,
                "reward_extra_info": {
                    "success": success,
                    "valid_for_training": float(valid),
                    "steps": float(len(outputs)),
                    **{key: float(value) for key, value in metrics.items()},
                },
                **({"error": error} if error else {}),
            },
        )


__all__ = ["DeepResearchAgentLoop", "extract_conclusion"]
