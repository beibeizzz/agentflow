from __future__ import annotations

import re
from time import monotonic
from typing import Any

from agentflow_rl.runtime.actions import ToolEvent
from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.runtime.memory import MemoryStore
from agentflow_rl.tasks.ticket.environment import TicketEnvironment, result as tool_result
from agentflow_rl.tasks.ticket.prompts import (
    TICKET_BASE_GENERATOR_SYSTEM_PROMPT,
    TICKET_EXECUTOR_SYSTEM_PROMPT,
    TICKET_GENERATOR_SYSTEM_PROMPT,
    TICKET_PLANNER_SYSTEM_PROMPT,
    TICKET_QUERY_ANALYSIS_SYSTEM_PROMPT,
    TICKET_VERIFIER_SYSTEM_PROMPT,
    render_executor_prompt,
    render_generator_prompt,
    render_next_step_prompt,
    render_query_analysis_prompt,
    render_verifier_prompt,
)
from agentflow_rl.tasks.ticket.schemas import TicketAction, TicketEpisode
from agentflow_rl.tasks.ticket.tools import ToolDispatchError, execute_tool
from agentflow_rl.tasks.ticket.verifier import verify_ticket
from agentflow_rl.verl.compat import AgentLoopOutput, register

from .base import AgentFlowLoopBase, config_value


CONCLUSION_RE = re.compile(r"Conclusion:\s*(STOP|CONTINUE)\b", re.IGNORECASE)


def extract_conclusion(text: str) -> str:
    matches = CONCLUSION_RE.findall(text)
    return matches[-1].upper() if matches else "CONTINUE"


@register("agentflow_ticket")
class TicketAgentLoop(AgentFlowLoopBase):
    def _view(self, memory: MemoryStore, role: str, *reserved_texts: str) -> str:
        recent = 3 if role in {"executor", "base_generator"} else 1000
        return self.role_memory_text(
            memory,
            role,
            *reserved_texts,
            default_max_tokens=1024,
            default_max_recent_entries=recent,
            required_tags=("identity",),
            required_latest_tags=("latest_ticket_state", "latest_generated_note", "latest_judgement"),
            include_roles=("query_analyzer", "executor", "verifier"),
            include_kinds=("analysis", "tool_event", "judgement"),
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> list[AgentLoopOutput]:
        uid = str(kwargs["uid"])
        session_id = int(kwargs.get("session_id", 0))
        episode = TicketEpisode.from_row(dict(kwargs["extra_info"]))
        environment = TicketEnvironment(
            initial_state=episode.initial_state.model_dump(mode="json"),
            goal_spec=episode.goal_spec,
        )
        deadline = monotonic() + float(config_value(self.config, "agentflow.max_time_s", 120.0))
        role_tokens = int(config_value(self.config, "agentflow.role_max_tokens", 512))
        outputs: list[AgentLoopOutput] = []
        events: list[ToolEvent] = []
        memory = MemoryStore()
        memory.add(turn_index=-1, role="user", kind="question", content=episode.user_request)
        terminal_reason = ""
        valid = True
        error: str | None = None
        final_text = ""

        try:
            analysis = await self.frozen_generate(
                deadline=deadline,
                prompt=render_query_analysis_prompt(episode.user_request),
                system_prompt=TICKET_QUERY_ANALYSIS_SYSTEM_PROMPT,
                think_mode="off",
                max_tokens=role_tokens,
            )
            memory.add(
                turn_index=-1,
                role="query_analyzer",
                kind="analysis",
                content=analysis,
                tags=("identity",),
            )
            for turn_index in range(episode.max_steps):
                planner_view = self._view(
                    memory,
                    "planner",
                    episode.user_request,
                    TICKET_PLANNER_SYSTEM_PROMPT,
                )
                prompt = render_next_step_prompt(
                    question=episode.user_request,
                    analysis=analysis,
                    memory=planner_view,
                )
                turn = await self.planner_generate(
                    uid=uid,
                    session_id=session_id,
                    turn_index=turn_index,
                    system_prompt=TICKET_PLANNER_SYSTEM_PROMPT,
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
                    proposed = TicketAction.parse(turn.response)
                    executor_text = await self.frozen_generate(
                        deadline=deadline,
                        prompt=render_executor_prompt(
                            turn.response,
                            self._view(
                                memory,
                                "executor",
                                turn.response,
                                TICKET_EXECUTOR_SYSTEM_PROMPT,
                            ),
                        ),
                        system_prompt=TICKET_EXECUTOR_SYSTEM_PROMPT,
                        think_mode="off",
                        max_tokens=role_tokens,
                    )
                    action = TicketAction.parse(executor_text)
                    if action.tool_name != proposed.tool_name:
                        raise ActionParseError("executor changed the Planner tool selection")
                    if action.tool_name == "Base_Generator_Tool":
                        result = {
                            "ok": True,
                            "data": await self.frozen_generate(
                                deadline=deadline,
                                prompt=(
                                    f"Sub-goal: {action.sub_goal}\n\nMemory:\n"
                                    f"{self._view(memory, 'base_generator', action.sub_goal)}"
                                ),
                                system_prompt=TICKET_BASE_GENERATOR_SYSTEM_PROMPT,
                                think_mode="off",
                                max_tokens=role_tokens,
                            ),
                        }
                    else:
                        result = execute_tool(environment, action)
                except (ActionParseError, ToolDispatchError, ValueError) as exc:
                    action = None
                    result = tool_result(False, "INVALID_ACTION", str(exc))

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
                    if action.tool_name in {
                        "Ticket_Query_Tool",
                        "Ticket_Update_Tool",
                        "Ticket_Finish_Tool",
                    }:
                        event_tags.append("latest_ticket_state")
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
                    prompt=render_verifier_prompt(
                        episode.user_request,
                        self._view(
                            memory,
                            "verifier",
                            episode.user_request,
                            TICKET_VERIFIER_SYSTEM_PROMPT,
                        ),
                    ),
                    system_prompt=TICKET_VERIFIER_SYSTEM_PROMPT,
                    think_mode="off",
                    max_tokens=role_tokens,
                )
                conclusion = extract_conclusion(verifier_text)
                memory.add(
                    turn_index=turn_index,
                    role="verifier",
                    kind="judgement",
                    content=verifier_text,
                    tags=("latest_judgement",),
                )
                outputs[-1].extra_fields.update({
                    "tool_event": event.model_dump(mode="json"),
                    "verifier_conclusion": conclusion,
                    "memory": memory.snapshot(),
                })
                if conclusion == "STOP":
                    terminal_reason = "verifier_stop"
                    break
                if turn_index + 1 == episode.max_steps:
                    terminal_reason = "step_limit"

            final_text = await self.frozen_generate(
                deadline=deadline,
                prompt=render_generator_prompt(
                    episode.user_request,
                    self._view(
                        memory,
                        "generator",
                        episode.user_request,
                        TICKET_GENERATOR_SYSTEM_PROMPT,
                    ),
                ),
                system_prompt=TICKET_GENERATOR_SYSTEM_PROMPT,
                think_mode="off",
                max_tokens=role_tokens,
            )
        except TimeoutError:
            terminal_reason = "time_limit"
        except (OSError, RuntimeError) as exc:
            valid = False
            terminal_reason = "infrastructure_failure"
            error = f"{type(exc).__name__}: {exc}"

        if valid and terminal_reason != "time_limit":
            verification = verify_ticket(
                environment,
                events,
                step_count=len(events),
                max_steps=episode.max_steps,
                lookup_mode=episode.lookup_mode,
            )
            reward = verification.reward
            verification_data = verification.model_dump(mode="json")
        elif valid:
            reward = 0.0
            verification_data = {"success": False, "failure_codes": ["TIME_LIMIT"]}
        else:
            reward = 0.0
            verification_data = {"success": False, "failure_codes": ["INFRASTRUCTURE_INVALID"]}
        return self.finalize_outputs(
            outputs,
            reward=reward,
            valid_for_training=valid,
            terminal_reason=terminal_reason or ("completed" if reward else "verification_failed"),
            final_fields={
                "episode_id": episode.episode_id,
                "final_answer": final_text,
                "verification": verification_data,
                "tool_events": [event.model_dump(mode="json") for event in events],
                "memory": memory.snapshot(),
                "environment_snapshot": environment.state_dict(),
                "reward_extra_info": {
                    "success": float(reward),
                    "valid_for_training": float(valid),
                    "steps": float(len(events)),
                    "direct": float(episode.curriculum_mode == "direct"),
                    "indirect": float(episode.curriculum_mode == "indirect"),
                },
                **({"error": error} if error else {}),
            },
        )


__all__ = ["TicketAgentLoop", "extract_conclusion"]
