from __future__ import annotations

from time import monotonic
from typing import Any

from agentflow_rl.runtime.actions import ToolAction, ToolEvent
from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.tasks.ticket.environment import TicketEnvironment, result as tool_result
from agentflow_rl.tasks.ticket.prompts import (
    TICKET_NEXT_STEP_SYSTEM_PROMPT,
    TICKET_QUERY_ANALYSIS_SYSTEM_PROMPT,
    render_next_step_prompt,
    render_query_analysis_prompt,
)
from agentflow_rl.tasks.ticket.schemas import TicketEpisode
from agentflow_rl.tasks.ticket.tools import ToolDispatchError, execute_tool
from agentflow_rl.tasks.ticket.verifier import verify_ticket
from agentflow_rl.verl.compat import AgentLoopOutput, register

from .base import AgentFlowLoopBase, config_value


@register("agentflow_ticket")
class TicketAgentLoop(AgentFlowLoopBase):
    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> list[AgentLoopOutput]:
        uid = str(kwargs["uid"])
        session_id = int(kwargs.get("session_id", 0))
        episode = TicketEpisode.from_row(dict(kwargs["extra_info"]))
        environment = TicketEnvironment(
            initial_state=episode.initial_state.model_dump(mode="json"),
            goal_spec=episode.goal_spec,
        )
        deadline = monotonic() + float(config_value(self.config, "agentflow.max_time_s", 120.0))
        outputs: list[AgentLoopOutput] = []
        events: list[ToolEvent] = []
        terminal_reason = ""
        valid = True
        error: str | None = None

        try:
            analysis = await self.frozen_generate(
                deadline=deadline,
                prompt=render_query_analysis_prompt(episode.user_request),
                system_prompt=TICKET_QUERY_ANALYSIS_SYSTEM_PROMPT,
                think_mode="on",
                max_tokens=192,
            )
            for turn_index in range(episode.max_steps):
                if monotonic() >= deadline:
                    terminal_reason = "time_limit"
                    break
                prompt = render_next_step_prompt(
                    question=episode.user_request,
                    analysis=analysis,
                    events=events,
                )
                turn = await self.planner_generate(
                    uid=uid,
                    session_id=session_id,
                    turn_index=turn_index,
                    system_prompt=TICKET_NEXT_STEP_SYSTEM_PROMPT,
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
                invalid = False
                try:
                    action = ToolAction.model_validate_json_response(turn.response)
                except ActionParseError as exc:
                    action = None
                    invalid = True
                    tool_name = "__INVALID_ACTION__"
                    arguments: dict[str, Any] = {}
                    response = tool_result(False, "INVALID_ACTION", str(exc))
                else:
                    try:
                        response = execute_tool(environment, action)
                    except ToolDispatchError as exc:
                        invalid = True
                        tool_name = "__INVALID_ACTION__"
                        arguments = action.arguments
                        response = tool_result(False, "INVALID_ACTION", str(exc))
                    else:
                        tool_name = action.tool_name
                        arguments = action.arguments
                event = ToolEvent(
                    turn_index=turn_index,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=response,
                    ok=response.get("ok") is True,
                )
                events.append(event)
                outputs[-1].metrics.tool_calls = float(not invalid)
                outputs[-1].extra_fields["tool_event"] = event.model_dump(mode="json")
                if invalid:
                    terminal_reason = "invalid_action"
                    break
                if response.get("ok") is not True:
                    terminal_reason = "tool_error"
                    break
                if action is not None and action.tool_name == "Ticket_Finish_Tool":
                    terminal_reason = "finish_submitted"
                    break
                if turn_index + 1 >= episode.max_steps:
                    terminal_reason = "step_limit"
        except TimeoutError:
            terminal_reason = "time_limit"
        except OSError as exc:
            valid = False
            terminal_reason = "infrastructure_failure"
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            valid = False
            terminal_reason = "infrastructure_failure"
            error = f"{type(exc).__name__}: {exc}"

        if valid:
            verification = verify_ticket(
                environment,
                events,
                step_count=len(events),
                max_steps=episode.max_steps,
            )
            reward = verification.reward
            verification_data = verification.model_dump(mode="json")
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
                "verification": verification_data,
                "tool_events": [event.model_dump(mode="json") for event in events],
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
