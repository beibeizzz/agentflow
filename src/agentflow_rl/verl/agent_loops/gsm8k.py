from __future__ import annotations

import json
import logging
import re
from time import monotonic
from typing import Any

from agentflow_rl.runtime.actions import ToolEvent
from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.runtime.memory import MemoryStore
from agentflow_rl.tasks.gsm8k.calculator import format_number, safe_eval_calculation
from agentflow_rl.tasks.gsm8k.prompts import (
    BASE_GENERATOR_SYSTEM,
    EXECUTOR_SYSTEM,
    FINAL_SYSTEM,
    PLANNER_SYSTEM,
    QUERY_SYSTEM,
    VERIFIER_SYSTEM,
    executor_prompt,
    final_prompt,
    planner_prompt,
    query_prompt,
    verifier_prompt,
)
from agentflow_rl.tasks.gsm8k.schemas import GSM8KAction
from agentflow_rl.tasks.gsm8k.verifier import (
    answers_match,
    extract_numeric_answer,
    extract_verifier_conclusion,
)
from agentflow_rl.verl.compat import AgentLoopOutput, register

from .base import AgentFlowLoopBase, config_value


logger = logging.getLogger(__name__)


def extract_legacy_expression(response: str) -> str:
    text = response.strip()
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    json_text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    json_text = re.sub(r"\s*```$", "", json_text)
    try:
        payload = json.loads(json_text)
    except (TypeError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("command"), str):
        text = payload["command"]
    match = re.search(
        r"execution\s*=\s*tool\.execute\(\s*expression\s*=\s*([\"'])(.*?)\1\s*\)",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("executor response has no Calculator_Tool expression command")
    return match.group(2).strip()


def parse_planner_response(response: str) -> tuple[str, str]:
    action = GSM8KAction.parse(response)
    if action.tool_name != "Calculator_Tool":
        raise ValueError("planner response does not select Calculator_Tool")
    return action.sub_goal, str(action.arguments["expression"])


@register("agentflow_gsm8k")
class GSM8KAgentLoop(AgentFlowLoopBase):
    def _view(self, memory: MemoryStore, role: str, *reserved_texts: str) -> str:
        recent = 3 if role in {"executor", "base_generator"} else 1000
        return self.role_memory_text(
            memory,
            role,
            *reserved_texts,
            default_max_tokens=1024,
            default_max_recent_entries=recent,
            required_tags=("identity",),
            required_latest_tags=("latest_tool_result", "latest_generated_note", "latest_judgement"),
            include_roles=("query_analyzer", "executor", "verifier"),
            include_kinds=("analysis", "tool_event", "judgement"),
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> list[AgentLoopOutput]:
        uid = str(kwargs["uid"])
        session_id = int(kwargs.get("session_id", 0))
        row = dict(kwargs["extra_info"])
        episode_id = str(row.get("episode_id") or row.get("id") or row.get("pid") or uid)
        question = str(row["question"])
        gold_answer = str(row.get("gold_answer") or row.get("answer"))
        max_steps = int(config_value(self.config, "agentflow.max_steps", 3))
        role_tokens = int(config_value(self.config, "agentflow.role_max_tokens", 512))
        executor_mode = str(config_value(
            self.config, "agentflow.gsm8k.executor_mode", "deterministic"
        ))
        deadline = monotonic() + float(config_value(self.config, "agentflow.max_time_s", 120.0))
        outputs: list[AgentLoopOutput] = []
        memory = MemoryStore()
        memory.add(turn_index=-1, role="user", kind="question", content=question)
        memory_actions: dict[str, dict[str, Any]] = {}
        events: list[ToolEvent] = []
        terminal_reason = ""
        valid = True
        error: str | None = None
        analysis = ""
        final_answer_text = ""

        try:
            analysis = await self.frozen_generate(
                deadline=deadline,
                prompt=query_prompt(question),
                system_prompt=QUERY_SYSTEM,
                think_mode="on",
                max_tokens=role_tokens,
            )
            memory.add(
                turn_index=-1,
                role="query_analyzer",
                kind="analysis",
                content=analysis,
                tags=("identity",),
            )
            for turn_index in range(max_steps):
                prompt = planner_prompt(
                    question,
                    self._view(memory, "planner", question, PLANNER_SYSTEM),
                )
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
                    proposed = GSM8KAction.parse(turn.response)
                    if executor_mode == "legacy_llm":
                        executor_text = await self.frozen_generate(
                            deadline=deadline,
                            prompt=executor_prompt(
                                question,
                                turn.response,
                                self._view(memory, "executor", turn.response, EXECUTOR_SYSTEM),
                            ),
                            system_prompt=EXECUTOR_SYSTEM,
                            think_mode="off",
                            max_tokens=role_tokens,
                        )
                        try:
                            action = GSM8KAction.parse(executor_text)
                        except ActionParseError:
                            if proposed.tool_name != "Calculator_Tool":
                                raise
                            action = proposed.model_copy(update={
                                "arguments": {"expression": extract_legacy_expression(executor_text)}
                            })
                        if action.tool_name != proposed.tool_name:
                            raise ActionParseError("executor changed the Planner tool selection")
                    elif executor_mode == "deterministic":
                        action = proposed
                    else:
                        raise ValueError(f"unsupported GSM8K executor mode: {executor_mode}")

                    if action.tool_name == "Base_Generator_Tool":
                        result = {
                            "ok": True,
                            "data": await self.frozen_generate(
                                deadline=deadline,
                                prompt=(
                                    f"Sub-goal: {action.sub_goal}\n\nMemory:\n"
                                    f"{self._view(memory, 'base_generator', action.sub_goal)}"
                                ),
                                system_prompt=BASE_GENERATOR_SYSTEM,
                                think_mode="off",
                                max_tokens=role_tokens,
                            ),
                        }
                    else:
                        expression = str(action.arguments["expression"])
                        try:
                            result = {
                                "ok": True,
                                "data": {"value": format_number(safe_eval_calculation(expression))},
                            }
                        except (ValueError, ZeroDivisionError) as exc:
                            result = {
                                "ok": False,
                                "code": "CALCULATION_ERROR",
                                "message": str(exc),
                            }
                except (ActionParseError, ValueError) as exc:
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
                    if action.tool_name == "Calculator_Tool":
                        event_tags.append("latest_tool_result")
                    else:
                        event_tags.append("latest_generated_note")
                memory.add(
                    turn_index=turn_index,
                    role="executor",
                    kind="tool_event",
                    content=event.model_dump(mode="json"),
                    tags=event_tags,
                )

                verifier_response = await self.frozen_generate(
                    deadline=deadline,
                    prompt=verifier_prompt(
                        question,
                        self._view(memory, "verifier", question, VERIFIER_SYSTEM),
                    ),
                    system_prompt=VERIFIER_SYSTEM,
                    think_mode="on",
                    max_tokens=role_tokens,
                )
                try:
                    conclusion = extract_verifier_conclusion(verifier_response)
                except ValueError:
                    conclusion = "CONTINUE"
                memory.add(
                    turn_index=turn_index,
                    role="verifier",
                    kind="judgement",
                    content=verifier_response,
                    tags=("latest_judgement",),
                )

                step_key = f"Action Step {turn_index + 1}"
                memory_actions[step_key] = {
                    "tool_name": event.tool_name,
                    "sub_goal": event.metadata.get("sub_goal"),
                    "command": event.arguments,
                    "result": event.result,
                    "judge": verifier_response,
                }
                outputs[-1].extra_fields.update({
                    "tool_event": event.model_dump(mode="json"),
                    "memory": memory.snapshot(),
                    "memory_actions": {key: dict(value) for key, value in memory_actions.items()},
                    "verifier_conclusion": conclusion,
                })
                if conclusion == "STOP":
                    terminal_reason = "verifier_stop"
                    break
                if turn_index + 1 == max_steps:
                    terminal_reason = "max_steps"

            final_answer_text = await self.frozen_generate(
                deadline=deadline,
                prompt=final_prompt(
                    question,
                    self._view(memory, "generator", question, FINAL_SYSTEM),
                ),
                system_prompt=FINAL_SYSTEM,
                think_mode="off",
                max_tokens=role_tokens,
            )
            predicted = extract_numeric_answer(final_answer_text)
            reward = 1.0 if answers_match(predicted, gold_answer) else 0.0
            verification = {"success": bool(reward), "predicted": predicted, "gold": gold_answer}
        except TimeoutError:
            reward = 0.0
            terminal_reason = "time_limit"
            verification = {"success": False, "failure_codes": ["TIME_LIMIT"]}
        except (OSError, RuntimeError) as exc:
            logger.exception(
                "GSM8K trajectory failed before completion: uid=%s session_id=%s",
                uid,
                session_id,
            )
            valid = False
            reward = 0.0
            terminal_reason = "infrastructure_failure"
            error = f"{type(exc).__name__}: {exc}"
            verification = {"success": False, "failure_codes": ["INFRASTRUCTURE_INVALID"]}

        return self.finalize_outputs(
            outputs,
            reward=reward,
            valid_for_training=valid,
            terminal_reason=terminal_reason or ("completed" if reward else "verification_failed"),
            final_fields={
                "episode_id": episode_id,
                "analysis": analysis,
                "memory": memory.snapshot(),
                "memory_actions": memory_actions,
                "tool_events": [event.model_dump(mode="json") for event in events],
                "final_answer": final_answer_text,
                "verification": verification,
                "reward_extra_info": {
                    "success": float(reward),
                    "valid_for_training": float(valid),
                    "steps": float(len(outputs)),
                    "verifier_stop": float(terminal_reason == "verifier_stop"),
                },
                **({"error": error} if error else {}),
            },
        )


__all__ = [
    "GSM8KAgentLoop",
    "extract_legacy_expression",
    "parse_planner_response",
]
