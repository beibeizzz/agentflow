from __future__ import annotations

import json
import logging
import re
from time import monotonic
from typing import Any

from agentflow_rl.tasks.gsm8k.calculator import format_number, safe_eval_calculation
from agentflow_rl.runtime.actions import strict_json_object
from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.tasks.gsm8k.prompts import (
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
    payload = strict_json_object(response)
    if not isinstance(payload, dict) or set(payload) not in (
        {"Sub_goal", "Calculation"},
        {"sub_goal", "calculation"},
    ):
        raise ValueError("planner response must be one exact object")
    return (
        str(payload.get("Sub_goal") or payload.get("sub_goal")),
        str(payload.get("Calculation") or payload.get("calculation")),
    )


@register("agentflow_gsm8k")
class GSM8KAgentLoop(AgentFlowLoopBase):
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
        memory: dict[str, dict[str, Any]] = {}
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
            for turn_index in range(max_steps):
                prompt = planner_prompt(question, analysis, memory)
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
                step_number = turn_index + 1
                step_key = f"Action Step {step_number}"
                try:
                    sub_goal, planner_expression = parse_planner_response(turn.response)
                except (ActionParseError, ValueError, json.JSONDecodeError) as exc:
                    outputs[-1].metrics.tool_calls = 0.0
                    memory_action: dict[str, Any] = {
                        "tool_name": None,
                        "sub_goal": None,
                        "command": "Planner output was invalid; no tool command generated.",
                        "result": "No calculator execution was attempted.",
                        "action_predictor_response": turn.response,
                    }
                    verifier_response = (
                        "Conclusion: CONTINUE\nlast Planner output was invalid. "
                        "check the Calculation first"
                    )
                    conclusion = "CONTINUE"
                    outputs[-1].extra_fields["planner_error"] = str(exc)
                else:
                    outputs[-1].metrics.tool_calls = 1.0
                    expression = planner_expression
                    command = f'execution = tool.execute(expression="{expression}")'
                    if executor_mode == "legacy_llm":
                        executor_response = await self.frozen_generate(
                            deadline=deadline,
                            prompt=executor_prompt(question, expression),
                            system_prompt=EXECUTOR_SYSTEM,
                            think_mode="off",
                            max_tokens=role_tokens,
                        )
                        try:
                            expression = extract_legacy_expression(executor_response)
                            command = f'execution = tool.execute(expression="{expression}")'
                        except ValueError:
                            expression = None
                            command = "No command found."
                    if expression is None:
                        result: Any = [] if executor_mode == "legacy_llm" else "No execution"
                    else:
                        try:
                            value = format_number(safe_eval_calculation(expression))
                            result = [value] if executor_mode == "legacy_llm" else value
                        except (ValueError, ZeroDivisionError) as exc:
                            value = f"Error: {exc}"
                            result = [value] if executor_mode == "legacy_llm" else value
                    memory_action = {
                        "tool_name": "Calculator_Tool",
                        "sub_goal": sub_goal,
                        "command": command,
                        "result": result,
                        "calculation": planner_expression,
                    }
                    provisional_memory = {**memory, step_key: memory_action}
                    verifier_response = await self.frozen_generate(
                        deadline=deadline,
                        prompt=verifier_prompt(question, analysis, provisional_memory),
                        system_prompt=VERIFIER_SYSTEM,
                        think_mode="on",
                        max_tokens=role_tokens,
                    )
                    try:
                        conclusion = extract_verifier_conclusion(verifier_response)
                    except ValueError:
                        verifier_response = "invalid verifier response"
                        conclusion = "CONTINUE"
                memory_action["judge"] = verifier_response
                memory[step_key] = memory_action
                outputs[-1].extra_fields["memory_actions"] = {
                    key: dict(value) for key, value in memory.items()
                }
                outputs[-1].extra_fields["verifier_conclusion"] = conclusion
                if conclusion == "STOP":
                    terminal_reason = "verifier_stop"
                    break
                if step_number >= max_steps:
                    terminal_reason = "max_steps"

            final_answer_text = await self.frozen_generate(
                deadline=deadline,
                prompt=final_prompt(question, analysis, memory),
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
        except Exception as exc:
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
                "memory_actions": memory,
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
