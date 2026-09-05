from __future__ import annotations

import asyncio
import re
from time import monotonic
from typing import Any

from pydantic import ValidationError

from agentflow_rl.runtime.actions import ToolEvent
from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.runtime.memory import MemoryStore
from agentflow_rl.tasks.coding.prompts import (
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
from agentflow_rl.tasks.coding.sandbox import CodeSandbox, DockerSandbox
from agentflow_rl.tasks.coding.schemas import CodeAction, CodeExample, FinalCode
from agentflow_rl.tasks.coding.tools import CodingEnvironment
from agentflow_rl.tasks.coding.verifier import evaluate_code
from agentflow_rl.verl.compat import AgentLoopOutput, register

from .base import AgentFlowLoopBase, config_value


CONCLUSION_RE = re.compile(r"Conclusion:\s*(STOP|CONTINUE)", re.IGNORECASE)


def extract_conclusion(text: str) -> str:
    matches = CONCLUSION_RE.findall(text)
    return matches[-1].upper() if matches else "CONTINUE"


@register("agentflow_coding")
class CodingAgentLoop(AgentFlowLoopBase):
    def __init__(self, *args: Any, code_sandbox: CodeSandbox | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.code_sandbox = code_sandbox or DockerSandbox(
            image=str(config_value(self.config, "agentflow.coding.sandbox_image", "agentflow-python-sandbox:3.11"))
        )
        self.sandbox_semaphore = asyncio.Semaphore(int(config_value(
            self.config,
            "agentflow.coding.max_concurrent_sandboxes",
            1,
        )))

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
            required_latest_tags=("current_code", "latest_test_result", "latest_judgement"),
            include_roles=("user", "query_analyzer", "executor", "verifier"),
            include_kinds=("starter_code", "analysis", "tool_event", "judgement"),
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> list[AgentLoopOutput]:
        uid = str(kwargs["uid"])
        session_id = int(kwargs.get("session_id", 0))
        example = CodeExample.from_row(dict(kwargs["extra_info"]))
        max_steps = int(config_value(self.config, "agentflow.max_steps", 5))
        role_tokens = int(config_value(self.config, "agentflow.role_max_tokens", 2048))
        test_timeout_s = float(config_value(self.config, "agentflow.coding.test_timeout_s", 10.0))
        deadline = monotonic() + float(config_value(self.config, "agentflow.max_time_s", 300.0))
        environment = CodingEnvironment(example, self.code_sandbox, test_timeout_s=test_timeout_s)
        memory = MemoryStore()
        memory.add(turn_index=-1, role="user", kind="question", content=example.question)
        if example.starter_code:
            memory.add(
                turn_index=-1,
                role="user",
                kind="starter_code",
                content=example.starter_code,
                tags=("identity", "current_code"),
            )
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
                prompt=query_prompt(example.question, example.starter_code),
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
                    proposed = CodeAction.parse(turn.response)
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
                    action = CodeAction.parse(executor_text)
                    if action.tool_name != proposed.tool_name:
                        raise ActionParseError("executor changed the Planner tool selection")
                    if action.tool_name == "Base_Generator_Tool":
                        generator_view = self._view(
                            memory, "base_generator", action.sub_goal, BASE_GENERATOR_SYSTEM
                        )
                        generated = await self.frozen_generate(
                            deadline=deadline,
                            prompt=f"Sub-goal: {action.sub_goal}\n\nMemory:\n{generator_view}",
                            system_prompt=BASE_GENERATOR_SYSTEM,
                            think_mode="off",
                            max_tokens=role_tokens,
                        )
                        result = {"ok": True, "data": generated}
                    else:
                        async with self.sandbox_semaphore:
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
                    if action.tool_name == "Code_Write_Tool":
                        event_tags.append("current_code")
                    elif action.tool_name == "Code_Run_Tests_Tool":
                        event_tags.append("latest_test_result")
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
                        self._view(
                            memory,
                            "verifier",
                            example.question,
                            VERIFIER_SYSTEM,
                        ),
                    ),
                    system_prompt=VERIFIER_SYSTEM,
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
                if turn_index + 1 == max_steps:
                    terminal_reason = "step_limit"

            final_text = await self.frozen_generate(
                deadline=deadline,
                prompt=generator_prompt(
                    example.question,
                    self._view(
                        memory,
                        "generator",
                        example.question,
                        GENERATOR_SYSTEM,
                    ),
                ),
                system_prompt=GENERATOR_SYSTEM,
                think_mode="off",
                max_tokens=role_tokens,
            )
            try:
                final_code = FinalCode.parse(final_text).code
                async with self.sandbox_semaphore:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError("AgentFlow trajectory deadline exceeded")
                    verification, hidden_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            evaluate_code,
                            final_code,
                            example,
                            self.code_sandbox,
                            timeout_s=test_timeout_s,
                        ),
                        timeout=remaining,
                    )
                reward = verification.reward
                verification_data = verification.model_dump(mode="json")
                verification_data["hidden_failures"] = [
                    {
                        "test_index": failure.get("test_index", -1),
                        "error_type": failure.get("error_type", "UNKNOWN"),
                    }
                    for failure in hidden_result.failures
                ]
            except (ActionParseError, ValidationError, ValueError):
                reward = 0.0
                verification_data = {"success": False, "reward": 0.0, "failure_codes": ["INVALID_FINAL_OUTPUT"]}
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
                "difficulty": example.difficulty,
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


__all__ = ["CodingAgentLoop", "extract_conclusion"]
