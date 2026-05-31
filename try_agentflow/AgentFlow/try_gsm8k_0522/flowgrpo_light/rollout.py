from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .calculator import format_number, safe_eval_calculation
from .frozen_client import FrozenClient
from .parsing import parse_planner_response
from .prompts import (
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    build_final_prompt,
    build_planner_prompt,
    build_query_analysis_prompt,
)
from .reward import compute_reward

if TYPE_CHECKING:
    from .policy import PlannerPolicy


@dataclass
class PlannerSample:
    prompt: str
    response: str


@dataclass
class RolloutResult:
    reward: float
    answer: str
    samples: list[PlannerSample] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    query_analysis: str = ""
    errors: list[str] = field(default_factory=list)


def run_rollout(
    row: dict[str, Any],
    *,
    policy: "PlannerPolicy",
    frozen_client: FrozenClient,
    max_steps: int,
    frozen_temperature: float,
    frozen_max_tokens: int,
    query_analysis_think_mode: str = "default",
    final_output_think_mode: str = "default",
) -> RolloutResult:
    question = str(row["question"])
    gold_answer = str(row.get("result") or row.get("gold_answer") or row.get("extra_info", {}).get("gold_answer"))

    query_analysis = frozen_client.chat(
        build_query_analysis_prompt(question),
        system_prompt=QUERY_ANALYSIS_SYSTEM_PROMPT,
        temperature=frozen_temperature,
        max_tokens=min(frozen_max_tokens, 512),
        think_mode=query_analysis_think_mode,
    )
    memory: dict[str, Any] = {}
    samples: list[PlannerSample] = []
    errors: list[str] = []

    for step in range(1, max_steps + 1):
        prompt = build_planner_prompt(
            question=question,
            query_analysis=query_analysis,
            memory=memory,
            step_count=step,
            max_steps=max_steps,
        )
        generated = policy.generate(prompt)
        samples.append(PlannerSample(prompt=generated.prompt, response=generated.response))
        try:
            sub_goal, calculation = parse_planner_response(generated.response)
            result = format_number(safe_eval_calculation(calculation))
            memory[f"Action Step {step}"] = {
                "sub_goal": sub_goal,
                "calculation": calculation,
                "result": result,
                "planner_response": generated.response,
            }
        except Exception as exc:
            errors.append(f"step {step}: {type(exc).__name__}: {exc}")
            memory[f"Action Step {step}"] = {
                "sub_goal": None,
                "calculation": None,
                "result": "No calculator execution was attempted.",
                "planner_response": generated.response,
                "error": str(exc),
            }

    final_answer = frozen_client.chat(
        build_final_prompt(question, query_analysis, memory),
        temperature=frozen_temperature,
        max_tokens=min(frozen_max_tokens, 128),
        think_mode=final_output_think_mode,
    )
    reward = compute_reward(final_answer, gold_answer)
    return RolloutResult(
        reward=reward,
        answer=final_answer,
        samples=samples,
        memory=memory,
        query_analysis=query_analysis,
        errors=errors,
    )
