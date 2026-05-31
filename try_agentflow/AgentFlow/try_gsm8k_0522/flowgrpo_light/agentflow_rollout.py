from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .policy import GeneratedResponse
from .rollout import PlannerSample, RolloutResult


def _vllm_engine_name(model_name: str) -> str:
    return model_name if model_name.startswith("vllm-") else f"vllm-{model_name}"


@dataclass
class AgentFlowPlannerEngine:
    policy: Any
    think_mode: str = "default"
    samples: list[PlannerSample] = field(default_factory=list)

    def __call__(self, prompt: Any, *, system_prompt: str | None = None, **_: Any) -> str:
        if isinstance(prompt, list):
            text_prompt = "\n".join(str(item) for item in prompt if isinstance(item, str))
        else:
            text_prompt = str(prompt)
        generated: GeneratedResponse = self.policy.generate_for_agentflow(
            text_prompt,
            system_prompt=system_prompt,
            think_mode=self.think_mode,
        )
        self.samples.append(PlannerSample(prompt=generated.prompt, response=generated.response))
        return generated.response

    def clear(self) -> None:
        self.samples.clear()


class AgentFlowRolloutRunner:
    def __init__(
        self,
        *,
        solver: Any,
        policy: Any,
        reset_solver: Callable[[Any], None],
        think_mode: str = "default",
    ) -> None:
        self.solver = solver
        self.policy_engine = AgentFlowPlannerEngine(policy, think_mode=think_mode)
        self.reset_solver = reset_solver
        self.solver.planner.llm_engine = self.policy_engine

    def run(self, row: dict[str, Any]) -> RolloutResult:
        from flowgrpo.reward import compute_result_reward

        question = str(row["question"])
        gold_answer = str(row.get("result") or row.get("gold_answer") or row.get("extra_info", {}).get("gold_answer"))
        self.reset_solver(self.solver)
        self.policy_engine.clear()
        try:
            result = self.solver.solve(question)
            reward, predicted_answer = compute_result_reward(result, gold_answer)
            answer: Any = result.get("direct_output") or result.get("final_output") or result.get("base_response")
            return RolloutResult(
                reward=reward,
                answer=str(answer if answer is not None else predicted_answer),
                samples=list(self.policy_engine.samples),
                memory=result.get("memory") or {},
                query_analysis=str(result.get("query_analysis") or ""),
                errors=[],
            )
        except Exception as exc:
            return RolloutResult(
                reward=0.0,
                answer="",
                samples=list(self.policy_engine.samples),
                memory={},
                query_analysis="",
                errors=[f"{type(exc).__name__}: {exc}"],
            )


def _make_frozen_engine(
    *,
    frozen_model: str,
    frozen_base_url: str,
    frozen_temperature: float,
    think_mode: str,
) -> Any:
    from agentflow.engine.factory import create_llm_engine

    return create_llm_engine(
        model_string=_vllm_engine_name(frozen_model),
        is_multimodal=False,
        base_url=frozen_base_url,
        temperature=frozen_temperature,
        think_mode=think_mode,
    )


def route_frozen_subagents(
    solver: Any,
    *,
    frozen_model: str,
    frozen_base_url: str,
    frozen_temperature: float,
    think_mode: str = "default",
) -> Any:
    frozen_engine = _make_frozen_engine(
        frozen_model=frozen_model,
        frozen_base_url=frozen_base_url,
        frozen_temperature=frozen_temperature,
        think_mode=think_mode,
    )
    solver.planner.llm_engine_fixed = frozen_engine
    solver.verifier.llm_engine_fixed = frozen_engine
    solver.executor.llm_generate_tool_command = frozen_engine
    return solver


def build_agentflow_rollout_runner(
    *,
    policy: Any,
    frozen_model: str,
    frozen_base_url: str,
    frozen_temperature: float,
    output_types: str,
    max_steps: int,
    max_time: int,
    max_tokens: int,
    subagent_config_path: Path | None = None,
    think_mode: str = "default",
    query_analysis_think_mode: str | None = None,
    final_output_think_mode: str | None = None,
    verifier_think_mode: str | None = None,
) -> AgentFlowRolloutRunner:
    import run_gsm8k_agentflow

    solver = run_gsm8k_agentflow.construct_solver(
        llm_engine_name=_vllm_engine_name(frozen_model),
        base_url=frozen_base_url,
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        temperature=frozen_temperature,
        subagent_config_path=subagent_config_path,
        think_mode=think_mode,
        query_analysis_think_mode=query_analysis_think_mode or think_mode,
        final_output_think_mode=final_output_think_mode or think_mode,
        verifier_think_mode=verifier_think_mode or think_mode,
    )
    route_frozen_subagents(
        solver,
        frozen_model=frozen_model,
        frozen_base_url=frozen_base_url,
        frozen_temperature=frozen_temperature,
        think_mode=think_mode,
    )
    return AgentFlowRolloutRunner(
        solver=solver,
        policy=policy,
        reset_solver=run_gsm8k_agentflow.reset_solver_memory,
        think_mode=think_mode,
    )
