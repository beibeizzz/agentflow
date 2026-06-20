from __future__ import annotations

import concurrent.futures
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .policy import GeneratedResponse
from .rollout import PlannerSample, RolloutResult


def _vllm_engine_name(model_name: str) -> str:
    return model_name if model_name.startswith("vllm-") else f"vllm-{model_name}"


def _policy_generation_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    generation_kwargs: dict[str, Any] = {}
    if "max_new_tokens" in kwargs and kwargs["max_new_tokens"] is not None:
        generation_kwargs["max_new_tokens"] = int(kwargs["max_new_tokens"])
    elif "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
        generation_kwargs["max_new_tokens"] = int(kwargs["max_tokens"])
    if "temperature" in kwargs and kwargs["temperature"] is not None:
        generation_kwargs["temperature"] = float(kwargs["temperature"])
    if "top_p" in kwargs and kwargs["top_p"] is not None:
        generation_kwargs["top_p"] = float(kwargs["top_p"])
    if "do_sample" in kwargs and kwargs["do_sample"] is not None:
        generation_kwargs["do_sample"] = bool(kwargs["do_sample"])
    return generation_kwargs


def _planner_sample(
    *,
    raw_prompt: str,
    system_prompt: str | None,
    generation_kwargs: dict[str, Any],
    generated: GeneratedResponse,
) -> PlannerSample:
    return PlannerSample(
        prompt=generated.prompt,
        rendered_prompt=generated.prompt,
        raw_prompt=raw_prompt,
        system_prompt=system_prompt,
        generation_kwargs=dict(generation_kwargs),
        response=generated.response,
    )


@dataclass
class AgentFlowPlannerEngine:
    policy: Any
    think_mode: str = "default"
    samples: list[PlannerSample] = field(default_factory=list)

    def __call__(self, prompt: Any, *, system_prompt: str | None = None, **kwargs: Any) -> str:
        if isinstance(prompt, list):
            text_prompt = "\n".join(str(item) for item in prompt if isinstance(item, str))
        else:
            text_prompt = str(prompt)
        generation_kwargs = _policy_generation_kwargs(kwargs)
        generated: GeneratedResponse = self.policy.generate_for_agentflow(
            text_prompt,
            system_prompt=system_prompt,
            think_mode=self.think_mode,
            **generation_kwargs,
        )
        self.samples.append(
            _planner_sample(
                raw_prompt=text_prompt,
                system_prompt=system_prompt,
                generation_kwargs=generation_kwargs,
                generated=generated,
            )
        )
        return generated.response

    def clear(self) -> None:
        self.samples.clear()


@dataclass
class _PlannerBatchRequest:
    prompt: str
    system_prompt: str | None
    generation_kwargs: dict[str, Any]
    samples: list[PlannerSample]
    future: concurrent.futures.Future[str]


class BatchedAgentFlowPlannerProxy:
    def __init__(self, engine: "BatchedAgentFlowPlannerEngine") -> None:
        self.engine = engine
        self.samples: list[PlannerSample] = []

    def __call__(self, prompt: Any, *, system_prompt: str | None = None, **kwargs: Any) -> str:
        if isinstance(prompt, list):
            text_prompt = "\n".join(str(item) for item in prompt if isinstance(item, str))
        else:
            text_prompt = str(prompt)
        return self.engine.submit(
            text_prompt,
            system_prompt=system_prompt,
            generation_kwargs=_policy_generation_kwargs(kwargs),
            samples=self.samples,
        )

    def clear(self) -> None:
        self.samples.clear()


class BatchedAgentFlowPlannerEngine:
    def __init__(
        self,
        policy: Any,
        *,
        think_mode: str = "default",
        max_batch_size: int = 1,
        batch_timeout_s: float = 0.01,
    ) -> None:
        self.policy = policy
        self.think_mode = think_mode
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_timeout_s = max(0.0, float(batch_timeout_s))
        self._queue: queue.Queue[_PlannerBatchRequest | None] = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._worker_loop, name="AgentFlowPlannerBatcher", daemon=True)
        self._thread.start()

    def create_proxy(self) -> BatchedAgentFlowPlannerProxy:
        return BatchedAgentFlowPlannerProxy(self)

    def submit(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        generation_kwargs: dict[str, Any],
        samples: list[PlannerSample],
    ) -> str:
        if self._closed:
            raise RuntimeError("Batched planner engine is closed")
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        self._queue.put(
            _PlannerBatchRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                generation_kwargs=generation_kwargs,
                samples=samples,
                future=future,
            )
        )
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=10)

    def _worker_loop(self) -> None:
        stopping = False
        while not stopping:
            item = self._queue.get()
            if item is None:
                break
            batch = [item]
            deadline = time.monotonic() + self.batch_timeout_s
            while len(batch) < self.max_batch_size:
                timeout = max(0.0, deadline - time.monotonic())
                if timeout == 0.0:
                    break
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is None:
                    stopping = True
                    break
                batch.append(item)
            self._generate_batch(batch)

    def _generate_batch(self, batch: list[_PlannerBatchRequest]) -> None:
        try:
            generation_kwargs = dict(batch[0].generation_kwargs)
            if any(request.generation_kwargs != generation_kwargs for request in batch):
                for request in batch:
                    generated = self.policy.generate_for_agentflow(
                        request.prompt,
                        system_prompt=request.system_prompt,
                        think_mode=self.think_mode,
                        **request.generation_kwargs,
                    )
                    request.samples.append(
                        _planner_sample(
                            raw_prompt=request.prompt,
                            system_prompt=request.system_prompt,
                            generation_kwargs=request.generation_kwargs,
                            generated=generated,
                        )
                    )
                    request.future.set_result(generated.response)
                return
            generated = self.policy.generate_many_for_agentflow(
                [request.prompt for request in batch],
                system_prompts=[request.system_prompt for request in batch],
                think_mode=self.think_mode,
                **generation_kwargs,
            )
            for request, item in zip(batch, generated, strict=True):
                request.samples.append(
                    _planner_sample(
                        raw_prompt=request.prompt,
                        system_prompt=request.system_prompt,
                        generation_kwargs=request.generation_kwargs,
                        generated=item,
                    )
                )
                request.future.set_result(item.response)
        except Exception as exc:
            for request in batch:
                request.future.set_exception(exc)


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
                valid_for_training=False,
            )


class AgentFlowRolloutWorker:
    def __init__(
        self,
        *,
        solver: Any,
        policy_engine: BatchedAgentFlowPlannerProxy,
        reset_solver: Callable[[Any], None],
    ) -> None:
        self.solver = solver
        self.policy_engine = policy_engine
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
                valid_for_training=False,
            )


class AgentFlowBatchRolloutRunner:
    def __init__(
        self,
        *,
        policy: Any,
        solver_factory: Callable[[], Any],
        reset_solver: Callable[[Any], None],
        rollout_concurrency: int = 1,
        planner_batch_size: int = 1,
        planner_batch_timeout_s: float = 0.01,
        think_mode: str = "default",
    ) -> None:
        self.rollout_concurrency = max(1, int(rollout_concurrency))
        self.planner_engine = BatchedAgentFlowPlannerEngine(
            policy,
            think_mode=think_mode,
            max_batch_size=planner_batch_size,
            batch_timeout_s=planner_batch_timeout_s,
        )
        self.workers = [
            AgentFlowRolloutWorker(
                solver=solver_factory(),
                policy_engine=self.planner_engine.create_proxy(),
                reset_solver=reset_solver,
            )
            for _ in range(self.rollout_concurrency)
        ]
        self._available_workers: queue.Queue[AgentFlowRolloutWorker] = queue.Queue()
        for worker in self.workers:
            self._available_workers.put(worker)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.rollout_concurrency)

    def run(self, row: dict[str, Any]) -> RolloutResult:
        return self.run_batch([row], group_size=1)[0][0]

    def run_batch(self, rows: list[dict[str, Any]], *, group_size: int) -> list[list[RolloutResult]]:
        if not rows:
            return []
        groups: list[list[RolloutResult | None]] = [[None for _ in range(group_size)] for _ in rows]
        futures: dict[concurrent.futures.Future[RolloutResult], tuple[int, int]] = {}
        for row_index, row in enumerate(rows):
            for sample_index in range(group_size):
                future = self._executor.submit(self._run_with_worker, row)
                futures[future] = (row_index, sample_index)
        for future in concurrent.futures.as_completed(futures):
            row_index, sample_index = futures[future]
            groups[row_index][sample_index] = future.result()
        return [[rollout for rollout in group if rollout is not None] for group in groups]

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self.planner_engine.close()

    def _run_with_worker(self, row: dict[str, Any]) -> RolloutResult:
        worker = self._available_workers.get()
        try:
            return worker.run(row)
        finally:
            self._available_workers.put(worker)


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
    rollout_concurrency: int = 1,
    planner_batch_size: int = 1,
    planner_batch_timeout_s: float = 0.01,
) -> AgentFlowBatchRolloutRunner:
    import run_gsm8k_agentflow

    def solver_factory() -> Any:
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
        return solver

    return AgentFlowBatchRolloutRunner(
        policy=policy,
        solver_factory=solver_factory,
        reset_solver=run_gsm8k_agentflow.reset_solver_memory,
        rollout_concurrency=rollout_concurrency,
        planner_batch_size=planner_batch_size,
        planner_batch_timeout_s=planner_batch_timeout_s,
        think_mode=think_mode,
    )
