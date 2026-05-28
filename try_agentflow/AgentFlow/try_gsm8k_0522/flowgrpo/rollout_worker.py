from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


FLOWGRPO_DIR = Path(__file__).resolve().parent
GSM8K_DIR = FLOWGRPO_DIR.parent
PROJECT_ROOT = GSM8K_DIR.parent
sys.path.insert(0, str(GSM8K_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import agentflow as _agentflow

INNER_AGENTFLOW = PROJECT_ROOT / "agentflow" / "agentflow"
if str(INNER_AGENTFLOW) not in _agentflow.__path__:
    _agentflow.__path__.append(str(INNER_AGENTFLOW))

from agentflow import LLM, LitAgent, NamedResources, Trainer, configure_logger, reward
from agentflow.models.memory import Memory

from flowgrpo.config_utils import config_value, export_env, load_yaml_config
from flowgrpo.reward import compute_result_reward
from flowgrpo.solver_factory import construct_flowgrpo_solver


configure_logger()


@reward
async def gsm8k_rule_reward(predicted_answer: str | None, gold_answer: str) -> float:
    from gsm8k_utils import answers_match

    return 1.0 if answers_match(predicted_answer, gold_answer) else 0.0


class GSM8KFlowGRPORollout(LitAgent):
    def __init__(
        self,
        *,
        frozen_model_name: str,
        frozen_base_url: str,
        max_steps: int,
        max_tokens: int,
        max_time: int,
        output_types: str,
        train_temperature: float,
        test_temperature: float,
        frozen_temperature: float,
        rollout_log_dir: Path,
    ) -> None:
        super().__init__(trained_agents="planner_next_step")
        self.frozen_model_name = frozen_model_name
        self.frozen_base_url = frozen_base_url
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_time = max_time
        self.output_types = output_types
        self.train_temperature = train_temperature
        self.test_temperature = test_temperature
        self.frozen_temperature = frozen_temperature
        self.rollout_log_dir = rollout_log_dir
        self.training_solver = None
        self.validation_solver = None

    def _get_solver(self, resources: NamedResources, *, val: bool):
        cached = self.validation_solver if val else self.training_solver
        if cached is not None:
            cached.memory = Memory()
            return cached

        llm: LLM = resources["main_llm"]
        solver = construct_flowgrpo_solver(
            trainable_model_name=llm.model,
            trainable_base_url=llm.endpoint,
            frozen_model_name=self.frozen_model_name,
            frozen_base_url=self.frozen_base_url,
            output_types=self.output_types,
            max_steps=self.max_steps,
            max_time=self.max_time,
            max_tokens=self.max_tokens,
            train_temperature=self.test_temperature if val else self.train_temperature,
            frozen_temperature=self.frozen_temperature,
        )
        solver.memory = Memory()
        if val:
            self.validation_solver = solver
        else:
            self.training_solver = solver
        return solver

    async def _solve_and_score(self, task: Any, rollout_id: str, resources: NamedResources, *, val: bool) -> float:
        solver = self._get_solver(resources, val=val)
        question = str(task["question"])
        gold_answer = str(task["result"])
        result: dict[str, Any]
        started_at = time.time()
        try:
            result = solver.solve(question)
            reward_value, predicted_answer = compute_result_reward(result, gold_answer)
        except Exception as exc:
            result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            predicted_answer = None
            reward_value = 0.0

        tracked_reward = await gsm8k_rule_reward(predicted_answer, gold_answer)
        self._write_rollout_log(
            rollout_id=rollout_id,
            task=task,
            result=result,
            predicted_answer=predicted_answer,
            reward_value=tracked_reward,
            val=val,
            wall_time=round(time.time() - started_at, 3),
        )
        return tracked_reward

    def _write_rollout_log(
        self,
        *,
        rollout_id: str,
        task: Any,
        result: dict[str, Any],
        predicted_answer: str | None,
        reward_value: float,
        val: bool,
        wall_time: float,
    ) -> None:
        split = "val" if val else "train"
        out_dir = self.rollout_log_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "rollout_id": rollout_id,
            "id": task.get("id"),
            "idx": task.get("extra_info", {}).get("idx"),
            "question": task.get("question"),
            "gold_answer": task.get("result"),
            "predicted_answer": predicted_answer,
            "reward": reward_value,
            "wall_time": wall_time,
            "result": result,
        }
        path = out_dir / f"{rollout_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> float:
        return await self._solve_and_score(task, rollout_id, resources, val=False)

    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> float:
        return await self._solve_and_score(task, rollout_id, resources, val=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSM8K AgentFlow rollout workers for Flow-GRPO.")
    parser.add_argument("--config", type=Path, default=FLOWGRPO_DIR / "config_smoke.yaml")
    parser.add_argument("--agentflow-port", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = load_yaml_config(config_path)
    export_env(config)

    agentflow_port = args.agentflow_port or int(config_value(config, "agentflow.port", 9999))
    n_workers = args.n_workers or int(config_value(config, "N_WORKERS", 1))
    exp_name = str(config_value(config, "EXPERIMENT_NAME", "gsm8k_flowgrpo"))
    rollout_log_dir = PROJECT_ROOT / "try_gsm8k_0522" / "flowgrpo" / "rollout_logs" / exp_name

    agent = GSM8KFlowGRPORollout(
        frozen_model_name=str(config_value(config, "FROZEN_MODEL", "Qwen3-0.6B-Frozen")),
        frozen_base_url=str(config_value(config, "FROZEN_BASE_URL", "http://localhost:8000/v1")),
        max_steps=int(config_value(config, "TOOL_STEPS", 3)),
        max_tokens=int(config_value(config, "data.max_response_length", 512)),
        max_time=int(config_value(config, "AGENT_MAX_TIMEOUT", 180)),
        output_types=str(config_value(config, "OUTPUT_TYPE", "direct")),
        train_temperature=float(config_value(config, "TRAIN_TEMPERATURE", 0.7)),
        test_temperature=float(config_value(config, "TEST_TEMPERATURE", 0.0)),
        frozen_temperature=float(config_value(config, "FROZEN_TEMPERATURE", 0.0)),
        rollout_log_dir=rollout_log_dir,
    )
    trainer = Trainer(n_workers=n_workers)
    trainer.fit(agent, f"http://localhost:{agentflow_port}/")


if __name__ == "__main__":
    main()
