from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agentflow_rl.rewards.schemas import ProcessTransition
from agentflow_rl.runtime.contracts import PublicTaskRecord
from agentflow_rl.runtime.loop import TrajectoryResult
from agentflow_rl.runtime.privacy import assert_public_payload
from agentflow_rl.roles.prompts import PROMPT_REVISION
from agentflow_rl.tools.catalog import CATALOG_REVISION
from agentflow_rl.runtime.observations import OBSERVATION_REVISION
from agentflow_rl.runtime.projections import MEMORY_VIEW_REVISION
from agentflow_rl.rewards.transition_view import PROCESS_VIEW_REVISION
from agentflow_rl.rewards.rubric import PROCESS_RUBRIC_REVISION


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def execution_metrics(result: TrajectoryResult) -> dict:
    """Recorded workload; failed dispatch internals and service retries require service logs."""
    tool_results = [e["content"] for e in result.memory if e["kind"] == "tool_result"]
    calls = [c for r in tool_results for c in r.get("backend_calls", [])]
    roles = [e["content"] for e in result.memory if e["kind"] == "model_call"]
    known_calls = [c for c in calls if c["stage"] != "dispatch_failure"]
    stages = {stage: sum(c["stage"] == stage for c in known_calls)
              for stage in ("search", "read", "run", "test", "generate")}
    models = roles + [c for c in calls if c["stage"] == "generate"]
    backend_models = [
        call for call in models if call.get("status") != "cache_hit"
    ]
    return {
        "planner_turns": len(result.turns), "tool_calls_recorded": len(tool_results),
        "backend_calls_recorded": len(known_calls), "backend_calls_by_stage": stages,
        "dispatch_failures_with_incomplete_accounting": sum(c["stage"] == "dispatch_failure" for c in calls),
        "frozen_model_calls_recorded": len(backend_models),
        "frozen_model_logical_requests_recorded": len(models),
        "frozen_model_cache_hits_recorded": len(models) - len(backend_models),
        "frozen_calls_missing_token_counts": sum(c.get("input_tokens") is None or c.get("output_tokens") is None for c in backend_models),
        "frozen_input_tokens_recorded": sum(c.get("input_tokens") or 0 for c in backend_models),
        "frozen_output_tokens_recorded": sum(c.get("output_tokens") or 0 for c in backend_models),
        "frozen_logical_input_tokens_recorded": sum(c.get("input_tokens") or 0 for c in models),
        "frozen_logical_output_tokens_recorded": sum(c.get("output_tokens") or 0 for c in models),
        "planner_input_tokens_recorded": sum(len(t.generation.prompt_ids) for t in result.turns),
        "planner_output_tokens_recorded": sum(len(t.generation.response_ids) for t in result.turns),
        "tool_latency_ms_sum": sum(r["latency_ms"] for r in tool_results),
        "backend_latency_ms_sum": sum(c["latency_ms"] for c in known_calls),
    }


class TrajectoryArtifactWriter:
    """Persist one public, replayable artifact per trajectory without shared appends."""

    schema_version = "agentflow-trajectory-v4"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(
        self,
        *,
        task: PublicTaskRecord,
        result: TrajectoryResult,
        transitions: tuple[ProcessTransition, ...],
    ) -> Path:
        record = {
            "schema_version": self.schema_version,
            "input_protocol": {"prompt": PROMPT_REVISION, "catalog": CATALOG_REVISION,
                               "observation": OBSERVATION_REVISION, "memory": MEMORY_VIEW_REVISION,
                               "process": PROCESS_VIEW_REVISION, "rubric": PROCESS_RUBRIC_REVISION},
            "task": task.model_dump(mode="json"),
            "identity": result.identity.model_dump(mode="json"),
            "terminal_reason": result.terminal_reason,
            "valid_for_training": result.valid_for_training,
            "verification": result.verification.model_dump(mode="json"),
            "execution_metrics": execution_metrics(result),
            "memory": list(result.memory),
            "transitions": [item.model_dump(mode="json") for item in transitions],
        }
        assert_public_payload(record, path="trajectory_artifact")
        record["content_sha256"] = _canonical_hash(record)
        return self._write_record(result.identity.trajectory_id, record)

    def write_timeout(
        self,
        *,
        task: PublicTaskRecord,
        trajectory_id: str,
        session_id: int,
    ) -> Path:
        record = {
            "schema_version": self.schema_version,
            "task": task.model_dump(mode="json"),
            "identity": {
                "task_id": task.task_id,
                "trajectory_id": trajectory_id,
                "session_id": session_id,
            },
            "terminal_reason": "trajectory_timeout",
            "valid_for_training": False,
            "verification": {
                "success": False,
                "reward": 0.0,
                "failure_codes": ["TRAJECTORY_TIMEOUT"],
                "metrics": {},
            },
            "memory": [],
            "transitions": [],
        }
        assert_public_payload(record, path="trajectory_artifact")
        record["content_sha256"] = _canonical_hash(record)
        return self._write_record(trajectory_id, record)

    def _write_record(self, trajectory_id: str, record: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()
        target = self.root / f"{name}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target


def load_process_transitions(path: str | Path) -> list[ProcessTransition]:
    def parse_transition(payload: dict[str, Any]) -> ProcessTransition:
        if "planner_memory_core" not in payload:
            raise ValueError(
                "Process transition predates v5 frozen Planner Memory; regenerate trajectories"
            )
        return ProcessTransition.model_validate(payload)

    source = Path(path)
    values: list[ProcessTransition] = []
    if source.is_dir():
        for artifact in sorted(source.glob("*.json")):
            row = json.loads(artifact.read_text(encoding="utf-8"))
            transitions = row.get("transitions", ())
            if not transitions:
                continue
            if row.get("schema_version") != TrajectoryArtifactWriter.schema_version:
                raise ValueError("Trajectory artifact schema is incompatible with the current PRM input")
            if row.get("input_protocol", {}).get("process") != PROCESS_VIEW_REVISION:
                raise ValueError("Trajectory artifact process-view revision is incompatible")
            values.extend(
                parse_transition(item)
                for item in transitions
            )
        return values
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            values.append(parse_transition(row.get("transition", row)))
    return values


__all__ = ["TrajectoryArtifactWriter", "load_process_transitions"]
