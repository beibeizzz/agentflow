from __future__ import annotations

from typing import Protocol

from agentflow_rl.runtime.contracts import PublicTaskRecord
from agentflow_rl.runtime.loop import PlannerTurnRecord

from .schemas import ProcessScore, ProcessTransition


class ProcessScorer(Protocol):
    revision: str

    async def score(self, transition: ProcessTransition) -> ProcessScore: ...

    async def score_many(
        self, transitions: tuple[ProcessTransition, ...]
    ) -> tuple[ProcessScore, ...]: ...


def build_process_transition(
    *,
    task: PublicTaskRecord,
    trajectory_id: str,
    turn: PlannerTurnRecord,
) -> ProcessTransition:
    return ProcessTransition(
        transition_id=f"{trajectory_id}:{turn.turn_index}",
        trajectory_id=trajectory_id,
        task=task,
        turn_index=turn.turn_index,
        memory_before_action=turn.memory_before_action,
        planner_response=turn.generation.response,
        planner_action=turn.action,
        tool_request=turn.tool_request,
        tool_result=turn.tool_result,
        verifier_decision=turn.verifier_decision,
        planner_memory_view=turn.planner_memory_view,
        planner_memory_event_ids=turn.planner_memory_event_ids,
        planner_memory_compacted_event_ids=turn.planner_memory_compacted_event_ids,
        planner_memory_core=turn.planner_memory_core,
    )


__all__ = ["ProcessScorer", "build_process_transition"]
