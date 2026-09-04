from __future__ import annotations

from typing import Iterable

from agentflow_rl.runtime.actions import ToolEvent
from agentflow_rl.tasks.base import BinaryVerificationResult

from .environment import TicketEnvironment


TICKET_TOOLS = {"Ticket_Query_Tool", "Ticket_Update_Tool", "Ticket_Finish_Tool"}


class TicketVerificationResult(BinaryVerificationResult):
    finish_outcome_correct: bool
    invalid_action_count: int
    tool_error_count: int
    collateral_mutations: int


def verify_ticket(
    environment: TicketEnvironment,
    events: Iterable[ToolEvent],
    *,
    step_count: int,
    max_steps: int,
) -> TicketVerificationResult:
    failures: list[str] = []
    actions = tuple(events)
    invalid_action_count = sum(1 for event in actions if event.tool_name not in TICKET_TOOLS)
    tool_error_count = sum(
        1
        for event in actions
        if event.tool_name in TICKET_TOOLS
        and (not isinstance(event.result, dict) or event.result.get("ok") is not True)
    )
    if invalid_action_count:
        failures.append("INVALID_ACTION")
    if tool_error_count:
        failures.append("TOOL_ERROR")
    if step_count > max_steps:
        failures.append("STEP_LIMIT")

    goal = environment.goal_spec
    target = environment.tickets.get(goal.target_ticket_id)
    if target is None or str(getattr(target, goal.field)) != goal.value:
        failures.append("GOAL_NOT_MET")

    diff = environment.state_diff()
    expected_target_diff = diff.get(goal.target_ticket_id, {})
    collateral = sum(
        len(fields) for ticket_id, fields in diff.items() if ticket_id != goal.target_ticket_id
    )
    collateral += sum(1 for field in expected_target_diff if field != goal.field)
    if collateral:
        failures.append("COLLATERAL_MUTATION")

    submission = environment.finish_submission
    finish_correct = bool(
        submission is not None
        and submission.ticket_id == goal.target_ticket_id
        and submission.outcome == goal.finish_outcome
    )
    if submission is None:
        failures.append("MISSING_FINISH")
    elif not finish_correct:
        failures.append("WRONG_FINISH")
    failure_codes = tuple(dict.fromkeys(failures))
    success = not failure_codes
    return TicketVerificationResult(
        success=success,
        reward=1.0 if success else 0.0,
        failure_codes=failure_codes,
        finish_outcome_correct=finish_correct,
        invalid_action_count=invalid_action_count,
        tool_error_count=tool_error_count,
        collateral_mutations=collateral,
    )
