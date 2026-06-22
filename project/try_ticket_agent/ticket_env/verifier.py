from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agentflow.models.formatters import MemoryVerification
from agentflow.models.memory import Memory
from agentflow.tools.ticket_common.backend import TicketBackend


TICKET_TOOLS = {"Ticket_Query_Tool", "Ticket_Update_Tool", "Ticket_Finish_Tool"}


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    failure_codes: tuple[str, ...]
    finish_outcome_correct: bool
    invalid_action_count: int
    tool_error_count: int
    collateral_mutations: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_codes"] = list(self.failure_codes)
        return payload


class TicketVerifier:
    def __init__(self, backend: TicketBackend, *, max_steps: int) -> None:
        self.backend = backend
        self.max_steps = int(max_steps)

    def verificate_context(
        self, question: str, image: object, query_analysis: str,
        memory: Memory, step_count: int = 0, json_data: Any = None,
    ) -> MemoryVerification:
        finished = self.backend.finish_submission is not None
        return MemoryVerification(
            analysis=(
                "A finish submission was received; verify the final sandbox state."
                if finished
                else "No finish submission yet; continue the ticket workflow."
            ),
            stop_signal=finished,
        )

    def extract_conclusion(self, response: MemoryVerification) -> tuple[str, str]:
        return response.analysis, "STOP" if response.stop_signal else "CONTINUE"

    def verify_final(self, memory: Memory, step_count: int) -> VerificationResult:
        goal = self.backend.goal_spec
        if goal is None:
            raise RuntimeError("Ticket backend has no hidden goal")
        failures: list[str] = []
        actions = list(memory.get_actions().values())
        invalid_action_count = sum(
            1 for action in actions if action.get("tool_name") not in TICKET_TOOLS
        )
        tool_error_count = sum(
            1
            for action in actions
            if action.get("tool_name") in TICKET_TOOLS
            and (
                not isinstance(action.get("result"), dict)
                or action["result"].get("ok") is not True
            )
        )
        if invalid_action_count:
            failures.append("INVALID_ACTION")
        if tool_error_count:
            failures.append("TOOL_ERROR")
        if step_count > self.max_steps:
            failures.append("STEP_LIMIT")

        target = self.backend.tickets.get(goal.target_ticket_id)
        if target is None or str(getattr(target, goal.field)) != goal.value:
            failures.append("GOAL_NOT_MET")

        diff = self.backend.state_diff()
        expected_target_diff = diff.get(goal.target_ticket_id, {})
        collateral = sum(
            len(fields)
            for ticket_id, fields in diff.items()
            if ticket_id != goal.target_ticket_id
        )
        collateral += sum(1 for field in expected_target_diff if field != goal.field)
        if collateral:
            failures.append("COLLATERAL_MUTATION")

        submission = self.backend.finish_submission
        finish_correct = bool(
            submission is not None
            and submission.ticket_id == goal.target_ticket_id
            and submission.outcome == goal.finish_outcome
        )
        if submission is None:
            failures.append("MISSING_FINISH")
        elif not finish_correct:
            failures.append("WRONG_FINISH")

        unique_failures = tuple(dict.fromkeys(failures))
        return VerificationResult(
            success=not unique_failures,
            failure_codes=unique_failures,
            finish_outcome_correct=finish_correct,
            invalid_action_count=invalid_action_count,
            tool_error_count=tool_error_count,
            collateral_mutations=collateral,
        )
