from __future__ import annotations

import json
from typing import Any, Iterable


TICKET_TOOL_SCHEMA = """Allowed tools:
- Ticket_Query_Tool: arguments={"lookup_by":"ticket_id|customer_id|order_id","value":"..."}
- Ticket_Update_Tool: arguments={"ticket_id":"...","field":"priority|assigned_team|status","value":"..."}
- Ticket_Finish_Tool: arguments={"ticket_id":"...","outcome":"completed"}
- Base_Generator_Tool: arguments={}
Output exactly {"sub_goal":"...","tool_name":"...","arguments":{...}}."""

TICKET_QUERY_ANALYSIS_SYSTEM_PROMPT = "Analyze the requested ticket workflow and required state transition."
TICKET_PLANNER_SYSTEM_PROMPT = f"Plan one ticket action.\n{TICKET_TOOL_SCHEMA}"
TICKET_EXECUTOR_SYSTEM_PROMPT = (
    f"Validate and concretize the proposed ticket action while preserving tool_name.\n{TICKET_TOOL_SCHEMA}"
)
TICKET_VERIFIER_SYSTEM_PROMPT = """Judge progress from executed ticket-tool results in memory.
Return STOP after the requested state update and a successful matching finish submission are present.
Return CONTINUE with the next required action otherwise. End with Conclusion: STOP or Conclusion: CONTINUE."""
TICKET_GENERATOR_SYSTEM_PROMPT = (
    "Use the request, executed tool results, and verifier judgements to give a concise final status."
)
TICKET_BASE_GENERATOR_SYSTEM_PROMPT = (
    "Analyze the supplied ticket state for the requested sub-goal. Output concise operational guidance."
)

TICKET_QUERY_ANALYSIS_PROMPT = """Summarize the ticket workflow in one short plan.
User request: {question}
Available tools: Ticket_Query_Tool, Ticket_Update_Tool, Ticket_Finish_Tool, Base_Generator_Tool

Rules:
- Direct workflow: update the ticket, then submit completion.
- Indirect workflow: query by customer_id or order_id, update the returned ticket_id, then submit completion.
- Track the lookup key, target field/value, and completed outcome.
- Keep it concise.
"""

TICKET_NEXT_STEP_PROMPT = """Choose the next ticket action.
Request: {question}
Memory: {memory}

Rules:
- A request containing ticket_id starts with Ticket_Update_Tool.
- A request containing customer_id or order_id starts with Ticket_Query_Tool.
- A successful query supplies data.ticket_id for Ticket_Update_Tool.
- A successful update is followed by Ticket_Finish_Tool with the same ticket_id.
- Base_Generator_Tool can analyze an unresolved sub-goal from current memory.
- Preserve each successful query and update; choose the next unfinished action.

{tool_schema}
"""


def render_query_analysis_prompt(question: str) -> str:
    return TICKET_QUERY_ANALYSIS_PROMPT.format(question=question)


def _event_payload(event: Any) -> Any:
    if hasattr(event, "model_dump"):
        return event.model_dump(mode="json")
    return event


def render_next_step_prompt(
    *,
    question: str,
    analysis: str,
    events: Iterable[Any] | None = None,
    memory: str | None = None,
) -> str:
    if memory is None:
        memory = json.dumps(
            {
                "query_analysis": analysis,
                "events": [_event_payload(event) for event in (events or ())],
            },
            ensure_ascii=False,
            separators=(",", ": "),
        )
    return TICKET_NEXT_STEP_PROMPT.format(
        question=question,
        memory=memory,
        tool_schema=TICKET_TOOL_SCHEMA,
    )


def render_executor_prompt(action: str, memory: str) -> str:
    return f"Proposed action:\n{action}\n\nRelevant memory:\n{memory}"


def render_verifier_prompt(question: str, memory: str) -> str:
    return f"Request:\n{question}\n\nExecuted workflow memory:\n{memory}"


def render_generator_prompt(question: str, memory: str) -> str:
    return f"Request:\n{question}\n\nCompleted workflow memory:\n{memory}"


__all__ = [
    "TICKET_BASE_GENERATOR_SYSTEM_PROMPT",
    "TICKET_EXECUTOR_SYSTEM_PROMPT",
    "TICKET_GENERATOR_SYSTEM_PROMPT",
    "TICKET_PLANNER_SYSTEM_PROMPT",
    "TICKET_QUERY_ANALYSIS_SYSTEM_PROMPT",
    "TICKET_VERIFIER_SYSTEM_PROMPT",
    "render_executor_prompt",
    "render_generator_prompt",
    "render_next_step_prompt",
    "render_query_analysis_prompt",
    "render_verifier_prompt",
]
