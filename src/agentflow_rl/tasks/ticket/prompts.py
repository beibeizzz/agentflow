from __future__ import annotations

import json
from typing import Any, Iterable


TICKET_QUERY_ANALYSIS_SYSTEM_PROMPT = "Plan concise ticket workflows."
TICKET_NEXT_STEP_SYSTEM_PROMPT = "Choose the next ticket tool call."

TICKET_QUERY_ANALYSIS_PROMPT = """Summarize the ticket workflow in one short plan.
User request: {question}
Available tools: Ticket_Query_Tool, Ticket_Update_Tool, Ticket_Finish_Tool

Rules:
- Direct: update the ticket, then finish.
- Indirect: query by customer_id or order_id, update the returned ticket, then finish.
- Track only lookup key, target field/value, and finish outcome completed.
- Keep it concise.
"""

TICKET_NEXT_STEP_PROMPT = """Next ticket action.
Request: {question}
Plan: {analysis}
Previous steps: {events}

Choose exactly one tool.
Return exactly one JSON object with exactly these top-level keys:
{{
  "tool_name": "Ticket_Query_Tool | Ticket_Update_Tool | Ticket_Finish_Tool",
  "arguments": {{}}
}}

Argument formats:
- Ticket_Query_Tool: {{"lookup_by": "ticket_id|customer_id|order_id", "value": "..."}}
- Ticket_Update_Tool: {{"ticket_id": "...", "field": "priority|assigned_team|status", "value": "..."}}
- Ticket_Finish_Tool: {{"ticket_id": "...", "outcome": "completed"}}

Rules:
- No previous step + ticket_id: use Ticket_Update_Tool.
- No previous step + customer_id/order_id: use Ticket_Query_Tool.
- If query OK: use Ticket_Update_Tool with result data.ticket_id.
- If update OK: use Ticket_Finish_Tool with the same ticket_id.
- Never repeat an OK query or update.
- Output no markdown or prose.
"""


def render_query_analysis_prompt(question: str) -> str:
    return TICKET_QUERY_ANALYSIS_PROMPT.format(question=question)


def _event_payload(event: Any) -> Any:
    if hasattr(event, "model_dump"):
        return event.model_dump(mode="json")
    return event


def render_next_step_prompt(*, question: str, analysis: str, events: Iterable[Any]) -> str:
    observations = json.dumps(
        [_event_payload(event) for event in events], ensure_ascii=False, separators=(",", ": ")
    )
    return TICKET_NEXT_STEP_PROMPT.format(
        question=question,
        analysis=analysis,
        events=observations,
    )
