from __future__ import annotations


def test_query_analysis_prompt_is_short_and_describes_both_workflows() -> None:
    from agentflow_rl.tasks.ticket.prompts import render_query_analysis_prompt

    prompt = render_query_analysis_prompt("change ticket")

    assert "Direct: update the ticket, then finish." in prompt
    assert "Indirect: query by customer_id or order_id" in prompt
    assert "curriculum_mode" not in prompt
    assert "not a math problem" not in prompt.lower()
    assert len(prompt.split()) < 80


def test_next_step_prompt_keeps_short_union_schema_and_observation() -> None:
    from agentflow_rl.tasks.ticket.prompts import render_next_step_prompt

    prompt = render_next_step_prompt(
        question="Locate C-1 and set priority high.",
        analysis="Query, update, finish.",
        events=[
            {
                "tool_name": "Ticket_Query_Tool",
                "result": {"ok": True, "code": "OK", "data": {"ticket_id": "T-1"}},
            }
        ],
    )

    assert '"tool_name": "Ticket_Query_Tool | Ticket_Update_Tool | Ticket_Finish_Tool"' in prompt
    assert '"arguments": {}' in prompt
    assert "result data.ticket_id" in prompt
    assert '"ticket_id": "T-1"' in prompt
    assert "curriculum_mode" not in prompt
    assert len(prompt.split()) < 190
