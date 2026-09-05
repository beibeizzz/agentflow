from __future__ import annotations


def test_query_analysis_prompt_is_short_and_describes_both_workflows() -> None:
    from agentflow_rl.tasks.ticket.prompts import render_query_analysis_prompt

    prompt = render_query_analysis_prompt("change ticket")

    assert "Direct workflow: update the ticket, then submit completion." in prompt
    assert "Indirect workflow: query by customer_id or order_id" in prompt
    assert "Base_Generator_Tool" in prompt
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

    assert "Ticket_Query_Tool" in prompt
    assert "Ticket_Update_Tool" in prompt
    assert "Ticket_Finish_Tool" in prompt
    assert "Base_Generator_Tool" in prompt
    assert '"arguments":{...}' in prompt
    assert "data.ticket_id" in prompt
    assert '"ticket_id": "T-1"' in prompt
    assert "curriculum_mode" not in prompt
    assert len(prompt.split()) < 190


def test_ticket_action_exposes_sub_goal_and_accepts_legacy_checkpoint_shape() -> None:
    from agentflow_rl.tasks.ticket.schemas import TicketAction

    action = TicketAction.parse(
        '{"sub_goal":"locate ticket","tool_name":"Ticket_Query_Tool",'
        '"arguments":{"lookup_by":"customer_id","value":"C-1"}}'
    )

    assert action.sub_goal == "locate ticket"
    assert action.tool_name == "Ticket_Query_Tool"
    legacy = TicketAction.parse(
        '{"tool_name":"Ticket_Update_Tool",'
        '"arguments":{"ticket_id":"T-1","field":"priority","value":"high"}}'
    )
    assert legacy.sub_goal == "execute the selected ticket operation"
