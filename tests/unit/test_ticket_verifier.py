from __future__ import annotations

from agentflow_rl.runtime.actions import ToolEvent

from test_ticket_environment import environment


def event(
    index: int, tool: str, result: dict, arguments: dict | None = None
) -> ToolEvent:
    return ToolEvent(
        turn_index=index,
        tool_name=tool,
        arguments=arguments or {},
        result=result,
        ok=result.get("ok") is True,
    )


def successful_run():
    env = environment()
    update = env.update("T-1", "priority", "high")
    finish = env.finish("T-1", "completed")
    return env, [event(0, "Ticket_Update_Tool", update), event(1, "Ticket_Finish_Tool", finish)]


def test_binary_verifier_accepts_exact_goal_and_finish() -> None:
    from agentflow_rl.tasks.ticket.verifier import verify_ticket

    env, events = successful_run()
    result = verify_ticket(env, events, step_count=2, max_steps=2)

    assert result.success
    assert result.reward == 1.0
    assert result.failure_codes == ()
    assert result.finish_outcome_correct
    assert result.workflow_order_correct
    assert result.lookup_correct


def test_verifier_rejects_finish_before_update() -> None:
    from agentflow_rl.tasks.ticket.verifier import verify_ticket

    env = environment()
    finish = env.finish("T-1", "completed")
    update = env.update("T-1", "priority", "high")
    events = [
        event(0, "Ticket_Finish_Tool", finish),
        event(1, "Ticket_Update_Tool", update),
    ]

    result = verify_ticket(env, events, step_count=2, max_steps=2)

    assert result.reward == 0.0
    assert result.workflow_order_correct is False
    assert result.failure_codes == ("WRONG_ACTION_ORDER",)


def test_verifier_checks_indirect_lookup_field_value_and_result() -> None:
    from agentflow_rl.tasks.ticket.verifier import verify_ticket

    env = environment()
    target = env.tickets["T-1"]
    query = env.query("customer_id", target.customer_id)
    update = env.update("T-1", "priority", "high")
    finish = env.finish("T-1", "completed")
    events = [
        event(
            0,
            "Ticket_Query_Tool",
            query,
            {"lookup_by": "customer_id", "value": target.customer_id},
        ),
        event(1, "Ticket_Update_Tool", update),
        event(2, "Ticket_Finish_Tool", finish),
    ]

    correct = verify_ticket(
        env,
        events,
        step_count=3,
        max_steps=3,
        lookup_mode="customer_id",
    )
    wrong = verify_ticket(
        env,
        events,
        step_count=3,
        max_steps=3,
        lookup_mode="order_id",
    )

    assert correct.reward == 1.0
    assert correct.lookup_correct
    assert wrong.reward == 0.0
    assert wrong.lookup_correct is False
    assert wrong.failure_codes == ("WRONG_LOOKUP",)


def test_verifier_rejects_invalid_action_tool_error_and_step_limit() -> None:
    from agentflow_rl.tasks.ticket.verifier import verify_ticket

    env, events = successful_run()
    events.extend(
        [
            event(2, "Other", {"ok": False, "code": "BAD"}),
            event(3, "Ticket_Query_Tool", {"ok": False, "code": "NOT_FOUND"}),
        ]
    )

    result = verify_ticket(env, events, step_count=4, max_steps=2)

    assert not result.success
    assert result.reward == 0.0
    assert result.failure_codes == (
        "INVALID_ACTION", "TOOL_ERROR", "STEP_LIMIT", "WRONG_ACTION_ORDER"
    )
    assert result.invalid_action_count == 1
    assert result.tool_error_count == 1


def test_verifier_rejects_collateral_mutation_and_wrong_finish() -> None:
    from agentflow_rl.tasks.ticket.verifier import verify_ticket

    env = environment()
    first = env.update("T-1", "priority", "high")
    collateral = env.update("T-2", "assigned_team", "billing")
    finish = env.finish("T-2", "completed")
    events = [
        event(0, "Ticket_Update_Tool", first),
        event(1, "Ticket_Update_Tool", collateral),
        event(2, "Ticket_Finish_Tool", finish),
    ]

    result = verify_ticket(env, events, step_count=3, max_steps=3)

    assert result.failure_codes == (
        "WRONG_ACTION_ORDER", "WRONG_LOOKUP", "COLLATERAL_MUTATION", "WRONG_FINISH"
    )
    assert result.collateral_mutations == 1


def test_verifier_rejects_missing_goal_and_finish() -> None:
    from agentflow_rl.tasks.ticket.verifier import verify_ticket

    result = verify_ticket(environment(), [], step_count=0, max_steps=2)

    assert result.failure_codes == (
        "WRONG_ACTION_ORDER", "GOAL_NOT_MET", "MISSING_FINISH"
    )
