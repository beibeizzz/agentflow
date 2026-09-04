from __future__ import annotations

from agentflow_rl.runtime.actions import ToolEvent

from test_ticket_environment import environment


def event(index: int, tool: str, result: dict) -> ToolEvent:
    return ToolEvent(
        turn_index=index,
        tool_name=tool,
        arguments={},
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
    assert result.failure_codes == ("INVALID_ACTION", "TOOL_ERROR", "STEP_LIMIT")
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

    assert result.failure_codes == ("COLLATERAL_MUTATION", "WRONG_FINISH")
    assert result.collateral_mutations == 1


def test_verifier_rejects_missing_goal_and_finish() -> None:
    from agentflow_rl.tasks.ticket.verifier import verify_ticket

    result = verify_ticket(environment(), [], step_count=0, max_steps=2)

    assert result.failure_codes == ("GOAL_NOT_MET", "MISSING_FINISH")
