from __future__ import annotations

from importlib import import_module

import pytest


def actions_module():
    try:
        return import_module("agentflow_rl.runtime.actions")
    except ModuleNotFoundError:
        pytest.fail("strict ToolAction parsing must be implemented", pytrace=False)


def test_tool_action_accepts_one_json_object_after_optional_think() -> None:
    module = actions_module()
    action = module.ToolAction.model_validate_json_response(
        '<think>choose lookup</think>\n{"tool_name":"Ticket_Query_Tool","arguments":{"value":"C-1"}}'
    )

    assert action.tool_name == "Ticket_Query_Tool"
    assert action.arguments == {"value": "C-1"}


def test_tool_action_accepts_one_fenced_json_object_after_optional_think() -> None:
    module = actions_module()
    action = module.ToolAction.model_validate_json_response(
        '<think>\n\n</think>\n```json\n{"tool_name":"Ticket_Query_Tool","arguments":{"value":"C-1"}}\n```'
    )

    assert action.tool_name == "Ticket_Query_Tool"


@pytest.mark.parametrize(
    "text",
    [
        'Use this: {"tool_name":"Ticket_Query_Tool","arguments":{}}',
        '{"tool_name":"Ticket_Query_Tool","arguments":{},"extra":1}',
        '{"tool_name":"A","arguments":{}} {"tool_name":"B","arguments":{}}',
        '<think>unfinished {"tool_name":"A","arguments":{}}',
    ],
)
def test_tool_action_rejects_prose_extra_keys_and_multiple_objects(text: str) -> None:
    module = actions_module()

    with pytest.raises(module.ActionParseError):
        module.ToolAction.model_validate_json_response(text)


def test_failure_kind_distinguishes_model_and_infrastructure_failures() -> None:
    module = import_module("agentflow_rl.runtime.errors")

    assert module.FailureKind.MODEL_VALID.value == "model_valid"
    assert module.FailureKind.INFRASTRUCTURE_INVALID.value == "infrastructure_invalid"


def test_tool_action_rejects_nonstandard_json_nan() -> None:
    module = actions_module()

    with pytest.raises(module.ActionParseError, match="exactly one JSON object"):
        module.ToolAction.model_validate_json_response(
            '{"tool_name":"x","arguments":{"value":NaN}}'
        )


def test_tool_event_exposes_task_operation_without_rollout_runtime() -> None:
    module = actions_module()
    event = module.ToolEvent(
        turn_index=0,
        tool_name="Ticket_Update_Tool",
        arguments={"ticket_id": "T-1"},
        result={"ok": True},
        ok=True,
    )

    assert event.operation == "update"
