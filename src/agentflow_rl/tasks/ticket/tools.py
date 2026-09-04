from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from agentflow_rl.runtime.actions import ToolAction
from agentflow_rl.runtime.errors import ModelValidError

from .environment import TicketEnvironment


class ToolDispatchError(ModelValidError):
    pass


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QueryArguments(ToolArguments):
    lookup_by: Literal["ticket_id", "customer_id", "order_id"]
    value: str


class UpdateArguments(ToolArguments):
    ticket_id: str
    field: Literal["status", "assigned_team", "priority"]
    value: str


class FinishArguments(ToolArguments):
    ticket_id: str
    outcome: Literal["completed"]


def execute_tool(environment: TicketEnvironment, action: ToolAction) -> dict:
    schemas = {
        "Ticket_Query_Tool": QueryArguments,
        "Ticket_Update_Tool": UpdateArguments,
        "Ticket_Finish_Tool": FinishArguments,
    }
    schema = schemas.get(action.tool_name)
    if schema is None:
        raise ToolDispatchError(f"unknown ticket tool: {action.tool_name}")
    try:
        arguments = schema.model_validate(action.arguments)
    except ValidationError as exc:
        raise ToolDispatchError("ticket tool arguments do not match the strict schema") from exc
    if isinstance(arguments, QueryArguments):
        return environment.query(arguments.lookup_by, arguments.value)
    if isinstance(arguments, UpdateArguments):
        return environment.update(arguments.ticket_id, arguments.field, arguments.value)
    return environment.finish(arguments.ticket_id, arguments.outcome)
