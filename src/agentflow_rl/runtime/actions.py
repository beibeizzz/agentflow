from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import ActionParseError


def strip_optional_think_prefix(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("<think>"):
        closing = candidate.find("</think>")
        if closing < 0:
            raise ActionParseError("unclosed think block")
        candidate = candidate[closing + len("</think>") :].strip()
    elif "<think>" in candidate or "</think>" in candidate:
        raise ActionParseError("think block must be one optional prefix")
    return candidate


def strict_json_object(text: str) -> dict[str, Any]:
    candidate = strip_optional_think_prefix(text)
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"} or lines[-1].strip() != "```":
            raise ActionParseError("JSON fence must wrap exactly one object")
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(
            candidate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ActionParseError("response must be exactly one JSON object") from exc
    if not isinstance(payload, dict):
        raise ActionParseError("response must be exactly one JSON object")
    return payload


class ToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]

    @classmethod
    def model_validate_json_response(cls, text: str) -> "ToolAction":
        payload = strict_json_object(text)
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ActionParseError("action does not match the strict schema") from exc


class ToolEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_index: int = Field(ge=0)
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    ok: bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def operation(self) -> str:
        return {
            "Ticket_Query_Tool": "query",
            "Ticket_Update_Tool": "update",
            "Ticket_Finish_Tool": "finish",
            "Calculator_Tool": "calculate",
            "Research_Search_Tool": "search",
            "Research_Read_Tool": "read",
            "Base_Generator_Tool": "generate",
            "Code_Write_Tool": "write_code",
            "Code_Run_Tests_Tool": "run_tests",
        }.get(self.tool_name, "invalid")
