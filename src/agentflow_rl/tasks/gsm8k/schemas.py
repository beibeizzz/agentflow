from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentflow_rl.runtime.actions import strict_json_object
from agentflow_rl.runtime.errors import ActionParseError


GSM8KToolName = Literal["Calculator_Tool", "Base_Generator_Tool"]


class GSM8KAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sub_goal: str = Field(min_length=1)
    tool_name: GSM8KToolName
    arguments: dict[str, Any]

    def model_post_init(self, __context: object) -> None:
        if self.tool_name == "Calculator_Tool":
            if set(self.arguments) != {"expression"} or not str(
                self.arguments["expression"]
            ).strip():
                raise ValueError("Calculator_Tool requires one non-empty expression")
        elif self.arguments:
            raise ValueError("Base_Generator_Tool arguments must be empty")

    @classmethod
    def parse(cls, text: str) -> "GSM8KAction":
        try:
            payload = strict_json_object(text)
            if set(payload) in ({"Sub_goal", "Calculation"}, {"sub_goal", "calculation"}):
                payload = {
                    "sub_goal": payload.get("Sub_goal") or payload.get("sub_goal"),
                    "tool_name": "Calculator_Tool",
                    "arguments": {
                        "expression": payload.get("Calculation") or payload.get("calculation")
                    },
                }
            return cls.model_validate(payload)
        except (ActionParseError, ValidationError, TypeError, ValueError) as exc:
            raise ActionParseError("GSM8K action must match one supported strict schema") from exc


__all__ = ["GSM8KAction", "GSM8KToolName"]
