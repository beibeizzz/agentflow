from __future__ import annotations

from agentflow.tools.base import BaseTool
from agentflow.tools.ticket_common.backend import TicketBackend, result

TOOL_NAME = "Ticket_Update_Tool"


class Ticket_Update_Tool(BaseTool):
    require_llm_engine = False

    def __init__(self, model_string: str | None = None) -> None:
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Atomically update one allowed field on one sandbox ticket.",
            tool_version="1.0.0",
            input_types={"ticket_id": "str", "field": "str", "value": "str"},
            output_type="dict",
            demo_commands=[],
            model_string=model_string,
        )
        self.backend: TicketBackend | None = None

    def bind_backend(self, backend: TicketBackend) -> None:
        self.backend = backend

    def execute(self, ticket_id: str, field: str, value: str) -> dict:
        if self.backend is None:
            return result(False, "BACKEND_NOT_BOUND", "Ticket backend is not bound.")
        return self.backend.update(ticket_id, field, value)
