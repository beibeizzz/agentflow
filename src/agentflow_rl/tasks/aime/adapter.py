from __future__ import annotations

from pydantic import ValidationError

from agentflow_rl.runtime.contracts import FinalAnswerEnvelope, PublicTaskRecord, TaskName
from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.runtime.parsing import strict_json_object


class AIMEAdapter:
    task_name = TaskName.AIME
    task_instructions = (
        "Solve the AIME problem and produce its integer answer. Use any shared tool "
        "that helps establish a reliable derivation."
    )
    final_answer_instructions = '{"answer":"integer","public_artifacts":{}}'

    def render_task(self, record: PublicTaskRecord) -> str:
        return record.prompt

    def parse_final_answer(self, text: str) -> FinalAnswerEnvelope:
        try:
            payload = strict_json_object(text)
            return FinalAnswerEnvelope(
                task_name=self.task_name,
                answer=str(payload["answer"]).strip(),
                public_artifacts=dict(payload.get("public_artifacts", {})),
            )
        except (ActionParseError, KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ActionParseError("AIME final answer violates the required schema") from exc


__all__ = ["AIMEAdapter"]
