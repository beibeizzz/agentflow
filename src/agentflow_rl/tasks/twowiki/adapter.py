from __future__ import annotations

from pydantic import ValidationError

from agentflow_rl.runtime.contracts import FinalAnswerEnvelope, PublicTaskRecord, TaskName
from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.runtime.parsing import strict_json_object


class TwoWikiAdapter:
    task_name = TaskName.TWOWIKI
    task_instructions = (
        "Answer the multi-hop question. Use the available tools to gather and verify "
        "the information needed for the final answer."
    )
    final_answer_instructions = '{"answer":"...","public_artifacts":{}}'

    def render_task(self, record: PublicTaskRecord) -> str:
        return record.prompt

    def parse_final_answer(self, text: str) -> FinalAnswerEnvelope:
        try:
            payload = strict_json_object(text)
            artifacts = dict(payload.get("public_artifacts", {}))
            return FinalAnswerEnvelope(
                task_name=self.task_name,
                answer=str(payload["answer"]).strip(),
                public_artifacts=artifacts,
            )
        except (ActionParseError, KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ActionParseError("2Wiki final answer violates the required schema") from exc


__all__ = ["TwoWikiAdapter"]
