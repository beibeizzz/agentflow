from __future__ import annotations

from pydantic import ValidationError

from agentflow_rl.runtime.contracts import FinalAnswerEnvelope, PublicTaskRecord, TaskName
from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.runtime.parsing import strict_json_object, strip_optional_think_prefix


class TACOAdapter:
    task_name = TaskName.TACO
    task_instructions = (
        "Produce a Python solution for the programming problem. Use public tests for "
        "diagnosis and preserve the accepted code revision in the final response."
    )
    final_answer_instructions = '{"answer":"complete Python code","public_artifacts":{}}'

    def render_task(self, record: PublicTaskRecord) -> str:
        return record.prompt

    def parse_final_answer(self, text: str) -> FinalAnswerEnvelope:
        try:
            payload = strict_json_object(text)
            return FinalAnswerEnvelope(
                task_name=self.task_name,
                answer=str(payload["answer"]),
                public_artifacts=dict(payload.get("public_artifacts", {})),
            )
        except (ActionParseError, KeyError, TypeError, ValueError, ValidationError) as json_error:
            candidate = strip_optional_think_prefix(text)
            lines = candidate.splitlines()
            if (
                len(lines) >= 3
                and lines[0].strip().lower() in {"```python", "```py"}
                and lines[-1].strip() == "```"
            ):
                return FinalAnswerEnvelope(
                    task_name=self.task_name,
                    answer="\n".join(lines[1:-1]).strip(),
                )
            raise ActionParseError("TACO final answer violates the required schema") from json_error


__all__ = ["TACOAdapter"]
