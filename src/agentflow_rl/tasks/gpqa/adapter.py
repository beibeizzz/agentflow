from __future__ import annotations

from agentflow_rl.runtime.contracts import FinalAnswerEnvelope, PublicTaskRecord, TaskName
from agentflow_rl.runtime.errors import ActionParseError
from agentflow_rl.runtime.parsing import strict_json_object


class GPQAAdapter:
    task_name = TaskName.GPQA
    task_instructions = (
        "Select the best option for the graduate-level multiple-choice question and "
        "ground the selection in reliable reasoning or evidence."
    )
    final_answer_instructions = '{"answer":"A|B|C|D","public_artifacts":{}}'

    def render_task(self, record: PublicTaskRecord) -> str:
        return record.prompt

    def parse_final_answer(self, text: str) -> FinalAnswerEnvelope:
        try:
            payload = strict_json_object(text)
            answer = str(payload["answer"]).strip().upper()
            if answer not in {"A", "B", "C", "D"}:
                raise ValueError("GPQA answer must be one option label")
            return FinalAnswerEnvelope(task_name=self.task_name, answer=answer)
        except (ActionParseError, KeyError, TypeError, ValueError) as exc:
            raise ActionParseError("GPQA final answer violates the required schema") from exc


__all__ = ["GPQAAdapter"]
