from __future__ import annotations

from agentflow_rl.tasks.taco.adapter import TACOAdapter
from agentflow_rl.runtime.contracts import FinalAnswerEnvelope, TaskName


class BigCodeBenchAdapter(TACOAdapter):
    task_name = TaskName.BIGCODEBENCH
    task_instructions = (
        "Implement the requested Python function with the declared dependencies and "
        "return complete executable code."
    )

    def parse_final_answer(self, text: str) -> FinalAnswerEnvelope:
        answer = super().parse_final_answer(text)
        return answer.model_copy(update={"task_name": self.task_name})


__all__ = ["BigCodeBenchAdapter"]
