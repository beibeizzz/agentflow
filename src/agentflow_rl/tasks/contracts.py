from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from agentflow_rl.runtime.contracts import (
    FinalAnswerEnvelope,
    PrivateEvaluationRecordRef,
    PublicTaskRecord,
    TaskName,
)


TERMINAL_EVALUATOR_REVISION = "terminal-evaluators-v3-taco-spj-safe"


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    reward: float = Field(ge=0.0, le=1.0)
    failure_codes: tuple[str, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)


class PrivateEvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_name: TaskName
    payload: dict[str, Any]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def model_post_init(self, __context: object) -> None:
        expected = private_record_sha256(
            task_id=self.task_id,
            task_name=self.task_name,
            payload=self.payload,
        )
        if self.sha256 != expected:
            raise ValueError("private evaluation record hash mismatch")


class PrivateEvaluationStore(Protocol):
    def get(self, reference: PrivateEvaluationRecordRef) -> PrivateEvaluationRecord: ...


class TaskAdapter(Protocol):
    task_name: TaskName
    task_instructions: str
    final_answer_instructions: str

    def render_task(self, record: PublicTaskRecord) -> str: ...

    def parse_final_answer(self, text: str) -> FinalAnswerEnvelope: ...


class TerminalEvaluator(Protocol):
    task_name: TaskName

    async def evaluate(
        self,
        answer: FinalAnswerEnvelope,
        reference: PrivateEvaluationRecordRef,
    ) -> VerificationResult: ...


def private_record_sha256(
    *, task_id: str, task_name: TaskName, payload: dict[str, Any]
) -> str:
    canonical = json.dumps(
        {
            "task_id": task_id,
            "task_name": task_name.value,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class InMemoryPrivateEvaluationStore:
    def __init__(self, records: tuple[PrivateEvaluationRecord, ...]) -> None:
        self._records = {record.key: record for record in records}
        if len(self._records) != len(records):
            raise ValueError("duplicate private evaluation key")

    def get(self, reference: PrivateEvaluationRecordRef) -> PrivateEvaluationRecord:
        record = self._records[reference.key]
        if record.sha256 != reference.sha256:
            raise ValueError("private evaluation reference hash mismatch")
        return record


class JsonlPrivateEvaluationStore(InMemoryPrivateEvaluationStore):
    def __init__(self, path: str | Path) -> None:
        records = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(PrivateEvaluationRecord.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(
                        f"invalid private evaluation record at line {line_number}"
                    ) from exc
        super().__init__(tuple(records))


class TaskRegistry:
    def __init__(
        self,
        adapters: tuple[TaskAdapter, ...],
        evaluators: tuple[TerminalEvaluator, ...],
    ) -> None:
        self._adapters = {item.task_name: item for item in adapters}
        self._evaluators = {item.task_name: item for item in evaluators}
        if len(self._adapters) != len(adapters):
            raise ValueError("duplicate task adapter")
        if len(self._evaluators) != len(evaluators):
            raise ValueError("duplicate terminal evaluator")
        if set(self._adapters) != set(self._evaluators):
            raise ValueError("task adapters and terminal evaluators must align")

    def adapter(self, task_name: TaskName) -> TaskAdapter:
        return self._adapters[task_name]

    def evaluator(self, task_name: TaskName) -> TerminalEvaluator:
        return self._evaluators[task_name]


__all__ = [
    "PrivateEvaluationRecord",
    "InMemoryPrivateEvaluationStore",
    "JsonlPrivateEvaluationStore",
    "PrivateEvaluationStore",
    "TaskAdapter",
    "TaskRegistry",
    "TERMINAL_EVALUATOR_REVISION",
    "TerminalEvaluator",
    "VerificationResult",
    "private_record_sha256",
]
