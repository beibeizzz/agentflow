from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentflow_rl.runtime.actions import strict_json_object, strip_optional_think_prefix
from agentflow_rl.runtime.errors import ActionParseError


CodeToolName = Literal[
    "Base_Generator_Tool",
    "Code_Write_Tool",
    "Code_Run_Tests_Tool",
]


class CodeTest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stdin: str | None = None
    expected_stdout: str | None = None
    fn_name: str | None = None
    args: Any = None
    expected: Any = None

    def model_post_init(self, __context: object) -> None:
        stdin_case = self.stdin is not None and self.expected_stdout is not None
        function_case = self.fn_name is not None
        if stdin_case == function_case:
            raise ValueError("test must define exactly one stdin or function case")


class CodeExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    question: str
    difficulty: Literal["EASY", "MEDIUM"]
    public_tests: tuple[CodeTest, ...]
    hidden_tests: tuple[CodeTest, ...]
    starter_code: str = ""
    source: str = "taco-verified"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CodeExample":
        identity = row.get("episode_id", row.get("id", row.get("task_id")))
        question = row.get("question", row.get("prompt"))
        if identity is None or not question:
            raise ValueError("coding row requires identity and question")
        public = tuple(CodeTest.model_validate(item) for item in row.get("public_tests", ()))
        hidden = tuple(CodeTest.model_validate(item) for item in row.get("hidden_tests", ()))
        if not public or not hidden:
            raise ValueError("coding row requires public and hidden tests")
        return cls(
            episode_id=str(identity),
            question=str(question),
            difficulty=str(row["difficulty"]).upper(),
            public_tests=public,
            hidden_tests=hidden,
            starter_code=str(row.get("starter_code", "")),
            source=str(row.get("source", "taco-verified")),
            metadata=dict(row.get("metadata", {})),
        )


class CodeAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sub_goal: str = Field(min_length=1)
    tool_name: CodeToolName
    arguments: dict[str, Any]

    @classmethod
    def parse(cls, text: str) -> "CodeAction":
        try:
            return cls.model_validate(strict_json_object(text))
        except (ActionParseError, ValidationError, TypeError, ValueError) as exc:
            raise ActionParseError("code action must be one strict JSON object") from exc


class FinalCode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)

    @classmethod
    def parse(cls, text: str) -> "FinalCode":
        try:
            return cls.model_validate(strict_json_object(text))
        except ActionParseError as json_error:
            candidate = strip_optional_think_prefix(text)
            lines = candidate.splitlines()
            if (
                len(lines) >= 3
                and lines[0].strip().lower() in {"```python", "```py"}
                and lines[-1].strip() == "```"
            ):
                return cls(code="\n".join(lines[1:-1]).strip())
            raise ActionParseError(
                "final code must be one JSON object or one Python fence"
            ) from json_error


def split_tests(
    tests: list[CodeTest], *, identity: str, public_fraction: float = 0.2
) -> tuple[tuple[CodeTest, ...], tuple[CodeTest, ...]]:
    if len(tests) < 2:
        raise ValueError("at least two tests are required")
    if not 0.0 < public_fraction < 1.0:
        raise ValueError("public_fraction must be within (0, 1)")
    ranked = sorted(
        enumerate(tests),
        key=lambda pair: hashlib.sha256(f"{identity}:{pair[0]}".encode()).digest(),
    )
    public_count = min(len(tests) - 1, max(1, round(len(tests) * public_fraction)))
    public_ids = {index for index, _ in ranked[:public_count]}
    public = tuple(test for index, test in enumerate(tests) if index in public_ids)
    hidden = tuple(test for index, test in enumerate(tests) if index not in public_ids)
    return public, hidden
