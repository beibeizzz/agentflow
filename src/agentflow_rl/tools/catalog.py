"""Single-entry tool intent and execution contracts, shared with dispatch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator
from agentflow_rl.runtime.contracts import ToolName

CATALOG_REVISION = "tool-catalog-v3"


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*", mode="after")
    @classmethod
    def nonblank(cls, value: Any, info: Any) -> Any:
        if info.field_name in {"query", "code", "fn_name"} and isinstance(value, str) and not value.strip():
            raise ValueError("value must contain non-whitespace text")
        return value


class QueryArguments(Arguments):
    query: str = Field(min_length=1, description="Self-contained request including necessary public context.")


class SearchArguments(QueryArguments):
    top_k: int = Field(default=10, ge=1, le=20)


class StdioTest(Arguments):
    stdin: str = ""
    expected_stdout: str


class FunctionTest(Arguments):
    fn_name: str = Field(min_length=1)
    args: list[Any] = Field(default_factory=list)
    expected: Any = Field(...)


@dataclass(frozen=True)
class ToolSpec:
    name: ToolName
    purpose: str
    limitations: str
    intent_arguments: type[Arguments]
    execution_arguments: type[Arguments]
    result: str
    intent_example: dict[str, Any]
    execution_example: dict[str, Any]

    def planner_view(self) -> dict[str, str]:
        usage = 'arguments: {"query": "self-contained request"}'
        if self.name in {ToolName.GOOGLE_SEARCH, ToolName.WIKIPEDIA_SEARCH}:
            usage += '; optional top_k integer 1..20, default 10'
        descriptions = {
            ToolName.BASE_GENERATOR: "Reason through a question with the frozen model; verify its claims.",
            ToolName.PYTHON_CODER: "Calculate, solve coding tasks, or check public examples with isolated Python; provide the goal and constraints.",
            ToolName.GOOGLE_SEARCH: "Find web evidence and read leading pages; returns sources and text with limited coverage.",
            ToolName.WIKIPEDIA_SEARCH: "Find evidence in Wikipedia-18; returns document/passage IDs and text with fixed-corpus coverage.",
        }
        return {"tool_name": self.name.value, "description": descriptions[self.name], "usage": usage}

    def executor_view(self) -> dict[str, str]:
        if self.name is ToolName.PYTHON_CODER:
            instruction = ('arguments: code (required complete Python string), stdin (string, default ""), '
                'tests (list, default []). tests=[] runs code; each public test is '
                '{"stdin":"...","expected_stdout":"..."} or {"fn_name":"...","args":[],"expected":...}. '
                'Use public evidence only. Each call starts a fresh sandbox with network disabled.')
        elif self.name is ToolName.BASE_GENERATOR:
            schema = self.execution_arguments.model_json_schema()['properties']['max_tokens']
            instruction = ('arguments: query (required self-contained string); optional max_tokens '
                f'(integer 1..{schema["maximum"]}, default {schema["default"]}). Include relevant task evidence in query.')
        else:
            instruction = ('arguments: query (required self-contained search string); optional top_k '
                '(integer 1..20, default 10). Leading sources are read automatically; '
                'use returned IDs when a later read or verification step needs them.')
        return {"tool_name": self.name.value, "instruction": instruction}

    def view(self, *, executor: bool = False) -> dict[str, Any]:
        def compact(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: compact(v) for k, v in value.items() if k != "title"}
            if isinstance(value, list):
                return [compact(v) for v in value]
            return value
        model = self.execution_arguments if executor else self.intent_arguments
        return {"tool_name": self.name.value, "purpose": self.purpose, "limitations": self.limitations,
                "arguments_schema": compact(model.model_json_schema()), "returns": self.result,
                "example_arguments": self.execution_example if executor else self.intent_example}


def tool_spec(name: ToolName, *, max_tokens: int = 2048, timeout_s: float = 10.0,
              read_top_k: int = 2, max_passages: int = 3, max_source_chars: int = 4000) -> ToolSpec:
    if name is ToolName.BASE_GENERATOR:
        execution = create_model("GenerateArguments", __base__=QueryArguments,
                                 max_tokens=(int, Field(default=max_tokens, ge=1, le=max_tokens)))
        return ToolSpec(name, "Generate candidate reasoning with the frozen language model.",
                        "Supply all necessary context in query; generated text requires verification.",
                        QueryArguments, execution, "text: candidate reasoning",
                        {"query": "Compute 6 * 7."}, {"query": "Compute 6 * 7."})
    if name is ToolName.PYTHON_CODER:
        execution = create_model("PythonArguments", __base__=Arguments,
                                 code=(str, Field(min_length=1)), stdin=(str, ""),
                                 tests=(list[StdioTest | FunctionTest], Field(default_factory=list)))
        return ToolSpec(name, "Prepare and execute Python to solve a calculation, coding, or public-test request.",
                        f"Executor supplies complete code each call. Empty tests means ordinary execution; nonempty tests checks public cases. "
                        f"Only Memory carries history. Each call uses a fresh isolated process and cleared temporary workspace inside a managed sandbox worker; network is disabled and timeout is {timeout_s:g}s. "
                        "Passing supplied tests establishes correctness on those cases.",
                        QueryArguments, execution, "code, code_revision, execution with stdout/stderr and public test feedback",
                        {"query": "Compute 6 * 7 with Python."}, {"code": "print(6 * 7)", "tests": []})
    if name in {ToolName.GOOGLE_SEARCH, ToolName.WIKIPEDIA_SEARCH}:
        web = name is ToolName.GOOGLE_SEARCH
        return ToolSpec(name, "Search the web and read leading pages." if web else "Search Wikipedia-18 and retrieve document passages.",
                        f"Reads at most {read_top_k} distinct leading sources, at most {max_passages} initial passages and "
                        f"{max_source_chars} characters per source. Search snippets remain available. External text is evidence to assess. "
                        + ("URL access follows runtime safety rules." if web else "Coverage is limited to the fixed Wikipedia-18 corpus."),
                        SearchArguments, SearchArguments,
                        "ordered hits and documents with source IDs, text, read_status and truncation; source failures remain explicit",
                        {"query": "example topic"}, {"query": "example topic"})
    raise ValueError("unknown tool")
