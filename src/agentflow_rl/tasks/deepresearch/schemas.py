from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentflow_rl.runtime.actions import strict_json_object
from agentflow_rl.runtime.errors import ActionParseError


ResearchToolName = Literal[
    "Research_Search_Tool",
    "Research_Read_Tool",
    "Base_Generator_Tool",
]


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    sentence_id: int = Field(ge=0)


class ResearchAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sub_goal: str = Field(min_length=1)
    tool_name: ResearchToolName
    arguments: dict[str, Any]

    @classmethod
    def parse(cls, text: str) -> "ResearchAction":
        try:
            return cls.model_validate(strict_json_object(text))
        except (ActionParseError, ValidationError, TypeError, ValueError) as exc:
            raise ActionParseError("research action must be one strict JSON object") from exc


class ResearchFinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str = Field(min_length=1)
    report: str = Field(min_length=1)
    citations: tuple[Citation, ...]

    @classmethod
    def parse(cls, text: str) -> "ResearchFinalAnswer":
        try:
            return cls.model_validate(strict_json_object(text))
        except (ActionParseError, ValidationError, TypeError, ValueError) as exc:
            raise ActionParseError(
                "research final answer must be one strict JSON object"
            ) from exc


class DeepResearchExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    dataset: Literal["hotpotqa", "2wiki"]
    question: str
    answer: str
    supporting_facts: tuple[Citation, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "DeepResearchExample":
        dataset = str(row.get("dataset", row.get("source", "hotpotqa"))).lower()
        dataset = "2wiki" if "2wiki" in dataset else "hotpotqa"
        facts = row.get("supporting_facts", ())
        if isinstance(facts, dict):
            titles = facts.get("title", ())
            sentence_ids = facts.get("sent_id", facts.get("sentence_id", ()))
            facts = list(zip(titles, sentence_ids, strict=False))
        citations = tuple(
            fact if isinstance(fact, Citation) else Citation(title=str(fact[0]), sentence_id=int(fact[1]))
            for fact in facts
        )
        identity = row.get("episode_id", row.get("id", row.get("_id")))
        if identity is None:
            raise ValueError("DeepResearch row requires a stable identity")
        return cls(
            episode_id=str(identity),
            dataset=dataset,
            question=str(row["question"]),
            answer=str(row["answer"]),
            supporting_facts=citations,
            metadata=dict(row.get("metadata", {})),
        )
