from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RoleName(StrEnum):
    QUERY_ANALYZER = "query_analyzer"
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    GENERATOR = "generator"


class PlannerGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str
    response: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    model_revision: str = Field(min_length=1)
    policy_update_id: str | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.prompt_ids:
            raise ValueError("Planner generation requires prompt tokens")
        if not self.response_ids:
            raise ValueError("Planner generation requires response tokens")
        if len(self.response_ids) != len(self.response_logprobs):
            raise ValueError("Planner response IDs and log-probabilities must align")


__all__ = ["PlannerGeneration", "RoleName"]
