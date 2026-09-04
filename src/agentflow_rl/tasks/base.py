from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    reward: float
    failure_codes: tuple[str, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        if not 0.0 <= self.reward <= 1.0:
            raise ValueError("reward must be within [0, 1]")


class BinaryVerificationResult(VerificationResult):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        expected = 1.0 if self.success else 0.0
        if self.reward != expected:
            raise ValueError("binary reward must equal success")
