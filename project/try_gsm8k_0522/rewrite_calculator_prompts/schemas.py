from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RewriteCandidate:
    rewritten_question: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RewriteCandidate":
        value = payload.get("rewritten_question")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("rewritten_question must be a non-empty string")
        return cls(rewritten_question=value.strip())


@dataclass(frozen=True)
class JudgeDecision:
    accepted: bool
    reasons: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "JudgeDecision":
        accepted = payload.get("accepted")
        reasons = payload.get("reasons", [])
        if not isinstance(accepted, bool):
            raise ValueError("accepted must be a boolean")
        if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
            raise ValueError("reasons must be a list of strings")
        return cls(accepted=accepted, reasons=tuple(item.strip() for item in reasons if item.strip()))


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...] = ()
    facts: tuple[str, ...] = ()
    question: str = ""

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    @property
    def messages(self) -> tuple[str, ...]:
        return tuple(issue.message for issue in self.issues)


@dataclass(frozen=True)
class ApiUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class JsonResponse:
    payload: dict[str, Any]
    usage: ApiUsage = field(default_factory=ApiUsage)

