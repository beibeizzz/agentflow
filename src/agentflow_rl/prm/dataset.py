from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from agentflow_rl.rewards.schemas import ProcessScore, ProcessTransition
from agentflow_rl.rewards.transition_view import PROCESS_VIEW_REVISION


class ProcessLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transition: ProcessTransition
    target: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    failure_code: str | None = None
    labeler_revision: str = Field(min_length=1)
    input_view_revision: str = "legacy"
    rubric_revision: str = "legacy"

    @classmethod
    def from_score(
        cls, transition: ProcessTransition, score: ProcessScore
    ) -> "ProcessLabel":
        return cls(
            transition=transition,
            target=score.score,
            confidence=score.confidence,
            reason=score.reason,
            failure_code=score.failure_code,
            labeler_revision=score.scorer_revision,
            input_view_revision=PROCESS_VIEW_REVISION,
            rubric_revision=score.rubric_revision,
        )


def prompt_fingerprint(transition: ProcessTransition) -> str:
    task = transition.task
    payload = json.dumps(
        {
            "task_name": task.task_name.value,
            "prompt": task.prompt,
            "public_payload": task.public_payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_labels_by_prompt(
    labels: list[ProcessLabel], *, dev_fraction: float = 0.1, seed: str = "agentflow"
) -> tuple[list[ProcessLabel], list[ProcessLabel]]:
    """Create deterministic splits while keeping every prompt in one split."""
    if not 0.0 <= dev_fraction <= 1.0:
        raise ValueError("dev_fraction must be in [0, 1]")

    groups: dict[str, list[ProcessLabel]] = defaultdict(list)
    for label in labels:
        groups[prompt_fingerprint(label.transition)].append(label)

    train: list[ProcessLabel] = []
    dev: list[ProcessLabel] = []
    cutoff = int(dev_fraction * 10_000)
    for prompt_id in sorted(groups):
        digest = hashlib.sha256(f"{seed}:{prompt_id}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % 10_000
        (dev if bucket < cutoff else train).extend(groups[prompt_id])
    return train, dev


def split_labels_by_trajectory(
    labels: list[ProcessLabel], *, dev_fraction: float = 0.1, seed: str = "agentflow"
) -> tuple[list[ProcessLabel], list[ProcessLabel]]:
    return split_labels_by_prompt(labels, dev_fraction=dev_fraction, seed=seed)


__all__ = [
    "ProcessLabel",
    "prompt_fingerprint",
    "split_labels_by_prompt",
    "split_labels_by_trajectory",
]
