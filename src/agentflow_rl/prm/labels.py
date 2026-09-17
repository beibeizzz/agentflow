from __future__ import annotations

import asyncio
from collections.abc import Iterable

from agentflow_rl.rewards.process import ProcessScorer
from agentflow_rl.rewards.schemas import ProcessTransition

from .dataset import ProcessLabel


async def label_transitions(
    transitions: Iterable[ProcessTransition],
    labeler: ProcessScorer,
    *,
    max_concurrency: int = 8,
) -> list[ProcessLabel]:
    if max_concurrency <= 0:
        raise ValueError("label concurrency must be positive")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def label_one(transition: ProcessTransition) -> ProcessLabel:
        async with semaphore:
            score = await labeler.score(transition)
            return ProcessLabel.from_score(transition, score)

    return list(await asyncio.gather(*(label_one(item) for item in transitions)))


__all__ = ["label_transitions"]
