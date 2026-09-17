from __future__ import annotations

import hashlib
import re

from agentflow_rl.rewards.schemas import ProcessScore, ProcessTransition

from .contracts import PageDocument, WikipediaDocument


class SmokeSearchBackend:
    revision = "smoke-empty-search-v1"

    async def search(self, *, query: str, top_k: int):
        return ()


class SmokePageReaderBackend:
    revision = "smoke-empty-page-reader-v1"

    async def open(self, *, url: str, result_id: str | None = None) -> PageDocument:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return PageDocument(
            result_id=result_id or digest[:16],
            canonical_url=url,
            title="Smoke backend",
            passages=("External page reading is disabled in this smoke run.",),
            content_sha256=digest,
        )


class SmokeWikipediaBackend:
    revision = "smoke-empty-wikipedia-v1"

    async def search(self, *, query: str, top_k: int):
        return ()

    async def read(
        self, *, doc_id: str, start_passage: int, max_passages: int
    ) -> WikipediaDocument:
        return WikipediaDocument(
            doc_id=doc_id,
            title="Smoke backend",
            passages=("Wikipedia retrieval is disabled in this smoke run.",),
            passage_ids=(f"{doc_id}:smoke",),
        )


class AlternatingSmokeProcessScorer:
    """Deterministic scorer used to exercise the real optimizer path in smoke runs."""

    revision = "smoke-alternating-process-v1"

    async def score(self, transition: ProcessTransition) -> ProcessScore:
        match = re.search(r":(\d+)$", transition.trajectory_id)
        session_id = int(match.group(1)) if match else 0
        return ProcessScore(
            score=float(session_id % 2),
            confidence=1.0,
            reason="deterministic smoke optimizer-path signal",
            scorer_revision=self.revision,
        )


__all__ = [
    "AlternatingSmokeProcessScorer",
    "SmokePageReaderBackend",
    "SmokeSearchBackend",
    "SmokeWikipediaBackend",
]
