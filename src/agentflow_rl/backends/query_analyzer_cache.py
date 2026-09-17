from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from .concurrency import CrossProcessSlotLimiter
from .sqlite_cache import SqliteResponseCache


@dataclass(frozen=True)
class QueryAnalysisLookup:
    text: str
    cache_key: str
    cache_hit: bool
    queue_wait_ms: float


class QueryAnalyzerCache:
    """Share deterministic Query Analyzer outputs across AgentLoop workers."""

    def __init__(
        self,
        path: str | Path,
        *,
        coordination_dir: str | Path,
        namespace: str = "query-analyzer-v1",
        lock_timeout_s: float = 120.0,
    ) -> None:
        if lock_timeout_s <= 0:
            raise ValueError("Query Analyzer cache lock timeout must be positive")
        self.cache = SqliteResponseCache(path, namespace=namespace)
        self.coordination_dir = Path(coordination_dir)
        self.lock_timeout_s = lock_timeout_s

    @staticmethod
    def key_for(
        *,
        model_revision: str,
        prompt_revision: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        request = {
            "model_revision": model_revision,
            "prompt_revision": prompt_revision,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        return hashlib.sha256(
            json.dumps(
                request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    async def get_or_compute(
        self,
        *,
        cache_key: str,
        compute: Callable[[], Awaitable[str]],
    ) -> QueryAnalysisLookup:
        cached = await asyncio.to_thread(self.cache.get, cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("text"), str):
            await asyncio.to_thread(self.cache.increment, "hits")
            return QueryAnalysisLookup(cached["text"], cache_key, True, 0.0)

        limiter = CrossProcessSlotLimiter(
            self.coordination_dir,
            name=f"query-analyzer-{cache_key}",
            limit=1,
        )
        async with limiter.slot(timeout_s=self.lock_timeout_s) as wait_ms:
            cached = await asyncio.to_thread(self.cache.get, cache_key)
            if isinstance(cached, dict) and isinstance(cached.get("text"), str):
                await asyncio.to_thread(self.cache.increment, "hits_after_wait")
                return QueryAnalysisLookup(cached["text"], cache_key, True, wait_ms)
            text = await compute()
            await asyncio.to_thread(self.cache.put, cache_key, {"text": text})
            await asyncio.to_thread(self.cache.increment, "misses")
            return QueryAnalysisLookup(text, cache_key, False, wait_ms)


__all__ = ["QueryAnalysisLookup", "QueryAnalyzerCache"]
