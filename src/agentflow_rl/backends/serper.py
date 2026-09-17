from __future__ import annotations

import asyncio
import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from time import sleep
from typing import Any

from agentflow_rl.runtime.errors import InfrastructureError, RateLimitError

from .concurrency import CrossProcessSlotLimiter
from .contracts import SearchHit
from .sqlite_cache import SqliteResponseCache


HttpPost = Callable[[str, dict[str, str], bytes, float], bytes]


def _post(url: str, headers: dict[str, str], body: bytes, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


class SerperSearchBackend:
    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = "https://google.serper.dev/search",
        timeout_s: float = 20.0,
        http_post: HttpPost = _post,
        revision: str = "serper-search-v1",
        cache_path: str | Path | None = None,
        coordination_dir: str | Path = "/tmp/agentflow-locks",
        max_concurrency: int = 40,
        max_attempts: int = 3,
        backoff_s: float = 0.5,
        cost_per_request_usd: float = 0.001,
    ) -> None:
        if not api_key:
            raise ValueError("Serper API key is required")
        if max_attempts <= 0 or backoff_s < 0 or cost_per_request_usd < 0:
            raise ValueError("Serper retry and cost settings are invalid")
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.http_post = http_post
        self.revision = revision
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.cost_per_request_usd = cost_per_request_usd
        self.cache = (
            SqliteResponseCache(cache_path, namespace=revision) if cache_path else None
        )
        self.limiter = CrossProcessSlotLimiter(
            coordination_dir, name="serper", limit=max_concurrency
        )
        self._inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._inflight_lock = asyncio.Lock()

    def _metric(self, name: str, value: float = 1.0) -> None:
        if self.cache is not None:
            self.cache.increment(name, value)

    @property
    def metrics(self) -> dict[str, float]:
        return self.cache.metrics() if self.cache is not None else {}

    def _request(self, body: bytes, headers: dict[str, str]) -> dict[str, Any]:
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._metric("api_attempts")
            self._metric("estimated_cost_usd", self.cost_per_request_usd)
            try:
                raw = self.http_post(
                    self.endpoint, headers, body, self.timeout_s
                )
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise json.JSONDecodeError("object required", str(payload), 0)
                return payload
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 429:
                    self._metric("rate_limited")
                    if attempt == self.max_attempts:
                        raise RateLimitError("Serper rate limit exhausted") from exc
                elif 500 <= exc.code < 600:
                    self._metric("server_errors")
                    if attempt == self.max_attempts:
                        break
                else:
                    raise InfrastructureError(
                        f"Serper rejected the request with HTTP {exc.code}"
                    ) from exc
            except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = exc
                self._metric("network_or_payload_errors")
                if attempt == self.max_attempts:
                    break
            self._metric("retries")
            sleep(self.backoff_s * attempt)
        raise InfrastructureError("Serper search request failed after retries") from last_error

    async def search(self, *, query: str, top_k: int) -> tuple[SearchHit, ...]:
        body = json.dumps({"q": query, "num": top_k}).encode("utf-8")
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        cache_key = hashlib.sha256(
            json.dumps(
                {"endpoint": self.endpoint, "query": query, "top_k": top_k},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        payload = self.cache.get(cache_key) if self.cache is not None else None
        if payload is not None:
            self._metric("cache_hits")
        else:
            async with self._inflight_lock:
                task = self._inflight.get(cache_key)
                owner = task is None
                if owner:
                    task = asyncio.create_task(
                        self._load_payload(cache_key, body, headers)
                    )
                    self._inflight[cache_key] = task
                    task.add_done_callback(
                        lambda completed, key=cache_key: asyncio.create_task(
                            self._clear_inflight(key, completed)
                        )
                    )
                else:
                    self._metric("inflight_deduplicated")
            payload = await asyncio.shield(task)
        hits = []
        for index, item in enumerate(payload.get("organic", ()), start=1):
            url = str(item.get("link", "")).strip()
            title = str(item.get("title", "")).strip()
            if not url or not title:
                continue
            result_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            hits.append(
                SearchHit(
                    result_id=result_id,
                    title=title,
                    url=url,
                    snippet=str(item.get("snippet", "")),
                    rank=index,
                    published_at=(
                        str(item["date"]) if item.get("date") is not None else None
                    ),
                )
            )
            if len(hits) == top_k:
                break
        return tuple(hits)

    async def _load_payload(
        self, cache_key: str, body: bytes, headers: dict[str, str]
    ) -> dict[str, Any]:
        async with self.limiter.slot(timeout_s=self.timeout_s) as wait_ms:
            self._metric("queue_wait_ms", wait_ms)
            payload = self.cache.get(cache_key) if self.cache is not None else None
            if payload is not None:
                self._metric("cache_hits")
                return payload
            payload = await asyncio.to_thread(self._request, body, headers)
            if self.cache is not None:
                self.cache.put(cache_key, payload)
            return payload

    async def _clear_inflight(
        self, cache_key: str, task: asyncio.Task[dict[str, Any]]
    ) -> None:
        async with self._inflight_lock:
            if self._inflight.get(cache_key) is task:
                self._inflight.pop(cache_key, None)


__all__ = ["SerperSearchBackend"]
