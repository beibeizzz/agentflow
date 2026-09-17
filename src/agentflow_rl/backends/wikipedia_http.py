from __future__ import annotations

import asyncio
import json
import urllib.request
from collections.abc import Callable

from agentflow_rl.runtime.errors import InfrastructureError, RevisionMismatchError

from .contracts import WikipediaDocument, WikipediaHit


HttpPost = Callable[[str, bytes, float], bytes]


def _post(url: str, body: bytes, timeout_s: float) -> bytes:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


class HttpWikipediaBackend:
    revision = "wikipedia-service-client-v1"

    def __init__(
        self,
        base_url: str,
        *,
        expected_revision: str,
        timeout_s: float = 30.0,
        http_post: HttpPost = _post,
    ) -> None:
        if not expected_revision:
            raise ValueError("Wikipedia index revision is required")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.http_post = http_post
        self.expected_revision = expected_revision
        self.revision = expected_revision

    async def _request(self, operation: str, payload: dict) -> dict:
        try:
            raw = await asyncio.to_thread(
                self.http_post,
                f"{self.base_url}/{operation}",
                json.dumps(payload).encode("utf-8"),
                self.timeout_s,
            )
            result = json.loads(raw)
        except Exception as exc:
            raise InfrastructureError("Wikipedia retrieval service failed") from exc
        if not isinstance(result, dict):
            raise InfrastructureError("Wikipedia service returned an invalid payload")
        actual_revision = result.get("revision")
        if actual_revision != self.expected_revision:
            raise RevisionMismatchError(
                "Wikipedia service index revision differs from the configured revision"
            )
        return result

    async def search(self, *, query: str, top_k: int) -> tuple[WikipediaHit, ...]:
        payload = await self._request("search", {"query": query, "top_k": top_k})
        return tuple(WikipediaHit.model_validate(item) for item in payload["hits"])

    async def read(
        self, *, doc_id: str, start_passage: int, max_passages: int
    ) -> WikipediaDocument:
        payload = await self._request(
            "read",
            {
                "doc_id": doc_id,
                "start_passage": start_passage,
                "max_passages": max_passages,
            },
        )
        return WikipediaDocument.model_validate(payload["document"])


__all__ = ["HttpWikipediaBackend"]
