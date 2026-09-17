from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Callable

from agentflow_rl.runtime.errors import InfrastructureError, RevisionMismatchError

from .contracts import SandboxRequest, SandboxResult


HttpPost = Callable[[str, bytes, float], bytes]


def _post(url: str, body: bytes, timeout_s: float) -> bytes:
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


class HttpSandboxBackend:
    revision = "python-sandbox-service-v1"

    def __init__(
        self,
        base_url: str,
        *,
        expected_revision: str = "python-sandbox-service-v1",
        timeout_s: float = 30.0,
        http_post: HttpPost = _post,
    ) -> None:
        if not base_url or not expected_revision or timeout_s <= 0:
            raise ValueError("sandbox service URL, revision and timeout are required")
        self.base_url = base_url.rstrip("/")
        self.expected_revision = expected_revision
        self.timeout_s = timeout_s
        self.http_post = http_post

    async def execute(self, request: SandboxRequest) -> SandboxResult:
        try:
            raw = await asyncio.to_thread(
                self.http_post,
                f"{self.base_url}/execute",
                request.model_dump_json().encode("utf-8"),
                self.timeout_s,
            )
            payload = json.loads(raw)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise InfrastructureError("Python sandbox service failed") from exc
        if not isinstance(payload, dict):
            raise InfrastructureError("Python sandbox service returned an invalid payload")
        if payload.get("revision") != self.expected_revision:
            raise RevisionMismatchError(
                "Python sandbox service revision differs from the configured revision"
            )
        try:
            return SandboxResult.model_validate(payload["result"])
        except Exception as exc:
            raise InfrastructureError("Python sandbox service returned an invalid result") from exc


__all__ = ["HttpSandboxBackend"]
