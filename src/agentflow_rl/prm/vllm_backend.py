from __future__ import annotations

import asyncio
import json
import math
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentflow_rl.rewards.rubric import PROCESS_RUBRIC_REVISION
from agentflow_rl.rewards.schemas import ProcessTransition
from agentflow_rl.rewards.transition_view import render_process_transition


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


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class VllmProcessRewardBackend:
    """Serve the scalar Qwen PRM through vLLM's classification endpoint.

    The checkpoint has one regression label. vLLM activation is disabled so
    that the wrapper can preserve the Transformers backend's sigmoid(logit)
    scoring contract; a one-class softmax would otherwise always equal one.
    """

    rubric_revision = PROCESS_RUBRIC_REVISION

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        tokenizer: Any,
        revision: str,
        max_length: int = 8192,
        timeout_s: float = 120.0,
        http_post: HttpPost = _post,
    ) -> None:
        if not endpoint or not model or not revision:
            raise ValueError("vLLM PRM endpoint, model, and revision are required")
        if max_length <= 0 or timeout_s <= 0:
            raise ValueError("vLLM PRM budgets must be positive")
        self.endpoint = endpoint
        self.model = model
        self.tokenizer = tokenizer
        self.revision = revision
        self.max_length = max_length
        self.timeout_s = timeout_s
        self.http_post = http_post

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_path: str | Path,
        model_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "VllmProcessRewardBackend":
        from transformers import AutoConfig, AutoTokenizer

        from agentflow_rl.prm.transformers_backend import validate_checkpoint_protocol

        checkpoint_path = str(model_path or tokenizer_path)
        config = AutoConfig.from_pretrained(
            checkpoint_path, local_files_only=True
        )
        validate_checkpoint_protocol(config)

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), use_fast=True, local_files_only=True
        )
        return cls(tokenizer=tokenizer, **kwargs)

    def _render(self, transitions: tuple[ProcessTransition, ...]) -> list[str]:
        texts = [
            render_process_transition(
                transition,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
            for transition in transitions
        ]
        for text in texts:
            token_count = len(self.tokenizer.encode(text, add_special_tokens=True))
            if token_count > self.max_length:
                raise ValueError("encoded vLLM PRM input exceeded its token budget")
        return texts

    async def predict(self, transition: ProcessTransition) -> float:
        return (await self.predict_many((transition,)))[0]

    async def predict_many(
        self, transitions: tuple[ProcessTransition, ...]
    ) -> tuple[float, ...]:
        if not transitions:
            return ()
        body = json.dumps(
            {
                "model": self.model,
                "input": self._render(transitions),
                "use_activation": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        raw = await asyncio.to_thread(
            self.http_post, self.endpoint, body, self.timeout_s
        )
        payload = json.loads(raw)
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != len(transitions):
            raise ValueError("vLLM PRM response length mismatch")
        ordered = sorted(rows, key=lambda row: int(row.get("index", -1)))
        logits: list[float] = []
        for expected_index, row in enumerate(ordered):
            if int(row.get("index", -1)) != expected_index:
                raise ValueError("vLLM PRM response indices are incomplete")
            values = row.get("probs")
            if not isinstance(values, list) or len(values) != 1:
                raise ValueError("vLLM PRM requires one raw classification logit")
            logits.append(float(values[0]))
        return tuple(_sigmoid(value) for value in logits)


__all__ = ["VllmProcessRewardBackend"]
