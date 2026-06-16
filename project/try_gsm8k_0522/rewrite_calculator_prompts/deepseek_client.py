from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from .prompts import build_judge_messages, build_rewrite_messages
from .schemas import ApiUsage, JsonResponse


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        rewrite_model: str = "deepseek-v4-flash",
        judge_model: str = "deepseek-v4-pro",
        max_transport_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        sdk_client: Any | None = None,
    ) -> None:
        if max_transport_attempts < 1:
            raise ValueError("max_transport_attempts must be at least 1")
        if sdk_client is None:
            resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not resolved_key:
                raise RuntimeError("DEEPSEEK_API_KEY is not set")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("The openai package is required to call DeepSeek") from exc
            sdk_client = OpenAI(api_key=resolved_key, base_url=base_url)
        self.sdk_client = sdk_client
        self.rewrite_model = rewrite_model
        self.judge_model = judge_model
        self.max_transport_attempts = max_transport_attempts
        self.sleep = sleep

    def rewrite(
        self,
        source: dict[str, Any],
        prior_failures: list[str] | None = None,
        *,
        temperature: float = 0.3,
    ) -> JsonResponse:
        return self._complete_json(
            messages=build_rewrite_messages(source, prior_failures),
            model=self.rewrite_model,
            temperature=temperature,
            thinking="disabled",
            reasoning_effort=None,
            max_tokens=1200,
        )

    def judge(
        self,
        source: dict[str, Any],
        rewritten_question: str,
    ) -> JsonResponse:
        return self._complete_json(
            messages=build_judge_messages(source, rewritten_question),
            model=self.judge_model,
            temperature=0.0,
            thinking="enabled",
            reasoning_effort="high",
            max_tokens=8192,
        )

    def _complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        thinking: str,
        reasoning_effort: str | None,
        max_tokens: int,
    ) -> JsonResponse:
        last_error: Exception | None = None
        accumulated = ApiUsage()
        current_max_tokens = max_tokens
        for attempt in range(1, self.max_transport_attempts + 1):
            try:
                request: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": temperature,
                    "max_tokens": current_max_tokens,
                    "stream": False,
                    "extra_body": {"thinking": {"type": thinking}},
                }
                if reasoning_effort is not None:
                    request["reasoning_effort"] = reasoning_effort
                response = self.sdk_client.chat.completions.create(**request)
                accumulated = _add_usage(accumulated, _extract_usage(response))
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    current_max_tokens = min(current_max_tokens * 2, 65536)
                    raise ValueError("DeepSeek JSON response was truncated")
                content = choice.message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("DeepSeek returned empty JSON content")
                payload = json.loads(content)
                if not isinstance(payload, dict):
                    raise ValueError("DeepSeek JSON response must be an object")
                return JsonResponse(payload=payload, usage=accumulated)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_transport_attempts:
                    self.sleep(float(2 ** (attempt - 1)))
        raise RuntimeError(
            f"DeepSeek request failed after {self.max_transport_attempts} attempts: {last_error}"
        ) from last_error


def _extract_usage(response: Any) -> ApiUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return ApiUsage()
    return ApiUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
    )


def _add_usage(left: ApiUsage, right: ApiUsage) -> ApiUsage:
    return ApiUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )
