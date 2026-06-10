from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class FrozenClient:
    base_url: str
    model: str
    timeout: int = 60
    think_mode: str = "default"

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        think_mode: str | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        effective_think_mode = self.think_mode if think_mode is None else think_mode
        if effective_think_mode in {"on", "off"}:
            payload["chat_template_kwargs"] = {
                "enable_thinking": effective_think_mode == "on",
            }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Frozen model HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Frozen model request failed: {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Frozen model response has no choices: {data}")
        return str(choices[0]["message"]["content"]).strip()

    def check_model(self) -> None:
        request = urllib.request.Request(self.base_url.rstrip("/") + "/models", method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ids = {item.get("id") for item in payload.get("data", [])}
        if self.model not in ids:
            raise RuntimeError(f"Frozen model {self.model!r} not found at {self.base_url}/models. Found: {sorted(ids)}")
