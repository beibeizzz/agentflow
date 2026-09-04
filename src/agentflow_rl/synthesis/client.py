from __future__ import annotations

import os
from typing import Any

from .schemas import TicketBlueprint


class OpenAIRequestRewriter:
    """Optional offline rewrite client; deterministic validators remain authoritative."""

    def __init__(self, client: Any, *, model: str) -> None:
        self.client = client
        self.model = model

    @classmethod
    def from_environment(cls, client_factory: Any, *, model: str) -> "OpenAIRequestRewriter":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for API rewriting")
        return cls(client_factory(api_key=api_key), model=model)

    def __call__(self, blueprint: TicketBlueprint, request: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the request concisely. Preserve every identifier, target field, "
                        "target value, and completion instruction exactly. Return text only."
                    ),
                },
                {"role": "user", "content": request},
            ],
        )
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty rewrite for {blueprint.episode_id}")
        return text.strip()
