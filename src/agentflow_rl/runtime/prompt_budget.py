from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def chat_token_count(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> int:
    """Count the complete served template; byte upper bound is for local fakes only."""
    if tokenizer is None:
        return sum(len(item["content"].encode("utf-8")) + 128 for item in messages)
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def role_token_count(tokenizer: Any, system: str, prompt: str) -> int:
    return chat_token_count(tokenizer, [{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}])
