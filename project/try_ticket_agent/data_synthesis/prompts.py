from __future__ import annotations

import json
from typing import Any


def build_rewrite_messages(
    source: dict[str, Any], prior_failures: list[str] | None = None
) -> list[dict[str, str]]:
    feedback = "\n".join(f"- {item}" for item in (prior_failures or [])) or "- None"
    return [
        {
            "role": "system",
            "content": (
                "Rewrite a synthetic support-ticket request in natural language. Preserve exactly the "
                "lookup facts and requested mutation. Do not mention hidden goals, tools, JSON, or a "
                "ticket ID that is absent from the public canonical request. Return exactly one JSON "
                "object with only the user_request string key."
            ),
        },
        {
            "role": "user",
            "content": (
                "Hidden blueprint for semantic preservation:\n"
                + json.dumps(source, ensure_ascii=False, sort_keys=True)
                + "\nPrevious validation feedback:\n"
                + feedback
            ),
        },
    ]


def build_judge_messages(source: dict[str, Any], user_request: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Judge whether the rewritten request preserves exactly one intended ticket mutation, "
                "the required lookup information, and the completion intent without exposing hidden "
                "state or tool instructions. Return exactly one JSON object with only the accepted "
                "boolean and reasons string-list keys."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"blueprint": source, "candidate_user_request": user_request},
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]
