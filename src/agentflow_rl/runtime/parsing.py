from __future__ import annotations

import json
from typing import Any

from .errors import ActionParseError


def strip_optional_think_prefix(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("<think>"):
        closing = candidate.find("</think>")
        if closing < 0:
            raise ActionParseError("unclosed think block")
        candidate = candidate[closing + len("</think>") :].strip()
    elif "<think>" in candidate or "</think>" in candidate:
        raise ActionParseError("think block must be one optional prefix")
    return candidate


def strict_json_object(text: str) -> dict[str, Any]:
    candidate = strip_optional_think_prefix(text)
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        valid_fence = (
            len(lines) >= 3
            and lines[0].strip().lower() in {"```", "```json"}
            and lines[-1].strip() == "```"
        )
        if not valid_fence:
            raise ActionParseError("JSON fence must wrap exactly one object")
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(
            candidate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ActionParseError("response must be exactly one JSON object") from exc
    if not isinstance(payload, dict):
        raise ActionParseError("response must be exactly one JSON object")
    return payload


__all__ = ["strict_json_object", "strip_optional_think_prefix"]
