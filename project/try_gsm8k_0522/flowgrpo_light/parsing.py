from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("<think>"):
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>") :].strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(fenced)
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse JSON object from planner response: {text!r}") from last_error


def parse_planner_response(text: str) -> tuple[str, str]:
    obj = extract_json_object(text)
    sub_goal = str(obj.get("Sub_goal") or obj.get("sub_goal") or "").strip()
    calculation = str(obj.get("Calculation") or obj.get("calculation") or "").strip()
    if not sub_goal or not calculation:
        raise ValueError(f"Planner response missing Sub_goal or Calculation: {text!r}")
    return sub_goal, calculation
