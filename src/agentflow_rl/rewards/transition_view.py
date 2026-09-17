from __future__ import annotations

import json
from typing import Any, Sequence

from .schemas import ProcessTransition
from agentflow_rl.runtime.observations import action_view, pick, tool_observation

PROCESS_VIEW_REVISION = "process-transition-view-v5"

def _model_source(transition: ProcessTransition) -> dict[str, Any]:
    source = transition.model_dump(mode="json")
    source["task"] = pick(source["task"], ("task_name", "prompt", "public_payload"))
    source["tool_request"] = action_view(source["tool_request"])
    # Start from the semantic tool observation, then let this module's shared
    # token-aware renderer preserve both the head and tail of oversized output.
    # The executed code is already present in tool_request and is removed here.
    source["tool_result"] = tool_observation(source["tool_result"])
    if (
        isinstance(source["tool_result"], dict)
        and isinstance(source["tool_result"].get("data"), dict)
    ):
        source["tool_result"]["data"].pop("code", None)
        source["tool_result"]["data"].pop("code_revision", None)
    return source


def _history(transition: ProcessTransition) -> list[dict]:
    return [dict(event) for event in transition.planner_memory_core]


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _compact(value: Any, max_chars: int) -> tuple[Any, bool]:
    raw = _json(value)
    if len(raw) <= max_chars:
        return value, False
    retained = max(128, (max_chars - 256) // 2)
    return (
        {
            "truncated": True,
            "original_chars": len(raw),
            "head": raw[:retained],
            "tail": raw[-retained:],
        },
        True,
    )


def process_transition_payload(
    transition: ProcessTransition,
    *,
    max_chars: int = 24_000,
) -> dict[str, Any]:
    if max_chars < 7_500:
        raise ValueError("process transition character budget is too small")
    source = _model_source(transition)
    history = _history(transition)
    field_budget = max_chars - 1_500
    weights = {
        "task": 0.15,
        "planner_response": 0.10,
        "planner_action": 0.15,
        "tool_request": 0.15,
        "tool_result": 0.45,
    }
    budgets = {
        field: max(512, int(field_budget * weight))
        for field, weight in weights.items()
    }
    current: dict[str, Any] = {
        "turn_index": source["turn_index"],
    }
    truncated_fields: list[str] = []
    for field, budget in budgets.items():
        current[field], truncated = _compact(source[field], budget)
        if truncated:
            truncated_fields.append(field)

    payload: dict[str, Any] = {
        "schema_version": PROCESS_VIEW_REVISION,
        "current": current,
        "history": [],
        "truncation": {
            "fields": truncated_fields,
            "history_events_total": len(history),
            "history_events_omitted": len(history),
        },
    }
    retained: list[Any] = []
    for event in reversed(history):
        compacted, _ = _compact(event, 2_000)
        candidate = [compacted, *retained]
        payload["history"] = candidate
        payload["truncation"]["history_events_omitted"] = (
            len(history) - len(candidate)
        )
        if len(_json(payload)) > max_chars:
            payload["history"] = retained
            payload["truncation"]["history_events_omitted"] = (
                len(history) - len(retained)
            )
            break
        retained = candidate
    return payload


def render_process_transition(
    transition: ProcessTransition,
    *,
    max_chars: int = 24_000,
    tokenizer: Any = None,
    max_length: int = 8192,
) -> str:
    instruction = "Assess the quality of this AgentFlow Planner transition.\n"
    payload = process_transition_payload(
        transition, max_chars=max_chars - len(instruction),
    )
    rendered = instruction + _json(payload)
    if tokenizer is None:
        return rendered
    if max_length <= 0:
        raise ValueError("process transition token budget must be positive")

    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=True))

    if token_count(rendered) <= max_length:
        return rendered

    source = _model_source(transition)
    history = _history(transition)
    weights = {
        "task": .15, "planner_response": .10, "planner_action": .15,
        "tool_request": .15, "tool_result": .45,
    }

    def preview(value: Any, edge: int) -> Any:
        raw = _json(value)
        if value is None or len(raw) <= 2 * edge:
            return value
        return {
            "truncated": True, "original_chars": len(raw),
            "head": raw[:edge], "tail": raw[-edge:] if edge else "",
        }

    # All five current evidence fields get a reservation before history or long fields
    # consume space. Every fit check uses the complete text and special tokens.
    current = payload["current"]
    payload["history"] = []
    payload["truncation"]["history_events_omitted"] = len(history)
    for field in weights:
        current[field] = min(
            (source[field], preview(source[field], 0)),
            key=lambda value: token_count(_json(value)),
        )

    def render() -> str:
        payload["truncation"]["fields"] = [
            field for field in weights if current[field] != source[field]
        ]
        return instruction + _json(payload)

    if token_count(render()) > max_length:
        raise ValueError("PRM token budget cannot retain all current transition fields")
    remaining_weight = sum(weights.values())
    for field, weight in weights.items():
        reserved = current[field]
        used = token_count(render())
        target = used + int((max_length - used) * weight / remaining_weight)
        remaining_weight -= weight
        current[field] = source[field]
        if token_count(render()) <= target:
            continue
        current[field] = reserved
        low, high = 0, (len(_json(source[field])) + 1) // 2
        best = reserved
        while low <= high:
            edge = (low + high) // 2
            current[field] = preview(source[field], edge)
            if token_count(render()) <= target:
                best = current[field]
                low = edge + 1
            else:
                high = edge - 1
        current[field] = best

    for event in reversed(history):
        previous = list(payload["history"])
        payload["history"] = [_compact(event, 2000)[0], *previous]
        payload["truncation"]["history_events_omitted"] -= 1
        if token_count(render()) > max_length:
            payload["history"] = previous
            payload["truncation"]["history_events_omitted"] += 1
            break
    rendered = render()
    if token_count(rendered) > max_length:
        raise ValueError("PRM transition exceeded its token budget")
    return rendered


def encode_process_transitions(
    transitions: Sequence[ProcessTransition], tokenizer: Any, *, max_length: int = 8192, **kwargs: Any,
) -> Any:
    """Shared training/serving encoding; preserve fields before tokenization."""
    texts = [
        render_process_transition(item, tokenizer=tokenizer, max_length=max_length)
        for item in transitions
    ]
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, **kwargs)
    if any(len(ids) > max_length for ids in encoded["input_ids"]):
        raise ValueError("encoded PRM input exceeded its token budget")
    return encoded


__all__ = ["process_transition_payload", "render_process_transition", "encode_process_transitions"]
