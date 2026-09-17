from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Iterable

from .contracts import MemoryAudience, MemoryEvent, MemoryVisibility
from .observations import event_content

_SEMANTIC_FIELDS = {"tool_name", "status", "failure_code", "outcome", "evidence_ids"}


@dataclass(frozen=True)
class MemoryView:
    text: str
    token_count: int
    included_event_ids: tuple[str, ...]
    omitted_events: int
    compacted_event_ids: tuple[str, ...] = ()


def approximate_token_count(text: str) -> int:
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def _visible(event: MemoryEvent, audience: MemoryAudience) -> bool:
    ranks = {
        MemoryVisibility.MODEL: 0,
        MemoryVisibility.SCORER: 1,
        MemoryVisibility.AUDIT: 2,
    }
    audience_rank = {
        MemoryAudience.MODEL: 0,
        MemoryAudience.SCORER: 1,
        MemoryAudience.AUDIT: 2,
    }[audience]
    return ranks[event.visibility] <= audience_rank


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _preview(value: Any, limit: int) -> Any:
    raw = value if isinstance(value, str) else _json_text(value)
    if len(raw) <= limit:
        return value
    edge = max(0, limit // 2)
    return {
        "chars": len(raw),
        "head": raw[:edge],
        "tail": raw[-edge:] if edge else "",
        "truncated": True,
    }


def _compact_required_event(
    event: MemoryEvent,
    *,
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> str | None:
    values = event.content if isinstance(event.content, dict) else {"value": event.content}

    def render(limit: int) -> str:
        payload = {
            "_event_ref": {
                "event_id": event.event_id,
            },
            "_compacted": True,
            "fields": {
                str(key): value if key in _SEMANTIC_FIELDS else _preview(value, limit)
                for key, value in sorted(values.items(), key=lambda pair: str(pair[0]))
            },
        }
        header = (
            f"[event={event.event_id} turn={event.turn_index} "
            f"role={event.role} kind={event.kind}]"
        )
        return f"{header}\n{_json_text(payload)}"

    minimum = render(0)
    if token_counter(minimum) > max_tokens:
        reference = _required_reference(event)
        return reference if token_counter(reference) <= max_tokens else None
    upper = max(
        (len(value) if isinstance(value, str) else len(_json_text(value)))
        for value in values.values()
    ) if values else 0
    low = 0
    while low < upper:
        middle = (low + upper + 1) // 2
        if token_counter(render(middle)) <= max_tokens:
            low = middle
        else:
            upper = middle - 1
    return render(low)


def _required_reference(event: MemoryEvent) -> str:
    fields = {key: value for key, value in event.content.items() if key in _SEMANTIC_FIELDS} if isinstance(event.content, dict) else {}
    return (
        f"[event={event.event_id} turn={event.turn_index} "
        f"role={event.role} kind={event.kind}]\n"
        + _json_text({"_event_ref": {
            "event_id": event.event_id,
        }, "_compacted": True, "fields": fields, "content_omitted": True})
    )


class MemoryStore:
    """Append-only trajectory memory with deterministic audience projections."""

    def __init__(self, events: Iterable[MemoryEvent] = ()) -> None:
        self._events: list[MemoryEvent] = []
        self._event_ids: set[str] = set()
        for event in events:
            self.append(event)

    @property
    def events(self) -> tuple[MemoryEvent, ...]:
        return tuple(event.model_copy(deep=True) for event in self._events)

    def append(self, event: MemoryEvent) -> MemoryEvent:
        if event.event_id in self._event_ids:
            raise ValueError(f"duplicate Memory event_id: {event.event_id}")
        if self._events:
            identity = (self._events[0].trajectory_id, self._events[0].task_id)
            if (event.trajectory_id, event.task_id) != identity:
                raise ValueError("Memory event identity differs from the trajectory")
            if event.turn_index < self._events[-1].turn_index:
                raise ValueError("Memory turn_index must be monotonic")
        stored = MemoryEvent.model_validate(
            event.model_dump(mode="python", exclude={"content_sha256"})
        )
        self._events.append(stored)
        self._event_ids.add(stored.event_id)
        return stored.model_copy(deep=True)

    def snapshot(self, *, audience: MemoryAudience = MemoryAudience.AUDIT) -> list[dict[str, Any]]:
        return [
            event.model_dump(mode="json")
            for event in self._events
            if _visible(event, audience)
        ]

    def project(
        self,
        *,
        audience: MemoryAudience,
        max_tokens: int,
        token_counter: Callable[[str], int] = approximate_token_count,
        required_tags: Iterable[str] = (),
        required_latest_tags: Iterable[str] = (),
        include_roles: Iterable[str] | None = None,
        include_kinds: Iterable[str] | None = None,
        max_recent_events: int | None = None,
        include_event_ids: Iterable[str] | None = None,
        header: str = "",
        content_projector: Callable[[str, Any], Any] | None = None,
    ) -> MemoryView:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_recent_events is not None and max_recent_events < 0:
            raise ValueError("max_recent_events must be non-negative")
        roles = set(include_roles) if include_roles is not None else None
        kinds = set(include_kinds) if include_kinds is not None else None
        event_ids = set(include_event_ids) if include_event_ids is not None else None
        eligible = [
            event
            for event in self._events
            if _visible(event, audience)
            and (roles is None or event.role in roles)
            and (kinds is None or event.kind in kinds)
            and (event_ids is None or event.event_id in event_ids)
        ]
        if audience is not MemoryAudience.AUDIT:
            project_content = content_projector or event_content
            eligible = [event.model_copy(update={"content": project_content(event.kind, event.content)})
                        for event in eligible]
        event_order = {event.event_id: i for i, event in enumerate(self._events)}
        wanted = set(required_tags)
        required = [event for event in eligible if wanted.intersection(event.tags)]
        required_latest: list[MemoryEvent] = []
        for tag in required_latest_tags:
            match = next((event for event in reversed(eligible) if tag in event.tags), None)
            if match is not None:
                required_latest.append(match)

        selected: list[tuple[MemoryEvent, str]] = []
        selected_ids: set[str] = set()
        compacted_ids: set[str] = set()
        required_order: list[MemoryEvent] = []
        required_seen: set[str] = set()
        for event in [*required_latest, *required]:
            if event.event_id not in required_seen:
                required_order.append(event)
                required_seen.add(event.event_id)

        def joined(items: list[tuple[MemoryEvent, str]]) -> str:
            ordered = sorted(items, key=lambda pair: event_order[pair[0].event_id])
            sections = [header.strip()] if header.strip() else []
            return "\n\n".join(sections + [text for _, text in ordered])

        # Reserve every required event before expanding any one of them. Count
        # the complete rendered view, including its header and separators.
        selected = [
            (event, min((event.render(), _required_reference(event)), key=token_counter))
            for event in required_order
        ]
        if token_counter(joined(selected)) > max_tokens:
            raise ValueError("memory budget cannot retain all required event references")
        for index, event in enumerate(required_order):
            def total_cost(text: str) -> int:
                candidate = list(selected)
                candidate[index] = (event, text)
                return token_counter(joined(candidate))

            rendered = event.render()
            if total_cost(rendered) > max_tokens:
                rendered = _compact_required_event(
                    event, max_tokens=max_tokens, token_counter=total_cost,
                ) or selected[index][1]
            selected[index] = (event, rendered)
            if rendered != event.render():
                compacted_ids.add(event.event_id)
            selected_ids.add(event.event_id)

        recent = [event for event in reversed(eligible) if event.event_id not in selected_ids]
        if max_recent_events is not None:
            recent = recent[:max_recent_events]
        for event in recent:
            rendered = event.render()
            if token_counter(joined([*selected, (event, rendered)])) <= max_tokens:
                selected.append((event, rendered))
                selected_ids.add(event.event_id)
        selected.sort(key=lambda pair: event_order[pair[0].event_id])
        text = joined(selected)
        return MemoryView(
            text=text,
            token_count=token_counter(text) if text else 0,
            included_event_ids=tuple(event.event_id for event, _ in selected),
            omitted_events=len(eligible) - len(selected),
            compacted_event_ids=tuple(
                event.event_id for event, _ in selected if event.event_id in compacted_ids
            ),
        )


__all__ = ["MemoryStore", "MemoryView", "approximate_token_count"]
