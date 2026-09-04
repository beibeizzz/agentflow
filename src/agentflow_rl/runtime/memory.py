from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field


class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_index: int = Field(ge=-1)
    role: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    content: Any
    tags: tuple[str, ...] = ()

    def render(self) -> str:
        body = (
            self.content
            if isinstance(self.content, str)
            else json.dumps(self.content, ensure_ascii=False, sort_keys=True)
        )
        return f"[turn={self.turn_index} role={self.role} kind={self.kind}]\n{body}"


@dataclass(frozen=True)
class MemoryView:
    text: str
    token_count: int
    included_entries: tuple[int, ...]
    omitted_entries: int


def approximate_token_count(text: str) -> int:
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


class MemoryStore:
    """Append-only episode memory with bounded, deterministic prompt views."""

    def __init__(self, entries: Iterable[MemoryEntry] = ()) -> None:
        self._entries = list(entries)

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    def add(
        self,
        *,
        turn_index: int,
        role: str,
        kind: str,
        content: Any,
        tags: Iterable[str] = (),
    ) -> MemoryEntry:
        entry = MemoryEntry(
            turn_index=turn_index,
            role=role,
            kind=kind,
            content=content,
            tags=tuple(tags),
        )
        self._entries.append(entry)
        return entry

    def snapshot(self) -> list[dict[str, Any]]:
        return [entry.model_dump(mode="json") for entry in self._entries]

    def project(
        self,
        *,
        max_tokens: int,
        token_counter: Callable[[str], int] = approximate_token_count,
        required_tags: Iterable[str] = (),
        header: str = "",
    ) -> MemoryView:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        wanted = set(required_tags)
        required = [
            (index, entry)
            for index, entry in enumerate(self._entries)
            if wanted.intersection(entry.tags)
        ]
        required_ids = {index for index, _ in required}
        recent = [
            (index, entry)
            for index, entry in reversed(list(enumerate(self._entries)))
            if index not in required_ids
        ]
        selected: list[tuple[int, MemoryEntry]] = []
        used = token_counter(header) if header else 0
        for index, entry in [*required, *recent]:
            rendered = entry.render()
            cost = token_counter(rendered)
            if used + cost <= max_tokens:
                selected.append((index, entry))
                used += cost
        selected.sort(key=lambda pair: pair[0])
        sections = [header.strip()] if header.strip() else []
        sections.extend(entry.render() for _, entry in selected)
        text = "\n\n".join(sections)
        return MemoryView(
            text=text,
            token_count=token_counter(text) if text else 0,
            included_entries=tuple(index for index, _ in selected),
            omitted_entries=len(self._entries) - len(selected),
        )


__all__ = [
    "MemoryEntry",
    "MemoryStore",
    "MemoryView",
    "approximate_token_count",
]
