from __future__ import annotations

import hashlib
from typing import Any, Iterable


def row_identity(row: dict[str, Any]) -> str:
    value = row.get("episode_id", row.get("id", row.get("_id")))
    if value is None:
        value = " ".join(str(row.get("question", "")).casefold().split())
    if not value:
        raise ValueError("evaluation row requires an ID or question")
    return str(value)


def split_labeled_rows(
    rows: Iterable[dict[str, Any]],
    *,
    validation_size: int,
    test_size: int,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("subset sizes must be positive")
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: hashlib.sha256(
            f"{seed}:{row_identity(row)}".encode("utf-8")
        ).digest(),
    )
    selected = validation_size + test_size
    if len(ranked) < selected:
        raise ValueError(f"need {selected} labeled rows, found {len(ranked)}")
    return ranked[:validation_size], ranked[validation_size:selected]


__all__ = ["row_identity", "split_labeled_rows"]
