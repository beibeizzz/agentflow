from __future__ import annotations

import bz2
import json
from pathlib import Path
from typing import Any, Iterator


def file_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("rows", payload.get("documents")))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array or JSONL file: {path}")
    yield from payload


def rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_dir():
        sources = sorted(
            child
            for child in path.rglob("*")
            if child.is_file() and child.suffix in {".bz2", ".json", ".jsonl"}
        )
        if not sources:
            raise ValueError(f"No JSON/JSONL/BZip2 corpus shards under {path}")
        for source in sources:
            yield from file_rows(source)
        return
    yield from file_rows(path)


def sentence_list(row: dict[str, Any]) -> list[str] | None:
    sentences = row.get("sentences")
    text = row.get("text")
    if sentences is None and isinstance(text, dict):
        sentences = text.get("sentences")
    if sentences is None and isinstance(text, list):
        if all(isinstance(value, str) for value in text):
            sentences = text
        else:
            sentences = next(
                (
                    paragraph
                    for paragraph in text
                    if isinstance(paragraph, list) and paragraph
                ),
                None,
            )
    if not isinstance(sentences, list):
        return None
    return [str(sentence) for sentence in sentences if str(sentence).strip()]


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    identity = row.get("id", row.get("doc_id"))
    title = row.get("title")
    normalized_sentences = sentence_list(row)
    if identity is None or title is None or not normalized_sentences:
        raise ValueError(
            "each corpus row requires id/doc_id, title, and an ordered sentence paragraph"
        )
    return {
        "id": str(identity),
        "title": str(title),
        "sentences": normalized_sentences,
        "contents": f"{title} {' '.join(normalized_sentences)}",
    }


__all__ = ["file_rows", "normalize", "rows", "sentence_list"]
