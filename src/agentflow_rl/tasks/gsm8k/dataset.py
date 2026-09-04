from __future__ import annotations

import json
from pathlib import Path


def load_gsm8k_rows(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("GSM8K JSON root must be a list")
        rows = value
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not {"question", "gold_answer"} <= row.keys():
            raise ValueError(f"invalid GSM8K row at item {number}")
    return rows
