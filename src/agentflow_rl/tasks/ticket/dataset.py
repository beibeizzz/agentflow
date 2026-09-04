from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from .schemas import TicketEpisode


def load_ticket_rows(path: str | Path) -> list[TicketEpisode]:
    episodes = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episodes.append(TicketEpisode.from_row(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid ticket row at line {line_number}") from exc
    return episodes


def balance_ticket_rows(
    rows: Iterable[dict[str, Any] | TicketEpisode],
    *,
    direct_fraction: float = 0.5,
    seed: int = 42,
) -> list[TicketEpisode]:
    if not 0.0 <= direct_fraction <= 1.0:
        raise ValueError("direct_fraction must be between zero and one")
    episodes = [row if isinstance(row, TicketEpisode) else TicketEpisode.from_row(row) for row in rows]
    direct = [episode for episode in episodes if episode.lookup_mode == "ticket_id"]
    indirect = [episode for episode in episodes if episode.lookup_mode != "ticket_id"]
    rng = random.Random(seed)
    rng.shuffle(direct)
    rng.shuffle(indirect)
    if direct_fraction == 1.0:
        selected = direct
    elif direct_fraction == 0.0:
        selected = indirect
    else:
        total = min(int(len(direct) / direct_fraction), int(len(indirect) / (1.0 - direct_fraction)))
        direct_count = min(len(direct), round(total * direct_fraction))
        indirect_count = min(len(indirect), total - direct_count)
        selected = direct[:direct_count] + indirect[:indirect_count]
    rng.shuffle(selected)
    return selected
