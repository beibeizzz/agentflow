from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    user_request: str
    lookup_mode: str
    max_steps: int
    initial_state: dict[str, Any]
    goal_spec: dict[str, Any]


def parse_episode(row: dict[str, Any]) -> EpisodeSpec:
    required = {
        "episode_id", "user_request", "lookup_mode", "max_steps",
        "initial_state", "goal_spec",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"Episode is missing fields: {sorted(missing)}")
    lookup_mode = str(row["lookup_mode"])
    if lookup_mode not in {"ticket_id", "customer_id", "order_id"}:
        raise ValueError(f"Unknown lookup_mode: {lookup_mode}")
    max_steps = int(row["max_steps"])
    if max_steps not in {2, 3}:
        raise ValueError("Ticket episodes must allow two or three steps")
    return EpisodeSpec(
        episode_id=str(row["episode_id"]),
        user_request=str(row["user_request"]),
        lookup_mode=lookup_mode,
        max_steps=max_steps,
        initial_state=dict(row["initial_state"]),
        goal_spec=dict(row["goal_spec"]),
    )
