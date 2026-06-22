from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EpisodeBlueprint:
    generator_version: str
    seed: int
    split: str
    index: int
    episode_id: str
    curriculum_mode: str
    lookup_mode: str
    user_request: str
    canonical_request: str
    initial_state: dict[str, Any]
    goal_spec: dict[str, str]
    max_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeBlueprint":
        return cls(
            generator_version=str(payload["generator_version"]),
            seed=int(payload["seed"]),
            split=str(payload["split"]),
            index=int(payload["index"]),
            episode_id=str(payload["episode_id"]),
            curriculum_mode=str(payload["curriculum_mode"]),
            lookup_mode=str(payload["lookup_mode"]),
            user_request=str(payload["user_request"]),
            canonical_request=str(payload["canonical_request"]),
            initial_state=dict(payload["initial_state"]),
            goal_spec={str(key): str(value) for key, value in dict(payload["goal_spec"]).items()},
            max_steps=int(payload["max_steps"]),
        )
