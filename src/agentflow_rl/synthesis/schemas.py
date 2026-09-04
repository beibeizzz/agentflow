from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


SCHEMA_VERSION = "2.0.0"


class TicketBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    split: str
    seed: int
    lookup_mode: Literal["ticket_id", "customer_id", "order_id"]
    target_index: int
    field: Literal["priority", "assigned_team", "status"]
    value: str


def blueprint_fingerprint(blueprint: TicketBlueprint) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "blueprint": blueprint.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
