from __future__ import annotations

import json
from pathlib import Path

from agentflow_rl.runtime.errors import InfrastructureError


def write_policy_update_id(path: str | Path, update_id: int | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"policy_update_id": str(update_id)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def read_policy_update_id(path: str | Path) -> str:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        value = str(payload["policy_update_id"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise InfrastructureError("synchronized Planner policy identity is unavailable") from exc
    if not value:
        raise InfrastructureError("synchronized Planner policy identity is empty")
    return value


__all__ = ["read_policy_update_id", "write_policy_update_id"]
