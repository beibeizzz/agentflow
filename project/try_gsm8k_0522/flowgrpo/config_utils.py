from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyyaml is required to read Flow-GRPO config files.") from exc

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a YAML object in {path}")
    return payload


def export_env(config: dict[str, Any]) -> None:
    for key, value in config.get("env", {}).items():
        os.environ[str(key)] = str(value)


def config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    if dotted_key in config.get("python_args", {}):
        return os.path.expandvars(str(config["python_args"][dotted_key]))
    if dotted_key in config.get("env", {}):
        return os.path.expandvars(str(config["env"][dotted_key]))
    return default

