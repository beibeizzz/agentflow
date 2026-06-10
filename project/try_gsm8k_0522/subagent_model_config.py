from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_SYSTEM_PROMPT = (
    "You are good at math problems. "
    "Use only the information in the problem. "
    "Keep the reasoning concise and arithmetic-focused."
)

BASELINE_GENERATION_CONFIG: dict[str, Any] = {
    "system_prompt": BASELINE_SYSTEM_PROMPT,
    "max_tokens": 512,
    "temperature": 0.0,
    "top_p": 0.95,
    "frequency_penalty": 0,
}

SUBAGENT_KEYS = [
    "query_analysis",
    "planner_next_step",
    "executor",
    "verifier",
    "generator",
]


def default_subagent_config() -> dict[str, dict[str, Any]]:
    return {key: dict(BASELINE_GENERATION_CONFIG) for key in SUBAGENT_KEYS}


def _merge_config(defaults: dict[str, dict[str, Any]], overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    merged = {key: dict(value) for key, value in defaults.items()}
    for key, value in overrides.items():
        if key not in merged or not isinstance(value, dict):
            continue
        merged[key].update(value)
    return merged


def load_subagent_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    defaults = default_subagent_config()
    if path is None:
        path = Path(__file__).with_name("subagent_model_config.json")
    if not path.exists():
        return defaults
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return _merge_config(defaults, payload)


def apply_subagent_config(solver: Any, config: dict[str, dict[str, Any]]) -> None:
    solver.planner.generation_configs = {
        "query_analysis": dict(config["query_analysis"]),
        "planner_next_step": dict(config["planner_next_step"]),
        "generator": dict(config["generator"]),
    }
    solver.executor.generation_configs = {
        "executor": dict(config["executor"]),
    }
    solver.verifier.generation_configs = {
        "verifier": dict(config["verifier"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or write AgentFlow GSM8K subagent generation config.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON to load.")
    parser.add_argument("--write-default", type=Path, default=None, help="Write the default config JSON to this path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_default is not None:
        args.write_default.parent.mkdir(parents=True, exist_ok=True)
        args.write_default.write_text(
            json.dumps(default_subagent_config(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote default subagent config to {args.write_default}")
        return
    config = load_subagent_config(args.config)
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
