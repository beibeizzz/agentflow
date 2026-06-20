from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


FLOWGRPO_DIR = Path(__file__).resolve().parent
GSM8K_DIR = FLOWGRPO_DIR.parent
PROJECT_ROOT = GSM8K_DIR.parent
sys.path.insert(0, str(GSM8K_DIR))

from flowgrpo.config_utils import export_env, load_yaml_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch verl Flow-GRPO training with a YAML config.")
    parser.add_argument("--config", type=Path, default=FLOWGRPO_DIR / "config_smoke.yaml")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = load_yaml_config(config_path)
    export_env(config)

    command = [sys.executable, "-m", "agentflow.verl"]
    for key, value in config.get("python_args", {}).items():
        expanded = os.path.expandvars(str(value))
        command.append(f"{key}={expanded}")
    command.extend(args.overrides)

    print("Starting Flow-GRPO trainer:")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=os.environ)


if __name__ == "__main__":
    main()
