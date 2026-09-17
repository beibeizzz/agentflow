from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from agentflow_rl.integrations.host_resources import snapshot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture Linux host capacity for the formal preflight."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 2,
        "timestamp_unix_s": time.time(),
        **snapshot(args.data_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
