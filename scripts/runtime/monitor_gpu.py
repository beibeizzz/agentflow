from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from agentflow_rl.integrations.host_resources import snapshot as host_snapshot


def sample() -> dict[str, dict[str, float]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    values = {}
    for line in output.splitlines():
        index, total, used = (part.strip() for part in line.split(","))
        values[f"gpu-{index}"] = {"total_mib": float(total), "used_mib": float(used)}
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        raise ValueError("GPU monitor requires a command and positive interval")
    process = subprocess.Popen(command)
    peaks: dict[str, dict[str, float]] = {}
    initial_host = host_snapshot(args.data_root)
    host = {
        "effective_cpu_cores": int(initial_host["effective_cpu_cores"]),
        "minimum_memory_available_bytes": int(
            initial_host["effective_memory_available_bytes"]
        ),
        "minimum_data_disk_free_bytes": int(initial_host["data_disk_free_bytes"]),
        "initial_swap_used_bytes": int(initial_host["swap_used_bytes"]),
        "maximum_swap_used_bytes": int(initial_host["swap_used_bytes"]),
    }

    def update_host() -> None:
        values = host_snapshot(args.data_root)
        host["effective_cpu_cores"] = min(
            host["effective_cpu_cores"], int(values["effective_cpu_cores"])
        )
        host["minimum_memory_available_bytes"] = min(
            host["minimum_memory_available_bytes"],
            int(values["effective_memory_available_bytes"]),
        )
        host["minimum_data_disk_free_bytes"] = min(
            host["minimum_data_disk_free_bytes"], int(values["data_disk_free_bytes"])
        )
        host["maximum_swap_used_bytes"] = max(
            host["maximum_swap_used_bytes"], int(values["swap_used_bytes"])
        )

    while process.poll() is None:
        for device, values in sample().items():
            current = peaks.setdefault(
                device,
                {"total_mib": values["total_mib"], "peak_used_mib": 0.0},
            )
            current["peak_used_mib"] = max(current["peak_used_mib"], values["used_mib"])
        update_host()
        time.sleep(args.interval)
    for device, values in sample().items():
        current = peaks.setdefault(
            device, {"total_mib": values["total_mib"], "peak_used_mib": 0.0}
        )
        current["peak_used_mib"] = max(current["peak_used_mib"], values["used_mib"])
    update_host()
    host["swap_growth_bytes"] = max(
        0, host["maximum_swap_used_bytes"] - host["initial_swap_used_bytes"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"schema_version": 2, "devices": peaks, "host": host},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
