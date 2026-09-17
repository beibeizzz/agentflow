from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
from typing import Any


_UNLIMITED_CGROUP_BYTES = 1 << 60


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_int(path: Path) -> int | None:
    value = _read_text(path)
    if value is None or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed < 0 or parsed >= _UNLIMITED_CGROUP_BYTES:
        return None
    return parsed


def _cgroup_paths(cgroup_root: Path, controller: str) -> tuple[Path, ...]:
    candidates = [cgroup_root]
    membership = _read_text(Path("/proc/self/cgroup"))
    if membership:
        for line in membership.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            controllers, relative = parts[1], parts[2].lstrip("/")
            if controllers == "":
                candidates.append(cgroup_root / relative)
            elif controller in controllers.split(","):
                candidates.extend(
                    (
                        cgroup_root / controllers / relative,
                        cgroup_root / controller / relative,
                    )
                )
    unique = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return tuple(unique)


def _cpu_quota(cgroup_root: Path) -> float | None:
    for base in _cgroup_paths(cgroup_root, "cpu"):
        cpu_max = _read_text(base / "cpu.max")
        if cpu_max:
            quota, period, *_ = cpu_max.split()
            if quota != "max" and int(period) > 0:
                return int(quota) / int(period)

        quota = _read_int(base / "cpu.cfs_quota_us")
        period = _read_int(base / "cpu.cfs_period_us")
        if quota is not None and period:
            return quota / period
    return None


def cpu_capacity(cgroup_root: Path = Path("/sys/fs/cgroup")) -> dict[str, Any]:
    logical = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = logical
    quota = _cpu_quota(cgroup_root)
    candidates = [float(logical), float(affinity)]
    if quota is not None:
        candidates.append(quota)
    effective = min(candidates)
    return {
        "logical_cpu_count": logical,
        "affinity_cpu_count": affinity,
        "cgroup_cpu_quota_cores": quota,
        # The scheduler can use a fractional quota.  The formal gate requires
        # whole schedulable cores, so a partial core does not satisfy it.
        "effective_cpu_cores": math.floor(effective),
    }


def _proc_memory(meminfo_path: Path) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values["MemAvailable"], values["SwapTotal"] - values["SwapFree"]


def memory_capacity(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    meminfo_path: Path = Path("/proc/meminfo"),
) -> dict[str, int | None]:
    system_available, system_swap_used = _proc_memory(meminfo_path)

    limit = current = swap_current = None
    for base in _cgroup_paths(cgroup_root, "memory"):
        candidate_limit = _read_int(base / "memory.max")
        candidate_current = _read_int(base / "memory.current")
        candidate_swap = _read_int(base / "memory.swap.current")
        if candidate_limit is None:
            candidate_limit = _read_int(base / "memory.limit_in_bytes")
            candidate_current = _read_int(base / "memory.usage_in_bytes")
            memsw = _read_int(base / "memory.memsw.usage_in_bytes")
            if memsw is not None and candidate_current is not None:
                candidate_swap = max(0, memsw - candidate_current)
        if candidate_limit is not None and candidate_current is not None:
            if limit is None or candidate_limit < limit:
                limit = candidate_limit
                current = candidate_current
                swap_current = candidate_swap

    cgroup_available = None
    if limit is not None and current is not None:
        cgroup_available = max(0, limit - current)
    effective_available = system_available
    if cgroup_available is not None:
        effective_available = min(effective_available, cgroup_available)

    return {
        "system_memory_available_bytes": system_available,
        "cgroup_memory_limit_bytes": limit,
        "cgroup_memory_current_bytes": current,
        "effective_memory_available_bytes": effective_available,
        "swap_used_bytes": (
            swap_current if swap_current is not None else system_swap_used
        ),
    }


def snapshot(data_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(data_root)
    cpu = cpu_capacity()
    memory = memory_capacity()
    return {
        **cpu,
        **memory,
        "data_root": str(data_root.resolve()),
        "data_disk_free_bytes": disk.free,
        "data_disk_total_bytes": disk.total,
    }


__all__ = ["cpu_capacity", "memory_capacity", "snapshot"]
