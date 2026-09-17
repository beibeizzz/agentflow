from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from time import monotonic
from typing import AsyncIterator, BinaryIO

from agentflow_rl.runtime.errors import InfrastructureError


def _try_lock(handle: BinaryIO) -> bool:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        try:
            handle.write(b"0")
            handle.flush()
        except OSError:
            # Another process may be initializing the same one-byte Windows
            # lock file. Treat this slot as busy and retry through the caller.
            return False
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class CrossProcessSlotLimiter:
    """Coordinate bounded host resources across AgentLoop worker processes."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str,
        limit: int,
        poll_interval_s: float = 0.02,
    ) -> None:
        if limit <= 0 or poll_interval_s <= 0:
            raise ValueError("slot limiter settings must be positive")
        self.root = Path(root)
        self.name = name
        self.limit = limit
        self.poll_interval_s = poll_interval_s
        self.root.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def slot(self, *, timeout_s: float) -> AsyncIterator[float]:
        if timeout_s <= 0:
            raise ValueError("slot acquisition timeout must be positive")
        started = monotonic()
        handle: BinaryIO | None = None
        while handle is None:
            for index in range(self.limit):
                candidate = (self.root / f"{self.name}-{index}.lock").open("a+b")
                if _try_lock(candidate):
                    handle = candidate
                    break
                candidate.close()
            if handle is not None:
                break
            if monotonic() - started >= timeout_s:
                raise InfrastructureError(f"timed out waiting for {self.name} capacity")
            await asyncio.sleep(self.poll_interval_s)
        try:
            yield (monotonic() - started) * 1000.0
        finally:
            _unlock(handle)
            handle.close()


__all__ = ["CrossProcessSlotLimiter"]
