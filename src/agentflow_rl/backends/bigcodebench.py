from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from functools import partial
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agentflow_rl.runtime.errors import InfrastructureError
from .concurrency import CrossProcessSlotLimiter
from .bounded_process import OutputLimitExceeded, run_bounded


class BigCodeBenchExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    status: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)
    timed_out: bool = False


class BigCodeBenchBackend(Protocol):
    revision: str

    async def evaluate(
        self,
        *,
        code: str,
        test_code: str,
        entry_point: str,
        gt_time_limit: float,
    ) -> BigCodeBenchExecution: ...


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class DockerBigCodeBenchBackend:
    """Run the pinned BigCodeBench `untrusted_check` harness in its own image."""

    def __init__(
        self,
        *,
        image: str,
        revision: str,
        docker_binary: str = "docker",
        cpus: float = 2.0,
        memory: str = "8g",
        timeout_s: float = 300.0,
        command_runner: CommandRunner | None = None,
        output_limit_bytes: int = 1_000_000,
        max_concurrency: int = 2,
        coordination_dir: str | None = None,
    ) -> None:
        if cpus <= 0 or timeout_s <= 0 or output_limit_bytes <= 0:
            raise ValueError("BigCodeBench resource limits must be positive")
        self.image = image
        self.revision = revision
        self.docker_binary = docker_binary
        self.cpus = cpus
        self.memory = memory
        self.timeout_s = timeout_s
        self.output_limit_bytes = output_limit_bytes
        self.command_runner = command_runner or partial(run_bounded, output_limit_bytes=output_limit_bytes)
        self.limiter = CrossProcessSlotLimiter(
            coordination_dir or f"{tempfile.gettempdir()}/agentflow-locks",
            name="bigcodebench",
            limit=max_concurrency,
        )
        self.requests = 0
        self.queue_wait_ms = 0.0

    def command(self, container_name: str | None = None) -> list[str]:
        identity = ["--name", container_name] if container_name else []
        return [
            self.docker_binary,
            "run",
            "--rm",
            *identity,
            "--network",
            "none",
            "--read-only",
            "--user",
            "65532:65532",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--cpus",
            str(self.cpus),
            "--memory",
            self.memory,
            "--memory-swap",
            self.memory,
            "--pids-limit",
            "256",
            "--ulimit",
            "nofile=128:128",
            "--tmpfs",
            "/tmp:rw,nosuid,size=2g",
            "-i",
            self.image,
        ]

    async def evaluate(
        self,
        *,
        code: str,
        test_code: str,
        entry_point: str,
        gt_time_limit: float,
    ) -> BigCodeBenchExecution:
        payload = {
            "code": code,
            "test_code": test_code,
            "entry_point": entry_point,
            "gt_time_limit": gt_time_limit,
        }
        async with self.limiter.slot(timeout_s=self.timeout_s + 10.0) as wait_ms:
            self.requests += 1
            self.queue_wait_ms += wait_ms
            return await asyncio.to_thread(self._execute, payload)

    def _execute(self, payload: dict[str, Any]) -> BigCodeBenchExecution:
        container_name = f"agentflow-bigcode-{uuid.uuid4().hex[:16]}"
        try:
            completed = self.command_runner(
                self.command(container_name),
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=True,
            )
        except (subprocess.TimeoutExpired, OutputLimitExceeded) as exc:
            try:
                self.command_runner(
                    [self.docker_binary, "rm", "-f", container_name],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    check=False,
                )
            except Exception:
                pass
            return BigCodeBenchExecution(
                passed=False,
                status="timeout" if isinstance(exc, subprocess.TimeoutExpired) else "output_limit",
                details={"outer_deadline": self.timeout_s} if isinstance(exc, subprocess.TimeoutExpired)
                        else {"output_limit_bytes": self.output_limit_bytes},
                timed_out=isinstance(exc, subprocess.TimeoutExpired),
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            raise InfrastructureError("BigCodeBench Docker evaluator failed") from exc
        if len((completed.stdout + completed.stderr).encode("utf-8")) > self.output_limit_bytes:
            return BigCodeBenchExecution(passed=False, status="output_limit")
        try:
            return BigCodeBenchExecution.model_validate_json(completed.stdout)
        except Exception as exc:
            raise InfrastructureError("BigCodeBench evaluator returned invalid JSON") from exc

    def metrics(self) -> dict[str, float]:
        return {
            "requests": float(self.requests),
            "queue_wait_ms": self.queue_wait_ms,
        }


__all__ = [
    "BigCodeBenchBackend",
    "BigCodeBenchExecution",
    "DockerBigCodeBenchBackend",
]
