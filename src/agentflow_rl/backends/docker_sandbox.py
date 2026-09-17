from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from functools import partial
from time import monotonic

from agentflow_rl.runtime.errors import InfrastructureError

from .contracts import SandboxRequest, SandboxResult
from .concurrency import CrossProcessSlotLimiter
from .bounded_process import OutputLimitExceeded, run_bounded


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class DockerSandboxBackend:
    revision = "docker-sandbox-v3"

    def __init__(
        self,
        *,
        image: str = "agentflow-python-sandbox:3.11",
        docker_binary: str = "docker",
        cpus: float = 1.0,
        memory: str = "1g",
        output_limit_bytes: int = 1_000_000,
        command_runner: CommandRunner | None = None,
        max_concurrency: int = 6,
        max_queue: int = 34,
        queue_admission_timeout_s: float = 0.05,
        coordination_dir: str | None = None,
    ) -> None:
        if (
            cpus <= 0
            or output_limit_bytes <= 0
            or max_concurrency <= 0
            or max_queue < 0
            or queue_admission_timeout_s <= 0
        ):
            raise ValueError("sandbox resource limits must be positive")
        self.image = image
        self.docker_binary = docker_binary
        self.cpus = cpus
        self.memory = memory
        self.output_limit_bytes = output_limit_bytes
        self.command_runner = command_runner or partial(run_bounded, output_limit_bytes=output_limit_bytes)
        limiter_root = coordination_dir or f"{tempfile.gettempdir()}/agentflow-locks"
        self.limiter = CrossProcessSlotLimiter(
            limiter_root,
            name="python-sandbox",
            limit=max_concurrency,
        )
        self.capacity = CrossProcessSlotLimiter(
            limiter_root,
            name="python-sandbox-capacity",
            limit=max_concurrency + max_queue,
        )
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.queue_admission_timeout_s = queue_admission_timeout_s
        self.requests = 0
        self.queue_rejections = 0
        self.inflight = 0
        self.peak_inflight = 0
        self.admission_wait_ms = 0.0
        self.queue_wait_ms = 0.0
        self.execution_ms = 0.0

    async def execute(self, request: SandboxRequest) -> SandboxResult:
        admitted = False
        try:
            async with self.capacity.slot(
                timeout_s=self.queue_admission_timeout_s
            ) as admission_wait_ms:
                admitted = True
                self.admission_wait_ms += admission_wait_ms
                self.inflight += 1
                self.peak_inflight = max(self.peak_inflight, self.inflight)
                try:
                    async with self.limiter.slot(
                        timeout_s=request.timeout_s + 10.0
                    ) as wait_ms:
                        self.requests += 1
                        self.queue_wait_ms += wait_ms
                        execution_started = monotonic()
                        result = await asyncio.to_thread(
                            self._execute_blocking, request
                        )
                        execution_ms = (monotonic() - execution_started) * 1000.0
                        self.execution_ms += execution_ms
                        return result.model_copy(
                            update={
                                "admission_wait_ms": admission_wait_ms,
                                "queue_wait_ms": wait_ms,
                                "execution_ms": execution_ms,
                            }
                        )
                finally:
                    self.inflight -= 1
        except InfrastructureError:
            if not admitted:
                self.queue_rejections += 1
            raise

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
            "128",
            "--ulimit",
            "nofile=64:64",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "-i",
            self.image,
        ]

    def _execute_blocking(self, request: SandboxRequest) -> SandboxResult:
        payload = request.model_dump(mode="json")
        payload["output_limit_bytes"] = self.output_limit_bytes
        container_name = f"agentflow-sandbox-{uuid.uuid4().hex[:16]}"
        try:
            completed = self.command_runner(
                self.command(container_name),
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=request.timeout_s + 5.0,
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
            return SandboxResult(
                ok=False,
                timed_out=isinstance(exc, subprocess.TimeoutExpired),
                stderr=("Docker sandbox exceeded its outer deadline" if isinstance(exc, subprocess.TimeoutExpired)
                        else "sandbox protocol output exceeded limit"),
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            raise InfrastructureError("Docker sandbox process failed") from exc

        if len((completed.stdout + completed.stderr).encode("utf-8")) > self.output_limit_bytes:
            return SandboxResult(ok=False, stderr="sandbox protocol output exceeded limit")
        try:
            return SandboxResult.model_validate_json(completed.stdout)
        except Exception as exc:
            raise InfrastructureError("Docker sandbox returned invalid JSON") from exc

    def metrics(self) -> dict[str, float]:
        return {
            "requests": float(self.requests),
            "queue_rejections": float(self.queue_rejections),
            "peak_inflight": float(self.peak_inflight),
            "admission_wait_ms": self.admission_wait_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "execution_ms": self.execution_ms,
        }


__all__ = ["DockerSandboxBackend"]
