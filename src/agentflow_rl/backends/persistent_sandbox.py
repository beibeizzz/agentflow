from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from time import monotonic
from typing import Callable

from agentflow_rl.runtime.errors import InfrastructureError

from .bounded_process import OutputLimitExceeded, run_bounded
from .contracts import SandboxRequest, SandboxResult


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass
class _Worker:
    index: int
    name: str
    reuses: int = 0


@dataclass
class _WorkItem:
    request: SandboxRequest
    admitted_at: float
    future: asyncio.Future[SandboxResult]


class PersistentDockerSandboxPool:
    """Six long-lived Docker envelopes with one fresh Python process per call.

    The service process is the sole queue owner. Accepted requests enter one FIFO;
    each worker consumes one item at a time. A client cancellation leaves the work
    item owned by the pool until execution and cleanup finish, so capacity cannot
    be released while an untrusted process is still alive.
    """

    revision = "persistent-docker-sandbox-pool-v1"

    def __init__(
        self,
        *,
        image: str,
        docker_binary: str = "docker",
        cpus: float = 1.0,
        memory: str = "1g",
        output_limit_bytes: int = 1_000_000,
        max_workers: int = 6,
        max_queue: int = 34,
        max_reuses: int = 64,
        queue_admission_timeout_s: float = 0.05,
        pool_name: str = "agentflow-python-sandbox",
        command_runner: CommandRunner | None = None,
    ) -> None:
        if (
            cpus <= 0
            or output_limit_bytes <= 0
            or max_workers <= 0
            or max_queue < 0
            or max_reuses <= 0
            or queue_admission_timeout_s <= 0
        ):
            raise ValueError("sandbox pool limits must be positive")
        self.image = image
        self.docker_binary = docker_binary
        self.cpus = cpus
        self.memory = memory
        self.output_limit_bytes = output_limit_bytes
        self.max_workers = max_workers
        self.max_queue = max_queue
        self.max_reuses = max_reuses
        self.queue_admission_timeout_s = queue_admission_timeout_s
        self.pool_name = pool_name
        self.command_runner = command_runner or run_bounded
        self._queue: asyncio.Queue[_WorkItem] = asyncio.Queue()
        self._condition = asyncio.Condition()
        self._inflight = 0
        self._started = False
        self._closed = False
        self._start_lock = asyncio.Lock()
        self._workers: list[_Worker] = []
        self._consumers: list[asyncio.Task[None]] = []
        self._requests = 0
        self._rejections = 0
        self._rotations = 0
        self._peak_inflight = 0
        self._queue_wait_ms = 0.0
        self._execution_ms = 0.0

    @property
    def capacity(self) -> int:
        return self.max_workers + self.max_queue

    async def start(self) -> None:
        async with self._start_lock:
            if self._closed:
                raise RuntimeError("sandbox pool is closed")
            if self._started:
                return
            workers: list[_Worker] = []
            try:
                for index in range(self.max_workers):
                    worker = _Worker(index=index, name=f"{self.pool_name}-{index}")
                    await asyncio.to_thread(self._create_worker, worker)
                    workers.append(worker)
            except Exception:
                for worker in workers:
                    await asyncio.to_thread(self._remove_worker, worker)
                raise
            self._workers = workers
            self._consumers = [
                asyncio.create_task(self._consume(worker), name=f"sandbox-worker-{worker.index}")
                for worker in workers
            ]
            self._started = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            await self._queue.join()
            for task in self._consumers:
                task.cancel()
            await asyncio.gather(*self._consumers, return_exceptions=True)
            await asyncio.gather(
                *(asyncio.to_thread(self._remove_worker, worker) for worker in self._workers),
                return_exceptions=True,
            )

    async def execute(self, request: SandboxRequest) -> SandboxResult:
        await self.start()
        admitted_started = monotonic()
        try:
            await asyncio.wait_for(
                self._admit(), timeout=self.queue_admission_timeout_s
            )
        except asyncio.TimeoutError as exc:
            self._rejections += 1
            raise InfrastructureError("Python sandbox service capacity is exhausted") from exc

        admitted_at = monotonic()
        admission_wait_ms = (admitted_at - admitted_started) * 1000.0
        future: asyncio.Future[SandboxResult] = asyncio.get_running_loop().create_future()
        await self._queue.put(_WorkItem(request=request, admitted_at=admitted_at, future=future))
        try:
            result = await asyncio.shield(future)
            return result.model_copy(update={"admission_wait_ms": admission_wait_ms})
        except asyncio.CancelledError:
            # The worker retains ownership and releases capacity after cleanup.
            future.add_done_callback(self._consume_abandoned_future)
            raise

    @staticmethod
    def _consume_abandoned_future(future: asyncio.Future[SandboxResult]) -> None:
        try:
            future.exception()
        except (asyncio.CancelledError, Exception):
            pass

    async def _admit(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._inflight < self.capacity)
            self._inflight += 1
            self._peak_inflight = max(self._peak_inflight, self._inflight)

    async def _consume(self, worker: _Worker) -> None:
        while True:
            item = await self._queue.get()
            queue_wait_ms = (monotonic() - item.admitted_at) * 1000.0
            started = monotonic()
            rotate = False
            result: SandboxResult | None = None
            failure: Exception | None = None
            replenishment_failure: Exception | None = None
            try:
                result, rotate = await asyncio.to_thread(
                    self._execute_on_worker, worker, item.request
                )
                execution_ms = (monotonic() - started) * 1000.0
                self._requests += 1
                self._queue_wait_ms += queue_wait_ms
                self._execution_ms += execution_ms
                result = result.model_copy(
                    update={"queue_wait_ms": queue_wait_ms, "execution_ms": execution_ms}
                )
            except Exception as exc:
                rotate = True
                failure = (
                    exc if isinstance(exc, InfrastructureError)
                    else InfrastructureError("persistent Docker sandbox failed")
                )
            finally:
                worker.reuses += 1
                if rotate or worker.reuses >= self.max_reuses:
                    try:
                        await asyncio.to_thread(self._replace_worker, worker)
                        self._rotations += 1
                    except Exception as exc:
                        replenishment_failure = InfrastructureError(
                            "sandbox worker replenishment failed"
                        )
                async with self._condition:
                    self._inflight -= 1
                    self._condition.notify(1)
                self._queue.task_done()
                if not item.future.done():
                    if replenishment_failure is not None:
                        item.future.set_exception(replenishment_failure)
                    elif failure is not None:
                        item.future.set_exception(failure)
                    else:
                        assert result is not None
                        item.future.set_result(result)
            if replenishment_failure is not None:
                raise replenishment_failure

    def _run(self, command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        return self.command_runner(command, **kwargs)

    def _create_worker(self, worker: _Worker) -> None:
        self._remove_worker(worker)
        command = [
            self.docker_binary, "run", "-d", "--name", worker.name,
            "--label", f"agentflow.sandbox.pool={self.pool_name}",
            "--network", "none", "--read-only", "--user", "65532:65532",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--cpus", str(self.cpus), "--memory", self.memory,
            "--memory-swap", self.memory, "--pids-limit", "128",
            "--ulimit", "nofile=64:64", "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
            "--entrypoint", "tail", self.image, "-f", "/dev/null",
        ]
        try:
            self._run(
                command, capture_output=True, text=True, timeout=30.0,
                output_limit_bytes=self.output_limit_bytes, check=True,
            )
        except Exception as exc:
            raise InfrastructureError(f"failed to create sandbox worker {worker.index}") from exc
        worker.reuses = 0

    def _remove_worker(self, worker: _Worker) -> None:
        try:
            self._run(
                [self.docker_binary, "rm", "-f", worker.name],
                capture_output=True, text=True, timeout=10.0,
                output_limit_bytes=self.output_limit_bytes, check=False,
            )
        except Exception:
            pass

    def _replace_worker(self, worker: _Worker) -> None:
        self._remove_worker(worker)
        self._create_worker(worker)

    def _execute_on_worker(
        self, worker: _Worker, request: SandboxRequest
    ) -> tuple[SandboxResult, bool]:
        payload = request.model_dump(mode="json")
        payload["output_limit_bytes"] = self.output_limit_bytes
        command = [
            self.docker_binary, "exec", "-i", worker.name,
            "python", "/opt/agentflow/public_runner.py",
        ]
        try:
            completed = self._run(
                command, input=json.dumps(payload), capture_output=True, text=True,
                timeout=request.timeout_s + 5.0,
                output_limit_bytes=self.output_limit_bytes, check=True,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False, timed_out=True,
                stderr="persistent Docker sandbox exceeded its outer deadline",
            ), True
        except OutputLimitExceeded:
            return SandboxResult(
                ok=False, stderr="sandbox protocol output exceeded limit"
            ), True
        except (subprocess.CalledProcessError, OSError) as exc:
            raise InfrastructureError("Docker sandbox worker process failed") from exc

        if len((completed.stdout + completed.stderr).encode("utf-8")) > self.output_limit_bytes:
            return SandboxResult(
                ok=False, stderr="sandbox protocol output exceeded limit"
            ), True
        try:
            result = SandboxResult.model_validate_json(completed.stdout)
        except Exception as exc:
            raise InfrastructureError("Docker sandbox worker returned invalid JSON") from exc
        clean = self._worker_is_clean(worker)
        failure_types = {
            str(item.get("error_type", "")) for item in result.failures
        }
        rotate = (
            result.timed_out
            or "output exceeded" in result.stderr
            or bool(failure_types & {"OUTPUT_LIMIT", "TIMEOUT"})
            or result.exit_code in {-9, 137}
            or not clean
        )
        return result, rotate

    def _worker_is_clean(self, worker: _Worker) -> bool:
        try:
            top = self._run(
                [self.docker_binary, "top", worker.name, "-eo", "comm"],
                capture_output=True, text=True, timeout=5.0,
                output_limit_bytes=64_000, check=True,
            )
            processes = [line.strip() for line in top.stdout.splitlines()[1:] if line.strip()]
            temp = self._run(
                [self.docker_binary, "exec", worker.name, "sh", "-c",
                 "test -z \"$(find /tmp -mindepth 1 -print -quit)\""],
                capture_output=True, text=True, timeout=5.0,
                output_limit_bytes=64_000, check=False,
            )
            return processes == ["tail"] and temp.returncode == 0
        except Exception:
            return False

    def metrics(self) -> dict[str, float | int | bool]:
        consumers_ready = bool(self._consumers) and all(
            not task.done() for task in self._consumers
        )
        return {
            "ready": self._started and not self._closed and consumers_ready,
            "workers": len(self._workers),
            "max_workers": self.max_workers,
            "max_queue": self.max_queue,
            "capacity": self.capacity,
            "inflight": self._inflight,
            "queued": self._queue.qsize(),
            "requests": self._requests,
            "queue_rejections": self._rejections,
            "worker_rotations": self._rotations,
            "peak_inflight": self._peak_inflight,
            "queue_wait_ms": self._queue_wait_ms,
            "execution_ms": self._execution_ms,
            "max_reuses": self.max_reuses,
        }


__all__ = ["PersistentDockerSandboxPool"]
