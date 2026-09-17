"""Bounded subprocess capture shared by the host and the sandbox image.

Standard library only: the Dockerfile copies this exact module into /opt/agentflow.
The byte budget is shared by stdout and stderr, including simultaneous writes.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from time import monotonic


class OutputLimitExceeded(subprocess.SubprocessError):
    pass


def run_bounded(command, *, input=None, timeout, output_limit_bytes=1_000_000,
                capture_output=True, text=True, check=False):
    if output_limit_bytes <= 0 or timeout <= 0:
        raise ValueError("process output limit and timeout must be positive")
    if not capture_output or not text:
        raise ValueError("bounded runner requires captured text output")
    deadline = monotonic() + timeout
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0, start_new_session=os.name == "posix",
    )
    buffers = [bytearray(), bytearray()]
    lock = threading.Lock()
    overflow = threading.Event()
    done = [threading.Event(), threading.Event()]
    errors = []
    used = 0

    def read_stream(pipe, index):
        nonlocal used
        try:
            while chunk := pipe.read(8192):
                with lock:
                    remaining = output_limit_bytes - used
                    buffers[index].extend(chunk[:remaining])
                    used += min(len(chunk), remaining)
                    if len(chunk) > remaining:
                        overflow.set()
                        return
        except OSError as exc:
            errors.append(exc)
        finally:
            pipe.close()
            done[index].set()

    def write_input():
        try:
            data = (input or "").encode("utf-8")
            view = memoryview(data)
            while view:
                count = process.stdin.write(view[:8192])
                if not count:
                    break
                view = view[count:]
        except (BrokenPipeError, OSError):
            pass
        finally:
            process.stdin.close()

    threads = [threading.Thread(target=read_stream, args=(pipe, index), daemon=True)
               for index, pipe in enumerate((process.stdout, process.stderr))]
    threads.append(threading.Thread(target=write_input, daemon=True))
    for thread in threads:
        thread.start()
    try:
        while True:
            if overflow.is_set():
                raise OutputLimitExceeded("combined stdout/stderr exceeded byte limit")
            if process.poll() is not None and all(event.is_set() for event in done):
                break
            if monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            overflow.wait(min(0.01, max(0, deadline - monotonic())))
        if overflow.is_set():
            raise OutputLimitExceeded("combined stdout/stderr exceeded byte limit")
        if errors:
            raise errors[0]
    finally:
        # Also remove descendants that survived normal parent termination.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=1)
    stdout, stderr = (bytes(value).decode("utf-8", errors="replace") for value in buffers)
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    result.output_bytes_count = used
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, stdout, stderr)
    return result
