from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from time import monotonic
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .schemas import CodeTest


@dataclass(frozen=True)
class TestRunResult:
    passed: int
    total: int
    failures: tuple[str, ...]
    timed_out: bool = False

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class CodeSandbox(Protocol):
    def run(self, code: str, tests: Sequence[CodeTest], *, timeout_s: float) -> TestRunResult: ...


def _normalize_stdout(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def _stdout_matches(actual: str, expected: str) -> bool:
    normalized_actual = _normalize_stdout(actual)
    normalized_expected = _normalize_stdout(expected)
    if normalized_actual == normalized_expected:
        return True
    actual_tokens = normalized_actual.split()
    expected_tokens = normalized_expected.split()
    if len(actual_tokens) != len(expected_tokens):
        return False
    try:
        return all(
            math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)
            for left, right in zip(actual_tokens, expected_tokens, strict=True)
        )
    except ValueError:
        return False


def _function_output_matches(actual: object, expected: object) -> bool:
    if isinstance(actual, tuple):
        actual = list(actual)
    if actual == expected:
        return True
    return isinstance(expected, list) and bool(expected) and actual == expected[0]


class LocalPythonSandbox:
    """Local test runner for trusted fixtures; formal trajectories use DockerSandbox."""

    def run(self, code: str, tests: Sequence[CodeTest], *, timeout_s: float = 10.0) -> TestRunResult:
        failures: list[str] = []
        passed = 0
        deadline = monotonic() + timeout_s
        with tempfile.TemporaryDirectory(prefix="agentflow-code-") as directory:
            root = Path(directory)
            solution = root / "solution.py"
            solution.write_text(code, encoding="utf-8")
            for index, test in enumerate(tests):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    failures.append(f"test_{index}: TIMEOUT")
                    return TestRunResult(passed, len(tests), tuple(failures), timed_out=True)
                try:
                    if test.fn_name:
                        runner = root / "runner.py"
                        runner.write_text(
                            "import importlib.util, json\n"
                            f"spec=importlib.util.spec_from_file_location('solution', {str(solution)!r})\n"
                            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
                            f"target=getattr(module, {test.fn_name!r}, None)\n"
                            "if target is None and hasattr(module, 'Solution'):\n"
                            f"    target=getattr(module.Solution(), {test.fn_name!r})\n"
                            "if target is None:\n"
                            f"    raise AttributeError({test.fn_name!r})\n"
                            f"args=json.loads({json.dumps(test.args)!r})\n"
                            "result=target(*args)\n"
                            "print(json.dumps(result, sort_keys=True))\n",
                            encoding="utf-8",
                        )
                        completed = subprocess.run(
                            [sys.executable, str(runner)], capture_output=True, text=True, timeout=remaining
                        )
                        actual = json.loads(completed.stdout) if completed.returncode == 0 else None
                        ok = completed.returncode == 0 and _function_output_matches(actual, test.expected)
                    else:
                        completed = subprocess.run(
                            [sys.executable, str(solution)],
                            input=test.stdin,
                            capture_output=True,
                            text=True,
                            timeout=remaining,
                        )
                        ok = completed.returncode == 0 and _stdout_matches(
                            completed.stdout, test.expected_stdout or ""
                        )
                    if ok:
                        passed += 1
                    else:
                        failures.append(f"test_{index}: {completed.stderr[-500:] or completed.stdout[-500:]}")
                except subprocess.TimeoutExpired:
                    failures.append(f"test_{index}: TIMEOUT")
                    return TestRunResult(passed, len(tests), tuple(failures), timed_out=True)
                except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                    failures.append(f"test_{index}: {type(exc).__name__}: {exc}")
        return TestRunResult(passed, len(tests), tuple(failures))


class DockerSandbox:
    def __init__(self, *, image: str = "agentflow-python-sandbox:3.11") -> None:
        self.image = image

    def run(self, code: str, tests: Sequence[CodeTest], *, timeout_s: float = 10.0) -> TestRunResult:
        request = json.dumps({
            "code": code,
            "tests": [test.model_dump(mode="json") for test in tests],
            "timeout_s": timeout_s,
        })
        command = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--cpus", "2", "--memory", "4g", "--pids-limit", "128",
            "--memory-swap", "4g", "--ulimit", "nofile=64:64",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
            "-i", self.image,
        ]
        try:
            completed = subprocess.run(
                command,
                input=request,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_s + 5,
            )
        except subprocess.TimeoutExpired:
            return TestRunResult(0, len(tests), ("SANDBOX_TIMEOUT",), timed_out=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "")[-1000:]
            raise RuntimeError(
                f"Docker sandbox failed with exit {exc.returncode}: {detail}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Docker sandbox process failed: {exc}") from exc
        try:
            payload = json.loads(completed.stdout)
            return TestRunResult(
                passed=int(payload["passed"]),
                total=int(payload["total"]),
                failures=tuple(str(value) for value in payload.get("failures", ())),
                timed_out=bool(payload.get("timed_out", False)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Docker sandbox returned an invalid result") from exc


__all__ = ["CodeSandbox", "DockerSandbox", "LocalPythonSandbox", "TestRunResult"]
