from __future__ import annotations

import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from time import monotonic
from typing import Any

SANDBOX_UID = 65534
OUTPUT_LIMIT_BYTES = 1_000_000


def drop_candidate_privileges() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT_BYTES, OUTPUT_LIMIT_BYTES))
    os.setgroups([])
    os.setgid(SANDBOX_UID)
    os.setuid(SANDBOX_UID)


def normalize_stdout(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def stdout_matches(actual: str, expected: str) -> bool:
    normalized_actual = normalize_stdout(actual)
    normalized_expected = normalize_stdout(expected)
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


def function_output_matches(actual: object, expected: object) -> bool:
    if isinstance(actual, tuple):
        actual = list(actual)
    if actual == expected:
        return True
    return isinstance(expected, list) and bool(expected) and actual == expected[0]


def run_candidate(
    command: list[str],
    *,
    root: Path,
    timeout_s: float,
    stdin: str | None = None,
) -> tuple[int, str, str]:
    stdout_path = root / "candidate.stdout"
    stderr_path = root / "candidate.stderr"
    with stdout_path.open("w+b") as stdout_handle, stderr_path.open("w+b") as stderr_handle:
        completed = subprocess.run(
            command,
            input=stdin.encode("utf-8") if stdin is not None else None,
            stdout=stdout_handle,
            stderr=stderr_handle,
            timeout=timeout_s,
            cwd=root,
            preexec_fn=drop_candidate_privileges,
        )
    stdout = stdout_path.read_bytes()[:OUTPUT_LIMIT_BYTES].decode("utf-8", errors="replace")
    stderr = stderr_path.read_bytes()[:OUTPUT_LIMIT_BYTES].decode("utf-8", errors="replace")
    return completed.returncode, stdout, stderr


def run_case(
    solution: Path, test: dict[str, Any], timeout_s: float
) -> tuple[bool, Any, str, str]:
    if test.get("fn_name"):
        runner = solution.parent / "function_case.py"
        runner.write_text(
            "import importlib.util, json\n"
            f"spec=importlib.util.spec_from_file_location('solution', {str(solution)!r})\n"
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
            f"target=getattr(module, {str(test['fn_name'])!r}, None)\n"
            "if target is None and hasattr(module, 'Solution'):\n"
            f"    target=getattr(module.Solution(), {str(test['fn_name'])!r})\n"
            "if target is None:\n"
            f"    raise AttributeError({str(test['fn_name'])!r})\n"
            f"args=json.loads({json.dumps(test.get('args'))!r})\n"
            "result=target(*args)\n"
            "print(json.dumps(result, sort_keys=True))\n",
            encoding="utf-8",
        )
        runner.chmod(0o444)
        returncode, stdout, stderr = run_candidate(
            [sys.executable, "-I", str(runner)],
            root=solution.parent,
            timeout=timeout_s,
        )
        if returncode:
            return False, None, stderr[-1000:], "RUNTIME_ERROR"
        try:
            actual = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return False, stdout[-1000:], str(exc), "OUTPUT_PARSE_ERROR"
        return function_output_matches(actual, test.get("expected")), actual, stderr[-1000:], "WRONG_ANSWER"

    returncode, stdout, stderr = run_candidate(
        [sys.executable, "-I", str(solution)],
        stdin=str(test.get("stdin", "")),
        root=solution.parent,
        timeout=timeout_s,
    )
    if returncode:
        return False, stdout[-1000:], stderr[-1000:], "RUNTIME_ERROR"
    return (
        stdout_matches(stdout, str(test.get("expected_stdout", ""))),
        stdout[-1000:],
        stderr[-1000:],
        "WRONG_ANSWER",
    )


def failure_payload(
    index: int,
    test: dict[str, Any],
    error_type: str,
    *,
    actual: Any = None,
    stderr: str = "",
    message: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "test_index": index,
        "error_type": error_type,
        "stderr": stderr[-1000:],
        "message": message,
    }
    if test.get("fn_name"):
        payload.update({
            "interface": "function",
            "fn_name": test.get("fn_name"),
            "args": test.get("args"),
            "expected": test.get("expected"),
            "actual": actual,
        })
    else:
        payload.update({
            "interface": "stdio",
            "input": test.get("stdin"),
            "expected": test.get("expected_stdout"),
            "actual": actual,
        })
    return payload


def main() -> int:
    payload = json.loads(sys.stdin.read())
    timeout_s = float(payload.get("timeout_s", 10.0))
    tests = list(payload.get("tests", ()))
    passed = 0
    failures = []
    timed_out = False
    deadline = monotonic() + timeout_s
    with tempfile.TemporaryDirectory(prefix="case-", dir="/tmp") as directory:
        solution = Path(directory) / "solution.py"
        solution.write_text(str(payload["code"]), encoding="utf-8")
        solution.chmod(0o444)
        # Root keeps write access for per-case harnesses; uid 65534 receives r-x.
        solution.parent.chmod(0o755)
        for index, test in enumerate(tests):
            remaining = deadline - monotonic()
            if remaining <= 0:
                failures.append(failure_payload(
                    index, test, "TIMEOUT", message="test budget exhausted"
                ))
                timed_out = True
                break
            try:
                ok, actual, stderr, error_type = run_case(solution, test, remaining)
                if ok:
                    passed += 1
                else:
                    failures.append(failure_payload(
                        index,
                        test,
                        error_type,
                        actual=actual,
                        stderr=stderr,
                    ))
            except subprocess.TimeoutExpired:
                failures.append(failure_payload(
                    index, test, "TIMEOUT", message="candidate timed out"
                ))
                timed_out = True
                break
            except (OSError, TypeError, ValueError) as exc:
                failures.append(failure_payload(
                    index,
                    test,
                    "HARNESS_ERROR",
                    message=f"{type(exc).__name__}: {exc}",
                ))
    print(json.dumps({
        "passed": passed,
        "total": len(tests),
        "failures": failures,
        "timed_out": timed_out,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
