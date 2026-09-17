from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from time import monotonic

try:
    from bounded_process import OutputLimitExceeded, run_bounded
except ModuleNotFoundError as exc:
    if exc.name != "bounded_process":
        raise
    # Direct source-tree execution in CPU tests; the image places both files together.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/agentflow_rl/backends"))
    from bounded_process import OutputLimitExceeded, run_bounded


def trim(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def run_stdio(code_path: Path, stdin: str, timeout_s: float, output_limit_bytes: int = 1_000_000):
    return run_bounded(
        [sys.executable, "-I", str(code_path)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        output_limit_bytes=output_limit_bytes,
    )


def run_function(code_path: Path, test: dict, timeout_s: float, output_limit_bytes: int = 1_000_000):
    runner = code_path.parent / "call.py"
    runner.write_text(
        "import importlib.util,json\n"
        f"spec=importlib.util.spec_from_file_location('solution',{str(code_path)!r})\n"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n"
        f"f=getattr(m,{test['fn_name']!r},None)\n"
        f"f=f if f is not None else getattr(m.Solution(),{test['fn_name']!r})\n"
        f"print(json.dumps(f(*json.loads({json.dumps(test.get('args', []))!r}))))\n",
        encoding="utf-8",
    )
    return run_bounded(
        [sys.executable, "-I", str(runner)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        output_limit_bytes=output_limit_bytes,
    )


def main() -> int:
    request = json.load(sys.stdin)
    limit = int(request.get("output_limit_bytes", 1_000_000))
    deadline = monotonic() + float(request["timeout_s"])
    failures = []
    passed = 0
    tests = request.get("tests", [])
    if request["mode"] == "test" and not tests:
        print(json.dumps({"ok": False, "passed": 0, "total": 0,
                          "failures": [{"error_type": "INVALID_ARGUMENTS"}],
                          "stderr": "test requires at least one public case"}))
        return 0
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="agentflow-") as directory:
        code_path = Path(directory) / "solution.py"
        code_path.write_text(request["code"], encoding="utf-8")
        if request["mode"] == "run":
            try:
                completed = run_stdio(
                    code_path, request.get("stdin", ""), max(0.01, deadline - monotonic()), limit
                )
                result = {
                    "ok": completed.returncode == 0,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "exit_code": completed.returncode,
                    "timed_out": False,
                }
            except subprocess.TimeoutExpired:
                result = {"ok": False, "timed_out": True, "stderr": "execution timed out"}
            except OutputLimitExceeded:
                result = {"ok": False, "timed_out": False, "stderr": "execution output exceeded limit"}
            print(json.dumps(result, ensure_ascii=False))
            return 0

        remaining_output = limit
        for index, test in enumerate(tests):
            remaining = deadline - monotonic()
            if remaining <= 0:
                failures.append({"test_index": index, "error_type": "TIMEOUT"})
                break
            try:
                if remaining_output <= 0:
                    raise OutputLimitExceeded("request output budget exhausted")
                if test.get("fn_name"):
                    completed = run_function(code_path, test, remaining, remaining_output)
                else:
                    completed = run_stdio(code_path, test.get("stdin", ""), remaining, remaining_output)
                remaining_output -= completed.output_bytes_count
                returncode, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
                if test.get("fn_name"):
                    actual = json.loads(stdout) if returncode == 0 else None
                    expected = test.get("expected")
                    correct = returncode == 0 and actual == expected
                else:
                    actual = normalize(stdout)
                    expected = normalize(str(test.get("expected_stdout", "")))
                    correct = returncode == 0 and actual == expected
                if correct:
                    passed += 1
                else:
                    failures.append(
                        {
                            "test_index": index,
                            "error_type": "WRONG_ANSWER" if returncode == 0 else "RUNTIME_ERROR",
                            "actual": actual,
                            "expected": expected,
                            "stderr": trim(stderr, limit),
                        }
                    )
            except subprocess.TimeoutExpired:
                failures.append({"test_index": index, "error_type": "TIMEOUT"})
                break
            except OutputLimitExceeded:
                failures.append({"test_index": index, "error_type": "OUTPUT_LIMIT"})
                break
            except Exception as exc:
                failures.append(
                    {"test_index": index, "error_type": "HARNESS_ERROR", "message": str(exc)}
                )
        print(
            json.dumps(
                {
                    "ok": passed == len(tests),
                    "passed": passed,
                    "total": len(tests),
                    "failures": failures,
                    "timed_out": any(item["error_type"] == "TIMEOUT" for item in failures),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
