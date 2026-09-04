from __future__ import annotations

import argparse
import json

from agentflow_rl.tasks.coding.sandbox import DockerSandbox
from agentflow_rl.tasks.coding.schemas import CodeTest


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the isolated Docker coding backend")
    parser.add_argument("--image", default="agentflow-python-sandbox:3.11")
    args = parser.parse_args()

    sandbox = DockerSandbox(image=args.image)
    function_result = sandbox.run(
        "class Solution:\n    def add(self, a, b):\n        return a + b\n",
        [CodeTest(fn_name="add", args=[20, 22], expected=[42])],
        timeout_s=5.0,
    )
    stdio_result = sandbox.run(
        "value = float(input())\nprint(value / 3)\n",
        [CodeTest(stdin="1\n", expected_stdout="0.333333")],
        timeout_s=5.0,
    )
    isolation_result = sandbox.run(
        "def request_is_hidden():\n"
        "    try:\n"
        "        open('/work/request.json').read()\n"
        "    except (PermissionError, FileNotFoundError):\n"
        "        return True\n"
        "    return False\n",
        [CodeTest(fn_name="request_is_hidden", args=[], expected=True)],
        timeout_s=5.0,
    )
    output_limit_result = sandbox.run(
        "print('x' * 1100000)\n",
        [CodeTest(stdin="", expected_stdout="unreachable")],
        timeout_s=5.0,
    )
    results = (function_result, stdio_result, isolation_result)
    if any(result.passed != 1 or result.total != 1 or result.timed_out for result in results):
        raise RuntimeError(f"coding sandbox health check failed: {results}")
    if output_limit_result.passed or output_limit_result.timed_out:
        raise RuntimeError(
            f"coding sandbox output limit check failed: {output_limit_result}"
        )
    print(json.dumps({
        "ready": True,
        "image": args.image,
        "checks": {
            "function_class": function_result.pass_rate,
            "stdio_numeric": stdio_result.pass_rate,
            "request_isolation": isolation_result.pass_rate,
            "output_limit": "enforced",
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
