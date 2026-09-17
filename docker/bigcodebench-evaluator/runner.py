from __future__ import annotations

import json
import sys

from bigcodebench.eval import PASS, untrusted_check


def main() -> int:
    request = json.load(sys.stdin)
    status, details = untrusted_check(
        code=request["code"],
        test_code=request["test_code"],
        entry_point=request["entry_point"],
        max_as_limit=30 * 1024,
        max_data_limit=30 * 1024,
        max_stack_limit=10,
        min_time_limit=1.0,
        gt_time_limit=float(request.get("gt_time_limit", 60.0)),
    )
    print(
        json.dumps(
            {
                "passed": status == PASS,
                "status": str(status),
                "details": dict(details),
                "timed_out": str(status).casefold() == "timeout",
            },
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
