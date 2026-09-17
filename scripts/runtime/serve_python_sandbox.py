from __future__ import annotations

import os

import uvicorn

from agentflow_rl.services.python_sandbox_app import create_app


def main() -> int:
    uvicorn.run(
        create_app(),
        host=os.environ.get("PYTHON_SANDBOX_HOST", "127.0.0.1"),
        port=int(os.environ.get("PYTHON_SANDBOX_PORT", "8005")),
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
