from __future__ import annotations

import os

import uvicorn

from agentflow_rl.services.wikipedia_app import create_app


def main() -> int:
    uvicorn.run(
        create_app(),
        host=os.environ.get("WIKIPEDIA_HOST", "127.0.0.1"),
        port=int(os.environ.get("WIKIPEDIA_PORT", "8002")),
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
