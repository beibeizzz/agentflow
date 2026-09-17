from __future__ import annotations

import os
from contextlib import asynccontextmanager

from agentflow_rl.backends.contracts import SandboxRequest
from agentflow_rl.backends.persistent_sandbox import PersistentDockerSandboxPool
from agentflow_rl.runtime.errors import InfrastructureError


SERVICE_REVISION = "python-sandbox-service-v1"


def create_app(pool: PersistentDockerSandboxPool | None = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - optional service dependency boundary
        raise RuntimeError("FastAPI is required for the Python sandbox service") from exc

    backend = pool or PersistentDockerSandboxPool(
        image=os.environ["SANDBOX_IMAGE"],
        cpus=float(os.environ.get("PYTHON_SANDBOX_CPUS", "1")),
        memory=os.environ.get("PYTHON_SANDBOX_MEMORY", "1g"),
        output_limit_bytes=int(os.environ.get("PYTHON_SANDBOX_OUTPUT_LIMIT_BYTES", "1000000")),
        max_workers=int(os.environ.get("PYTHON_SANDBOX_WORKERS", "6")),
        max_queue=int(os.environ.get("PYTHON_SANDBOX_MAX_QUEUE", "34")),
        max_reuses=int(os.environ.get("PYTHON_SANDBOX_MAX_REUSES", "64")),
        queue_admission_timeout_s=float(
            os.environ.get("PYTHON_SANDBOX_ADMISSION_TIMEOUT_S", "0.05")
        ),
        pool_name=os.environ.get("PYTHON_SANDBOX_POOL_NAME", "agentflow-python-sandbox"),
    )

    @asynccontextmanager
    async def lifespan(_app):
        await backend.start()
        yield
        await backend.close()

    app = FastAPI(title="AgentFlow Python Sandbox", version="1", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "revision": SERVICE_REVISION,
            "pool_revision": backend.revision,
            "image": backend.image,
            **backend.metrics(),
        }

    @app.post("/execute")
    async def execute(request: SandboxRequest):
        try:
            result = await backend.execute(request)
        except InfrastructureError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"revision": SERVICE_REVISION, "result": result.model_dump(mode="json")}

    return app


__all__ = ["SERVICE_REVISION", "create_app"]
