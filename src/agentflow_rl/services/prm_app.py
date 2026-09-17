from __future__ import annotations

import os
from contextlib import asynccontextmanager

from pydantic import BaseModel

from agentflow_rl.prm.transformers_backend import TransformersProcessRewardBackend
from agentflow_rl.prm.vllm_backend import VllmProcessRewardBackend
from agentflow_rl.prm.service import AsyncProcessRewardBatcher
from agentflow_rl.rewards.schemas import ProcessTransition


class ScoreRequest(BaseModel):
    transition: ProcessTransition


class ScoreBatchRequest(BaseModel):
    transitions: tuple[ProcessTransition, ...]


def build_backend_from_env():
    backend = os.environ.get("PRM_BACKEND", "transformers").strip().lower()
    common = {
        "revision": os.environ["PRM_REVISION"],
        "max_length": int(os.environ.get("PRM_MAX_LENGTH", "8192")),
    }
    if backend == "vllm":
        model_path = os.environ["PRM_MODEL_PATH"]
        return VllmProcessRewardBackend.from_pretrained(
            os.environ.get("PRM_TOKENIZER_PATH", model_path),
            model_path=model_path,
            endpoint=os.environ.get(
                "PRM_VLLM_ENDPOINT", "http://127.0.0.1:8004/classify"
            ),
            model=os.environ.get("PRM_VLLM_MODEL", "AgentFlow-PRM"),
            timeout_s=float(os.environ.get("PRM_VLLM_TIMEOUT_S", "120")),
            **common,
        )
    if backend == "transformers":
        return TransformersProcessRewardBackend.from_pretrained(
            os.environ["PRM_MODEL_PATH"],
            device=os.environ.get("PRM_DEVICE", "cuda:0"),
            **common,
        )
    raise ValueError(f"unsupported PRM backend: {backend}")


def create_app(
    backend=None,
    *,
    batch_max_size: int | None = None,
    batch_wait_ms: float | None = None,
    batch_max_queue: int | None = None,
):
    from fastapi import FastAPI

    scorer = backend or build_backend_from_env()
    batcher = AsyncProcessRewardBatcher(
        scorer,
        max_batch_size=(
            batch_max_size
            if batch_max_size is not None
            else int(os.environ.get("PRM_BATCH_MAX_SIZE", "32"))
        ),
        batch_wait_ms=(
            batch_wait_ms
            if batch_wait_ms is not None
            else float(os.environ.get("PRM_BATCH_WAIT_MS", "10"))
        ),
        max_queue=(
            batch_max_queue
            if batch_max_queue is not None
            else int(os.environ.get("PRM_BATCH_MAX_QUEUE", "256"))
        ),
    )
    identity = {
        "revision": scorer.revision,
        "rubric_revision": getattr(scorer, "rubric_revision", "legacy"),
    }
    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await batcher.close()

    app = FastAPI(title="AgentFlow Process Reward", version="1", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", **identity, "batch_metrics": batcher.metrics()}

    @app.post("/score")
    async def score(request: ScoreRequest):
        return {
            "score": await batcher.predict(request.transition),
            **identity,
        }

    @app.post("/score_batch")
    async def score_batch(request: ScoreBatchRequest):
        scores = await batcher.predict_many(request.transitions)
        return {"scores": list(scores), **identity}

    return app


__all__ = [
    "ScoreBatchRequest",
    "ScoreRequest",
    "build_backend_from_env",
    "create_app",
]
