from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
import urllib.request

from agentflow_rl.integrations.bootstrap import build_runtime_bundle
from agentflow_rl.integrations.verl_main import load_config
from agentflow_rl.runtime.contracts import ToolName
from agentflow_rl.tools.contracts import ToolRequest, ToolStatus


class UnusedPlanner:
    pass


def accepted_result(tool_name: ToolName, result) -> bool:
    success = result.status is ToolStatus.SUCCESS
    if tool_name in {ToolName.GOOGLE_SEARCH, ToolName.WIKIPEDIA_SEARCH}:
        success = success and bool(result.data.get("hits")) and any(
            doc.get("read_status") == "success" and doc.get("passages")
            for doc in result.data.get("documents", [])
        )
    if tool_name is ToolName.PYTHON_CODER:
        success = success and "42" in str(result.data)
    return success


def request_for(tool_name: ToolName, index: int) -> ToolRequest:
    arguments = {
        ToolName.PYTHON_CODER: {"code": "print(6 * 7)"},
        ToolName.GOOGLE_SEARCH: {
            "query": (
                "OpenAI official website"
                if index == 0
                else f"OpenAI API official documentation {index}"
            ),
            "top_k": 2,
        },
        ToolName.WIKIPEDIA_SEARCH: {
            "query": ("Albert Einstein", "Marie Curie", "Alan Turing", "Ada Lovelace")[
                index
            ],
            "top_k": 2,
        },
    }[tool_name]
    timeout = 30 if tool_name is ToolName.PYTHON_CODER else 60
    return ToolRequest(
        request_id=f"preflight-{tool_name.value}-{index}",
        trajectory_id=f"preflight-concurrency-{index}",
        task_id=f"preflight-{index}",
        turn_index=0,
        tool_name=tool_name,
        arguments=arguments,
        timeout_s=timeout,
    )


async def concurrency_probe(registry, tool_name: ToolName, concurrency: int) -> dict:
    requests = tuple(request_for(tool_name, index) for index in range(concurrency))
    started = time.monotonic()
    results = await asyncio.gather(*(registry.dispatch(request) for request in requests))
    payload = {
        "requested_concurrency": concurrency,
        "success_count": sum(accepted_result(tool_name, result) for result in results),
        "wall_time_s": time.monotonic() - started,
        "request_latency_ms": [float(result.latency_ms) for result in results],
        "failure_codes": [
            result.failure_code for result in results if result.status is not ToolStatus.SUCCESS
        ],
    }
    if tool_name is ToolName.PYTHON_CODER:
        payload["sandbox_timing_ms"] = [
            {
                key: float(result.data.get("execution", {}).get(key, 0.0))
                for key in ("admission_wait_ms", "queue_wait_ms", "execution_ms")
            }
            for result in results
        ]
    return payload


def get_json(url: str, timeout_s: float = 20.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ValueError(f"service metadata at {url} is not an object")
    return payload


def post_json(url: str, payload: dict, timeout_s: float = 20.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError(f"service response at {url} is not an object")
    return result


async def probe(config_path: Path, overrides: list[str]) -> dict:
    config = load_config(config_path, overrides)
    bundle = build_runtime_bundle(config, planner=UnusedPlanner())
    requests = (
        ToolRequest(
            request_id="preflight-base", trajectory_id="preflight", task_id="preflight",
            turn_index=0, tool_name=ToolName.BASE_GENERATOR,
            arguments={"query": "Return the integer 42.", "max_tokens": 32}, timeout_s=120,
        ),
        ToolRequest(
            request_id="preflight-python-warmup", trajectory_id="preflight", task_id="preflight",
            turn_index=0, tool_name=ToolName.PYTHON_CODER,
            arguments={"code": "print(6 * 7)"}, timeout_s=30,
        ),
        ToolRequest(
            request_id="preflight-google", trajectory_id="preflight", task_id="preflight",
            turn_index=0, tool_name=ToolName.GOOGLE_SEARCH,
            arguments={"query": "OpenAI official website", "top_k": 2}, timeout_s=60,
        ),
        ToolRequest(
            request_id="preflight-wikipedia", trajectory_id="preflight", task_id="preflight",
            turn_index=0, tool_name=ToolName.WIKIPEDIA_SEARCH,
            arguments={"query": "Albert Einstein", "top_k": 2}, timeout_s=60,
        ),
    )
    results = {}
    for request in requests:
        result = await bundle.loop.tools.dispatch(request)
        success = accepted_result(request.tool_name, result)
        results[request.tool_name.value] = {
            "status": "success" if success else "failure",
            "tool_status": result.status.value,
            "failure_code": result.failure_code,
            "backend_revision": result.backend_revision,
            "latency_ms": result.latency_ms,
            "backend_calls": [call.model_dump(mode="json") for call in result.backend_calls],
        }
    tool_config = config.agentflow.tools
    sandbox_health = await asyncio.to_thread(
        get_json, f"{str(tool_config.sandbox_service_url).rstrip('/')}/health"
    )
    sandbox_expected = {
        "revision": str(tool_config.sandbox_service_revision),
        "max_workers": int(tool_config.sandbox_max_concurrency),
        "max_queue": int(tool_config.sandbox_max_queue),
        "max_reuses": int(tool_config.sandbox_max_reuses),
    }
    if any(sandbox_health.get(key) != value for key, value in sandbox_expected.items()):
        raise ValueError("Python sandbox service settings differ from formal configuration")
    if not sandbox_health.get("ready"):
        raise ValueError("Python sandbox service has fewer ready workers than configured")
    wikipedia_health = await asyncio.to_thread(
        get_json, f"{str(tool_config.wikipedia_service_url).rstrip('/')}/health"
    )
    sandbox_url = str(tool_config.sandbox_service_url).rstrip("/")
    sandbox_health_before_recovery = await asyncio.to_thread(
        get_json, f"{sandbox_url}/health"
    )
    initial_rotations = int(sandbox_health.get("worker_rotations", 0))
    rotations_before = int(sandbox_health_before_recovery.get("worker_rotations", 0))
    timeout_probe = await asyncio.to_thread(
        post_json,
        f"{sandbox_url}/execute",
        {
            "code": "import time; time.sleep(1)",
            "mode": "run",
            "stdin": "",
            "tests": [],
            "timeout_s": 0.1,
        },
    )
    sandbox_health_after = await asyncio.to_thread(
        get_json, f"{sandbox_url}/health"
    )
    recovery_probe = {
        "unexpected_rotations_before_timeout": rotations_before - initial_rotations,
        "timed_out": bool(timeout_probe.get("result", {}).get("timed_out")),
        "worker_rotation_delta": int(sandbox_health_after.get("worker_rotations", 0))
        - rotations_before,
        "ready_after": bool(sandbox_health_after.get("ready")),
    }
    expected = {
        "index_type": str(tool_config.wikipedia_index_type),
        "hnsw_m": int(tool_config.wikipedia_hnsw_m),
        "hnsw_ef_search": int(tool_config.wikipedia_hnsw_ef_search),
        "max_concurrency": int(tool_config.wikipedia_max_concurrency),
        "max_queue": int(tool_config.wikipedia_max_queue),
        "batch_max_size": int(tool_config.wikipedia_batch_max_size),
        "batch_wait_ms": float(tool_config.wikipedia_batch_wait_ms),
    }
    if any(wikipedia_health.get(key) != value for key, value in expected.items()):
        raise ValueError("Wikipedia service HNSW settings differ from formal configuration")
    if wikipedia_health.get("revision") != str(tool_config.wikipedia_revision):
        raise ValueError("Wikipedia service revision differs from formal configuration")
    targets = {
        ToolName.PYTHON_CODER: int(tool_config.sandbox_max_concurrency),
        ToolName.GOOGLE_SEARCH: int(tool_config.serper_max_concurrency),
        ToolName.WIKIPEDIA_SEARCH: int(tool_config.wikipedia_max_concurrency),
    }
    concurrency_checks = {
        tool_name.value: await concurrency_probe(
            bundle.loop.tools, tool_name, concurrency
        )
        for tool_name, concurrency in targets.items()
    }
    wikipedia_health = await asyncio.to_thread(
        get_json, f"{str(tool_config.wikipedia_service_url).rstrip('/')}/health"
    )
    python_result = results[ToolName.PYTHON_CODER.value]
    warmups = {
        "python_sandbox": {
            "status": python_result["status"],
            "method": "persistent_pool_health_and_exec",
            "latency_ms": float(python_result["latency_ms"]),
        }
    }
    return {
        "schema_version": 2,
        "tools": results,
        "warmups": warmups,
        "concurrency_checks": concurrency_checks,
        "sandbox_recovery_probe": recovery_probe,
        "services": {
            "python_sandbox_initial": sandbox_health,
            "python_sandbox_before_recovery": sandbox_health_before_recovery,
            "python_sandbox_after": sandbox_health_after,
            "wikipedia": wikipedia_health,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    payload = asyncio.run(probe(args.config, args.override))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return int(any(row["status"] != "success" for row in payload["tools"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
