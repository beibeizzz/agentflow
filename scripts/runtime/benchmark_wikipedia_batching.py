from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import urllib.request


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(q * len(ordered) + 0.999999) - 1))
    return ordered[position]


def get_json(url: str, timeout_s: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ValueError(f"response from {url} is not an object")
    return payload


def post_json(url: str, payload: dict, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError(f"response from {url} is not an object")
    return result


def load_queries(path: Path, top_k: int) -> tuple[dict, ...]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, str):
            query = value
            row_top_k = top_k
        elif isinstance(value, dict):
            query = str(value["query"])
            row_top_k = int(value.get("top_k", top_k))
        else:
            raise ValueError("each query row must be a JSON string or object")
        queries.append({"query": query, "top_k": row_top_k})
    if not queries:
        raise ValueError("query set is empty")
    return tuple(queries)


def normalized_hits(response: dict) -> tuple[tuple, ...]:
    return tuple(
        (
            hit.get("doc_id"),
            hit.get("passage_id"),
            hit.get("rank"),
            hit.get("fused_score"),
            hit.get("bm25_score"),
            hit.get("dense_score"),
        )
        for hit in response.get("hits", ())
    )


def execute(
    service_url: str,
    queries: tuple[dict, ...],
    *,
    concurrency: int,
    timeout_s: float,
) -> tuple[list[dict], list[float], float]:
    endpoint = f"{service_url.rstrip('/')}/search"

    def one(payload: dict) -> tuple[dict, float]:
        started = time.perf_counter()
        result = post_json(endpoint, payload, timeout_s)
        return result, time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(one, queries))
    wall_time = time.perf_counter() - started
    return [row[0] for row in rows], [row[1] for row in rows], wall_time


async def benchmark(args) -> dict:
    service_url = args.service_url.rstrip("/")
    queries = load_queries(args.queries, args.top_k)[: args.query_count]
    health_before = await asyncio.to_thread(
        get_json, f"{service_url}/health", args.timeout_s
    )
    if int(health_before.get("batch_max_size", 0)) != args.expected_batch_max_size:
        raise ValueError("service batch_max_size differs from the benchmark expectation")

    sequential, sequential_latencies, sequential_wall = await asyncio.to_thread(
        execute,
        service_url,
        queries,
        concurrency=1,
        timeout_s=args.timeout_s,
    )
    health_pre_burst = await asyncio.to_thread(
        get_json, f"{service_url}/health", args.timeout_s
    )
    batched, batched_latencies, batched_wall = await asyncio.to_thread(
        execute,
        service_url,
        queries,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
    )
    health_after = await asyncio.to_thread(
        get_json, f"{service_url}/health", args.timeout_s
    )

    before_metrics = health_pre_burst.get("batch_metrics", {})
    after_metrics = health_after.get("batch_metrics", {})
    burst_queries = int(after_metrics.get("query_count", 0)) - int(
        before_metrics.get("query_count", 0)
    )
    burst_batches = int(after_metrics.get("batch_count", 0)) - int(
        before_metrics.get("batch_count", 0)
    )
    matches = [
        normalized_hits(left) == normalized_hits(right)
        for left, right in zip(sequential, batched, strict=True)
    ]
    return {
        "schema_version": 1,
        "service_url": service_url,
        "revision": health_after.get("revision"),
        "query_count": len(queries),
        "concurrency": args.concurrency,
        "configured_batch_max_size": health_after.get("batch_max_size"),
        "configured_batch_wait_ms": health_after.get("batch_wait_ms"),
        "response_match_rate": statistics.fmean(matches),
        "sequential": {
            "wall_time_s": sequential_wall,
            "throughput_qps": len(queries) / sequential_wall,
            "latency_s_p50": percentile(sequential_latencies, 0.50),
            "latency_s_p95": percentile(sequential_latencies, 0.95),
        },
        "batched": {
            "wall_time_s": batched_wall,
            "throughput_qps": len(queries) / batched_wall,
            "latency_s_p50": percentile(batched_latencies, 0.50),
            "latency_s_p95": percentile(batched_latencies, 0.95),
            "query_delta": burst_queries,
            "batch_delta": burst_batches,
            "mean_realized_batch_size": (
                burst_queries / burst_batches if burst_batches else 0.0
            ),
            "max_observed_batch_size": after_metrics.get(
                "max_observed_batch_size", 0
            ),
            "peak_pending": after_metrics.get("peak_pending", 0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sequential and concurrent requests against one Wikipedia "
            "service and verify dynamic-batch result parity."
        )
    )
    parser.add_argument("--service-url", default="http://127.0.0.1:8002")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--expected-batch-max-size", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.query_count <= 0
        or args.top_k <= 0
        or args.concurrency <= 0
        or args.expected_batch_max_size <= 0
        or args.timeout_s <= 0
    ):
        raise ValueError("numeric benchmark settings must be positive")
    result = asyncio.run(benchmark(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if result["response_match_rate"] != 1.0:
        return 1
    if result["batched"]["query_delta"] != result["query_count"]:
        return 1
    if result["batched"]["mean_realized_batch_size"] <= 1.0:
        return 1
    if result["batched"]["max_observed_batch_size"] > args.expected_batch_max_size:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
