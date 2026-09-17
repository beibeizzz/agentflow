from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

from agentflow_rl.integrations.artifacts import load_process_transitions


def _score_batch(url: str, transitions: list[dict], timeout_s: float) -> tuple[list[float], dict]:
    endpoint = url.rstrip("/")
    if endpoint.endswith("/score"):
        endpoint = endpoint[: -len("/score")]
    body = json.dumps({"transitions": transitions}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/score_batch",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read())
    scores = [float(value) for value in payload["scores"]]
    if len(scores) != len(transitions):
        raise ValueError("PRM benchmark response length mismatch")
    return scores, payload


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else float(left == right)


def _run(url: str, rows: list[dict], batch_size: int, timeout_s: float) -> tuple[list[float], float, dict]:
    scores: list[float] = []
    identity: dict | None = None
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch_scores, payload = _score_batch(url, rows[offset : offset + batch_size], timeout_s)
        current = {
            "revision": payload.get("revision"),
            "rubric_revision": payload.get("rubric_revision"),
        }
        if identity is not None and current != identity:
            raise ValueError("PRM identity changed during benchmark")
        identity = current
        scores.extend(batch_scores)
    return scores, time.perf_counter() - started, identity or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Transformers and vLLM PRM services")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-mean-error", type=float, default=0.001)
    parser.add_argument("--max-error", type=float, default=0.01)
    parser.add_argument("--min-rank-correlation", type=float, default=0.999)
    args = parser.parse_args()
    if args.limit <= 0 or args.batch_size <= 0:
        raise ValueError("limit and batch size must be positive")

    transitions = load_process_transitions(args.input)[: args.limit]
    if not transitions:
        raise ValueError("PRM benchmark input contains no transitions")
    rows = [item.model_dump(mode="json") for item in transitions]
    reference, reference_s, reference_identity = _run(
        args.reference_url, rows, args.batch_size, args.timeout
    )
    candidate, candidate_s, candidate_identity = _run(
        args.candidate_url, rows, args.batch_size, args.timeout
    )
    if reference_identity != candidate_identity:
        raise ValueError("PRM reference and candidate identities differ")
    errors = [abs(left - right) for left, right in zip(reference, candidate, strict=True)]
    rank_correlation = _correlation(_average_ranks(reference), _average_ranks(candidate))
    report = {
        "count": len(rows),
        "batch_size": args.batch_size,
        "identity": reference_identity,
        "mean_absolute_error": sum(errors) / len(errors),
        "maximum_absolute_error": max(errors),
        "rank_correlation": rank_correlation,
        "reference_elapsed_s": reference_s,
        "candidate_elapsed_s": candidate_s,
        "reference_sequences_per_s": len(rows) / reference_s,
        "candidate_sequences_per_s": len(rows) / candidate_s,
    }
    report["passed"] = (
        report["mean_absolute_error"] <= args.max_mean_error
        and report["maximum_absolute_error"] <= args.max_error
        and rank_correlation >= args.min_rank_correlation
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
