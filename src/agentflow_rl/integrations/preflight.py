from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from agentflow_rl.runtime.contracts import TaskName, ToolName


TRAIN_TASKS = (TaskName.AIME, TaskName.TWOWIKI, TaskName.TACO)
IDENTITY_KEYS = {
    "config_sha256",
    "data_sha256",
    "private_data_sha256",
    "planner_revision",
    "frozen_revision",
    "process_revision",
    "wikipedia_revision",
    "wikipedia_index_type",
    "wikipedia_passage_count",
    "wikipedia_hnsw_m",
    "wikipedia_hnsw_ef_search",
    "wikipedia_index_sha256",
    "wikipedia_corpus_sha256",
    "wikipedia_arrow_fingerprint",
    "wikipedia_encoder_revision",
    "wikipedia_benchmark_sha256",
    "wikipedia_faiss_version",
    "sandbox_runtime_image_id",
    "sandbox_public_image_id",
    "sandbox_private_image_id",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def freeze_preflight_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(payload)
    report.pop("report_sha256", None)
    report["report_sha256"] = canonical_sha256(report)
    return report


def validate_preflight_report(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, str] | None = None,
) -> None:
    report = dict(payload)
    digest = report.pop("report_sha256", None)
    if digest != canonical_sha256(report):
        raise ValueError("preflight report hash mismatch")
    identity = report.get("identity")
    if not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS or any(
        not isinstance(value, str) or not value or "unresolved" in value
        for value in identity.values()
    ):
        raise ValueError("preflight identity is incomplete")
    if expected_identity is not None and identity != dict(expected_identity):
        raise ValueError("preflight identity differs from the formal run")
    if (
        identity["wikipedia_index_type"] != "hnsw64"
        or int(identity["wikipedia_passage_count"]) <= 0
        or identity["wikipedia_hnsw_m"] != "64"
        or identity["wikipedia_hnsw_ef_search"] != "256"
        or len(identity["wikipedia_index_sha256"]) != 64
        or len(identity["wikipedia_corpus_sha256"]) != 64
        or not identity["wikipedia_arrow_fingerprint"]
        or len(identity["wikipedia_benchmark_sha256"]) != 64
    ):
        raise ValueError("preflight Wikipedia HNSW identity is invalid")
    retrieval = report.get("wikipedia_index_quality", {})
    if (
        int(retrieval.get("query_count", 0)) <= 0
        or float(retrieval.get("recall_at_10_mean", 0.0)) < 0.97
        or int(retrieval.get("hnsw_search_concurrency", 0)) != 4
        or float(retrieval.get("hnsw_search_p95_s", float("inf"))) > 1.0
        or int(retrieval.get("hnsw_m", 0)) != 64
        or int(retrieval.get("hnsw_ef_search", 0)) != 256
        or int(retrieval.get("top_k", 0)) != 10
        or int(retrieval.get("passage_count", 0))
        != int(identity["wikipedia_passage_count"])
        or retrieval.get("hnsw_index_sha256") != identity["wikipedia_index_sha256"]
        or retrieval.get("corpus_sha256") != identity["wikipedia_corpus_sha256"]
        or retrieval.get("arrow_fingerprint")
        != identity["wikipedia_arrow_fingerprint"]
        or len(str(retrieval.get("flat_index_sha256", ""))) != 64
    ):
        raise ValueError("preflight Wikipedia HNSW quality gate failed")

    counts = report.get("prompt_counts", {})
    for task in TRAIN_TASKS:
        if int(counts.get(task.value, 0)) < 32:
            raise ValueError(f"preflight requires 32 prompts for {task.value}")
    group_size = int(report.get("rollout_group_size", 0))
    expected_trajectories = sum(int(counts[task.value]) for task in TRAIN_TASKS) * group_size
    if group_size != 5 or int(report.get("trajectory_count", 0)) < expected_trajectories:
        raise ValueError("preflight rollout coverage is incomplete")
    lora = report.get("planner_lora", {})
    if (
        int(lora.get("rank", 0)) != 64
        or int(lora.get("alpha", 0)) != 128
        or lora.get("target_modules") != "all-linear"
    ):
        raise ValueError("preflight Planner LoRA identity is invalid")
    training_batch = report.get("planner_training_batch", {})
    if (
        int(training_batch.get("turn_mini_batch_max", 0)) != 32
        or float(training_batch.get("rollout_gpu_memory_utilization", 0.0)) != 0.50
        or int(training_batch.get("actor_token_budget_per_gpu", 0)) != 40960
        or int(training_batch.get("old_log_prob_token_budget_per_gpu", 0)) != 40960
        or training_batch.get("dynamic_batching") is not True
    ):
        raise ValueError("preflight Planner training batch identity is invalid")
    if int(report.get("dynamic_generation_batches", 0)) < 1:
        raise ValueError("preflight recorded no DAPO generation batch")
    if int(report.get("collection_batches", 0)) < 1:
        raise ValueError("preflight recorded no collection batch")

    tool_checks = report.get("tool_checks", {})
    if any(tool_checks.get(tool.value) != "success" for tool in ToolName):
        raise ValueError("preflight four-tool probe is incomplete")
    python_warmup = report.get("tool_warmups", {}).get("python_sandbox", {})
    if (
        python_warmup.get("status") != "success"
        or python_warmup.get("method") != "persistent_pool_health_and_exec"
        or float(python_warmup.get("latency_ms", 0.0)) <= 0.0
    ):
        raise ValueError("preflight Python sandbox readiness probe is incomplete")
    sandbox_service = report.get("python_sandbox_service", {})
    if (
        sandbox_service.get("revision") != "python-sandbox-service-v1"
        or sandbox_service.get("pool_revision") != "persistent-docker-sandbox-pool-v1"
        or sandbox_service.get("ready") is not True
        or int(sandbox_service.get("workers", 0)) != 6
        or int(sandbox_service.get("max_workers", 0)) != 6
        or int(sandbox_service.get("max_queue", -1)) != 34
        or int(sandbox_service.get("capacity", 0)) != 40
        or int(sandbox_service.get("max_reuses", 0)) != 64
        or int(sandbox_service.get("peak_inflight", 0)) < 6
    ):
        raise ValueError("preflight Python sandbox service gate failed")
    sandbox_recovery = report.get("python_sandbox_recovery", {})
    if (
        int(sandbox_recovery.get("unexpected_rotations_before_timeout", -1)) != 0
        or sandbox_recovery.get("timed_out") is not True
        or int(sandbox_recovery.get("worker_rotation_delta", 0)) < 1
        or sandbox_recovery.get("ready_after") is not True
    ):
        raise ValueError("preflight Python sandbox recovery gate failed")
    concurrent_tools = report.get("tool_concurrency_checks", {})
    concurrency_targets = {
        ToolName.PYTHON_CODER: 6,
        ToolName.GOOGLE_SEARCH: 40,
        ToolName.WIKIPEDIA_SEARCH: 4,
    }
    for tool, target in concurrency_targets.items():
        check = concurrent_tools.get(tool.value, {})
        if (
            int(check.get("requested_concurrency", 0)) != target
            or int(check.get("success_count", 0)) != target
            or len(check.get("request_latency_ms", ())) != target
        ):
            raise ValueError(
                f"preflight {tool.value} configured concurrency gate failed"
            )
    reward_mode = str(report.get("reward_mode", ""))
    if reward_mode not in {"terminal", "prm", "judge"}:
        raise ValueError("preflight reward mode is invalid")
    if reward_mode in {"prm", "judge"} and int(report.get("scored_turns", 0)) <= 0:
        raise ValueError("preflight process scorer produced no scores")
    if report.get("dynamic_sampling_complete") is not True:
        raise ValueError("preflight dynamic sampling did not complete")
    if int(report.get("successful_optimizer_updates", 0)) < 1:
        raise ValueError("preflight completed no optimizer update")
    if report.get("checkpoint_saved") is not True:
        raise ValueError("preflight checkpoint was not saved")
    if report.get("checkpoint_reloaded") is not True:
        raise ValueError("preflight checkpoint was not reloaded")
    if int(report.get("fresh_rollout_count", 0)) < 1:
        raise ValueError("preflight produced no rollout after checkpoint reload")

    total = int(report.get("total_trajectories", report.get("trajectory_count", 0)))
    failures = int(report.get("infrastructure_failures", 0))
    if total <= 0 or failures / total > 0.01:
        raise ValueError("preflight infrastructure failure rate exceeds 1%")
    devices = report.get("gpu_memory", {})
    if len(devices) < 2:
        raise ValueError("preflight GPU memory report requires two devices")
    for device, values in devices.items():
        total_mib = float(values["total_mib"])
        peak_mib = float(values["peak_used_mib"])
        if peak_mib >= 75 * 1024 or total_mib - peak_mib < 5 * 1024:
            raise ValueError(f"GPU memory gate failed for {device}")
    host = report.get("host_resources", {})
    if (
        int(host.get("effective_cpu_cores", 0)) < 28
        or int(host.get("minimum_memory_available_bytes", 0)) < 32 * 1024**3
        or int(host.get("minimum_data_disk_free_bytes", 0)) < 100 * 1024**3
        or int(host.get("swap_growth_bytes", 1)) != 0
    ):
        raise ValueError("preflight host resource gate failed")


def write_preflight_report(payload: Mapping[str, Any], path: str | Path) -> Path:
    report = freeze_preflight_report(payload)
    validate_preflight_report(report)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return target


def read_preflight_report(
    path: str | Path, *, expected_identity: Mapping[str, str] | None = None
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_preflight_report(payload, expected_identity=expected_identity)
    return payload


__all__ = [
    "TRAIN_TASKS",
    "IDENTITY_KEYS",
    "canonical_sha256",
    "freeze_preflight_report",
    "read_preflight_report",
    "sha256_file",
    "validate_preflight_report",
    "write_preflight_report",
]
