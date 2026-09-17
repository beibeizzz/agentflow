from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from agentflow_rl.integrations.preflight import sha256_file, write_preflight_report


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_summary(root: Path) -> tuple[int, int]:
    count = failures = 0
    for path in root.glob("*.json"):
        row = load(path)
        count += 1
        codes = {str(value).upper() for value in row.get("verification", {}).get("failure_codes", ())}
        reason = str(row.get("terminal_reason", "")).lower()
        if "infrastructure" in reason or "timeout" in reason or any(
            "TIMEOUT" in code or "BACKEND" in code or "INFRASTRUCTURE" in code
            for code in codes
        ):
            failures += 1
    return count, failures


def latest_json(root: Path, pattern: str) -> dict:
    paths = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime_ns)
    if not paths:
        raise ValueError(f"missing preflight evidence: {root / pattern}")
    return load(paths[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-mode", choices=("terminal", "prm", "judge"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--tool-probe", type=Path, required=True)
    parser.add_argument("--wikipedia-benchmark", type=Path, required=True)
    parser.add_argument("--gpu-report", type=Path, action="append", required=True)
    parser.add_argument("--host-report", type=Path, action="append", required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--reload-trajectory-dir", type=Path, required=True)
    parser.add_argument("--planner-revision", required=True)
    parser.add_argument("--frozen-revision", required=True)
    parser.add_argument("--process-revision", required=True)
    parser.add_argument("--wikipedia-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load(args.data_manifest)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    images = load(args.image_manifest)
    tool_probe = load(args.tool_probe)
    tools = tool_probe["tools"]
    wikipedia = tool_probe["services"]["wikipedia"]
    python_sandbox = tool_probe["services"]["python_sandbox_after"]
    wikipedia_benchmark = load(args.wikipedia_benchmark)
    if (
        wikipedia_benchmark.get("hnsw_index_sha256") != wikipedia.get("index_sha256")
        or wikipedia_benchmark.get("corpus_sha256") != wikipedia.get("corpus_sha256")
        or wikipedia_benchmark.get("arrow_fingerprint")
        != wikipedia.get("arrow_fingerprint")
    ):
        raise ValueError("Wikipedia benchmark artifacts differ from the running service")
    states = sorted(
        args.train_dir.glob("global_step_*/agentflow_state.json"),
        key=lambda path: int(path.parent.name.rsplit("_", 1)[1]),
    )
    if not states:
        raise ValueError("preflight saved no AgentFlow checkpoint state")
    state = load(states[0])
    metrics = load(
        args.train_dir
        / "agentflow_metrics"
        / f"step_{int(state['collection_step'])}.json"
    )
    reload_marker = load(args.train_dir / "agentflow_reload.json")
    trajectories, failures = trajectory_summary(args.trajectory_dir)
    reload_trajectories, reload_failures = trajectory_summary(args.reload_trajectory_dir)

    gpu_memory = {}
    monitored_hosts = []
    for path in args.gpu_report:
        gpu_report = load(path)
        for device, values in gpu_report["devices"].items():
            current = gpu_memory.setdefault(
                device,
                {"total_mib": float(values["total_mib"]), "peak_used_mib": 0.0},
            )
            current["peak_used_mib"] = max(
                current["peak_used_mib"], float(values["peak_used_mib"])
            )
        monitored_hosts.append(gpu_report["host"])
    host_samples = [load(path) for path in args.host_report]
    initial_swap = int(host_samples[0]["swap_used_bytes"])
    host_resources = {
        "effective_cpu_cores": min(
            [int(sample["effective_cpu_cores"]) for sample in host_samples]
            + [int(sample["effective_cpu_cores"]) for sample in monitored_hosts]
        ),
        "minimum_memory_available_bytes": min(
            [
                int(sample["effective_memory_available_bytes"])
                for sample in host_samples
            ]
            + [
                int(sample["minimum_memory_available_bytes"])
                for sample in monitored_hosts
            ]
        ),
        "minimum_data_disk_free_bytes": min(
            [int(sample["data_disk_free_bytes"]) for sample in host_samples]
            + [
                int(sample["minimum_data_disk_free_bytes"])
                for sample in monitored_hosts
            ]
        ),
        "swap_growth_bytes": max(
            [
                max(
                    0,
                    max(int(sample["swap_used_bytes"]) for sample in host_samples)
                    - initial_swap,
                )
            ]
            + [int(sample["swap_growth_bytes"]) for sample in monitored_hosts]
        ),
    }

    process_available = int(metrics.get("agentflow/process_available_turn_count", 0))
    identity = {
        "config_sha256": sha256_file(args.config),
        "data_sha256": str(data["source_sha256"]),
        "private_data_sha256": sha256_file(args.private_data),
        "planner_revision": args.planner_revision,
        "frozen_revision": args.frozen_revision,
        "process_revision": args.process_revision,
        "wikipedia_revision": args.wikipedia_revision,
        "wikipedia_index_type": str(wikipedia["index_type"]),
        "wikipedia_passage_count": str(wikipedia["passage_count"]),
        "wikipedia_hnsw_m": str(wikipedia["hnsw_m"]),
        "wikipedia_hnsw_ef_search": str(wikipedia["hnsw_ef_search"]),
        "wikipedia_index_sha256": str(wikipedia["index_sha256"]),
        "wikipedia_corpus_sha256": str(wikipedia["corpus_sha256"]),
        "wikipedia_arrow_fingerprint": str(wikipedia["arrow_fingerprint"]),
        "wikipedia_encoder_revision": str(wikipedia["encoder_revision"]),
        "wikipedia_benchmark_sha256": sha256_file(args.wikipedia_benchmark),
        "wikipedia_faiss_version": str(wikipedia["faiss_version"]),
        "sandbox_runtime_image_id": str(images["runtime_image_id"]),
        "sandbox_public_image_id": str(images["public_image"]["id"]),
        "sandbox_private_image_id": str(images["private_image"]["id"]),
    }
    payload = {
        "schema_version": 1,
        "reward_mode": args.reward_mode,
        "identity": identity,
        "prompt_counts": data["prompt_counts"],
        "rollout_group_size": int(config["actor_rollout_ref"]["rollout"]["n"]),
        "planner_lora": {
            "rank": int(config["actor_rollout_ref"]["model"]["lora_rank"]),
            "alpha": int(config["actor_rollout_ref"]["model"]["lora_alpha"]),
            "target_modules": config["actor_rollout_ref"]["model"]["target_modules"],
        },
        "planner_training_batch": {
            "turn_mini_batch_max": int(config["agentflow"]["turn_mini_batch_size"]),
            "rollout_gpu_memory_utilization": float(
                config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"]
            ),
            "actor_token_budget_per_gpu": int(
                config["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"]
            ),
            "old_log_prob_token_budget_per_gpu": int(
                config["actor_rollout_ref"]["rollout"]["log_prob_max_token_len_per_gpu"]
            ),
            "dynamic_batching": bool(
                config["actor_rollout_ref"]["actor"]["use_dynamic_bsz"]
                and config["actor_rollout_ref"]["rollout"]["log_prob_use_dynamic_bsz"]
            ),
        },
        "dynamic_generation_batches": int(
            metrics.get("agentflow/dynamic_generation_batches", 0)
        ),
        "collection_batches": int(state.get("collection_batches", state["collection_step"])),
        "trajectory_count": trajectories,
        "total_trajectories": trajectories + reload_trajectories,
        "infrastructure_failures": failures + reload_failures,
        "tool_checks": {
            name: str(value["status"]) for name, value in tools.items()
        },
        "tool_warmups": tool_probe["warmups"],
        "tool_concurrency_checks": tool_probe["concurrency_checks"],
        "python_sandbox_service": python_sandbox,
        "python_sandbox_recovery": tool_probe["sandbox_recovery_probe"],
        "wikipedia_index_quality": {
            "query_count": int(wikipedia_benchmark["query_count"]),
            "passage_count": int(wikipedia_benchmark["passage_count"]),
            "recall_at_10_mean": float(wikipedia_benchmark["recall_at_k_mean"]),
            "hnsw_search_concurrency": int(
                wikipedia_benchmark["hnsw_search_concurrency"]
            ),
            "hnsw_search_p95_s": float(wikipedia_benchmark["hnsw_latency_s_p95"]),
            "hnsw_m": int(wikipedia_benchmark["hnsw_m"]),
            "hnsw_ef_search": int(wikipedia_benchmark["hnsw_ef_search"]),
            "top_k": int(wikipedia_benchmark["top_k"]),
            "flat_index_sha256": str(wikipedia_benchmark["flat_index_sha256"]),
            "hnsw_index_sha256": str(wikipedia_benchmark["hnsw_index_sha256"]),
            "corpus_sha256": str(wikipedia_benchmark["corpus_sha256"]),
            "arrow_fingerprint": str(wikipedia_benchmark["arrow_fingerprint"]),
        },
        "scored_turns": process_available,
        "dynamic_sampling_complete": bool(
            metrics.get("agentflow/dynamic_sampling_complete", 0.0) == 1.0
        ),
        "successful_optimizer_updates": int(state["successful_updates"]),
        "checkpoint_saved": bool(states),
        "checkpoint_reloaded": (
            int(reload_marker["loaded_collection_step"])
            == int(state["collection_step"])
            and int(reload_marker["loaded_successful_updates"])
            == int(state["successful_updates"])
        ),
        "fresh_rollout_count": reload_trajectories,
        "gpu_memory": gpu_memory,
        "host_resources": host_resources,
        "evidence_sha256": {
            "data_manifest": sha256_file(args.data_manifest),
            "image_manifest": sha256_file(args.image_manifest),
            "tool_probe": sha256_file(args.tool_probe),
            "wikipedia_benchmark": sha256_file(args.wikipedia_benchmark),
            **{
                f"gpu_report_{index}": sha256_file(path)
                for index, path in enumerate(args.gpu_report)
            },
            **{
                f"host_report_{index}": sha256_file(path)
                for index, path in enumerate(args.host_report)
            },
        },
    }
    write_preflight_report(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
