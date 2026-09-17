from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

import yaml

from agentflow_rl.integrations.preflight import read_preflight_report, sha256_file


def get_json(url: str, timeout_s: float = 20.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ValueError(f"service metadata at {url} is not an object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=("terminal", "prm", "judge"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--private-data", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--wikipedia-benchmark", type=Path, required=True)
    parser.add_argument("--sandbox-image", required=True)
    parser.add_argument("--bigcodebench-image", required=True)
    parser.add_argument("--planner-revision", required=True)
    parser.add_argument("--frozen-revision", required=True)
    parser.add_argument("--process-revision", required=True)
    parser.add_argument("--wikipedia-revision", required=True)
    args = parser.parse_args()
    images = json.loads(args.image_manifest.read_text(encoding="utf-8"))
    actual_public = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.sandbox_image],
        text=True,
    ).strip()
    actual_private = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.bigcodebench_image],
        text=True,
    ).strip()
    if actual_public != images["public_image"]["id"]:
        raise ValueError("public sandbox image ID differs from its manifest")
    if actual_private != images["private_image"]["id"]:
        raise ValueError("private evaluator image ID differs from its manifest")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    wikipedia_config = config["agentflow"]["tools"]
    wikipedia = get_json(
        f"{str(wikipedia_config['wikipedia_service_url']).rstrip('/')}/health"
    )
    if wikipedia.get("revision") != args.wikipedia_revision:
        raise ValueError("Wikipedia service revision differs from the formal run")
    expected_retrieval = {
        "index_type": str(wikipedia_config["wikipedia_index_type"]),
        "hnsw_m": int(wikipedia_config["wikipedia_hnsw_m"]),
        "hnsw_ef_search": int(wikipedia_config["wikipedia_hnsw_ef_search"]),
    }
    if any(wikipedia.get(key) != value for key, value in expected_retrieval.items()):
        raise ValueError("Wikipedia service HNSW settings differ from the formal run")
    identity = {
        "config_sha256": sha256_file(args.config),
        "data_sha256": sha256_file(args.train_data),
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
    report = read_preflight_report(args.report, expected_identity=identity)
    if report["reward_mode"] != args.reward_mode:
        raise ValueError("preflight reward mode differs from the formal run")
    print(json.dumps({"preflight": "accepted", "report": str(args.report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
