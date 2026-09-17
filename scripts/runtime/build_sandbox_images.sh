#!/usr/bin/env bash
set -euo pipefail

: "${BIGCODEBENCH_COMMIT:?Set BIGCODEBENCH_COMMIT to an immutable git commit}"
project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${project_root}"

runtime_tag="${RUNTIME_TAG:-agentflow-python-runtime:${BIGCODEBENCH_COMMIT:0:12}}"
public_tag="${SANDBOX_IMAGE:-agentflow-python-sandbox:${BIGCODEBENCH_COMMIT:0:12}}"
private_tag="${BIGCODEBENCH_IMAGE:-agentflow-bigcodebench-evaluator:${BIGCODEBENCH_COMMIT:0:12}}"
lock_path="docker/python-runtime/requirements-eval.lock"
lock_sha="$(sha256sum "${lock_path}" | awk '{print $1}')"

docker build \
  --file docker/python-runtime/Dockerfile \
  --build-arg BIGCODEBENCH_COMMIT="${BIGCODEBENCH_COMMIT}" \
  --build-arg REQUIREMENTS_LOCK_SHA256="${lock_sha}" \
  --tag "${runtime_tag}" .
runtime_id="$(docker image inspect --format '{{.Id}}' "${runtime_tag}")"

docker build \
  --file docker/unified-python-sandbox/Dockerfile \
  --build-arg AGENTFLOW_PYTHON_RUNTIME_IMAGE="${runtime_id}" \
  --tag "${public_tag}" .
docker build \
  --file docker/bigcodebench-evaluator/Dockerfile \
  --build-arg AGENTFLOW_PYTHON_RUNTIME_IMAGE="${runtime_id}" \
  --tag "${private_tag}" .

mkdir -p outputs/environment
python - "${runtime_id}" "${public_tag}" "${private_tag}" "${BIGCODEBENCH_COMMIT}" "${lock_sha}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

runtime_id, public_tag, private_tag, revision, lock_sha = sys.argv[1:]

def image_id(tag: str) -> str:
    return subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag], text=True
    ).strip()

payload = {
    "schema_version": 1,
    "runtime_image_id": runtime_id,
    "public_image": {"reference": public_tag, "id": image_id(public_tag)},
    "private_image": {"reference": private_tag, "id": image_id(private_tag)},
    "bigcodebench_revision": revision,
    "requirements_lock_sha256": lock_sha,
}
Path("outputs/environment/sandbox_images.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "wrote outputs/environment/sandbox_images.json"
