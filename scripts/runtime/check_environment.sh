#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:?Set PYTHON to the runtime Python executable}"
project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "${project_root}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
docker info >/dev/null
df -h "${DATA_ROOT:-${project_root}/data}"
"${PYTHON}" - <<'PY'
import importlib
import json

required = [
    "torch",
    "transformers",
    "accelerate",
    "vllm",
    "verl",
    "ray",
    "pyarrow",
    "pydantic",
    "flash_attn",
]
versions = {}
for name in required:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")
print(json.dumps(versions, indent=2, sort_keys=True))
PY
"${PYTHON}" scripts/runtime/check_verl_api.py
