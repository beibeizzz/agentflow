#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"

MODEL_PATH="${MODEL_PATH:-/home/north/vllm_test/models/Qwen/Qwen3-0.6B}"
FROZEN_BASE_URL="${FROZEN_BASE_URL:-http://127.0.0.1:8000/v1}"
FROZEN_MODEL="${FROZEN_MODEL:-Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-try_gsm8k_0522/data/gsm8k_test_train_1000.json}"

PYTHON_BIN="${PYTHON:-/home/north/vllm_test/.venv/bin/python}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "FROZEN_BASE_URL=${FROZEN_BASE_URL}"
echo "FROZEN_MODEL=${FROZEN_MODEL}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "PYTHON=${PYTHON_BIN}"

test -d "$MODEL_PATH"
if [[ ! -f "$TRAIN_FILE" ]]; then
  bash try_gsm8k_0522/flowgrpo_light_40g/prepare_split_1000.sh
fi
test -f "$TRAIN_FILE"
test -f try_gsm8k_0522/data/gsm8k_test_eval_rest.json
test -x "$PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import importlib
for name in ["torch", "transformers", "peft", "yaml"]:
    mod = importlib.import_module(name)
    print(f"{name}: ok {getattr(mod, '__version__', '')}")
PY

curl --noproxy "*" -fsS "${FROZEN_BASE_URL}/models"
echo
echo "40G environment check passed."
