#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON:-/home/north/vllm_test/.venv/bin/python}"

"$PYTHON_BIN" try_gsm8k_0522/flowgrpo_light_40g/prepare_split.py \
  --input try_gsm8k_0522/data/gsm8k_test.json \
  --train-output try_gsm8k_0522/data/gsm8k_test_train_1000.json \
  --eval-output try_gsm8k_0522/data/gsm8k_test_eval_rest.json \
  --train-size 1000

