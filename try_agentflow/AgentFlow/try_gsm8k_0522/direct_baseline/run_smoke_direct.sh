#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/north/vllm_test/.venv/bin/python}"

cd "$BASE_DIR"
mkdir -p direct_baseline/logs direct_baseline/results direct_baseline/summary data

SMOKE_SIZE="${SMOKE_SIZE:-50}"
MODEL="${MODEL:-Qwen3-0.6B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"

"$PYTHON" prepare_gsm8k_json.py --smoke-size "$SMOKE_SIZE"

DATA_FILE="data/gsm8k_smoke_${SMOKE_SIZE}.json"
RESULT_DIR="direct_baseline/results/smoke_direct_vllm"
LOG_DIR="direct_baseline/logs/smoke_direct_vllm"
SUMMARY_FILE="direct_baseline/summary/smoke_direct_vllm_summary.json"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "=== Direct vLLM baseline smoke model=$MODEL max_tokens=$MAX_TOKENS smoke_size=$SMOKE_SIZE ==="
"$PYTHON" direct_baseline/run_direct_vllm.py \
  --data-file "$DATA_FILE" \
  --output-dir "$RESULT_DIR" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --overwrite \
  2>&1 | tee "$LOG_DIR/run.log"

"$PYTHON" score_gsm8k.py \
  --data-file "$DATA_FILE" \
  --result-dir "$RESULT_DIR" \
  --response-field direct_output \
  --output-file "$SUMMARY_FILE" \
  2>&1 | tee "$LOG_DIR/score.log"
