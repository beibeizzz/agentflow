#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON:-python}"
CONFIG_FILE="${CONFIG_FILE:-try_ticket_agent/baseline/config_agentflow_baseline.yaml}"
MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
LLM_ENGINE_NAME="${LLM_ENGINE_NAME:-vllm-Qwen3-0.6B}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
DATA_FILE="${DATA_FILE:-try_ticket_agent/data/generated/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-try_ticket_agent/baseline/outputs/test}"
MAX_STEPS="${MAX_STEPS:-3}"
MAX_TIME="${MAX_TIME:-120}"
MAX_TOKENS="${MAX_TOKENS:-512}"

[[ -f "$DATA_FILE" ]] || { echo "Data file does not exist: $DATA_FILE" >&2; exit 1; }
"$PYTHON_BIN" try_ticket_agent/run_ticket_agentflow.py \
  --config "$CONFIG_FILE" --data-file "$DATA_FILE" --output-dir "$OUTPUT_DIR" \
  --llm-engine-name "$LLM_ENGINE_NAME" --base-url "$BASE_URL" \
  --max-steps "$MAX_STEPS" --max-time "$MAX_TIME" --max-tokens "$MAX_TOKENS" \
  --temperature 0.0 --think-mode off --query-analysis-think-mode on --overwrite
