#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/inspire/hdd/project/fdu-aidake-cfff/public/zhangjiangbei/agentflow/AgentFlow/.venv/bin/python}"

export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"

DATA_FILE="${DATA_FILE:-data/gsm8k_test_eval_rest.json}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_ENGINE_NAME="${LLM_ENGINE_NAME:-vllm-Qwen3-0.6B}"
OUTPUT_TYPES="${OUTPUT_TYPES:-direct}"
RESPONSE_FIELD="${RESPONSE_FIELD:-direct_output}"

MAX_STEPS="${MAX_STEPS:-4}"
MAX_TIME="${MAX_TIME:-120}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
THINK_MODE="${THINK_MODE:-off}"
QUERY_ANALYSIS_THINK_MODE="${QUERY_ANALYSIS_THINK_MODE:-on}"
FINAL_OUTPUT_THINK_MODE="${FINAL_OUTPUT_THINK_MODE:-off}"
VERIFIER_THINK_MODE="${VERIFIER_THINK_MODE:-on}"

RESULT_DIR="${RESULT_DIR:-try_gsm8k_0522/results/eval_rest_baseline_steps${MAX_STEPS}}"
LOG_DIR="${LOG_DIR:-try_gsm8k_0522/logs/eval_rest_baseline_steps${MAX_STEPS}}"
SUMMARY_FILE="${SUMMARY_FILE:-try_gsm8k_0522/summary/eval_rest_baseline_steps${MAX_STEPS}_summary.json}"

echo "Using PYTHON=${PYTHON_BIN}"
echo "Using DATA_FILE=${DATA_FILE}"
echo "Using BASE_URL=${BASE_URL}"
echo "Using LLM_ENGINE_NAME=${LLM_ENGINE_NAME}"
echo "Using RESULT_DIR=${RESULT_DIR}"
echo "Using SUMMARY_FILE=${SUMMARY_FILE}"
echo "Using MAX_STEPS=${MAX_STEPS}"
echo "Using MAX_TOKENS=${MAX_TOKENS}"
echo "Using THINK_MODE=${THINK_MODE}"
echo "Using QUERY_ANALYSIS_THINK_MODE=${QUERY_ANALYSIS_THINK_MODE}"
echo "Using FINAL_OUTPUT_THINK_MODE=${FINAL_OUTPUT_THINK_MODE}"
echo "Using VERIFIER_THINK_MODE=${VERIFIER_THINK_MODE}"

test -f "$DATA_FILE"
mkdir -p "$RESULT_DIR" "$LOG_DIR/problems" "$(dirname "$SUMMARY_FILE")"

"$PYTHON_BIN" run_gsm8k_agentflow.py \
  --data-file "$DATA_FILE" \
  --output-dir "$RESULT_DIR" \
  --solver-log-dir "$LOG_DIR/problems" \
  --llm-engine-name "$LLM_ENGINE_NAME" \
  --base-url "$BASE_URL" \
  --output-types "$OUTPUT_TYPES" \
  --max-steps "$MAX_STEPS" \
  --max-time "$MAX_TIME" \
  --max-tokens "$MAX_TOKENS" \
  --think-mode "$THINK_MODE" \
  --query-analysis-think-mode "$QUERY_ANALYSIS_THINK_MODE" \
  --final-output-think-mode "$FINAL_OUTPUT_THINK_MODE" \
  --verifier-think-mode "$VERIFIER_THINK_MODE" \
  --overwrite \
  2>&1 | tee "$LOG_DIR/run.log"

"$PYTHON_BIN" score_gsm8k.py \
  --data-file "$DATA_FILE" \
  --result-dir "$RESULT_DIR" \
  --response-field "$RESPONSE_FIELD" \
  --output-file "$SUMMARY_FILE" \
  2>&1 | tee "$LOG_DIR/score.log"
