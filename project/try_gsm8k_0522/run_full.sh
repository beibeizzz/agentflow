#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/north/vllm_test/.venv/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p logs/full/problems results/full summary data

"$PYTHON" prepare_gsm8k_json.py --smoke-size 20

TOTAL="${TOTAL:-1319}"
WORKERS="${WORKERS:-4}"
THINK_MODE="${THINK_MODE:-default}"
QUERY_ANALYSIS_THINK_MODE="${QUERY_ANALYSIS_THINK_MODE:-$THINK_MODE}"
FINAL_OUTPUT_THINK_MODE="${FINAL_OUTPUT_THINK_MODE:-$THINK_MODE}"
VERIFIER_THINK_MODE="${VERIFIER_THINK_MODE:-$THINK_MODE}"
CHUNK_SIZE=$(( (TOTAL + WORKERS - 1) / WORKERS ))

echo "Running full GSM8K with TOTAL=$TOTAL WORKERS=$WORKERS CHUNK_SIZE=$CHUNK_SIZE THINK_MODE=$THINK_MODE QUERY_ANALYSIS_THINK_MODE=$QUERY_ANALYSIS_THINK_MODE FINAL_OUTPUT_THINK_MODE=$FINAL_OUTPUT_THINK_MODE VERIFIER_THINK_MODE=$VERIFIER_THINK_MODE" | tee logs/full/run.log

pids=()
for worker in $(seq 0 $((WORKERS - 1))); do
  start=$((worker * CHUNK_SIZE))
  if [ "$start" -ge "$TOTAL" ]; then
    continue
  fi
  limit="$CHUNK_SIZE"
  worker_log="logs/full/worker_${worker}.log"
  echo "Starting worker=$worker start=$start limit=$limit log=$worker_log" | tee -a logs/full/run.log
  "$PYTHON" run_gsm8k_agentflow.py \
    --data-file try_gsm8k_0522/data/gsm8k_test_eval_rest.json \
    --output-dir results/full \
    --solver-log-dir "logs/full/problems/worker_${worker}" \
    --output-types direct \
    --start "$start" \
    --limit "$limit" \
    --max-steps "${MAX_STEPS:-2}" \
    --max-time "${MAX_TIME:-120}" \
    --max-tokens "${MAX_TOKENS:-2048}" \
    --think-mode "$THINK_MODE" \
    --query-analysis-think-mode "$QUERY_ANALYSIS_THINK_MODE" \
    --final-output-think-mode "$FINAL_OUTPUT_THINK_MODE" \
    --verifier-think-mode "$VERIFIER_THINK_MODE" \
    > "$worker_log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All full-run workers completed" | tee -a logs/full/run.log

"$PYTHON" score_gsm8k.py \
  --data-file data/gsm8k_test.json \
  --result-dir results/full \
  --output-file summary/full_summary.json \
  2>&1 | tee logs/full/score.log
