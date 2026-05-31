#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve relative path before cd
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON="$(pwd)/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

cd "$SCRIPT_DIR"
mkdir -p logs/smoke results summary data

"$PYTHON" prepare_gsm8k_json.py --smoke-size 20
THINK_MODE="${THINK_MODE:-off}"
QUERY_ANALYSIS_THINK_MODE="${QUERY_ANALYSIS_THINK_MODE:-on}"
FINAL_OUTPUT_THINK_MODE="${FINAL_OUTPUT_THINK_MODE:-off}"
VERIFIER_THINK_MODE="${VERIFIER_THINK_MODE:-on}"

run_variant() {
  local name="$1"
  local output_types="$2"
  local response_field="$3"
  local max_steps="$4"
  local result_dir="results/${name}"
  local log_dir="logs/${name}"
  local summary_file="summary/${name}_summary.json"

  mkdir -p "$result_dir" "$log_dir/problems"

  echo "=== Smoke variant: $name output_types=$output_types max_steps=$max_steps max_tokens=${MAX_TOKENS:-256} think_mode=$THINK_MODE query_analysis_think_mode=$QUERY_ANALYSIS_THINK_MODE final_output_think_mode=$FINAL_OUTPUT_THINK_MODE verifier_think_mode=$VERIFIER_THINK_MODE ==="
  "$PYTHON" run_gsm8k_agentflow.py \
    --data-file data/gsm8k_smoke_20.json \
    --output-dir "$result_dir" \
    --solver-log-dir "$log_dir/problems" \
    --output-types "$output_types" \
    --max-steps "$max_steps" \
    --max-time "${MAX_TIME:-120}" \
    --max-tokens "${MAX_TOKENS:-256}" \
    --think-mode "$THINK_MODE" \
    --query-analysis-think-mode "$QUERY_ANALYSIS_THINK_MODE" \
    --final-output-think-mode "$FINAL_OUTPUT_THINK_MODE" \
    --verifier-think-mode "$VERIFIER_THINK_MODE" \
    --overwrite \
    2>&1 | tee "$log_dir/run.log"

  "$PYTHON" score_gsm8k.py \
    --data-file data/gsm8k_smoke_20.json \
    --result-dir "$result_dir" \
    --response-field "$response_field" \
    --output-file "$summary_file" \
    2>&1 | tee "$log_dir/score.log"
}

run_variant "smoke_calculator_steps1" "direct" "direct_output" 1
run_variant "smoke_calculator_steps2" "direct" "direct_output" 2
run_variant "smoke_calculator_steps3" "direct" "direct_output" 3
run_variant "smoke_calculator_steps4" "direct" "direct_output" 4
run_variant "smoke_calculator_steps5" "direct" "direct_output" 5
