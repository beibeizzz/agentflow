#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/gsm8k/baseline.yaml
  configs/gsm8k/local_single_gpu_smoke.yaml
  configs/gsm8k/smoke.yaml
  configs/gsm8k/train.yaml
  configs/gsm8k/eval.yaml
  configs/ticket/baseline.yaml
  configs/ticket/smoke.yaml
  configs/ticket/train.yaml
  configs/ticket/eval.yaml
  configs/deepresearch/baseline.yaml
  configs/deepresearch/preflight.yaml
  configs/deepresearch/train.yaml
  configs/deepresearch/eval.yaml
  configs/coding/baseline.yaml
  configs/coding/preflight.yaml
  configs/coding/train.yaml
  configs/coding/eval.yaml
)

log_file="$(mktemp)"
trap 'rm -f "${log_file}"' EXIT

for config in "${configs[@]}"; do
  echo "validating ${config}"
  if ! python -m agentflow_rl.verl.main --config "${config}" --dry-run >"${log_file}" 2>&1; then
    cat "${log_file}" >&2
    exit 1
  fi
done

echo "validated ${#configs[@]} veRL configs"
