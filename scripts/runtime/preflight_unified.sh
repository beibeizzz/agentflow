#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:?Set PYTHON to the runtime Python executable}"
: "${SANDBOX_IMAGE:?Set SANDBOX_IMAGE to the built public image}"
: "${BIGCODEBENCH_IMAGE:?Set BIGCODEBENCH_IMAGE to the built private image}"
project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${project_root}"

required_files=(
  "${PLANNER_MODEL_PATH}/config.json"
  "${FROZEN_MODEL_PATH}/config.json"
  "data/verl/train_mixed.parquet"
  "data/verl/development_mixed.parquet"
  "data/private/all.jsonl"
)
for path in "${required_files[@]}"; do
  test -f "${path}" || { echo "missing required file: ${path}" >&2; exit 1; }
done

curl --fail --silent http://127.0.0.1:8001/health >/dev/null
curl --fail --silent http://127.0.0.1:8002/health >/dev/null
curl --fail --silent "${PYTHON_SANDBOX_URL:-http://127.0.0.1:8005}/health" >/dev/null
if [[ "${REWARD_MODE:-prm}" == "prm" ]]; then
  if [[ "${PRM_BACKEND:-vllm}" == "vllm" ]]; then
    curl --fail --silent "${PRM_VLLM_HEALTH_URL:-http://127.0.0.1:8004/health}" >/dev/null
  fi
  curl --fail --silent http://127.0.0.1:8003/health >/dev/null
fi
docker image inspect "${SANDBOX_IMAGE}" >/dev/null
docker image inspect "${BIGCODEBENCH_IMAGE}" >/dev/null

overrides=()
case "${REWARD_MODE:-prm}" in
  terminal)
    overrides+=(--override agentflow.lambda_process=0.0)
    overrides+=(--override agentflow.process_reward.mode=none)
    overrides+=(--override agentflow.process_reward.revision=terminal-only)
    ;;
  prm) ;;
  judge)
    : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY for online Judge mode}"
    : "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL for online Judge mode}"
    : "${JUDGE_MODEL:?Set JUDGE_MODEL for online Judge mode}"
    : "${JUDGE_REVISION:?Set JUDGE_REVISION for online Judge mode}"
    overrides+=(--override agentflow.process_reward.mode=online_judge)
    overrides+=(--override agentflow.process_reward.base_url="${JUDGE_BASE_URL}")
    overrides+=(--override agentflow.process_reward.model="${JUDGE_MODEL}")
    overrides+=(--override agentflow.process_reward.revision="${JUDGE_REVISION}")
    overrides+=(--override agentflow.process_reward.cache_path="${JUDGE_CACHE_PATH:-outputs/judge/cache.jsonl}")
    ;;
  *) echo "unsupported REWARD_MODE: ${REWARD_MODE}" >&2; exit 1 ;;
esac

"${PYTHON}" -m agentflow_rl.integrations.verl_main \
  --config configs/train/unified_terminal_prm.yaml \
  --dry-run \
  --override "agentflow.tools.sandbox_image=${SANDBOX_IMAGE}" \
  --override "agentflow.tools.bigcodebench_image=${BIGCODEBENCH_IMAGE}" \
  --override "agentflow.tools.bigcodebench_revision=${BIGCODEBENCH_REVISION:?Set BIGCODEBENCH_REVISION}" \
  "${overrides[@]}"
