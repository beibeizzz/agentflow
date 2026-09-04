#!/usr/bin/env bash
set -euo pipefail

VENV_ROOT="${VENV_ROOT:-/home/north/agentflow_local}"
BASE_VENV="${BASE_VENV:-/home/north/vllm_test/.venv}"
VERL_SOURCE="${VERL_SOURCE:-/mnt/c/tmp/verl-v0.8.0}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/c/Users/north/Desktop/agentflow/project_alpha}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TRANSFER_QUEUE_SOURCE="${TRANSFER_QUEUE_SOURCE:-/mnt/c/all_software/anaconda3/envs/all-in-rag/Lib/site-packages}"

test -x "${BASE_VENV}/bin/python"
test -d "${VERL_SOURCE}"
test -d "${PROJECT_ROOT}"

if [[ ! -x "${VENV_ROOT}/.venv/bin/python" ]]; then
  uv venv --python "${PYTHON_VERSION}" "${VENV_ROOT}/.venv"
fi
python_path="${VENV_ROOT}/.venv/bin/python"
site_packages="$(${python_path} -c 'import site; print(site.getsitepackages()[0])')"
base_site="$(${BASE_VENV}/bin/python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${base_site}" > "${site_packages}/vllm_test_base.pth"
printf '%s\n' "${VERL_SOURCE}" > "${site_packages}/verl_source.pth"
printf '%s\n' "${PROJECT_ROOT}/src" > "${site_packages}/project_alpha.pth"
cp -a "${TRANSFER_QUEUE_SOURCE}/transfer_queue" "${site_packages}/"
cp -a "${TRANSFER_QUEUE_SOURCE}/transferqueue-0.1.6.dist-info" "${site_packages}/"

"${python_path}" -c \
  'import torch, verl, vllm, transfer_queue; from verl.experimental.agent_loop.agent_loop import AgentLoopBase; import verl.trainer.main_ppo_sync; print(torch.__version__, vllm.__version__, verl.__file__)'
