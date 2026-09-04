#!/usr/bin/env bash
set -euo pipefail

echo "== OS =="
cat /etc/os-release
echo "== CPU and memory =="
lscpu | grep -E 'Model name|^CPU\(s\)'
free -h
cpu_count="$(nproc)"
memory_kib="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
if [[ "${cpu_count}" -lt 22 ]]; then
  echo "remote gate requires at least 22 logical CPUs" >&2
  exit 1
fi
if [[ "${memory_kib}" -lt 104857600 ]]; then
  echo "remote gate requires at least 100 GiB of host memory" >&2
  exit 1
fi
echo "== Disks =="
df -h / "${DATA_ROOT:-.}"
data_root="${DATA_ROOT:-.}"
data_available_gib="$(df --output=avail -BG "${data_root}" | tail -n 1 | tr -dc '0-9')"
if [[ "${data_available_gib}" -lt 200 ]]; then
  echo "${data_root} provides ${data_available_gib} GiB free; expand the data volume to at least 200 GiB free" >&2
  exit 1
fi
project_device="$(df --output=source . | tail -n 1 | xargs)"
data_device="$(df --output=source "${data_root}" | tail -n 1 | xargs)"
if [[ "${project_device}" != "${data_device}" ]]; then
  echo "project root is on ${project_device}, while DATA_ROOT is on ${data_device}" >&2
  echo "place project_alpha on DATA_ROOT because model/data/outputs are repository-relative" >&2
  exit 1
fi
echo "== GPUs =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${gpu_count}" -lt 2 ]]; then
  echo "remote gate requires at least two GPUs" >&2
  exit 1
fi
while IFS=, read -r index memory_mib compute_cap; do
  memory_mib="${memory_mib// /}"
  compute_cap="${compute_cap// /}"
  if [[ "${memory_mib}" -lt 80000 ]]; then
    echo "GPU ${index} provides ${memory_mib} MiB; the formal gate requires 80000 MiB" >&2
    exit 1
  fi
  if [[ "${compute_cap}" != "8.0" ]]; then
    echo "GPU ${index} compute capability is ${compute_cap}; expected Ampere sm_80" >&2
    exit 1
  fi
done < <(nvidia-smi --query-gpu=index,memory.total,compute_cap --format=csv,noheader,nounits)
echo "== Toolchain =="
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python -c "import verl, vllm; print('verl', getattr(verl, '__version__', 'unknown')); print('vllm', vllm.__version__)"
python scripts/remote/check_verl_install.py
python -c "import pyserini; from pyserini.search.lucene import LuceneSearcher; print('pyserini', getattr(pyserini, '__version__', 'installed'), 'lucene_ready', LuceneSearcher is not None)"
docker --version
docker info --format 'docker_server={{.ServerVersion}}'
java -version
java_major="$(java -version 2>&1 | awk -F'[\".]' '/version/ {print $2; exit}')"
if [[ "${java_major}" -lt 21 ]]; then
  echo "Pyserini 0.44 requires Java 21 or newer" >&2
  exit 1
fi
bash scripts/remote/validate_all_configs.sh
