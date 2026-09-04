#!/usr/bin/env bash
set -euo pipefail
docker build -t "${SANDBOX_IMAGE:-agentflow-python-sandbox:3.11}" docker/code-sandbox
