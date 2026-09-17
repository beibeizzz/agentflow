ARG PYTHON_BASE_IMAGE=python:3.10-slim@sha256:8fb43f7da946cbd603699e22290fd964266854a49510cfbd840608893b5d8d01
FROM ${PYTHON_BASE_IMAGE}

ARG BIGCODEBENCH_COMMIT
ARG REQUIREMENTS_LOCK_SHA256

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN test -n "$BIGCODEBENCH_COMMIT" \
    && test -n "$REQUIREMENTS_LOCK_SHA256" \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
       build-essential git libgdal-dev libfreetype6-dev libpng-dev \
       pkg-config python3-dev python3-tk r-base zip unzip \
    && rm -rf /var/lib/apt/lists/*

COPY docker/python-runtime/requirements-eval.lock /tmp/requirements-eval.lock
RUN echo "${REQUIREMENTS_LOCK_SHA256}  /tmp/requirements-eval.lock" | sha256sum --check - \
    && python -m pip install --no-cache-dir --requirement /tmp/requirements-eval.lock \
    && python -m pip install --no-cache-dir --no-deps \
       "git+https://github.com/bigcode-project/bigcodebench.git@${BIGCODEBENCH_COMMIT}" \
    && python -m pip check \
    && python -m pip freeze --all | LC_ALL=C sort > /opt/agentflow-runtime.lock \
    && rm /tmp/requirements-eval.lock

RUN useradd --uid 65532 --create-home runner

LABEL org.opencontainers.image.title="AgentFlow shared Python runtime" \
      org.opencontainers.image.source="https://github.com/beibeizzz/agentflow" \
      agentflow.bigcodebench.revision="${BIGCODEBENCH_COMMIT}" \
      agentflow.requirements.sha256="${REQUIREMENTS_LOCK_SHA256}"
