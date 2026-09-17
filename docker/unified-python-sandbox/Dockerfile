ARG AGENTFLOW_PYTHON_RUNTIME_IMAGE
FROM ${AGENTFLOW_PYTHON_RUNTIME_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY docker/unified-python-sandbox/runner.py /opt/agentflow/public_runner.py
COPY src/agentflow_rl/backends/bounded_process.py /opt/agentflow/bounded_process.py
RUN chmod 0555 /opt/agentflow/public_runner.py

USER 65532:65532
ENTRYPOINT ["python", "/opt/agentflow/public_runner.py"]
