ARG AGENTFLOW_PYTHON_RUNTIME_IMAGE
FROM ${AGENTFLOW_PYTHON_RUNTIME_IMAGE}

COPY docker/bigcodebench-evaluator/runner.py /opt/agentflow/private_evaluator.py
USER 65532:65532
WORKDIR /tmp
ENTRYPOINT ["python", "/opt/agentflow/private_evaluator.py"]
