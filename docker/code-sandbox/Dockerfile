FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY runner.py /opt/runner.py
RUN chmod 0555 /opt/runner.py

ENTRYPOINT ["python", "/opt/runner.py"]
