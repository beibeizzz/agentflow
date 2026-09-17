from __future__ import annotations

import importlib.util
from pathlib import Path


def load_cli_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "eval" / "run_evaluation.py"
    spec = importlib.util.spec_from_file_location("agentflow_run_evaluation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required_args() -> list[str]:
    return [
        "--condition",
        "E0_direct",
        "--input",
        "input.parquet",
        "--private-records",
        "private.jsonl",
        "--shared-manifest",
        "manifest.json",
        "--output",
        "output",
        "--planner-tokenizer",
        "tokenizer",
        "--planner-revision",
        "planner-r1",
        "--wikipedia-revision",
        "wiki-r1",
        "--serper-cache",
        "serper.sqlite3",
        "--sandbox-image",
        "sandbox@sha256:1",
        "--bigcodebench-image",
        "bigcode@sha256:1",
        "--bigcodebench-revision",
        "bigcode-r1",
    ]


def test_evaluation_cli_sandbox_defaults_reach_runtime_config() -> None:
    module = load_cli_module()
    args = module.parse_args(required_args())
    tools = module.runtime_config(args)["agentflow"]["tools"]
    assert tools["sandbox_service_url"] == "http://127.0.0.1:8005"
    assert tools["sandbox_service_revision"] == "python-sandbox-service-v1"
    assert tools["sandbox_service_timeout_s"] == 30.0


def test_evaluation_cli_sandbox_overrides_reach_runtime_config() -> None:
    module = load_cli_module()
    args = module.parse_args(
        required_args()
        + [
            "--sandbox-service-url",
            "http://sandbox:9000",
            "--sandbox-service-revision",
            "sandbox-r9",
            "--sandbox-service-timeout-s",
            "19.5",
        ]
    )
    tools = module.runtime_config(args)["agentflow"]["tools"]
    assert tools["sandbox_service_url"] == "http://sandbox:9000"
    assert tools["sandbox_service_revision"] == "sandbox-r9"
    assert tools["sandbox_service_timeout_s"] == 19.5
