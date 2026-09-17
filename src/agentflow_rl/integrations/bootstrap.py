from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentflow_rl.backends.sandbox_http import HttpSandboxBackend
from agentflow_rl.backends.bigcodebench import DockerBigCodeBenchBackend
from agentflow_rl.backends.frozen_llm import (
    BaseGeneratorGatewayAdapter,
    OpenAICompatibleFrozenGateway,
)
from agentflow_rl.backends.serper import SerperSearchBackend
from agentflow_rl.backends.web_reader import WebPageReaderBackend
from agentflow_rl.backends.wikipedia_http import HttpWikipediaBackend
from agentflow_rl.backends.query_analyzer_cache import QueryAnalyzerCache
from agentflow_rl.prm import HttpProcessRewardBackend, LearnedProcessScorer
from agentflow_rl.rewards.online_judge import JsonlScoreCache, OpenAICompatibleProcessJudge
from agentflow_rl.runtime.loop import UnifiedAgentFlowLoop
from agentflow_rl.runtime.projections import RoleMemoryProjector
from agentflow_rl.tasks.aime import AIMEAdapter, AIMEEvaluator
from agentflow_rl.tasks.bigcodebench import BigCodeBenchAdapter, BigCodeBenchEvaluator
from agentflow_rl.tasks.contracts import JsonlPrivateEvaluationStore, TaskRegistry
from agentflow_rl.tasks.gpqa import GPQAAdapter, GPQAEvaluator
from agentflow_rl.tasks.taco import TACOAdapter, TACOEvaluator
from agentflow_rl.tasks.twowiki import TwoWikiAdapter, TwoWikiEvaluator
from agentflow_rl.tools import (
    BaseGeneratorTool,
    GoogleSearchTool,
    PythonCoderTool,
    ToolRegistry,
    WikipediaSearchTool,
)

from .verl_agent_loop import RuntimeBundle, config_value
from .artifacts import TrajectoryArtifactWriter


def _required(config: Any, path: str) -> Any:
    value = config_value(config, path)
    if value is None or value == "":
        raise ValueError(f"missing required configuration: {path}")
    return value


def _openai_client(*, base_url: str, api_key: str, timeout_s: float):
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)


def build_runtime_bundle(config: Any, *, planner) -> RuntimeBundle:
    mode = str(config_value(config, "agentflow.environment_mode", "formal"))
    if mode != "formal":
        raise ValueError("beta release supports agentflow.environment_mode=formal")
    timeout_s = float(config_value(config, "agentflow.http_timeout_s", 60.0))
    coordination_dir = str(
        config_value(config, "agentflow.tools.coordination_dir", "/tmp/agentflow-locks")
    )
    frozen_model = str(_required(config, "agentflow.frozen_model"))
    frozen_revision = str(
        config_value(config, "agentflow.frozen_revision", frozen_model)
    )
    frozen_client = _openai_client(
        base_url=str(_required(config, "agentflow.frozen_base_url")),
        api_key=str(config_value(config, "agentflow.frozen_api_key", "not-required")),
        timeout_s=timeout_s,
    )
    frozen_gateway = OpenAICompatibleFrozenGateway(
        frozen_client,
        model=frozen_model,
        revision=frozen_revision,
        temperature=0.0,
        max_input_tokens=int(config_value(config, "agentflow.frozen_max_input_tokens",
                                          config_value(config, "data.max_prompt_length", 8192))),
        role_max_input_tokens=config_value(
            config, "agentflow.role_max_input_tokens", None
        ),
        tokenizer=_frozen_tokenizer(config),
    )
    sandbox_image = _required(config, "agentflow.tools.sandbox_image")
    sandbox = HttpSandboxBackend(
        str(_required(config, "agentflow.tools.sandbox_service_url")),
        expected_revision=str(
            config_value(
                config,
                "agentflow.tools.sandbox_service_revision",
                "python-sandbox-service-v1",
            )
        ),
        timeout_s=float(
            config_value(config, "agentflow.tools.sandbox_service_timeout_s", 30.0)
        ),
    )
    bigcodebench_image = _required(config, "agentflow.tools.bigcodebench_image")
    bigcodebench_revision = _required(config, "agentflow.tools.bigcodebench_revision")
    bigcodebench_backend = DockerBigCodeBenchBackend(
        image=str(bigcodebench_image),
        revision=str(bigcodebench_revision),
        cpus=float(config_value(config, "agentflow.tools.sandbox_cpus", 1.0)),
        memory=str(config_value(config, "agentflow.tools.bigcodebench_memory", "8g")),
        output_limit_bytes=int(config_value(config, "agentflow.tools.bigcodebench_output_limit_bytes", 1_000_000)),
        max_concurrency=int(
            config_value(config, "agentflow.tools.bigcodebench_max_concurrency", 2)
        ),
        coordination_dir=str(
            config_value(
                config,
                "agentflow.tools.coordination_dir",
                "/tmp/agentflow-locks",
            )
        ),
    )

    key_env = str(config_value(config, "agentflow.tools.serper_key_env", "SERPER_API_KEY"))
    search_backend = SerperSearchBackend(
        api_key=os.environ.get(key_env, ""),
        revision=str(
            config_value(
                config, "agentflow.tools.serper_revision", "serper-search-v1"
            )
        ),
        cache_path=str(_required(config, "agentflow.tools.serper_cache_path")),
        coordination_dir=str(
            config_value(
                config,
                "agentflow.tools.coordination_dir",
                "/tmp/agentflow-locks",
            )
        ),
        max_concurrency=int(
            config_value(config, "agentflow.tools.serper_max_concurrency", 40)
        ),
        max_attempts=int(
            config_value(config, "agentflow.tools.serper_max_attempts", 3)
        ),
        backoff_s=float(
            config_value(config, "agentflow.tools.serper_backoff_s", 0.5)
        ),
        cost_per_request_usd=float(
            config_value(
                config, "agentflow.tools.serper_cost_per_request_usd", 0.001
            )
        ),
    )
    page_reader = WebPageReaderBackend(
        timeout_s=timeout_s,
        max_concurrency=int(
            config_value(config, "agentflow.tools.web_reader_max_concurrency", 12)
        ),
        coordination_dir=coordination_dir,
    )
    wikipedia_backend = HttpWikipediaBackend(
        str(_required(config, "agentflow.tools.wikipedia_service_url")),
        expected_revision=str(
            _required(config, "agentflow.tools.wikipedia_revision")
        ),
        timeout_s=timeout_s,
    )

    retrieval_policy = {
        "read_top_k": int(config_value(config, "agentflow.tools.retrieval_read_top_k", 2)),
        "max_passages": int(config_value(config, "agentflow.tools.retrieval_max_passages", 3)),
        "max_source_chars": int(config_value(config, "agentflow.tools.retrieval_max_source_chars", 4000)),
        "read_timeout_s": float(config_value(config, "agentflow.tools.retrieval_read_timeout_s", 8.0)),
    }
    tools = ToolRegistry(
        (
            BaseGeneratorTool(
                BaseGeneratorGatewayAdapter(frozen_gateway),
                default_max_tokens=int(
                    config_value(
                        config,
                        "agentflow.tools.base_generator_max_tokens",
                        768,
                    )
                ),
            ),
            PythonCoderTool(sandbox),
            GoogleSearchTool(search_backend, page_reader, **retrieval_policy),
            WikipediaSearchTool(wikipedia_backend, **retrieval_policy),
        )
    )
    store = JsonlPrivateEvaluationStore(
        Path(str(_required(config, "agentflow.private_records_path")))
    )
    registry = TaskRegistry(
        (
            AIMEAdapter(),
            TwoWikiAdapter(),
            TACOAdapter(),
            GPQAAdapter(),
            BigCodeBenchAdapter(),
        ),
        (
            AIMEEvaluator(store),
            TwoWikiEvaluator(store),
            TACOEvaluator(store, sandbox),
            GPQAEvaluator(store),
            BigCodeBenchEvaluator(store, bigcodebench_backend),
        ),
    )

    process_mode = str(config_value(config, "agentflow.process_reward.mode", "none"))
    process_scorer = None
    if process_mode in {"prm", "online_judge"}:
        from agentflow_rl.rewards.rubric import PROCESS_RUBRIC_REVISION, validate_rubric_revision
        validate_rubric_revision(str(config_value(config, "agentflow.process_reward.rubric_revision", PROCESS_RUBRIC_REVISION)))
    if process_mode == "prm":
        process_scorer = LearnedProcessScorer(
            HttpProcessRewardBackend(
                str(_required(config, "agentflow.process_reward.endpoint")),
                revision=str(_required(config, "agentflow.process_reward.revision")),
                timeout_s=timeout_s,
            )
        )
    elif process_mode == "online_judge":
        key_env = str(
            config_value(config, "agentflow.process_reward.api_key_env", "DEEPSEEK_API_KEY")
        )
        judge_client = _openai_client(
            base_url=str(_required(config, "agentflow.process_reward.base_url")),
            api_key=os.environ.get(key_env, ""),
            timeout_s=timeout_s,
        )
        process_scorer = OpenAICompatibleProcessJudge(
            judge_client,
            model=str(_required(config, "agentflow.process_reward.model")),
            revision=str(_required(config, "agentflow.process_reward.revision")),
            cache=JsonlScoreCache(
                Path(str(_required(config, "agentflow.process_reward.cache_path")))
            ),
            max_concurrency=int(
                config_value(config, "agentflow.process_reward.max_concurrency", 8)
            ),
            max_attempts=int(config_value(config, "agentflow.process_reward.max_attempts", 3)),
            backoff_s=float(config_value(config, "agentflow.process_reward.backoff_s", 0.5)),
            coordination_dir=str(
                config_value(
                    config,
                    "agentflow.tools.coordination_dir",
                    "/tmp/agentflow-locks",
                )
            ),
        )
    elif process_mode != "none":
        raise ValueError(f"unsupported process reward mode: {process_mode}")

    analyzer_cache = None
    if bool(config_value(config, "agentflow.query_analyzer_cache.enabled", False)):
        analyzer_cache = QueryAnalyzerCache(
            str(_required(config, "agentflow.query_analyzer_cache.path")),
            coordination_dir=coordination_dir,
            namespace=str(
                config_value(
                    config,
                    "agentflow.query_analyzer_cache.namespace",
                    "query-analyzer-v1",
                )
            ),
            lock_timeout_s=float(
                config_value(
                    config,
                    "agentflow.query_analyzer_cache.lock_timeout_s",
                    120.0,
                )
            ),
        )
    role_token_config = config_value(config, "agentflow.role_max_tokens", 2048)
    if not isinstance(role_token_config, int):
        role_token_config = {
            str(key): int(value) for key, value in role_token_config.items()
        }
    loop = UnifiedAgentFlowLoop(
        planner=planner,
        frozen_roles=frozen_gateway,
        tools=tools,
        tasks=registry,
        projector=RoleMemoryProjector(
            token_counter=lambda text: len(
                planner.tokenizer.encode(text, add_special_tokens=False)
            )
            if hasattr(planner, "tokenizer")
            else max(1, (len(text.encode("utf-8")) + 3) // 4)
        ),
        max_turns=int(config_value(config, "agentflow.max_turns", 5)),
        max_role_tokens=role_token_config,
        query_analyzer_cache=analyzer_cache,
    )
    artifact_dir = config_value(config, "agentflow.trajectory_artifact_dir")
    trajectory_writer = (
        TrajectoryArtifactWriter(str(artifact_dir)) if artifact_dir else None
    )
    return RuntimeBundle(
        loop=loop,
        process_scorer=process_scorer,
        trajectory_writer=trajectory_writer,
    )


__all__ = ["build_runtime_bundle"]


@lru_cache(maxsize=4)
def _load_tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)


def _frozen_tokenizer(config):
    path = config_value(config, "agentflow.frozen_tokenizer_path") or os.environ.get("FROZEN_MODEL_PATH")
    if path:
        return _load_tokenizer(str(path))
    raise ValueError("Set agentflow.frozen_tokenizer_path or FROZEN_MODEL_PATH to the served frozen checkpoint")
