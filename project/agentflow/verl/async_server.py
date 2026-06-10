import inspect
import logging

import ray
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from agentflow.instrumentation.vllm import instrument_vllm
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


logger = logging.getLogger(__file__)


@ray.remote(num_cpus=1)
class PatchedvLLMServer(vLLMHttpServer):
    async def run_server(self, args):
        instrument_vllm()

        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            **kwargs,
        )

        await engine_client.reset_mm_cache()
        await engine_client.collective_rpc(
            method="monkey_patch_model",
            kwargs={"vocab_size": len(self.model_config.tokenizer)},
        )

        build_app_sig = inspect.signature(build_app)
        supported_tasks = ()
        if "supported_tasks" in build_app_sig.parameters:
            supported_tasks = await engine_client.get_supported_tasks()
            app = build_app(args, supported_tasks)
        else:
            app = build_app(args)

        init_app_sig = inspect.signature(init_app_state)
        if "vllm_config" in init_app_sig.parameters:
            await init_app_state(engine_client, vllm_config, app.state, args)
        elif "supported_tasks" in init_app_sig.parameters:
            await init_app_state(engine_client, app.state, args, supported_tasks)
        else:
            await init_app_state(engine_client, app.state, args)

        app.state.server = self

        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)
