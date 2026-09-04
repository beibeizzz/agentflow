# 04 运行时技术栈

## 1. 技术栈总表

| 层 | 主要技术 | 项目用途 |
|---|---|---|
| Python 进程与并发 | Python 3.11、`asyncio`、`asyncio.Semaphore`、`asyncio.to_thread`、`subprocess` | AgentLoop 协程、并发限流、阻塞工具卸载、Docker 子进程 |
| 推理引擎 | vLLM | GPU 0 OpenAI 兼容冻结服务；GPU 1 veRL 内部 Planner rollout |
| 强化学习框架 | veRL v0.8.0 接口 | RLHFDataset、AgentLoop、PPOTrainer、GSPO、checkpoint、validation |
| 训练框架 | PyTorch、FSDP、PEFT LoRA | 自动微分、混合精度、参数分片、Planner 参数高效更新 |
| 分布式计算 | Ray、veRL ResourcePoolManager | worker 生命周期、GPU 资源映射、任务调度 |
| 训练数据总线 | TransferQueue 0.1.6、TensorDict、DataProto | 变长 turn 数据交换、字段拉取与写回、actor 输入 |
| 配置 | Hydra、OmegaConf、YAML | veRL 基础配置合并、任务覆盖、命令行 override |
| 数据处理 | pandas、PyArrow、Hugging Face Datasets、JSON/JSONL | 数据下载、规范化、拆分、Parquet 转换与哈希 |
| 数据模式 | Pydantic v2 | action、event、episode、final answer 的严格校验 |
| 检索 | Pyserini、Apache Lucene、BM25 | DeepResearch 全库搜索、文档读取和 supporting-fact 审计 |
| 沙箱 | Docker Engine、Python 3.11 slim | Coding 公共/隐藏测试隔离执行 |
| 通信 | OpenAI Python SDK、HTTP/JSON、veRL server manager | 冻结角色请求、Planner token-level 请求、工具结果传递 |
| 测试 | pytest、pytest-asyncio | 单元、异步 AgentLoop、集成与语义一致性测试 |

参考：[veRL/HybridFlow](https://arxiv.org/abs/2409.19256)、[vLLM](https://docs.vllm.ai/)、[Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html)、[PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)。

## 2. Python 进程管理

### 进程角色

```text
shell experiment runner
├── vLLM OpenAI API server                 GPU 0
└── Python veRL launcher                    GPU 1 visible
    ├── Ray driver / AgentFlowTaskRunner
    ├── ActorRolloutRefWorker
    ├── Planner vLLM rollout engine
    ├── AgentLoop workers
    └── Docker / Lucene blocking operations CPU side
```

Shell 脚本管理服务阶段、环境变量、GPU 可见性和输出路径。`AgentFlowTaskRunner` 通过 Ray remote actor 创建 worker、资源池和 Trainer，并在 `finally` 阶段关闭 replay buffer 与 TransferQueue。

AgentLoop 使用协程并发：

- `asyncio.wait_for` 将冻结模型和 Planner 调用绑定到 trajectory deadline；
- `asyncio.to_thread` 承载 Lucene 查询和 Docker 执行等阻塞操作；
- `asyncio.Semaphore` 控制检索 JVM 与容器并发；
- `subprocess.run` 负责 Docker 生命周期和超时回收。

## 3. 双 vLLM 推理面

### 冻结角色服务

GPU 0 独立运行 `vllm.entrypoints.openai.api_server`。`AsyncOpenAI.chat.completions.create` 发送 system/user messages、输出上限、温度和 Qwen thinking 开关。Query Analyzer、Executor、Verifier、Generator 与 Base Generator 共享一个模型实例。

### Planner rollout 服务

GPU 1 的 vLLM 由 veRL `ActorRolloutRefWorker` 管理。项目先调用 tokenizer chat template，随后向 `LLMServerClient.generate()` 发送精确 prompt IDs，并保存 response IDs 与每 token logprob。相同 token 序列进入旧策略打分和 actor loss，减少 retokenization 引入的概率偏差。

vLLM 提供 PagedAttention、continuous batching 和高吞吐 KV cache 管理。项目使用 async rollout、chunked prefill、prefix caching 和受控 `gpu_memory_utilization`。官方能力说明见 [vLLM 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)。

## 4. veRL 强化学习控制面

veRL 管理以下职责：

1. `RLHFDataset` 读取 Parquet 并生成 prompt batch；
2. AgentLoop manager 为每个问题展开多条 session；
3. TransferQueue 保存变长 Planner-turn 数据；
4. Actor worker 重算 old logprob；
5. 原生 GSPO 计算序列概率比、裁剪目标和梯度；
6. optimizer、checkpoint、validation 与日志统一执行。

项目在 `AgentFlowPPOTrainer` 中补充四项语义：

- 按 session 最后一轮读取一次终局 reward；
- 按问题组计算 trajectory advantage 并传播到真实 Planner turn；
- 过滤基础设施无效 turn；
- 按实际 turn 数选择可整除的 optimizer mini-batch，并适配变长 AgentLoop metrics。

## 5. PyTorch、FSDP 与 LoRA

Planner 训练采用 FSDP strategy、BF16、remove padding 和每 GPU 一行的 micro-batch。LoRA 配置为 rank 64、alpha 128、`all-linear`，优化器更新 adapter 参数，阶段切换时导出标准 PEFT adapter。

FSDP 负责参数、梯度和优化器状态的分片语义；当前正式拓扑为单训练 GPU，FSDP 同时保持与 veRL 标准 worker/checkpoint 接口一致。模型扩展到多训练 GPU 时可直接复用其分片和 collective 通信路径。

## 6. Ray 与资源映射

`ResourcePoolManager` 将 actor/rollout worker 映射到 `global_pool`。启动脚本设置 `CUDA_VISIBLE_DEVICES=1`，Ray 进程看到的 local device 0 对应物理 GPU 1。GPU 0 的冻结服务通过 `127.0.0.1:8000/v1` 访问，形成清晰的服务侧与训练侧资源边界。

Ray 负责 coarse-grained worker 调度；`asyncio` 负责单个 AgentLoop 内的 fine-grained I/O 调度；vLLM 负责模型请求的 dynamic batching。三层并发各自承担资源分配、轨迹协程和 GPU batch 合并。

## 7. 配置与版本约束

`agentflow_rl.verl.main.load_config` 读取固定 veRL checkout 的 `ppo_trainer` 配置，再合并项目 YAML 和命令行覆盖。`validate_config` 在分配训练资源前检查 batch、worker、rollout 和 loss 契约。

远程主环境使用 Python 3.11、Linux CUDA、固定 veRL commit、vLLM 0.12.0、TransferQueue 0.1.6、Java 21 和 Docker Engine。`scripts/remote/audit_environment.sh` 将版本、导入、硬件和 17 份配置组合成启动门禁。

