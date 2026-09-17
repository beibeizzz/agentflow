# AgentFlow Beta

AgentFlow Beta 是统一、模块化的 reasoning agent 训练与评测实现。系统通过冻结角色和共享工具完成任务，只优化 Planner，并研究 Planner 强化学习对完整 Agentic System 的影响。

本分支包含可运行源码、正式配置、数据处理、PRM 训练、AgentFlow 强化学习、评测入口、服务脚本和容器定义。测试代码、实验记录、预测结果、模型权重、优化器检查点、数据集、索引和运行日志不进入仓库。

## 系统流程

一条轨迹依次经过 Query Analyzer、Planner、Executor、工具、Verifier 和 Generator。Planner 最多决策五轮；终局 Evaluator 使用私有答案或隐藏测试评分。Memory 是 append-only event log，各角色和 PRM 使用确定性的专属投影，隐藏评测信息只进入终局 Evaluator。

四个统一工具：

- `Base_Generator_Tool`：冻结 Qwen3-8B 通用推理。
- `Python_Coder_Tool`：隔离 Python 沙箱。
- `Google_Search_Tool`：Serper 搜索与安全网页读取。
- `Wikipedia_Search_Tool`：Wikipedia-18 BM25 与 E5-HNSW64 混合检索。

训练使用 Qwen3-4B Planner LoRA（rank 64、alpha 128）、GSPO sequence-level loss、RTG+LOO 优势和 DAPO 在线动态采样。冻结角色与 Base Generator 使用 Qwen3-8B，PRM 使用 Qwen3-0.6B。

优势算法 revision 为 `terminal_prm_rtg_loo_v1`。每条有效轨迹的原始 PRM 分数按逆序累加，并除以固定的五轮上限，得到各轮过程 reward-to-go。终局奖励和过程 RTG 分别减去同一 prompt group 中其他有效完整轨迹的 leave-one-out 均值；较短轨迹结束后的过程回报按零参与后续轮次的基线。`lambda_process=0.30` 加权过程优势。组内任一有效 turn 缺少 PRM 分数时，整个 group 使用 terminal-only LOO。checkpoint 恢复要求算法 revision、最大轮数和过程权重完全一致。

## 目录

```text
configs/                  正式数据、训练、实验和 E0-E3 评测配置
docker/                   Python 工具与 BigCodeBench 隔离执行镜像
scripts/data/             数据准备
scripts/prm/              PRM 标注与训练
scripts/eval/             E0-E3 评测
scripts/runtime/          服务、索引、preflight 和训练入口
scripts/tools/            工具目录导出
src/agentflow_rl/runtime  AgentLoop、Memory、投影、解析和隐私边界
src/agentflow_rl/roles    角色 schema、prompt 和冻结模型网关
src/agentflow_rl/tools    四个工具及注册表
src/agentflow_rl/backends 模型、Serper、Wikipedia 和 Docker 后端
src/agentflow_rl/tasks    五项任务适配器与 Evaluator
src/agentflow_rl/rewards  奖励、rubric 和优势构造
src/agentflow_rl/prm      PRM 数据、训练与推理
src/agentflow_rl/sampling DAPO 动态采样
src/agentflow_rl/data     切分、去重、泄漏检查与 veRL 数据
src/agentflow_rl/integrations veRL/vLLM 和训练器
src/agentflow_rl/evaluation   统一评测与指标
```

## 安装

正式路径面向 Linux、CUDA、Docker 和可容纳 Qwen3-4B/Qwen3-8B 的多 GPU 环境。先安装匹配的 PyTorch、CUDA 和 FlashAttention 2：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,retrieval,services,runtime]"
export PYTHON="$(pwd)/.venv/bin/python"
export PYTHONPATH="$(pwd)/src"
bash scripts/runtime/check_environment.sh
```

`runtime` extra 固定 veRL、vLLM 和 TransferQueue 版本。模型、数据、索引和镜像均使用不可变 revision。

## 数据

```bash
cp configs/data/formal_sources.example.yaml configs/data/formal_sources.yaml
# 填写路径、revision、URI 和许可证
$PYTHON scripts/data/prepare_formal_data.py \
  --config configs/data/formal_sources.yaml --output data
```

输出包括混合训练集、开发集、五项任务测试集、Evaluator 私有记录、数据 manifest 和泄漏报告。2Wiki 与 TACO 分别使用 800 条训练、80 条开发、80 条测试样本；GPQA Diamond 与 BigCodeBench-Hard 只用于评测。

## 沙箱与工具服务

```bash
export BIGCODEBENCH_COMMIT="<pinned-commit>"
export SANDBOX_IMAGE="agentflow-python-sandbox:<revision>"
export BIGCODEBENCH_IMAGE="agentflow-bigcodebench-evaluator:<revision>"
bash scripts/runtime/build_sandbox_images.sh
export SANDBOX_IMAGE_MANIFEST=outputs/environment/sandbox_images.json
$PYTHON scripts/runtime/serve_python_sandbox.py
```

Python 工具使用六个预热 worker，每个 1 CPU 和 1 GiB 内存，最多接受 34 个排队请求。容器内网络关闭，CPU、内存、时间、进程和输出均受限。

Google Search 从 `SERPER_API_KEY` 读取凭据，允许 40 个搜索并发和 12 个网页读取并发。

## Wikipedia

规范语料为固定 revision 的 `wiki-18.jsonl`。Arrow、BM25 和 FAISS 通过同一 corpus row 对齐：

```bash
$PYTHON scripts/runtime/build_wikipedia_hnsw.py \
  --corpus /data/wikipedia/wiki-18.jsonl \
  --arrow-cache-dir /data/wikipedia/arrow-cache \
  --model /data/models/intfloat-e5-base-v2 \
  --model-revision "<e5-revision>" \
  --index-revision "<index-revision>" \
  --index-output /data/wikipedia/wiki18-hnsw64.faiss \
  --flat-reference-output /data/wikipedia/wiki18-flat.faiss \
  --manifest-output /data/wikipedia/wiki18-hnsw64.manifest.json \
  --device cpu --threads 20 --m 64 --ef-construction 256
```

为同一语料建立 Pyserini BM25 索引，设置 `WIKIPEDIA_CORPUS_PATH`、`WIKIPEDIA_ARROW_CACHE_DIR`、`WIKIPEDIA_ARROW_FINGERPRINT`、`WIKIPEDIA_BM25_INDEX`、`WIKIPEDIA_FAISS_INDEX`、`WIKIPEDIA_E5_MODEL`、`WIKIPEDIA_E5_REVISION` 和 `WIKIPEDIA_INDEX_REVISION`，再运行：

```bash
$PYTHON scripts/runtime/serve_wikipedia.py
```

服务默认监听 8002，同时处理四个检索批次，在 5 ms 内合并最多 16 个查询。

## 模型服务

```bash
export FROZEN_MODEL_PATH=/data/models/Qwen3-8B
export FROZEN_MODEL_REVISION="<frozen-revision>"
export FROZEN_CUDA_VISIBLE_DEVICES=1
bash scripts/runtime/serve_frozen_qwen3_8b.sh

export PRM_MODEL_PATH=/data/models/agentflow-prm
export PRM_REVISION="<prm-revision>"
export PRM_CUDA_VISIBLE_DEVICES=1
bash scripts/runtime/serve_prm_vllm.sh

export PRM_BACKEND=vllm
export PRM_VLLM_ENDPOINT=http://127.0.0.1:8004
$PYTHON scripts/runtime/serve_prm.py
```

冻结 8B、PRM wrapper 和 PRM vLLM 默认监听 8001、8003 和 8004。

## PRM 标注与训练

```bash
export DEEPSEEK_API_KEY="<api-key>"
$PYTHON scripts/prm/label_transitions.py \
  --input outputs/prm/transitions.jsonl \
  --output outputs/prm/formal/labels.jsonl \
  --cache outputs/prm/formal/judge-cache.jsonl \
  --metrics outputs/prm/formal/label-metrics.json \
  --base-url "<judge-api-url>" --revision "<judge-revision>" \
  --tokenizer /data/models/Qwen3-0.6B

export PRM_BASE_MODEL=/data/models/Qwen3-0.6B
export PRM_LABELS=outputs/prm/formal/labels.jsonl
export PRM_OUTPUT=outputs/prm/formal/training-ddp
bash scripts/runtime/train_prm_ddp.sh
```

PRM 输入包括任务 query、统一评分指令、当前决策前的有效 Memory、Planner action、Executor 核心请求和工具核心结果。训练与在线打分使用同一个序列化协议。

## Preflight 与 Planner 训练

设置 `PLANNER_MODEL_PATH`、`PLANNER_REVISION`、`FROZEN_MODEL_PATH`、`FROZEN_MODEL_REVISION`、`PRM_REVISION`、`WIKIPEDIA_INDEX_REVISION`、`WIKIPEDIA_BENCHMARK_REPORT`、`SANDBOX_IMAGE`、`BIGCODEBENCH_IMAGE`、`BIGCODEBENCH_REVISION` 和 `SANDBOX_IMAGE_MANIFEST`，启动服务后执行：

```bash
export TRAIN_CUDA_VISIBLE_DEVICES=0
export REWARD_MODE=prm
bash scripts/runtime/run_real_preflight.sh
bash scripts/runtime/run_unified_train.sh
```

正式配置采集 8 个候选 prompt group，每组 5 条轨迹；DAPO 只按终局奖励方差过滤，并补采样到 4 个合格 group 或达到生成上限。Actor 使用动态 token batch、`ppo_mini_batch_size=32`、`ppo_max_token_len_per_gpu=40960`、学习率 `1e-6`、一个数据 epoch 和零 KL。默认训练目录分别为 `outputs/train/rtg_loo_terminal` 与 `outputs/train/rtg_loo_prm`。

## E0-E3 评测

四个条件为 Qwen3-4B 直接生成、初始 Planner AgentFlow、终局奖励 RL、终局奖励加 PRM RL：

```bash
cp configs/eval/shared_manifest.example.yaml configs/eval/shared_manifest.yaml
$PYTHON scripts/eval/prepare_evaluation_manifest.py \
  --config configs/eval/shared_manifest.yaml \
  --output outputs/evaluation/shared_manifest.json

export EVAL_PLANNER_MODEL_PATH=/data/models/or/checkpoints/planner
export EVAL_PLANNER_CUDA_VISIBLE_DEVICES=0
bash scripts/runtime/serve_planner_eval.sh
```

使用 `scripts/eval/run_evaluation.py` 运行 `E0_direct`、`E1_initial_agentflow`、`E2_terminal_rl` 和 `E3_terminal_prm_rl`。五个 `data/verl/test_*.parquet` 逐项追加 `--input`；各条件共享测试数据、私有记录、工具预算、解码参数和冻结 revision，默认单种子、并发 8。使用 `scripts/eval/compare_evaluations.py` 汇总结果。

AIME、2Wiki 和 GPQA 使用任务对应的规范化答案评测；TACO 与 BigCodeBench 使用隔离测试执行器。隐藏答案和隐藏测试只存在于 Evaluator 边界。

## 复现与安全

- manifest 保存模型、数据、语料、索引、prompt、Evaluator、镜像和环境哈希。
- Serper 与 Judge 凭据只从环境变量读取。
- 网页读取拒绝私有、回环、链路本地和重定向后的受限地址。
- 模型、语料、轨迹、缓存、索引、checkpoint 和日志位于 Git 忽略路径。

项目使用 Apache-2.0。第三方声明见 `LICENSES/`、`NOTICE` 和 `THIRD_PARTY_NOTICES.md`。
