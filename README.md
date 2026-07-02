# AgentFlow Turn-Level GSPO Experiments

本仓库是在 [lupantech/AgentFlow](https://github.com/lupantech/AgentFlow) 基础上扩展的实验工程，主要研究小模型在多步工具调用任务中的规划、执行和强化学习优化。

当前代码重点覆盖两个任务方向：

- **GSM8K AgentFlow**：使用 Calculator 工具完成多步数学推理。
- **Ticket AgentFlow**：在完全隔离的合成工单环境中完成查询、更新和提交任务。

项目重点是构造一个可验证、可复现、支持并发 rollout 的 AgentFlow 实验环境，并在此基础上训练 Planner 的下一步工具调用策略。

> 当前 fork 主要维护 **language-only AgentFlow 工作流**。上游项目中的部分多模态引擎、示例和集成已被删除或简化，因此本仓库不应被视为上游 AgentFlow 全部功能的直接替代。

---

## 1. 项目目标

AgentFlow 将复杂任务拆分为 Planner、Executor、Verifier 等组件，并通过多轮工具调用完成任务。

本项目主要研究以下问题：

> 对同一个 query 采样多条 AgentFlow rollout，将它们组成一个 group，使用任务级 reward 计算组内 advantage，再对每个 Planner turn 的下一步动作进行强化学习更新，能否提升小模型的工具调用规划能力？

当前 ticket-agent 实验不使用 SFT 冷启动，实验顺序为：

```text
Frozen AgentFlow baseline
        ↓
Grouped AgentFlow rollouts
        ↓
Binary verifier reward
        ↓
Group-normalized advantage
        ↓
Planner-only LoRA GSPO update
        ↓
Controlled baseline / adapter evaluation
```

---

## 2. 当前支持的任务

### 2.1 GSM8K AgentFlow

目录：

```text
project/try_gsm8k_0522/
```

该任务使用 Calculator 工具完成多步数学推理，包含：

- AgentFlow baseline；
- direct generation baseline；
- rollout 与数值答案评分；
- Planner-only LoRA 训练；
- 轻量 GRPO / GSPO 目标函数；
- 训练与评估公共组件。

其中：

```text
project/try_gsm8k_0522/flowgrpo_light/
```

提供了当前 ticket-agent 训练复用的轻量训练基础设施，包括：

- 本地 Transformers + PEFT Planner policy；
- 冻结的 OpenAI-compatible vLLM 客户端；
- AgentFlow rollout runner；
- grouped advantage；
- sequence log-probability；
- clipped GRPO / GSPO objective。

更多信息见：

```text
project/try_gsm8k_0522/flowgrpo_light/README.md
```

### 2.2 Ticket AgentFlow

目录：

```text
project/try_ticket_agent/
```

这是一个不依赖真实业务系统的合成工单任务。每条 episode 包含：

- 初始 ticket 状态；
- 用户请求；
- 隐藏目标 `goal_spec`；
- 最大允许步数；
- 确定性 verifier。

Agent 可以调用三个工具：

- `Ticket_Query_Tool`：根据 `ticket_id`、`customer_id` 或 `order_id` 查询 ticket；
- `Ticket_Update_Tool`：修改目标 ticket 的一个允许字段；
- `Ticket_Finish_Tool`：提交任务完成结果。

任务分为两类：

#### Direct

用户请求直接提供 `ticket_id`，典型流程为：

```text
Update → Finish
```

#### Indirect

用户只提供 `customer_id` 或 `order_id`，典型流程为：

```text
Query → Update returned ticket → Finish
```

Verifier 会检查：

- 是否修改了正确的 ticket；
- 是否修改了正确字段和值；
- 是否正确调用 Finish；
- 是否超过最大步数；
- 是否错误修改了其他 ticket；
- 是否出现非法工具调用。

只有全部条件满足时，reward 才为 `1`，否则为 `0`。

更多信息见：

```text
project/try_ticket_agent/README.md
```

---

## 3. Turn-Level GSPO 实现

训练时，对同一个 query 采样多条 rollout，并组成一个 rollout group。

每条 rollout 获得一个任务级 binary reward。基础流程如下：

1. 从同一个 query 采样多条 AgentFlow trajectory；
2. 移除基础设施失败的 trajectory；
3. 在剩余 trajectory 内对 reward 做组内标准化；
4. 得到每条 trajectory 的 rollout-level advantage；
5. 将该 advantage 广播到该 trajectory 的所有 Planner turns；
6. 使用 Planner 真实生成时记录的 prompt 和 response token IDs 计算 log-probability；
7. 对每个 Planner turn 构造长度归一化的 sequence-level importance ratio；
8. 使用非对称 clipping 更新 Planner LoRA 参数。

需要注意：

- 训练样本的粒度是 **Planner turn**；
- token IDs 用于精确重建响应并计算 sequence log-probability；
- 梯度会经过 Planner response tokens；
- importance ratio 和 clipping 的粒度是每个 turn 的完整响应序列，而不是每个 token 独立计算。

当前默认 clipping 范围为：

```text
lower clip: 0.001
upper clip: 0.003
```

当一个 group 内所有有效 trajectory 的 reward 相同时，组内 advantage 为零，该 group 不产生有效 policy update。

---

## 4. 目录结构

```text
.
├── README.md
├── requirements.txt
├── DATA_SOURCES.md
├── LICENSE
├── NOTICE.md
└── project
    ├── agentflow
    │   └── agentflow
    │       ├── engine
    │       ├── models
    │       ├── tools
    │       └── ...
    ├── try_gsm8k_0522
    │   ├── direct_baseline
    │   ├── flowgrpo_light
    │   ├── flowgrpo_general_2x40g
    │   ├── tests
    │   └── ...
    └── try_ticket_agent
        ├── baseline
        ├── data
        ├── data_synthesis
        ├── flowgrpo_general_2x40g
        ├── scripts
        ├── tests
        ├── ticket_env
        └── run_ticket_agentflow.py
```

关键实现位置：

```text
project/agentflow/agentflow/tools/ticket_*
```

Ticket 工具实现。

```text
project/try_ticket_agent/ticket_env/
```

Ticket backend、episode schema、runtime 构造和确定性 verifier。

```text
project/try_ticket_agent/ticket_env/solver_factory.py
```

构造 ticket-agent 专用 AgentFlow runtime、Planner prompt 和工具配置。

```text
project/try_gsm8k_0522/flowgrpo_light/
```

Planner policy、rollout runner、log-probability 和轻量 GRPO / GSPO 训练组件。

```text
project/try_ticket_agent/flowgrpo_general_2x40g/
```

Ticket AgentFlow 的训练、配置和受控评估入口。

---

## 5. 环境要求

推荐环境：

- Linux；
- Python 3.10 或 3.11；
- CUDA GPU；
- 支持 BF16 的 GPU；
- PyTorch、Transformers、PEFT；
- OpenAI-compatible vLLM 服务。

核心 AgentFlow package 要求：

```text
Python >= 3.10
```

创建环境并安装基础依赖：

```bash
git clone https://github.com/beibeizzz/agentflow.git
cd agentflow

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

`vLLM` 与 CUDA、PyTorch 和操作系统版本强相关，建议根据目标机器环境单独安装：

```bash
pip install vllm
```

模型权重不会随仓库提交，需要自行准备，例如：

```text
/path/to/Qwen3-0.6B
```

### 默认 GPU 布局

完整 ticket-agent 训练默认按照双 GPU 布局设计：

```text
GPU 0：冻结的 Qwen3-0.6B vLLM 服务
GPU 1：本地 Transformers + PEFT Planner LoRA 训练
```

单 GPU 机器需要降低采样并发、batch size，或者将 rollout 与训练分阶段执行。

---

## 6. Quick Start：Ticket AgentFlow

以下命令默认从仓库根目录开始。

### 6.1 进入项目目录

```bash
cd project
```

### 6.2 生成 Ticket blueprints

仓库中已经包含生成后的 ticket 数据。如需重新生成，可以执行：

```bash
python try_ticket_agent/scripts/generate_blueprints.py \
  --seed 42 \
  --smoke 32 \
  --train 2500 \
  --validation 256 \
  --test 512 \
  --output-dir try_ticket_agent/data/blueprints
```

校验 blueprints：

```bash
python try_ticket_agent/scripts/validate_dataset.py \
  --blueprints try_ticket_agent/data/blueprints
```

### 6.3 使用 LLM 合成自然语言请求

复制配置文件：

```bash
cp \
  try_ticket_agent/config_synthesis.example.yaml \
  try_ticket_agent/config_synthesis.yaml
```

根据实际 API 修改：

```text
try_ticket_agent/config_synthesis.yaml
```

然后执行：

```bash
DEEPSEEK_API_KEY=your_api_key \
python try_ticket_agent/scripts/synthesize_dataset.py \
  --config try_ticket_agent/config_synthesis.yaml
```

校验生成结果：

```bash
python try_ticket_agent/scripts/validate_dataset.py \
  --dataset try_ticket_agent/data/generated
```

数据合成流程为：

```text
Blueprint
→ LLM rewrite
→ Deterministic validator
→ LLM judge
→ Registered-tool reference execution
```

LLM judge 仅用于数据合成阶段，不参与训练和测试阶段的 reward 计算。

### 6.4 启动冻结 vLLM 服务

设置模型路径：

```bash
export MODEL_PATH=/absolute/path/to/Qwen3-0.6B
```

在 GPU 0 上启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --max-model-len 4096
```

根据显存情况，可以额外设置：

```text
--gpu-memory-utilization
--max-model-len
--max-num-seqs
```

检查服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

返回结果中应包含：

```text
Qwen3-0.6B
```

### 6.5 运行 Frozen AgentFlow baseline

在另一个终端中：

```bash
cd project
```

运行 baseline：

```bash
bash try_ticket_agent/baseline/run_agentflow_baseline.sh
```

可覆盖的主要环境变量包括：

```bash
BASE_URL=http://127.0.0.1:8000/v1
LLM_ENGINE_NAME=vllm-Qwen3-0.6B
DATA_FILE=try_ticket_agent/data/generated/test.jsonl
OUTPUT_DIR=try_ticket_agent/baseline/outputs/test
MAX_STEPS=3
MAX_TIME=120
MAX_TOKENS=512
```

例如：

```bash
DATA_FILE=try_ticket_agent/data/generated/smoke.jsonl \
OUTPUT_DIR=try_ticket_agent/baseline/outputs/smoke \
bash try_ticket_agent/baseline/run_agentflow_baseline.sh
```

当前 baseline shell 脚本固定使用：

```text
temperature = 0.0
think_mode = off
query_analysis_think_mode = on
```

Baseline 输出：

```text
try_ticket_agent/baseline/outputs/test/
├── baseline_details.jsonl
└── baseline_summary.json
```

### 6.6 运行最小 Smoke Training

建议先使用最小配置验证：

- vLLM 服务可访问；
- rollout 可以正常执行；
- loss、ratio 和 KL 为有限值；
- LoRA 参数发生更新；
- adapter 可以正确保存。

```bash
MODEL_PATH=/absolute/path/to/Qwen3-0.6B \
CUDA_VISIBLE_DEVICES=1 \
QUESTION_BATCH_SIZE=2 \
GROUP_SIZE=2 \
ROLLOUT_CONCURRENCY=4 \
PLANNER_BATCH_SIZE=4 \
MAX_TRAIN_ITEMS=2 \
EPOCHS=1 \
bash try_ticket_agent/flowgrpo_general_2x40g/run_train_general_2x40g.sh
```

### 6.7 运行完整 Ticket GSPO Training

```bash
MODEL_PATH=/absolute/path/to/Qwen3-0.6B \
CUDA_VISIBLE_DEVICES=1 \
bash try_ticket_agent/flowgrpo_general_2x40g/run_train_general_2x40g.sh
```

训练入口：

```text
try_ticket_agent/flowgrpo_general_2x40g/train_ticket_gspo.py
```

默认配置文件：

```text
try_ticket_agent/flowgrpo_general_2x40g/config_train_general_2x40g.yaml
```

默认输出目录：

```text
try_ticket_agent/flowgrpo_general_2x40g/outputs/train_general_2x40g/
```

最终 adapter：

```text
try_ticket_agent/flowgrpo_general_2x40g/outputs/train_general_2x40g/final_adapter/
```

### Shell 参数优先级

通过 `run_train_general_2x40g.sh` 启动时，shell 脚本传入的 CLI 参数会覆盖 YAML 中对应字段。

常用覆盖变量：

```bash
MODEL_PATH
TRAIN_FILE
OUTPUT_DIR
FROZEN_BASE_URL
FROZEN_MODEL
QUESTION_BATCH_SIZE
GROUP_SIZE
ROLLOUT_CONCURRENCY
PLANNER_BATCH_SIZE
MAX_STEPS
CLIP_RANGE_LOW
CLIP_RANGE_HIGH
POLICY_EPOCHS
MAX_TRAIN_ITEMS
EPOCHS
```

例如：

```bash
MODEL_PATH=/absolute/path/to/Qwen3-0.6B \
QUESTION_BATCH_SIZE=4 \
GROUP_SIZE=8 \
ROLLOUT_CONCURRENCY=32 \
PLANNER_BATCH_SIZE=32 \
MAX_TRAIN_ITEMS=2500 \
bash try_ticket_agent/flowgrpo_general_2x40g/run_train_general_2x40g.sh
```

---

## 7. Controlled Evaluation

评估脚本同时支持：

- frozen baseline；
- LoRA adapter。

评估入口：

```text
try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh
```

### 7.1 创建本地评估配置

默认评估 YAML 中的 `model_path` 可能指向开发机器上的绝对路径，因此建议创建本地配置：

```bash
cp \
  try_ticket_agent/flowgrpo_general_2x40g/config_eval_learnable_general_2x40g.yaml \
  try_ticket_agent/flowgrpo_general_2x40g/config_eval.local.yaml
```

编辑：

```text
try_ticket_agent/flowgrpo_general_2x40g/config_eval.local.yaml
```

至少修改：

```yaml
model_path: /absolute/path/to/Qwen3-0.6B
frozen_base_url: http://127.0.0.1:8000/v1
frozen_model: Qwen3-0.6B
eval_file: try_ticket_agent/data/generated/test.jsonl
```

### 7.2 评估 Frozen baseline

```bash
CONFIG_FILE=try_ticket_agent/flowgrpo_general_2x40g/config_eval.local.yaml \
EVAL_MODE=baseline \
ADAPTER_PATH=false \
bash try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh
```

### 7.3 评估 LoRA adapter

```bash
CONFIG_FILE=try_ticket_agent/flowgrpo_general_2x40g/config_eval.local.yaml \
EVAL_MODE=adapter \
ADAPTER_PATH=try_ticket_agent/flowgrpo_general_2x40g/outputs/train_general_2x40g/final_adapter \
CUDA_VISIBLE_DEVICES=1 \
bash try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh
```

评估输出包括：

- overall episode success rate；
- direct success rate；
- indirect success rate；
- invalid-action rate；
- infrastructure-failure rate；
- 每条 episode 的 Planner response；
- Planner response token IDs；
- verifier 结果；
- rollout errors。

默认输出目录：

```text
try_ticket_agent/flowgrpo_general_2x40g/outputs/
├── eval_baseline
└── eval_adapter
```

---

## 8. 运行测试

在 `project/` 目录下执行：

```bash
python -m unittest discover try_ticket_agent/tests
```

运行 GSM8K 相关测试：

```bash
python -m unittest discover try_gsm8k_0522/tests
```

运行全部测试：

```bash
python -m unittest discover try_ticket_agent/tests
python -m unittest discover try_gsm8k_0522/tests
```

部分测试只验证本地逻辑、schema 和配置，不能替代目标 GPU 环境上的端到端训练验证。

---

## 9. 训练指标

训练过程会记录以下类型的指标：

### Rollout 和 reward

- rollout group 数；
- trajectory 数；
- valid trajectory 数；
- infrastructure-failed trajectory 数；
- reward mean / std；
- nonzero-advantage trajectory 数；
- skipped equal-reward group 数。

### Planner turn

- Planner turn 数；
- response token 数；
- nonzero-advantage turn 数；
- invalid Planner action 数。

### Policy update

- loss；
- policy loss；
- ratio mean / min / max；
- clip fraction；
- approximate KL；
- effective sample count；
- policy epoch 数；
- gradient norm；
- checkpoint 保存状态。

建议不要只观察最终 success rate，还应检查：

- loss 是否为有限值；
- ratio 是否异常爆炸；
- clip fraction 是否长期接近 0 或 1；
- approximate KL 是否持续增大；
- 有效 trajectory 比例；
- invalid-action rate；
- adapter 保存和重新加载是否正常。

---

## 10. 已知限制

- Ticket 任务是合成沙盒环境，不连接真实工单系统。
- 数据中不包含真实客户、订单或工单信息。
- 当前主要面向 Qwen3-0.6B 级别的小模型设计。
- 小模型对 Planner prompt、工具 schema、较敏感。
- Equal-reward group 不产生有效 advantage。
- 当前 fork 主要支持 language-only 工作流，不保证兼容上游全部多模态功能。

---

## 11. 数据来源

本项目的数据来源和生成方式见：

```text
DATA_SOURCES.md
```

简要说明：

- GSM8K 实验数据来自 GSM8K 数据集及本地预处理、改写流程；
- ticket-agent 数据为纯合成沙盒数据；
- ticket-agent 不包含真实客户、订单或工单记录；
- ticket 自然语言请求可由 LLM 合成，但最终通过确定性规则和 reference execution 校验。

---

## 12. License and Attribution

This repository is based on [lupantech/AgentFlow](https://github.com/lupantech/AgentFlow), which is licensed under the MIT License.

Original AgentFlow copyright:

```text
Copyright (c) 2025 the AgentFlow Team
```

This repository preserves the upstream MIT license notice and adds experimental code for GSM8K and ticket-agent GSPO workflows.

Additional modifications are licensed under the same MIT License unless a file states otherwise.

See:

```text
LICENSE
NOTICE.md
```
