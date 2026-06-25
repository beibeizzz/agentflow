# AgentFlow GSPO Experiments

本仓库是在 AgentFlow 基础上扩展的实验工程，用于研究小模型在多步工具调用任务中的规划、执行与强化学习优化。当前代码重点覆盖两个任务方向：GSM8K 计算推理任务，以及新增的沙盒工单处理任务。工程目标不是接入真实业务系统，而是构造可验证、可复现、可并发运行的 AgentFlow 任务环境，并在此基础上进行 turn-level GSPO 风格训练。

## 背景

原始 AgentFlow 将复杂问题拆成 planner、executor、verifier 等模块，并通过工具调用完成多步任务。本项目在此基础上关注一个具体问题：当 planner 每轮输出下一步工具调用时，是否可以把同一 query 下的多条 rollout 作为一组，用二值或任务级 reward 计算组内 advantage，再对 planner token 做轻量强化学习更新。

项目中的 GSPO 改造保持了 turn-level 训练风格：训练样本来自 AgentFlow rollout 中每一轮 planner next step 的真实 prompt 与 response token IDs；reward/advantage 按 query rollout group 聚合；目标函数采用带 old-logprob ratio 与非对称 clip 的 GSPO/PPO 风格形式。

## 任务

### GSM8K AgentFlow

`project/try_gsm8k_0522` 保留了 GSM8K 上的 AgentFlow baseline、直接 baseline、rollout、评分和轻量 GRPO/GSPO 训练流程。该任务主要用于验证多步 calculator 工具调用与数值答案评分。

### Ticket AgentFlow

`project/try_ticket_agent` 是新增的无真实业务依赖沙盒工单任务。每条样本包含隔离的初始 ticket 状态、用户请求、隐藏目标和最大步数。Agent 需要使用三个工具完成任务：

- `Ticket_Query_Tool`：按 `ticket_id`、`customer_id` 或 `order_id` 查询 ticket。
- `Ticket_Update_Tool`：更新目标 ticket 的一个允许字段。
- `Ticket_Finish_Tool`：提交完成结果。

任务分为 direct 与 indirect 两类：

- direct：用户请求直接给出 `ticket_id`，流程通常是 Update -> Finish。
- indirect：用户请求只给出 `customer_id` 或 `order_id`，流程通常是 Query -> Update returned ticket -> Finish。

reward 使用 verifier 根据最终状态、提交结果和步数约束做确定性二值判定。

## 思路

训练时对同一个 query 采样多条 rollout，形成 group。每条 rollout 得到一个任务级 reward。组内 reward 标准化后得到 rollout-level advantage，再复制到该 rollout 的每个 planner turn。这样保留了 AgentFlow 的多步结构，同时让每轮 planner 输出都能参与 token-level policy update。

当前 ticket 任务面向 Qwen3-0.6B 这类小模型设计，因此工具数、流程长度和 prompt 信息量都被控制在较小范围内。工程上优先保证任务可验证、采样可并发、失败模式可分析。

## 工程实现

主要目录：

```text
project/agentflow/agentflow/          AgentFlow 核心框架与工具实现
project/try_gsm8k_0522/               GSM8K AgentFlow 与 GSPO/GRPO 实验
project/try_ticket_agent/             沙盒工单 AgentFlow 任务
project/try_ticket_agent/data/        ticket 数据生成、校验与说明
project/try_ticket_agent/baseline/    ticket AgentFlow baseline 配置与脚本
project/try_ticket_agent/flowgrpo_general_2x40g/  ticket turn-level GSPO 训练与评估
```

关键实现点：

- ticket 工具在 `project/agentflow/agentflow/tools/ticket_*` 中实现。
- ticket backend 和确定性 verifier 用于隔离状态、执行工具、计算 binary reward。
- `try_ticket_agent/ticket_env/solver_factory.py` 构造 ticket 专用 AgentFlow runtime 和 planner prompt。
- `try_gsm8k_0522/flowgrpo_light` 提供轻量 policy、rollout runner、GSPO/GRPO objective 与训练公共组件。
- ticket 训练脚本复用上述轻量组件，并接入 ticket runtime、数据集、二值 reward 和 turn-level planner token 训练。

## 环境依赖

基础 Python 依赖记录在 [requirements.txt](requirements.txt)。建议在独立虚拟环境中安装：

```bash
pip install -r requirements.txt
```

`vLLM` 与 CUDA/PyTorch 版本强相关，建议根据远程实验机器的 CUDA 环境单独安装。模型权重不随仓库提交。

## 使用流程

以下命令默认在 `project/` 目录下运行，并假设已有可用 Python 环境、PyTorch/Transformers/PEFT 等依赖，以及本地或远程 vLLM 服务。

### 1. 生成或校验 ticket 数据

```bash
python try_ticket_agent/scripts/generate_blueprints.py --seed 42 --smoke 32 --train 2500 --validation 256 --test 512 --output-dir try_ticket_agent/data/blueprints
python try_ticket_agent/scripts/validate_dataset.py --blueprints try_ticket_agent/data/blueprints
python try_ticket_agent/scripts/validate_dataset.py --dataset try_ticket_agent/data/generated
```

如需调用大模型合成自然语言请求，可参考：

```bash
cp try_ticket_agent/config_synthesis.example.yaml try_ticket_agent/config_synthesis.yaml
DEEPSEEK_API_KEY=... python try_ticket_agent/scripts/synthesize_dataset.py --config try_ticket_agent/config_synthesis.yaml
```

### 2. 运行 ticket AgentFlow baseline

```bash
bash try_ticket_agent/baseline/run_agentflow_baseline.sh
```

可通过环境变量覆盖模型路径、数据文件、输出目录、temperature、think mode 等参数。

### 3. 运行 ticket GSPO 训练

通常先启动 vLLM 服务，再运行训练脚本：

```bash
bash try_ticket_agent/flowgrpo_general_2x40g/run_vllm_gpu1.sh
bash try_ticket_agent/flowgrpo_general_2x40g/run_train_general_2x40g.sh
```

训练配置位于：

```text
try_ticket_agent/flowgrpo_general_2x40g/config_train_general_2x40g.yaml
```

### 4. 评估 adapter

```bash
bash try_ticket_agent/flowgrpo_general_2x40g/run_eval_ticket_agent.sh
```

### 5. 运行测试

```bash
python -m unittest discover try_ticket_agent/tests
python -m unittest discover try_gsm8k_0522/tests
```

## 数据来源

本项目的数据来源和生成方式见 [DATA_SOURCES.md](DATA_SOURCES.md)。简要说明：GSM8K 实验数据来自 GSM8K 及本地预处理/改写流程；ticket-agent 数据是纯合成沙盒数据，不包含真实客户、订单或工单记录。

## License and Attribution

This repository is based on [lupantech/AgentFlow](https://github.com/lupantech/AgentFlow), which is licensed under the MIT License.

Original AgentFlow copyright:

```text
Copyright (c) 2025 the AgentFlow Team
```

This repository preserves the upstream MIT license notice and adds experimental code for GSM8K and ticket-agent GSPO workflows. Additional modifications are licensed under the same MIT License unless a file states otherwise. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for details.

## 注意事项

- ticket 任务是沙盒环境，不连接真实工单系统，不包含真实客户或订单数据。
- 小模型实验对 prompt 长度、工具 schema 和采样温度较敏感，建议同时查看 rollout details 与 metrics，而不是只看最终准确率。
