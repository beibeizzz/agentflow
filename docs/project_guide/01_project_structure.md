# 01 项目结构

## 1. 目录总览

```text
project_remote_20260904/
├── configs/                    # veRL 配置、任务配置、AgentLoop 注册
├── data/                       # 源数据、veRL Parquet、数据来源清单
├── docker/code-sandbox/        # Coding 隔离执行镜像与容器入口
├── docs/                       # 架构、部署、验证、设计规格和本导读
├── scripts/                    # 数据准备、服务启动、训练、评测、门禁
├── src/agentflow_rl/           # 项目主体 Python 包
│   ├── runtime/                # Action、ToolEvent、Memory、错误语义
│   ├── synthesis/              # Ticket 合成数据客户端与校验管线
│   ├── tasks/                  # 四个任务环境、工具、Prompt、Verifier
│   └── verl/                   # AgentLoop、模型端口、advantage、Trainer
├── tests/                      # 单元、集成、语义一致性测试
├── verification_evidence/      # 本地真实模型 smoke 与 actor update 证据
├── README.md                   # 项目入口
├── REMOTE_HANDOFF.md           # 远程交接与正式启动顺序
└── PACKAGE_MANIFEST.sha256     # 交付副本逐文件哈希
```

## 2. 主体代码

| 路径 | 职责 | 关键对象 |
|---|---|---|
| `src/agentflow_rl/verl/agent_loops/` | 四任务轨迹状态机 | `GSM8KAgentLoop`、`TicketAgentLoop`、`DeepResearchAgentLoop`、`CodingAgentLoop` |
| `src/agentflow_rl/verl/ports.py` | 训练 Planner 与冻结角色的模型端口 | `generate_planner_turn`、`AsyncFrozenModel` |
| `src/agentflow_rl/verl/advantage.py` | 按问题分组、按完整轨迹归一化、向 Planner 各轮传播 advantage | `normalize_trajectory_turns` |
| `src/agentflow_rl/verl/trainer.py` | veRL Trainer 的语义适配 | `AgentFlowPPOTrainer` |
| `src/agentflow_rl/verl/main.py` | Hydra 配置合并、Ray 资源映射、训练入口 | `build_task_runner`、`run`、`validate` |
| `src/agentflow_rl/runtime/` | 跨任务协议 | `ToolAction`、`ToolEvent`、`MemoryStore` |
| `src/agentflow_rl/tasks/<task>/` | 数据模式、Prompt、工具环境和终局验证 | 各任务 `schemas`、`tools`、`verifier` |

## 3. 配置层

`configs/agent_loops.yaml` 将任务名注册到具体 AgentLoop。每个任务目录保存 baseline、train、eval 配置，DeepResearch 和 Coding 额外保存 preflight 配置，GSM8K 额外保存单卡真实模型 smoke 配置。

配置加载路线：

```text
veRL ppo_trainer 默认配置
  + configs/<task>/<mode>.yaml
  + 命令行 --override
  -> OmegaConf 最终配置
  -> veRL validate_config
  -> Ray TaskRunner
```

任务配置负责声明模型路径、LoRA、采样、GSPO、数据、日志、AgentLoop 和环境参数。Shell 脚本负责声明 GPU 可见性、服务地址、输出目录和实验阶段。

## 4. 数据与产物

| 类型 | 路径 | 生命周期 |
|---|---|---|
| 可审计源数据 | `data/gsm8k/`、`data/ticket/` | 随交付副本保存 |
| 远程下载数据 | `data/raw/` | 远程数据准备阶段生成 |
| 任务标准数据 | `data/deepresearch/`、`data/coding/` | 标准化脚本生成 |
| 检索索引 | `data/indexes/` | Lucene 构建脚本生成 |
| veRL 输入 | `data/verl/<task>/*.parquet` | `prepare_verl_data.py` 生成 |
| 轨迹与指标 | `outputs/<task>/<mode>/` | rollout、validation、metrics |
| 模型产物 | `outputs/**/global_step_*`、导出的 PEFT adapter | 训练和阶段切换生成 |

`data/README.md` 记录数据来源、行数、哈希、拆分和过滤规则。`data/verl/manifest.json` 记录 Parquet 行数与哈希。根目录清单覆盖交付文件完整性。

## 5. 测试分层

| 测试层 | 覆盖内容 |
|---|---|
| `tests/unit/` | Action 解析、Memory、工具环境、Verifier、advantage、Trainer 辅助函数 |
| `tests/integration/` | AgentLoop 到 veRL 输出协议、真实 Planner turn 展开语义 |
| `tests/parity/` | Ticket 环境和历史任务语义一致性 |
| `verification_evidence/` | Qwen3-0.6B、vLLM、veRL、GSPO actor update 的动态证据 |
| `scripts/remote/check_*.py` | 搜索索引、Docker、冻结模型、preflight、veRL 安装门禁 |

## 6. 远程运行的有效入口

1. `scripts/remote/audit_environment.sh`：软硬件与 17 份配置总门禁。
2. `scripts/serve_frozen_vllm.sh`：GPU 0 冻结角色服务。
3. `scripts/run_<task>_baseline.sh`：训练前基线。
4. `scripts/run_<task>_preflight.sh` 或 smoke：小规模闭环验证。
5. `scripts/run_<task>_train.sh`：正式训练。
6. `scripts/run_<task>_eval.sh`：独立测试集评测。

`docs/plans/` 与 `docs/specs/` 保存设计背景和实现契约；正式执行以当前源码、当前 YAML 和 `REMOTE_HANDOFF.md` 为准。

