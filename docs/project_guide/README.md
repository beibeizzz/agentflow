# AgentFlow RL v3 项目导读

本目录提供远程运行副本的系统化说明，内容以当前源码、正式配置和已完成验证为准。

## 阅读路线

| 文档 | 主题 | 适合读者 |
|---|---|---|
| [01 项目结构](01_project_structure.md) | 目录职责、核心入口、有效资产 | 接手工程的开发者 |
| [02 框架主线](02_architecture_mainline.md) | AgentFlow 状态机、训练闭环、奖励与更新层级 | 算法与系统开发者 |
| [03 四任务流程](03_four_task_flows.md) | GSM8K、Ticket、DeepResearch、Coding | 任务环境开发者 |
| [04 运行时技术栈](04_runtime_technology_stack.md) | Python、vLLM、veRL、FSDP、Ray、TransferQueue | 训练平台开发者 |
| [05 数据、存储、沙箱与通信](05_data_storage_sandbox_communication.md) | 数据流水线、Lucene、Docker、进程间协议 | 数据与基础设施开发者 |
| [06 软硬件执行流](06_software_hardware_flow.md) | 双 A800 拓扑、进程生命周期、显存与磁盘流 | 远程实验执行者 |
| [07 大模型后训练理论](07_llm_post_training_theory.md) | GRPO/GSPO、LoRA、FSDP、策略新鲜度与指标 | 后训练算法开发者 |
| [08 Agentic RL 前沿](08_agentic_rl_frontier.md) | harness、credit assignment、工具学习与研究边界 | 研究与算法开发者 |
| [09 DeepResearch 端到端](09_deepresearch_end_to_end.md) | 一条搜索轨迹从 Parquet、Lucene 到 GSPO 更新的逐函数调用链 | 搜索 Agent 与训练开发者 |
| [10 Coding 端到端](10_coding_end_to_end.md) | 一条代码轨迹从公开测试、Docker 到隐藏奖励和 GSPO 更新 | Coding Agent 与训练开发者 |

## 内容状态

- **已实现**：代码路径、配置和测试共同支持的能力。
- **远程门禁**：需要在双 A800 Linux 主机上完成的动态验证。
- **研究方向**：具备明确接口和理论依据的扩展项。

正式启动顺序见根目录 [REMOTE_HANDOFF.md](../../REMOTE_HANDOFF.md)，具体命令见 [alpha_deployment.md](../alpha_deployment.md)，本地与远程验证证据见 [alpha_verification.md](../alpha_verification.md)。
