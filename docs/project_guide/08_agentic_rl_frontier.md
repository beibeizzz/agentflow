# 08 Agentic RL 前沿

本文档按 2026 年 9 月可获得的一手论文与官方框架资料定位项目。

## 1. 从单模型策略到模块化 Agent 系统优化

[AgentFlow](https://arxiv.org/abs/2510.05592) 将 Planner、Executor、Verifier、Generator 通过 evolving memory 组成多轮系统，并在真实交互 loop 内直接优化 Planner。其 Flow-GRPO 使用终局可验证结果对多轮 Planner 决策进行 credit assignment。

本项目继承模块化角色、共享 Memory、Planner-only optimization 和终局奖励主线，并完成两项工程适配：

- 每个 Planner turn 形成独立 veRL row；
- trajectory advantage 传播到各 turn 后，使用 veRL 原生 sequence-level GSPO 更新。

研究问题从“训练一个会交替思考和调用工具的单模型”扩展为“训练 Agent 系统中的可控决策模块”。

## 2. Harnessed Agentic RL

[Agent Lightning v1.0](https://arxiv.org/abs/2608.17528) 将近期范式概括为 harnessed agentic RL：部署时 harness 管理工具、上下文和控制流，trainer 观察 LLM request-response 片段并完成训练。论文强调 retokenization、sample merging、advantage、loss normalization 和 backend scheduling 会直接影响训练稳定性。

本项目具备典型 harness 特征：

```text
AgentFlow harness owns:
  role loop + memory + tools + environment + terminal verification

veRL trainer owns:
  rollout workers + token data + advantage fields + GSPO + optimizer + checkpoint
```

项目通过精确 token IDs、response mask、session/turn identity、TransferQueue 和基础设施有效性标记连接两侧。这个接口也是后续接入新任务和新 trainer 的主要扩展点。

[Agent Lightning](https://arxiv.org/abs/2508.03680) 进一步展示了训练执行与 agent 运行解耦、MDP 化轨迹和层级 credit assignment 的通用方向。项目当前采用同一主机内的服务化解耦，扩展目标可覆盖远程 rollout service 与独立 learner 集群。

## 3. 长轨迹 credit assignment

终局奖励具有低标注成本和目标一致性，长轨迹会引入归因模糊。当前 trajectory advantage 向全部 Planner turn 传播，使每轮 action 朝完整任务成功方向更新。

前沿研究集中在四类细化方法：

1. **Outcome redistribution**：根据 turn、工具结果或价值估计重新分配终局结果；
2. **Process reward**：对 action 合法性、检索增益、测试修复或状态进展提供中间信号；
3. **Counterfactual credit**：替换或移除某一 turn，估计其对终局结果的边际贡献；
4. **Hierarchical RL**：高层 Planner 学子目标，低层 Executor 学动作执行，分别建立 advantage。

当前实现适合作为 outcome-only 基线。每个 turn 的独立 row、Memory snapshot 和 ToolEvent 已提供过程奖励与反事实实验需要的数据接口。

## 4. 从 token ratio 到 sequence ratio

[GSPO](https://arxiv.org/abs/2507.18071) 使用长度归一化 sequence likelihood ratio 和 sequence-level clipping，目标单位与完整语言 action 对齐。Agent 场景中的一次 Planner 输出通常包含子目标、工具名和参数，完整 JSON 共同决定 action 语义，因此 turn-level sequence objective 与 Planner action 边界具有直接对应关系。

后续研究重点包括：

- 不同长度 action 的归一化与 clip 灵敏度；
- 多轮 turn 数变化对 loss weighting 的影响；
- rollout engine 与 actor engine 的 logprob mismatch；
- 多 actor epochs 和异步 rollout 带来的策略陈旧度。

项目已经记录精确 token、rollout logprob、actor old logprob 和 turn identity，可直接支持这些诊断。

## 5. 可验证工具学习

[ReTool](https://arxiv.org/abs/2504.11536) 展示了在线代码执行与 outcome reward 对工具选择、自我修正和数学推理的促进作用。[Search-R1](https://arxiv.org/abs/2503.09516) 展示了多轮搜索交互、检索 token 处理和 outcome reward 驱动的搜索策略学习。

四任务构成逐步增强的可验证环境谱系：

| 环境 | 状态空间 | 工具反馈 | 奖励密度 |
|---|---|---|---|
| GSM8K | 短数学状态 | 精确计算结果 | 二元 |
| Ticket | 结构化可变状态 | 查询、更新、finish | 二元 |
| DeepResearch | 大规模外部语料 | 排名结果、文档句子 | 连续 joint F1 |
| Coding | 程序与测试状态 | pass/failure、错误信息 | 连续 hidden pass rate |

该谱系支持研究 action space、horizon、tool latency、reward sparsity 和环境复杂度对 Agentic RL 的影响。

## 6. DeepResearch RL 的最新经验

[How to Train Your Deep Research Agent?](https://arxiv.org/abs/2602.19526) 从 prompt、reward 和 policy optimization 三个维度分析 Search-R1。其结果指出模板、F1/EM 奖励形态、action penalty 和优化算法都会显著影响稳定性，并观察到部分设置中的 answer avoidance 和训练退化。

这组发现对本项目形成四个直接实验：

1. 对比 fast-thinking/no-think 与显式长思考模板；
2. 对比 joint F1、joint EM 和带 action-cost 的组合奖励；
3. 按课程阶段监控搜索次数、Read 覆盖和答案回避；
4. 对比 GSPO、REINFORCE-style objective 与 outcome-only baseline。

当前配置的 no-think、最大 5 轮、joint F1、检索限流和完整 ToolEvent 日志提供可复现起点。

## 7. 系统与算法协同

[HybridFlow/veRL](https://arxiv.org/abs/2409.19256) 将 RLHF 表述为多模型分布式计算图，并通过分层 API 解耦计算与数据依赖。Agentic RL 进一步加入外部服务、工具环境、变长轨迹和动态 turn 数，系统行为会改变实际采样分布。

关键协同问题：

- rollout throughput 与策略新鲜度共同决定训练有效性；
- 工具并发和超时策略会改变 trajectory validity；
- prompt 截断与 Memory projection 会改变 agent 可见状态；
- hidden tests、检索索引和环境版本决定 reward 的可复现性；
- checkpoint 到 rollout engine 的同步决定 on-policy 程度。

项目使用 Ray、vLLM、TransferQueue、FSDP、Lucene 和 Docker 建立显式边界，并用 preflight 将系统门禁和算法门禁合并。

## 8. 当前落地与研究路线

| 方向 | 当前落地 | 下一组实验 |
|---|---|---|
| 模块化系统优化 | Planner-only，冻结其余角色 | Planner/Verifier 交替优化 |
| Credit assignment | trajectory advantage 传播到全部 turn | 基于 verifier/tool delta 的 turn 权重 |
| 策略目标 | turn-level GSPO | GSPO 与 REINFORCE/GRPO 消融 |
| 策略新鲜度 | 同步 rollout/update/weight sync | 异步队列 staleness 分桶与校正 |
| Memory | append-only log + bounded view | learned memory selection 与压缩 |
| 奖励 | 终局规则奖励 | 过程奖励、成本奖励、多目标 Pareto |
| 任务泛化 | 四任务独立 checkpoint | 共享 Planner、多任务 curriculum |
| 工具安全 | schema、deadline、Docker/Lucene gate | adversarial tool input 与 reward hacking 测试 |

最优先的研究顺序是：完成四任务 preflight，建立 outcome-only GSPO 基线，开展 DeepResearch reward/optimizer 消融，再引入 turn-level process credit。这个顺序保持每次结论都由可复现对照实验支撑。

## 9. 主要参考来源

- [In-the-Flow Agentic System Optimization for Effective Planning and Tool Use](https://arxiv.org/abs/2510.05592)
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [Agent Lightning](https://arxiv.org/abs/2508.03680)
- [Agent Lightning v1.0: Towards Harnessed Agentic RL](https://arxiv.org/abs/2608.17528)
- [Search-R1](https://arxiv.org/abs/2503.09516)
- [How to Train Your Deep Research Agent?](https://arxiv.org/abs/2602.19526)
- [ReTool](https://arxiv.org/abs/2504.11536)
- [HybridFlow / veRL](https://arxiv.org/abs/2409.19256)
- [vLLM documentation](https://docs.vllm.ai/)
- [Ray documentation](https://docs.ray.io/)
- [PyTorch FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html)

