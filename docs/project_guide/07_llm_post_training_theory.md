# 07 大模型后训练理论

## 1. 项目采用的后训练范式

项目从具备基础指令遵循和 agent action 能力的 Qwen checkpoint 直接进入在线强化学习。每次训练迭代使用当前 Planner 与真实工具环境生成新轨迹，确定性终局 Verifier 计算奖励，veRL 随后更新 Planner LoRA。

```text
current policy
  -> online multi-turn rollout
  -> terminal verifiable reward
  -> query-group relative advantage
  -> turn-level sequence objective
  -> LoRA optimizer step
  -> rollout weight synchronization
```

SFT、过程奖励和多角色联合训练属于可控对比或后续扩展。当前主线集中验证终局结果对系统内特定模块的优化能力。

## 2. GRPO 的组内相对 advantage

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 提出的 GRPO 使用同一问题的多条采样结果估计相对 advantage，从而省去独立 value model。项目沿用这项分组思想，并将样本单位定义为完整 AgentFlow session。

对问题 `q` 采样 `K` 条有效轨迹，终局奖励为 `R_1 ... R_K`：

```math
\mu_q = \frac{1}{K}\sum_{i=1}^{K} R_i
```

```math
\sigma_q = \sqrt{\frac{1}{K}\sum_{i=1}^{K}(R_i-\mu_q)^2}
```

```math
A_i = \frac{R_i-\mu_q}{\sigma_q}
```

代码使用 population standard deviation。有效轨迹少于 2 条或 `sigma < 1e-6` 时，该问题组提供零 advantage。二元奖励任务需要同组同时出现成功和失败轨迹，才能形成有效相对信号。

## 3. 奖励、advantage、turn 与 token

项目的数据层级如下：

| 层级 | 数值 | 作用 |
|---|---|---|
| 完整 session | 一个终局 reward | 衡量系统最终任务效果 |
| query group | reward mean/std | 形成同题相对基线 |
| 完整 session | 一个标准化 advantage | 表示该轨迹相对同组的质量 |
| Planner turn | 继承所属 session advantage | 训练每一轮 Planner action |
| Planner response token | 由 response mask 覆盖 | 参与该 turn 的序列 log-likelihood 与梯度 |

终局 reward 在 session 最后一行读取一次。`normalize_trajectory_turns` 计算 trajectory advantage，再把同一 advantage 附加到该 session 的每个真实 Planner turn。Trainer 将标量 advantage 乘 response mask 形成 veRL 输入，GSPO 使用整段 Planner action 的序列概率比进行优化。

这套机制可称为 **trajectory-level outcome、trajectory-level advantage、turn-level GSPO、token-masked gradient**。

## 4. GSPO 序列目标

[GSPO](https://arxiv.org/abs/2507.18071) 将 importance ratio 定义在序列级，并执行序列级 clipping、rewarding 和 optimization。项目将一个 Planner turn 视为一个 action sequence。

对 Planner 第 `t` 轮生成序列 `y_{i,t}`，长度为 `L_{i,t}`，序列概率比写为：

```math
\rho_{i,t}(\theta)=\exp\left(
\frac{1}{L_{i,t}}\sum_{j=1}^{L_{i,t}}
\left[\log\pi_\theta(y_j\mid x,y_{<j})-
\log\pi_{old}(y_j\mid x,y_{<j})\right]
\right)
```

裁剪目标可概括为：

```math
L_{GSPO}=-\mathbb{E}_{i,t}\left[
\min\left(
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon_{low},1+\epsilon_{high})A_i
\right)
\right]
```

GSM8K/Ticket 的 `epsilon_low/epsilon_high` 为 `0.001/0.003`，DeepResearch/Coding 为 `0.0003/0.0004`。长度归一化降低长 action 因 token 数量获得更大概率比尺度的影响，序列裁剪直接约束完整 Planner action 的更新幅度。

## 5. 策略新鲜度与 on-policy 链路

项目通过以下闭环维护策略新鲜度：

1. Planner rollout 由 veRL 管理的当前 actor vLLM engine 生成；
2. prompt IDs、response IDs 和 rollout logprob 原样保存；
3. actor update 前对同一 token IDs 重算 proximal `old_log_probs`；
4. GSPO 在固定 rollout batch 上执行配置的 actor epochs；
5. optimizer step 后将更新权重同步到 Planner rollout engine；
6. 下一批问题由更新后的 Planner 重新 rollout。

策略新鲜度可通过四类证据判断：rollout/actor logprob correlation、importance ratio 分布、old-logprob 行数与 trainable-turn 数一致性、权重同步后的下一批行为变化。PPO epochs 增加会扩大同一批数据的复用程度，clip ratio 负责约束批内策略漂移。

## 6. KL 系数为 0

当前实验使用 `use_kl_in_reward=false`、`use_kl_loss=false` 和 entropy coefficient 0。该设置把优化信号集中在可验证终局结果与 GSPO clipping，并降低 reference model 的额外显存和计算开销。LoRA、窄 clip、短训练阶段、独立 validation 和轨迹审计共同约束策略变化。

训练中持续观察 response length、格式有效率、工具调用率、reward、importance ratio、gradient norm 和跨任务基线。出现语言质量、格式稳定性或通用能力下降时，可依次增加 reference KL、混入通用任务、缩小学习率与 clip、提前停止或采用多目标 reward。

## 7. LoRA 与优化对象

LoRA 将低秩矩阵注入 Planner 的线性层，参数更新可写为：

```math
W' = W + \frac{\alpha}{r}BA
```

项目使用 `r=64`、`alpha=128` 和 all-linear target。基础模型权重保持稳定，optimizer 主要保存 adapter 参数状态。该选择降低单卡训练显存、缩小 checkpoint、支持 DeepResearch 课程阶段间快速导出和加载。

Planner-only 优化提供清晰的因果归因：系统环境、冻结角色和终局规则保持固定，性能变化主要来自 Planner 决策分布变化。

## 8. FSDP、micro-batch 与变长 turn

FSDP 提供模型参数、梯度和优化器状态的分片接口。当前 GPU 1 单卡运行保持标准 veRL FSDP worker 和 checkpoint 语义，后续多卡训练可扩展到真实分片。

一批 prompt 展开后，各 session 的 Planner turn 数可能不同。项目先过滤有效 turn，再选择小于等于配置上限且整除实际 turn 数的最大 mini-batch。每 GPU micro-batch 为 1，梯度在有效 turn 上累积。该策略保持 GSPO 分母与真实训练行数一致，并控制长 prompt/response 的峰值显存。

## 9. 终局奖励的性质

| 任务 | reward 类型 | 优点 | 主要风险 |
|---|---|---|---|
| GSM8K | 数值 exact-style 二元奖励 | 客观、低成本 | 同组零方差、等价表达提取 |
| Ticket | 状态与流程联合二元奖励 | 覆盖副作用和 finish | 稀疏、indirect 链路更难 |
| DeepResearch | answer/supporting-fact joint F1 | 连续、引用可验证 | 部分匹配信号与目标偏差 |
| Coding | hidden-test pass rate | 连续、功能导向 | 测试覆盖决定 reward 完整性 |

模型行为错误作为有效低奖励样本，基础设施错误退出训练统计。这个划分保护算法信号，同时保留真实失败轨迹用于诊断。

## 10. 训练指标

核心指标分为五组：

- **任务结果**：reward、success、EM/F1、hidden pass rate、direct/indirect 正确率；
- **探索质量**：组内 reward std、zero-variance group fraction、skipped group fraction；
- **轨迹健康**：有效轨迹率、平均轮数、工具调用率、终止原因；
- **优化健康**：loss、gradient norm、importance ratio、old-logprob 行数、trainable-turn 数；
- **系统效率**：rollout latency、tool latency、tokens/s、GPU memory、checkpoint 时间。

preflight 以“真实工具调用 + reward 方差 + 有限梯度 + checkpoint + 权重同步”作为进入正式训练的联合条件。

