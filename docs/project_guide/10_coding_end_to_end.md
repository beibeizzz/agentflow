# 10 Coding 一次 AgentFlow 推理与优化全流程

## 1. 场景与运行边界

本文跟踪一个已下载到本地的 TACO-Verified Easy/Medium 样本，从标准化和 Parquet 转换开始，经过一次典型“分析、写代码、跑公开测试、检查错误、修复、提交”AgentFlow rollout，再进入隐藏测试 reward、组内 advantage、turn-level GSPO、LoRA 更新和 vLLM 权重同步。

标准样本结构：

```json
{
  "episode_id": "coding-example-id",
  "question": "编程题面、输入输出和约束",
  "difficulty": "MEDIUM",
  "public_tests": [
    {"stdin": "...", "expected_stdout": "..."}
  ],
  "hidden_tests": [
    {"stdin": "...", "expected_stdout": "..."}
  ],
  "starter_code": "",
  "source": "taco-verified",
  "metadata": {
    "fingerprint": "...",
    "question_fingerprint": "..."
  }
}
```

一次正式训练 step 读取 4 个 prompt，每个 prompt 采样 6 条 session，共 24 条完整 Coding 轨迹。每条轨迹最多包含 5 个 Planner turn。Planner 使用 Qwen3-4B LoRA；Query Analyzer、Executor、Verifier、Generator 和 Base Generator 共享 GPU 0 上的 Qwen3-8B；候选代码在 CPU 侧 Docker 容器中执行。

## 2. 代码地图

| 阶段 | 文件 | 类或函数 | 具体职责 |
|---|---|---|---|
| TACO 解析 | [`tasks/coding/dataset.py`](../../src/agentflow_rl/tasks/coding/dataset.py) | `parse_taco_tests`、`standardize_taco_row` | 解析 stdio/call-based tests，筛选 Easy/Medium |
| 去重与拆分 | 同上 | `problem_fingerprint`、`question_fingerprint`、`split_verified_rows` | 内容身份、跨 split 审计、稳定 80/10/10 |
| 测试拆分 | [`tasks/coding/schemas.py`](../../src/agentflow_rl/tasks/coding/schemas.py) | `split_tests` | 按样本 ID 稳定拆成 public/hidden tests |
| Parquet 转换 | [`verl/data.py`](../../src/agentflow_rl/verl/data.py) | `coding_to_verl_row`、`convert_file` | 构建 veRL row 与 manifest |
| 主状态机 | [`agent_loops/coding.py`](../../src/agentflow_rl/verl/agent_loops/coding.py) | `CodingAgentLoop.run` | 驱动角色、代码状态、公开/隐藏测试和 reward |
| 代码环境 | [`tasks/coding/tools.py`](../../src/agentflow_rl/tasks/coding/tools.py) | `CodingEnvironment.execute` | Write、Run Tests、Inspect Error、Finish |
| 宿主沙箱接口 | [`tasks/coding/sandbox.py`](../../src/agentflow_rl/tasks/coding/sandbox.py) | `DockerSandbox.run` | 启动受限容器并解析测试结果 |
| 容器执行器 | [`docker/code-sandbox/runner.py`](../../docker/code-sandbox/runner.py) | `main`、`run_case`、`run_candidate` | 降权执行 stdio 或函数测试 |
| 终局评测 | [`tasks/coding/verifier.py`](../../src/agentflow_rl/tasks/coding/verifier.py) | `evaluate_code` | 隐藏测试通过率与失败码 |
| 共享模型端口 | [`verl/ports.py`](../../src/agentflow_rl/verl/ports.py) | `generate_planner_turn`、`AsyncFrozenModel.generate` | Planner token 调用与冻结 HTTP 调用 |
| 共享 Memory | [`runtime/memory.py`](../../src/agentflow_rl/runtime/memory.py) | `MemoryStore.add/project` | 题面、代码、测试反馈与 judgement |
| Advantage | [`verl/advantage.py`](../../src/agentflow_rl/verl/advantage.py) | `normalize_trajectory_turns` | 同题完整轨迹标准化与 turn 传播 |
| Trainer | [`verl/trainer.py`](../../src/agentflow_rl/verl/trainer.py) | `AgentFlowPPOTrainer` | old logprob、GSPO 输入与 actor update |
| 启动脚本 | [`run_coding_train.sh`](../../scripts/run_coding_train.sh) | shell 主流程 | 冻结服务检查、镜像构建、沙箱门禁和训练 |

## 3. 本地 TACO 数据进入训练集

### 3.1 标准化

假设本地源文件位于：

```text
data/raw/taco_verified_train.jsonl
```

`scripts/prepare_coding_data.py` 对每条记录调用 `prepare_split()` 和 `standardize_taco_row()`：

1. 保留 `EASY`、`MEDIUM`；
2. 根据题面模式过滤 interactive、special judge、图像依赖等任务；
3. `parse_taco_tests()` 将 `input_output` 转成 `CodeTest`；
4. 支持 stdio case 和 `fn_name + args + expected` 的 call-based case；
5. 保留至少两条测试的题目；
6. `split_tests()` 按 `sha256(episode_id:test_index)` 排序，默认取约 20% 为 public tests，其余为 hidden tests；
7. `problem_fingerprint()` 绑定规范化题面与测试；
8. `question_fingerprint()` 绑定规范化题面，用于跨 split 去重。

`split_verified_rows()` 使用 seed 42 和 question fingerprint 稳定形成 80/10/10 train/validation/test。`deterministic_limit()` 依据 episode ID 哈希应用可复现子集上限。最终生成：

```text
data/coding/train.jsonl
data/coding/validation.jsonl
data/coding/test.jsonl
data/coding/easy.jsonl
data/coding/medium.jsonl
data/coding/preflight.jsonl
data/coding/manifest.json
```

### 3.2 Pydantic 数据契约

`CodeExample.from_row()` 校验 episode ID、题面、难度、public tests 和 hidden tests。`CodeTest.model_post_init()` 要求每条测试精确属于一种接口：

```text
stdio: stdin + expected_stdout
call-based: fn_name + args + expected
```

这个数据契约将测试接口错误提前到数据准备阶段，容器运行阶段只处理已标准化 case。

### 3.3 转为 veRL Parquet

`scripts/prepare_verl_data.py --task coding` 调用 `coding_to_verl_row()`：

```python
{
    "data_source": "coding",
    "prompt": [{"role": "user", "content": question}],
    "ability": "coding",
    "reward_model": {"style": "rule", "ground_truth": "hidden_tests"},
    "extra_info": complete_code_example,
    "agent_name": "agentflow_coding",
}
```

PyArrow `write_parquet()` 使用 Zstd 压缩，并把文件 SHA-256 写入 `data/verl/manifest.json`。hidden tests 存在于 host 侧 `extra_info`，Agent prompt 只使用题面、starter code、Memory 和 public-test feedback。

## 4. 服务、沙箱和训练进程

### 4.1 GPU 0 冻结 vLLM

`serve_frozen_vllm.sh` 加载 Qwen3-8B，通过 OpenAI-compatible HTTP 服务角色请求。`AsyncFrozenModel.generate()` 使用 `AsyncOpenAI.chat.completions.create()`，temperature 0 和 Qwen no-think 模式保证冻结角色输出稳定。

### 4.2 Coding Docker image

`run_coding_train.sh` 先执行：

```bash
bash scripts/build_code_sandbox.sh
python scripts/remote/check_code_sandbox.py
```

镜像 `agentflow-python-sandbox:3.11` 基于 `python:3.11-slim`，入口为 `/opt/runner.py`。门禁覆盖 `Solution` 类、stdio case、请求隔离和输出上限。

### 4.3 GPU 1 veRL 训练进程

训练入口：

```bash
CUDA_VISIBLE_DEVICES=1 python -m agentflow_rl.verl.main \
  --config configs/coding/train.yaml
```

`main()` 依次调用 `load_config()`、`run()`、veRL `run_ppo()` 和 Ray remote `AgentFlowTaskRunner.run()`。`AgentFlowPPOTrainer.init_workers()` 初始化 FSDP actor、Planner vLLM、AgentLoop workers、TransferQueue 和 checkpoint manager。

### 4.4 正式参数

| 参数 | 值 | 具体含义 |
|---|---:|---|
| prompt batch | 4 | 每 step 四道题 |
| group size | 6 | 每题六条完整 Coding session |
| AgentLoop workers | 4 | Ray 侧轨迹 worker 数 |
| sandbox concurrency | 1 | 每个 AgentLoop 实例的容器 semaphore 上限 |
| max steps | 5 | 单 session Planner action 上限 |
| trajectory deadline | 300 s | 角色、工具和终局测试共享总时限 |
| test suite timeout | 10 s | 每次 public/hidden suite 的总预算 |
| prompt/response | 4096/1024 | 模型输入与单角色输出上限 |
| rollout sampling | temp 1.0, top-p 1.0 | 同题 session 探索 |
| turn mini-batch | 8 上限 | 动态 Planner-turn optimizer batch |
| PPO epochs | 1 | 每批 rollout turn 使用一次 |
| learning rate | `1e-6` | LoRA 更新步长 |
| GSPO clip | `3e-4/4e-4` | sequence ratio 非对称裁剪 |

## 5. 一个 prompt group 如何并发运行

固定 veRL 的 `AgentLoopManagerTQ.generate_sequences()` 把 4 个 prompt 分块发送给 4 个 Ray `AgentLoopWorkerTQ`。每个 worker 的 `generate_sequences()` 使用 `asyncio.create_task()` 启动 prompt，`_run_prompt()` 再按 `rollout.n=6` 为每题启动 session 0 到 5，并通过 `asyncio.gather()` 等待同题 session 完成。

每条 session 创建自己的：

```python
example = CodeExample.from_row(extra_info)
environment = CodingEnvironment(example, DockerSandbox(...))
memory = MemoryStore()
events = []
outputs = []
```

Qwen3-8B 服务、Qwen3-4B Planner vLLM 和 Docker Engine 在 session 间共享。`CodingEnvironment.code`、`last_result`、Memory 和 ToolEvent 保持 session 隔离。

## 6. 一条典型五轮 Coding 轨迹

### 6.1 初始化题面和 starter code

`CodingAgentLoop.run()` 把 question 写入 `MemoryStore`。starter code 存在时，以带 `identity` tag 的条目写入。`_view()` 调用 `bounded_memory_text()`，在 4,096 token prompt 上限内保留身份信息、近期工具事件和 Verifier judgement。

### 6.2 Query Analyzer 提取算法约束

`query_prompt()` 组合题面与 starter code，`frozen_generate()` 通过 GPU 0 Qwen3-8B 生成接口、约束、边界和候选算法分析。分析以 `role="query_analyzer"` 写入 Memory，并在每轮 Planner prompt 中保持可见。

### 6.3 Turn 0：Planner 请求初始实现

`planner_prompt()` 组合 problem 和 Memory view。`planner_generate()` 调用共享 `generate_planner_turn()`：

1. `tokenizer.apply_chat_template(..., enable_thinking=False)` 生成精确 prompt IDs；
2. veRL `server_manager.generate()` 调用 GPU 1 Planner vLLM；
3. 保存 response token IDs 和逐 token logprob；
4. `planner_output()` 创建 `AgentLoopOutput`，Planner response mask 全部为 1。

Planner 可先选择 `Base_Generator_Tool` 获取实现建议，也可直接选择 `Code_Write_Tool`。典型 Write action：

```json
{
  "sub_goal": "实现满足约束的完整 Python 解法",
  "tool_name": "Code_Write_Tool",
  "arguments": {"code": "import sys\n..."}
}
```

`CodeAction.parse()` 通过 `strict_json_object()` 和 Pydantic 校验工具名、sub-goal 和 arguments。

### 6.4 Executor 保持工具 ownership

冻结 Executor 接收 proposed action 与当前 Memory。输出再次进入 `CodeAction.parse()`，并执行：

```python
if action.tool_name != proposed.tool_name:
    raise ActionParseError("executor changed the Planner tool selection")
```

Planner 对工具选择承担训练责任，Executor 对参数和完整代码进行规范化。`Code_Write_Tool` 进入 `CodingEnvironment.execute()`，把 `environment.code` 更新为完整代码，并返回字符数。

### 6.5 Turn 1：运行公开测试

Planner 选择：

```json
{
  "sub_goal": "运行公开测试并定位当前实现问题",
  "tool_name": "Code_Run_Tests_Tool",
  "arguments": {}
}
```

`CodingEnvironment.execute()` 调用：

```python
self.last_result = self.sandbox.run(
    self.code,
    self.example.public_tests,
    timeout_s=self.test_timeout_s,
)
```

AgentLoop 在 `sandbox_semaphore` 内使用 `run_blocking()` 和 `asyncio.to_thread()` 执行这段同步逻辑，保持 event loop 可继续处理其他 session。

### 6.6 DockerSandbox 的宿主调用

`DockerSandbox.run()` 将 code、public tests 和 timeout 序列化为 JSON，通过 stdin 发送给一次性容器：

```text
docker run --rm
  --network none
  --read-only
  --cap-drop ALL
  --security-opt no-new-privileges
  --cpus 2
  --memory 4g --memory-swap 4g
  --pids-limit 128
  --ulimit nofile=64:64
  --tmpfs /tmp:rw,noexec,nosuid,size=256m
  -i agentflow-python-sandbox:3.11
```

Docker Engine 提供文件系统、网络、进程、CPU 和内存隔离。宿主 `subprocess.run()` 使用 `timeout_s + 5` 约束容器总生命周期，并捕获 stdout/stderr。

### 6.7 容器 runner 执行测试

`docker/code-sandbox/runner.py:main()` 从 stdin 读取请求，在 `/tmp/case-*` 创建只读 `solution.py`。每个 case 调用 `run_case()`：

- stdio：`python -I solution.py`，向 stdin 写入测试输入；
- call-based：生成 `function_case.py`，动态加载模块，查找顶层函数或 `Solution().method`，再用 JSON 传参与比较输出。

`run_candidate()` 通过 `preexec_fn=drop_candidate_privileges` 将候选进程切换到 UID/GID 65534，清空 supplementary groups，并设置 1 MB 文件大小上限。stdout/stderr 各截取最多 1 MB。

`stdout_matches()` 先比较规范化文本，再按 `rel_tol=1e-6`、`abs_tol=1e-6` 支持浮点 token 比较。`function_output_matches()` 支持 tuple/list 和包装 expected 结构。

容器 stdout 返回：

```json
{
  "passed": 2,
  "total": 3,
  "failures": ["test_2: ..."],
  "timed_out": false
}
```

宿主 `DockerSandbox.run()` 将其解析为 `TestRunResult`，`CodingEnvironment` 保存为 `last_result`。

### 6.8 ToolEvent、Memory 和 Verifier

AgentLoop 把工具名、arguments、测试结果和 `ok` 封装成 `ToolEvent`，再用 `MemoryStore.add()` 追加到完整轨迹。

冻结 Verifier 根据题面、当前代码和测试 Memory 输出 `Conclusion: STOP/CONTINUE`。`extract_conclusion()` 解析最后一个结论并写入当前 turn metadata。

public test failure 属于可学习环境反馈，Planner 在下一轮可见失败摘要。Docker daemon、镜像或容器协议异常通过 `RuntimeError` 标记为基础设施无效轨迹。

### 6.9 Turn 2：检查错误

Planner 可选择 `Code_Inspect_Error_Tool`。`CodingEnvironment.execute()` 从 `last_result.failures` 返回最近失败，避免重复运行测试。该反馈进入 Memory 后，Planner 能形成针对性修复计划。

### 6.10 Turn 3：写入修复代码

Planner 再次选择 `Code_Write_Tool`，完整替换 `environment.code`。这种 full-code state 更新让每轮 action 都对应明确的程序版本，轨迹日志可重建代码演化过程。

### 6.11 Turn 4：Finish

Planner 选择 `Code_Finish_Tool`，或 Verifier 在代码准备完成后输出 STOP。循环也会在第 5 轮或 300 秒 trajectory deadline 到达时结束。

## 7. Generator 与隐藏测试 reward

`generator_prompt()` 接收题面、最终 bounded Memory 和 `environment.code`。冻结 Generator 返回：

```json
{"code": "完整 Python 3 解法"}
```

`FinalCode.parse()` 同时支持严格 JSON object 和单个 Python fenced block。解析后的 `final_code` 进入：

```python
verification, hidden_result = evaluate_code(
    final_code,
    example,
    DockerSandbox(...),
    timeout_s=10.0,
)
```

`evaluate_code()` 只传入 `example.hidden_tests`。隐藏测试在新的受限容器中执行，轨迹中的 Planner、Executor 和 Verifier保持只见 public feedback。

终局定义：

```text
reward  = hidden_passed / hidden_total
success = hidden_total > 0 and hidden_passed == hidden_total
```

`HIDDEN_TEST_FAILURE` 表示部分或全部隐藏 case 失败，`HIDDEN_TEST_TIMEOUT` 表示 suite 超时。连续 pass-rate reward 为同题六条 session 提供更细粒度的相对质量。

## 8. AgentLoopOutput 与 TransferQueue

每次 Planner 调用都产生一个独立 `AgentLoopOutput`，内容包括：

- 精确 prompt IDs 和 response IDs；
- Planner response logprob；
- 全 1 response mask；
- `uid/session_id/turn_index`；
- Planner prompt、Planner response、ToolEvent、Verifier conclusion 和 Memory snapshot。

`finalize_outputs()` 把终局 reward 写入最后一轮，并为所有 turn 填充 `valid_for_training` 与 `terminal_reason`。

固定 veRL 的 `AgentLoopWorkerTQ._agent_loop_postprocess()` 将 final reward 复制到本 session 其他 turn row，组合 input/position/loss masks，以 `{uid}_{session_id}_{turn_index}` 为 key 调用 `tq.async_kv_batch_put()`。

Trainer 的 advantage 代码按 session 最大 turn index 读取一次终局 reward，因此 TransferQueue 的字段复制服务于 row 一致性，同时保持完整轨迹只贡献一次 reward 统计。

## 9. old logprob 与 Coding group advantage

`AgentFlowPPOTrainer._compute_old_log_prob()` 先筛选 `valid_for_training=True` 的 row，再由 GPU 1 FSDP actor对保存 token IDs 重算 old logprob。

`_compute_advantage()` 执行：

```text
TransferQueue rm_scores/response_mask/extra_fields
  -> build_trajectory_turns()
  -> key.rsplit("_", 2) 得到 uid/session/turn
  -> 每个 session 选最大 turn index 的 reward
  -> 同 uid 的 6 条 session 形成 group
  -> population mean/std
  -> A_session = (R_session - mean) / std
  -> A_session 传播到该 session 全部 Planner turns
```

例如六条轨迹的 hidden pass rate 形成多个水平时，完整通过的轨迹得到正 advantage，部分通过或失败轨迹根据同组均值获得较低 advantage。每条轨迹内部的 Write、Run、Inspect、Rewrite、Finish turn 继承同一个 trajectory advantage。

基础设施无效 session 退出 group 统计。有效 session 少于 2 条或 reward std 低于 `1e-6` 时，该题组 actor update 跳过。

## 10. turn-level GSPO 与 LoRA backward

`AgentFlowPPOTrainer._update_actor()`：

1. 选择 `normalize_trajectory_turns()` 产生的 trainable keys；
2. `build_unpadded_attention_mask()` 适配变长 prompt/response；
3. `effective_turn_mini_batch_size()` 选取小于等于 8 且整除实际 turn 数的最大值；
4. `actor_update_metadata()` 设置 1 epoch、temperature 1.0、shuffle 和 mini-batch；
5. `actor_rollout_wg.update_actor()` 进入 veRL `ActorRolloutRefWorker.update_actor()`；
6. actor `train_mini_batch()` 使用原生 `loss_mode=gspo`；
7. 每个 Planner turn 形成长度归一化 sequence likelihood ratio；
8. clip `0.0003/0.0004` 与 trajectory advantage 形成 surrogate objective；
9. PyTorch autograd 计算 gradient，AdamW 以 `1e-6` 更新 LoRA；
10. metrics 返回 `actor/loss`、`actor/grad_norm` 等值。

一次 Planner JSON action 是 GSPO 的序列优化单位。response mask 将梯度限定在 Planner 生成 token；题面、Memory、冻结角色文本、公开测试和容器输出构成条件上下文。

## 11. 更新后权重同步

veRL `PPOTrainer.fit()` 在 actor update 后调用 `checkpoint_manager.update_weights()`。`ActorRolloutRefWorker.update_weights()` 从 FSDP actor 获取 LoRA 参数和 PEFT config，调用 Planner vLLM rollout backend 的 `update_weights()`。

下一批四道题由更新后的 Planner 生成六条新 session，形成同步 rollout-update-rollout 闭环。Coding checkpoint 每 64 step 保存，validation 每 64 step运行；`resume_mode=auto` 从已有训练状态继续。

## 12. 技术栈在本流程中的具体作用

| 技术 | 具体调用点 | 解决的问题 |
|---|---|---|
| Python/asyncio | AgentLoop `create_task/wait_for/to_thread/Semaphore` | 24 条 session 调度、总 deadline、容器阻塞卸载 |
| Pydantic | `CodeExample`、`CodeTest`、`CodeAction`、`FinalCode` | 数据、工具调用和终局代码格式约束 |
| OpenAI SDK | `AsyncFrozenModel.generate` | GPU 0 冻结角色统一 HTTP 客户端 |
| vLLM | 冻结 OpenAI server、Planner `server_manager.generate` | continuous batching、KV cache、精确 Planner token/logprob |
| CodingEnvironment | `execute`、`code`、`last_result` | 每条 session 的代码和公开测试状态机 |
| Docker Engine | `DockerSandbox.run` | 候选代码隔离、资源限制和一次性回收 |
| Linux resource API | `drop_candidate_privileges` | UID/GID 降权和文件大小上限 |
| subprocess | 宿主 `docker run`、容器 candidate Python | 进程生命周期、stdin/stdout、超时和 exit code |
| MemoryStore | `add/project` | 保留题面、版本、测试反馈与 judgement |
| Ray | AgentLoop worker 与 actor worker | GPU placement 和并发任务分发 |
| veRL | RLHFDataset、AgentLoopManagerTQ、PPOTrainer | prompt group、rollout、训练、验证和 checkpoint |
| TransferQueue/TensorDict | turn rows、nested token 字段 | 变长轨迹汇聚和 learner 数据交换 |
| PyTorch/FSDP | old logprob、GSPO backward、optimizer | Qwen3-4B Planner 训练 |
| LoRA/PEFT | rank 64 all-linear adapter | 单卡参数高效更新和小体积 checkpoint |
| PyArrow/Parquet | `write_parquet` 与 RLHFDataset | 本地标准数据到训练 batch |
| Hydra/OmegaConf | config merge 与 CLI override | 模型、数据、batch、loss 和输出配置 |
| Bash | build/check/train entrypoint | 镜像门禁、GPU 选择、日志和正式启动 |

## 13. 关键观测指标

一次健康的 Coding step 应观察：

- public test passed/total、failure type、timeout；
- hidden pass rate、hidden passed/total 和 success；
- 每题 reward std、zero-variance group fraction；
- valid trajectory fraction、Planner turn 数、tool-call 数；
- Write/Run/Inspect/Finish action 分布；
- old-logprob row count、trainable-turn count；
- actor loss、gradient norm、importance ratio；
- Docker startup latency、suite runtime、container failures；
- prompt/response length、Memory 截断条目；
- checkpoint、validation、weight sync 和磁盘增长。

Coding preflight 需要出现真实容器执行、公开测试反馈进入下一轮 Planner prompt、隐藏测试 reward 方差、有限 loss/gradient 和有效 checkpoint，再进入正式训练。

