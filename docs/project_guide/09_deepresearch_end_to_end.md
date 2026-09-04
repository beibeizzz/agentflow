# 09 DeepResearch 一次 AgentFlow 推理与优化全流程

## 1. 场景与运行边界

本文跟踪一个已保存在本地的 DeepResearch 样本，从本地 JSONL 转为 veRL Parquet，再经历一次典型多跳搜索 AgentFlow rollout、终局评测、组内 advantage、turn-level GSPO、LoRA 更新和 vLLM 权重同步。

典型样本包含：

```json
{
  "episode_id": "example-id",
  "dataset": "2wiki",
  "question": "一个需要连接两个实体关系的多跳问题",
  "answer": "gold answer",
  "supporting_facts": [
    {"title": "Document A", "sentence_id": 2},
    {"title": "Document B", "sentence_id": 1}
  ],
  "metadata": {}
}
```

一次正式训练 step 读取 4 个 prompt，每个 prompt 采样 6 条 session，因此最多并发形成 24 条完整轨迹。每条轨迹最多包含 5 个 Planner turn。DeepResearch 使用 Qwen3-4B Planner LoRA，Query Analyzer、Executor、Verifier、Generator 和 Base Generator 共享 GPU 0 上的 Qwen3-8B 冻结服务。

## 2. 代码地图

| 阶段 | 文件 | 类或函数 | 具体职责 |
|---|---|---|---|
| 本地数据标准化 | [`tasks/deepresearch/dataset.py`](../../src/agentflow_rl/tasks/deepresearch/dataset.py) | `standardize_example`、`deterministic_subset` | 统一 HotpotQA/2Wiki schema，稳定限额 |
| 样本模式 | [`tasks/deepresearch/schemas.py`](../../src/agentflow_rl/tasks/deepresearch/schemas.py) | `DeepResearchExample.from_row` | 校验 ID、问题、答案和 supporting facts |
| Parquet 转换 | [`verl/data.py`](../../src/agentflow_rl/verl/data.py) | `deepresearch_to_verl_row`、`convert_file`、`write_parquet` | 构建 veRL row、Zstd Parquet 和 SHA-256 |
| 配置入口 | [`verl/main.py`](../../src/agentflow_rl/verl/main.py) | `load_config`、`run`、`build_task_runner` | 合并 Hydra 配置、初始化 Ray/veRL |
| 主状态机 | [`agent_loops/deepresearch.py`](../../src/agentflow_rl/verl/agent_loops/deepresearch.py) | `DeepResearchAgentLoop.run` | 驱动角色、工具、Memory 和终局评测 |
| 模型端口 | [`verl/ports.py`](../../src/agentflow_rl/verl/ports.py) | `generate_planner_turn`、`AsyncFrozenModel.generate` | Planner token 调用与冻结 HTTP 调用 |
| Memory | [`runtime/memory.py`](../../src/agentflow_rl/runtime/memory.py) | `MemoryStore.add`、`MemoryStore.project` | 保存完整轨迹并生成 bounded view |
| 检索后端 | [`tasks/deepresearch/retrieval.py`](../../src/agentflow_rl/tasks/deepresearch/retrieval.py) | `PyseriniResearchIndex.search/read` | 调用 Lucene BM25 与 stored document |
| 工具环境 | [`tasks/deepresearch/tools.py`](../../src/agentflow_rl/tasks/deepresearch/tools.py) | `DeepResearchEnvironment.execute` | 分发 Search、Read、Finish |
| 终局评测 | [`tasks/deepresearch/verifier.py`](../../src/agentflow_rl/tasks/deepresearch/verifier.py) | `evaluate_research_answer` | 计算 answer、supporting facts 和 joint F1 |
| Advantage | [`verl/advantage.py`](../../src/agentflow_rl/verl/advantage.py) | `normalize_trajectory_turns` | 以完整 session 为单位做同题标准化 |
| Trainer 适配 | [`verl/trainer.py`](../../src/agentflow_rl/verl/trainer.py) | `AgentFlowPPOTrainer` | old logprob、advantage、有效 turn、actor update |
| 三阶段训练 | [`run_deepresearch_train.sh`](../../scripts/run_deepresearch_train.sh) | `run_stage` | local context、2Wiki、FullWiki 课程衔接 |

## 3. 本地数据进入 veRL

### 3.1 任务标准数据

假设本地已经存在：

```text
data/deepresearch/2wiki.jsonl
data/deepresearch/2wiki_validation.jsonl
data/indexes/2wiki/
```

`scripts/prepare_verl_data.py --task deepresearch` 遍历 DeepResearch split，并调用 `agentflow_rl.verl.data.convert_file()`：

1. `load_rows()` 逐行读取 JSONL；
2. `convert_rows()` 为每条记录调用 `deepresearch_to_verl_row()`；
3. `_base_row()` 生成 veRL 约定字段；
4. `write_parquet()` 使用 PyArrow 和 Zstd 写入 Parquet；
5. `convert_file()` 返回 source、target、rows 和 SHA-256；
6. 脚本将结果写入 `data/verl/manifest.json`。

形成的 veRL row 结构：

```python
{
    "data_source": "deepresearch",
    "prompt": [{"role": "user", "content": question}],
    "ability": "deepresearch",
    "reward_model": {"style": "rule", "ground_truth": answer},
    "extra_info": complete_deepresearch_row,
    "agent_name": "agentflow_deepresearch",
}
```

`extra_info` 保存 supporting facts 和本地检索上下文，AgentLoop 在运行时从这里重建环境。

### 3.2 数据技术栈的作用

- **JSONL**：保留逐样本可读记录，方便失败样本定位；
- **Pydantic**：`DeepResearchExample` 和 `Citation` 保证字段与类型完整；
- **PyArrow/Parquet**：提供列式存储和 veRL `RLHFDataset` 输入；
- **SHA-256 manifest**：把训练输入身份固定到具体文件内容；
- **Hydra/OmegaConf**：将 Parquet 路径和课程阶段通过配置覆盖传入训练器。

## 4. 进程和模型启动

### 4.1 GPU 0：冻结角色服务

`scripts/serve_frozen_vllm.sh` 启动 OpenAI-compatible vLLM server，加载 Qwen3-8B。AgentLoop 通过 `AsyncOpenAI(base_url="http://127.0.0.1:8000/v1")` 访问该进程。

冻结服务承担五种 prompt：Query Analyzer、Executor、Verifier、Generator、Base Generator。共享模型服务减少角色模型副本，并让 vLLM 对并发角色请求执行 continuous batching 和 KV cache 管理。

### 4.2 GPU 1：Planner rollout 与训练

`run_deepresearch_train.sh` 使用：

```bash
CUDA_VISIBLE_DEVICES=1 python -m agentflow_rl.verl.main \
  --config configs/deepresearch/train.yaml
```

调用链：

```text
main()
  -> load_config()
  -> run()
  -> veRL run_ppo()
  -> AgentFlowTaskRunner.run()
  -> AgentFlowPPOTrainer.init_workers()
  -> AgentFlowPPOTrainer.fit()
```

`build_task_runner()` 创建 Ray remote `AgentFlowTaskRunner`，使用 `ResourcePoolManager` 将 `AgentFlowActorRolloutRefWorker` 映射到 GPU 1。该 worker 同时持有 PyTorch/FSDP actor 和 veRL 管理的 Planner vLLM rollout engine。

### 4.3 关键配置

| 配置 | 值 | 作用 |
|---|---:|---|
| `train_batch_size` | 4 prompts | 每个 learner step 的问题数 |
| `rollout.n` | 6 | 每题完整 session 数 |
| `max_steps` | 5 | 每条 session 的 Planner 轮数上限 |
| `role_max_tokens` | 1024 | 每个角色单次输出上限 |
| `max_prompt_length` | 4096 | Planner/角色 prompt 总预算 |
| `memory_view_tokens` | 3000 | Memory view 上限 |
| `temperature` | 1.0 | Planner group 探索 |
| `turn_mini_batch_size` | 8 | actor 更新 mini-batch 上限 |
| `ppo_epochs` | 1 | 每批 turn 数据使用一次 |
| `lr` | `1e-6` | LoRA optimizer 学习率 |
| GSPO clip | `3e-4/4e-4` | 序列概率比非对称裁剪 |
| LoRA | `r=64, alpha=128` | 更新 Qwen3-4B all-linear adapter |

## 5. 一个 prompt group 如何展开

veRL 的 `RLHFDataset` 提供一个 4-prompt batch。固定 veRL 中的 `AgentLoopManagerTQ.generate_sequences()` 把 prompt 分配给 2 个 AgentLoop worker，`AgentLoopWorkerTQ.generate_sequences()` 为每个 prompt 创建后台任务。

`AgentLoopWorkerTQ._run_prompt()` 读取 `rollout.n=6`，执行：

```python
for session_id in range(6):
    asyncio.create_task(
        self._run_agent_loop(..., session_id=session_id, **prompt)
    )
```

因此同一个 `uid` 获得 session 0 到 5。每条 session 调用独立的 `DeepResearchAgentLoop.run()`，共享模型服务和检索 index，Memory、events 和终局结果保持 session 级隔离。

Ray 管理 worker placement，`asyncio` 管理 session 协程，vLLM 合并 GPU 推理请求，`asyncio.Semaphore` 管理 Lucene 并发。四层调度分别控制设备、轨迹、模型 batch 和工具压力。

## 6. 一条典型搜索轨迹

下面展示一条四轮 session。实际 action 和轮数由 Planner 采样决定。

### 6.1 初始化任务环境

`DeepResearchAgentLoop.run()` 执行：

```python
example = DeepResearchExample.from_row(dict(kwargs["extra_info"]))
environment = DeepResearchEnvironment(
    self._research_index(example),
    top_k=10,
)
memory = MemoryStore()
memory.add(turn_index=-1, role="user", kind="question", content=example.question)
```

`_research_index()` 根据课程阶段选择实现：

- `local_context`：从 `metadata.retrieval_documents` 创建 `InMemoryBM25Index`；
- `global`：使用 worker 初始化时创建的 `PyseriniResearchIndex`。

2Wiki 和 FullWiki 的典型正式搜索使用 `global`，Lucene index 路径由 `run_stage()` 覆盖。

### 6.2 Query Analyzer 建立研究计划

`frozen_generate()` 调用 `AsyncFrozenModel.generate()`，后者通过 OpenAI Python SDK 请求 GPU 0：

```python
response = await client.chat.completions.create(
    model="Qwen3-8B",
    messages=[system_message, user_message],
    max_tokens=1024,
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

分析结果以 `role="query_analyzer"`、`kind="analysis"` 写入 `MemoryStore`，并带有 `identity` tag。后续 Memory view 优先保留该条目。

### 6.3 Turn 0：Planner 选择 Search

`_view()` 调用共享函数 `bounded_memory_text()`：

1. tokenizer 精确统计保留文本 token；
2. 从 4,096 prompt 上限扣除系统 prompt、问题、当前 action 和 256 reserve；
3. `MemoryStore.project()` 优先选择 identity entries 与近期事件；
4. 生成最多 3,000 token 的确定性 view。

`planner_prompt()` 将 question 和 Memory view 组合。`planner_generate()` 进入 `generate_planner_turn()`：

```python
prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
)
output = await server_manager.generate(
    request_id=f"{uid}-{session_id}-{turn_index}",
    prompt_ids=prompt_ids,
    sampling_params=sampling_params,
)
```

这里的 `server_manager` 指向 GPU 1 上的 Planner vLLM。返回值包含 response token IDs 和逐 token logprob。`planner_output()` 立即构建一行 `AgentLoopOutput`，`response_mask` 对 Planner response 全部置 1。

典型 Planner action：

```json
{
  "sub_goal": "找到第一个实体对应的桥接实体",
  "tool_name": "Research_Search_Tool",
  "arguments": {"query": "entity A relation"}
}
```

`ResearchAction.parse()` 先调用 `strict_json_object()`，再通过 Pydantic 拒绝额外字段、空 sub-goal 和未知工具。

### 6.4 Executor 固化工具调用

Planner action 和 bounded Memory 进入 GPU 0 的冻结 Executor。`executor_prompt()` 提供 proposed action，`ResearchAction.parse()` 再次校验 Executor 输出。

代码检查：

```python
if action.tool_name != proposed.tool_name:
    raise ActionParseError("executor changed the Planner tool selection")
```

Planner 决定工具类型，Executor负责补全与规范化参数。工具类型一致性检查保持优化对象的 action ownership。

### 6.5 Pyserini/Lucene 执行 Search

Search action 在 `retrieval_semaphore` 内调用：

```python
result = await self.run_blocking(
    deadline=deadline,
    operation=environment.execute,
    args=(action,),
)
```

`run_blocking()` 使用 `asyncio.to_thread()`，使 Lucene 阻塞查询离开 event loop。`DeepResearchEnvironment.execute()` 提取 query，然后调用 `PyseriniResearchIndex.search(query, top_k=10)`。

`PyseriniResearchIndex.search()` 调用 `LuceneSearcher.search()`，再通过 `_document()` 读取 stored JSON，返回：

```json
{
  "ok": true,
  "data": [
    {"doc_id": "...", "title": "...", "score": 12.4, "snippet": "..."}
  ]
}
```

Pyserini 提供 Python API，Lucene 提供 BM25 倒排检索，Java 21 和 JVM heap 管理搜索运行时，独立 2Wiki/HotpotQA index 保持 benchmark 的 document 与 sentence 语义。

### 6.6 ToolEvent、Memory 和 Verifier

`ToolEvent` 记录 turn index、工具、参数、结果和 `ok`。`MemoryStore.add()` 将完整事件追加为 `role="executor"`、`kind="tool_event"`。

冻结 Verifier 接收 question 和最新 bounded Memory，通过 `extract_conclusion()` 解析最后一个 `Conclusion: STOP/CONTINUE`。judgement 同样写入 Memory，并保存到该 turn 的 `extra_fields`。

### 6.7 Turn 1：Read 精确证据

下一轮 Planner 从 Search hits 中选择 `doc_id`：

```json
{
  "sub_goal": "读取 Document A 中与桥接实体相关的句子",
  "tool_name": "Research_Read_Tool",
  "arguments": {
    "doc_id": "document-a-id",
    "start_sentence": 0,
    "max_sentences": 20
  }
}
```

`DeepResearchEnvironment.execute()` 调用 `ResearchIndex.read()`，返回 title、带全局 `sentence_id` 的句子、`next_start_sentence` 和总句数。citation 后续直接引用这些 sentence IDs。

### 6.8 Turn 2/3：第二跳搜索与结束判断

Planner 基于第一跳证据形成第二个 query，再执行 Search 和 Read。Verifier 在证据覆盖答案与 supporting facts 后输出 STOP；Planner 也可选择 `Research_Finish_Tool`。循环还会在第 5 轮或 300 秒 deadline 到达时结束。

`Base_Generator_Tool` 提供另一条动作路径：该工具调用 GPU 0 的冻结 Base Generator，对指定 sub-goal 和现有证据生成简短事实笔记，结果继续写入 Memory。

## 7. Generator 与终局 reward

循环结束后，`generator_prompt()` 把 question 与最终 bounded Memory 发送给 Qwen3-8B Generator。输出格式由 `ResearchFinalAnswer.parse()` 校验：

```json
{
  "answer": "predicted answer",
  "report": "基于两条来源形成的结论",
  "citations": [
    {"title": "Document A", "sentence_id": 2},
    {"title": "Document B", "sentence_id": 1}
  ]
}
```

`evaluate_research_answer()` 分别调用：

- `answer_scores()`：计算 answer EM、precision、recall、F1；
- `supporting_fact_scores()`：对 `(title, sentence_id)` 集合计算 EM、precision、recall、F1。

最终计算：

```text
joint_precision = answer_precision * supporting_fact_precision
joint_recall    = answer_recall * supporting_fact_recall
reward          = joint_f1(joint_precision, joint_recall)
success         = answer_em * supporting_fact_em == 1
```

reward 可取 `[0,1]` 连续值，同题 6 条 session 更容易产生方差。`ANSWER_MISMATCH` 和 `SUPPORTING_FACT_MISMATCH` 支持分类型诊断。

## 8. AgentLoopOutput 如何进入 TransferQueue

`finalize_outputs()` 对本 session 的每个 Planner turn 写入：

```text
valid_for_training
terminal_reason
uid / session_id / turn_index
```

终局 reward 先写在最后一个 `AgentLoopOutput.reward_score`，完整 verification、Memory、ToolEvent 和 final answer 写入最后一行 metadata。

固定 veRL 的 `AgentLoopWorkerTQ._agent_loop_postprocess()` 随后：

1. 将 final reward 复制到同 session 的前序 turn row，满足 TransferQueue 字段一致性；
2. 为每个 turn 组合 `input_ids = prompt_ids + response_ids`；
3. 生成 attention/position/loss mask；
4. 使用 key `{uid}_{session_id}_{turn_index}`；
5. 调用 `tq.async_kv_batch_put()` 写入 `train` partition。

reward 的统计单位仍为完整 session。下一阶段的 `normalize_trajectory_turns()` 只选择每个 session 的最大 turn index 参与 reward mean/std，避免复制字段形成重复计数。

## 9. 从 group reward 到 trajectory advantage

当 4 个 prompt 的 session 都完成后，ReplayBuffer 返回本 step 的动态 turn keys。`AgentFlowPPOTrainer.step()` 按以下顺序执行。

### 9.1 重算 old logprob

`_compute_old_log_prob()`：

1. 从 TransferQueue 读取 `extra_fields`；
2. `valid_training_keys()` 选择基础设施有效 row；
3. 调用 veRL `PPOTrainer._compute_old_log_prob()`；
4. Actor 对保存的精确 token IDs 重算 proximal old logprob；
5. 记录 `agentflow/old_log_prob_row_count`。

这一步使用 GPU 1 上的 PyTorch/FSDP actor，提供 GSPO importance ratio 的分母。

### 9.2 计算 advantage

`_compute_advantage()` 从 TransferQueue 读取 `response_mask`、`rm_scores` 和 `extra_fields`。调用链：

```text
build_trajectory_turns()
  -> TrajectoryTurn.identity 解析 uid/session/turn
  -> normalize_trajectory_turns()
     -> 每个 session 选择最大 turn index
     -> 按 uid 形成 6-session group
     -> 过滤基础设施无效 session
     -> 计算 population mean/std
     -> 为每条 session 计算 A_i
     -> 将 A_i 传播到该 session 的全部真实 turn keys
```

Trainer 将每个 turn 的标量 advantage 乘 `response_mask`，形成 nested `advantages` 和 `returns`，再通过 `tq.kv_batch_put()` 写回 TransferQueue。

组内 reward 方差低于 `1e-6` 时，该组 turn 进入 skipped keys。其余模型失败继续以低 reward 参与同题相对比较。

## 10. turn-level GSPO 与 LoRA 更新

`_update_actor()` 取得 `trainable_keys`，随后执行：

1. `build_unpadded_attention_mask()` 为 nested input IDs 建立全序列 mask；
2. `effective_turn_mini_batch_size()` 从实际 trainable turn 数中选择小于等于 8 的最大整除数；
3. `actor_update_metadata()` 设置 mini-batch、1 个 PPO epoch、seed、temperature；
4. `self.actor_rollout_wg.update_actor(train_batch)` 进入 veRL worker；
5. `ActorRolloutRefWorker.update_actor()` 调用 actor 的 `train_mini_batch()`；
6. veRL 原生 `loss_mode=gspo` 对每个 Planner turn 计算长度归一化 sequence ratio；
7. 非对称 clip `0.0003/0.0004` 和 trajectory advantage 形成 policy loss；
8. PyTorch autograd 执行 backward，optimizer 以 `1e-6` 更新 LoRA 参数。

一个 session 的多轮 Planner action 分别进入目标函数，每轮都使用所属 session 的同一个 advantage。每个 turn 内的 token 通过 response mask 参与序列 log-likelihood；Query Analyzer、Executor、Verifier、Generator 和工具返回文本只作为 prompt context。

## 11. 权重同步、保存与下一步 rollout

veRL `PPOTrainer.fit()` 在每个 step 中执行：

```text
rollout + training
  -> 可选 checkpoint
  -> checkpoint_manager.update_weights()
  -> 可选 validation
  -> metrics / rollout dump
  -> TransferQueue 与 ReplayBuffer 清理
```

`ActorRolloutRefWorker.update_weights()` 从 FSDP actor 取得当前参数和 PEFT config，再调用 vLLM rollout backend 的 `update_weights()`。下一个 prompt batch 因此使用更新后的 Planner LoRA 采样，构成同步 on-policy 闭环。

DeepResearch 每 64 step 保存 checkpoint 和运行 validation。`run_deepresearch_train.sh` 在阶段结束后使用 `find_latest_adapter.py` 导出最新 PEFT adapter，并按以下顺序继续：

```text
HotpotQA distractor local context
  -> 2Wiki global Lucene
  -> HotpotQA FullWiki global Lucene
```

每阶段继承 Planner adapter，optimizer 重新初始化，数据和检索难度逐步提高。

## 12. 技术栈在本流程中的具体作用

| 技术 | 具体调用点 | 解决的问题 |
|---|---|---|
| Python/asyncio | `create_task`、`gather`、`wait_for`、`to_thread`、`Semaphore` | 多 session 并发、deadline、阻塞检索卸载、后端限流 |
| Pydantic | `DeepResearchExample.from_row`、`ResearchAction.parse`、`ResearchFinalAnswer.parse` | 输入、action、引用和终局输出的严格 schema |
| OpenAI SDK | `AsyncFrozenModel.generate` | 统一访问 GPU 0 冻结 vLLM 服务 |
| vLLM | 外部 OpenAI server、内部 `server_manager.generate` | 冻结角色 continuous batching、Planner token/logprob rollout |
| Pyserini/Lucene | `PyseriniResearchIndex.search/read` | 百万文档 BM25 检索和 doc/sentence 定位 |
| MemoryStore | `add`、`project` | 保存完整轨迹、构造预算内 prompt state |
| Ray | `AgentFlowTaskRunner`、AgentLoop workers | GPU worker placement 与任务分发 |
| veRL | `run_ppo`、AgentLoopManagerTQ、PPOTrainer | rollout、learner、checkpoint 和 validation 主循环 |
| TransferQueue | `async_kv_batch_put`、`kv_batch_get/put` | 变长 Planner turn 的异步汇聚和字段交换 |
| PyTorch/FSDP | old logprob、autograd、optimizer、checkpoint | Planner actor 训练与模型状态管理 |
| LoRA/PEFT | rank 64 adapter 与阶段导出 | 低显存 Planner 更新、课程阶段衔接 |
| Hydra/OmegaConf | `load_config` 和 `--override` | veRL 默认配置、任务配置、阶段参数合并 |
| PyArrow/Parquet | `write_parquet`、RLHFDataset | 本地任务数据到训练 batch 的高效转换 |
| Bash | `run_stage`、GPU 环境变量、输出目录 | 服务编排、课程切换、恢复与日志路径 |

## 13. 关键观测指标

一次健康的 DeepResearch step 应看到：

- `answer_em/f1`、`supporting_fact_em/f1`、`joint_em/f1`；
- 每题 reward mean/std 和 zero-variance group fraction；
- valid trajectory fraction、平均 Planner turns、tool calls；
- old-logprob row count 与 trainable-turn count；
- actor loss、gradient norm、importance ratio；
- Search/Read latency、trajectory deadline、Lucene errors；
- rollout token 长度、Memory omitted entries 和 prompt ceiling；
- checkpoint、validation 和 weight sync 时间。

远程 preflight 需要出现真实 Search/Read、组内 reward 方差、有限 loss/gradient、有效 checkpoint 和包含检索结果的后续 Planner prompt，再进入三阶段正式训练。

