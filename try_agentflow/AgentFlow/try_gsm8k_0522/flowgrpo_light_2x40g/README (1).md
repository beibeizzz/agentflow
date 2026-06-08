# FlowGRPO Light 2x40G

This setup uses two visible GPUs by splitting work across processes:

- GPU 1: frozen vLLM server.
- GPU 0: trainable planner LoRA training.

The training entrypoint is `train_light_grpo_quiet.py`, which keeps AgentFlow rollout logs out of the terminal and writes training metrics into the output directory.

## Start vLLM

```bash
cd /home/north/vllm_test/new_agentflow0527/agentflow/try_agentflow/AgentFlow
CUDA_VISIBLE_DEVICES=1 bash try_gsm8k_0522/flowgrpo_light_2x40g/run_vllm_gpu1.sh
```

Equivalent manual command:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /home/north/vllm_test/models/Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --attention-backend FLASH_ATTN
```

## Train

Default train file:

```text
try_gsm8k_0522/data/gsm8k_train_learnable.json
```

Run:

```bash
cd /home/north/vllm_test/new_agentflow0527/agentflow/try_agentflow/AgentFlow
CUDA_VISIBLE_DEVICES=0 bash try_gsm8k_0522/flowgrpo_light_2x40g/run_train_2x40g.sh
```

Reduce pressure if needed:

```bash
GROUP_SIZE=4 ROLLOUT_CONCURRENCY=16 PLANNER_BATCH_SIZE=16 \
  bash try_gsm8k_0522/flowgrpo_light_2x40g/run_train_2x40g.sh
```

## Outputs

Default output directory:

```text
try_gsm8k_0522/flowgrpo_light_2x40g/outputs/train_2x40g
```

Important files:

```text
metrics.jsonl
summary_metrics.jsonl
train_summary.json
rollout_logs/step_000001.log
checkpoint_step_<n>/
final_adapter/
```

Terminal output is limited to one high-level line per training step, including reward mean/std, valid rollout count, non-zero advantage count, effective samples, loss, GPU memory, and timing.
