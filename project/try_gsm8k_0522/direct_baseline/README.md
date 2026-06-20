# Direct vLLM GSM8K Baseline

This folder runs the local vLLM-served model directly on GSM8K without AgentFlow.

The prompt asks the model to solve the problem and end with:

```text
<answer>NUMBER</answer>
```

The parent `score_gsm8k.py` scorer reads `direct_output` and compares the extracted numeric answer with `gold_answer`.

Run smoke:

```bash
cd /home/north/vllm_test/try_agentflow/AgentFlow/try_gsm8k_0522
bash direct_baseline/run_smoke_direct.sh
```

Defaults:

- Data: `data/gsm8k_smoke_50.json`
- Results: `direct_baseline/results/smoke_direct_vllm`
- Summary: `direct_baseline/summary/smoke_direct_vllm_summary.json`
- Model: `Qwen3-0.6B-Instruct`
- Base URL: `http://localhost:8000/v1`
- Max tokens: `2048`
- Temperature: `0.0`

Override example:

```bash
SMOKE_SIZE=20 MAX_TOKENS=1024 bash direct_baseline/run_smoke_direct.sh
```

Latest smoke result on `gsm8k_smoke_50.json`:

- Correct / total: `28 / 50`
- Accuracy: `56.0%`
- Missing / errored / unparseable: `0 / 0 / 0`
- All 50 outputs included a parseable `<answer>...</answer>` tag.
