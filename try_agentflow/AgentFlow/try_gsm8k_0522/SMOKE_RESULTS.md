# GSM8K Smoke Results - 2026-05-22

Model endpoint: `http://localhost:8000/v1`
Served model: `Qwen3-0.6B-Instruct`
Sample: `data/gsm8k_smoke_20.json`
Prompt style: `Planner` / `Executor` / `Verifier` / `Generator`
Generation: `temperature=0.0`, `max_tokens=2048`
Available AgentFlow tool: `Calculator_Tool`

| Variant | Result directory | Response field | Max steps | Correct / Total | Accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| Calculator AgentFlow | `results/smoke_calculator_steps1` | `direct_output` | 1 | 2 / 20 | 10.0% |
| Calculator AgentFlow | `results/smoke_calculator_steps2` | `direct_output` | 2 | 1 / 20 | 5.0% |
| Calculator AgentFlow | `results/smoke_calculator_steps3` | `direct_output` | 3 | 0 / 20 | 0.0% |

Summary files:

- `summary/smoke_calculator_steps1_summary.json`
- `summary/smoke_calculator_steps2_summary.json`
- `summary/smoke_calculator_steps3_summary.json`

Observed failure mode after the prompt-contract update: the local vLLM service is reachable and the only available AgentFlow tool is `Calculator_Tool`, but the small model still often fails the Planner output format. Most failed steps do not parse into a valid `Tool Name: Calculator_Tool`, so no calculator command is generated and Memory records a missing tool call.

Correct pids:

- `max_steps=1`: `1`, `19`
- `max_steps=2`: `1`
- `max_steps=3`: none

Tool-call health:

- `max_steps=1`: 16 / 20 action steps had `tool_name=null` and no command.
- `max_steps=2`: 27 / 29 action steps had `tool_name=null` and no command.
- `max_steps=3`: 21 / 25 action steps had `tool_name=null` and no command.

Representative failure: for pid `0`, the Planner response did not include the required `Context:` section and instead produced a direct answer-like response, so `extract_context_subgoal_and_tool()` returned `None` and the calculator was never called.

Verification commands run:

```bash
/home/north/vllm_test/.venv/bin/python -m unittest discover -s try_agentflow/AgentFlow/try_gsm8k_0522/tests -v
/home/north/vllm_test/.venv/bin/python -m py_compile try_agentflow/AgentFlow/agentflow/agentflow/models/planner.py try_agentflow/AgentFlow/agentflow/agentflow/models/executor.py try_agentflow/AgentFlow/agentflow/agentflow/models/verifier.py
bash -n try_agentflow/AgentFlow/try_gsm8k_0522/run_smoke.sh try_agentflow/AgentFlow/try_gsm8k_0522/run_full.sh
curl --noproxy '*' -sS --max-time 3 http://127.0.0.1:8000/v1/models
```

Smoke command:

```bash
cd /home/north/vllm_test/try_agentflow/AgentFlow/try_gsm8k_0522
bash run_smoke.sh
```
