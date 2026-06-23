# Ticket AgentFlow + turn-level GSPO

This is an isolated synthetic ticket task built on the repository's existing AgentFlow implementation. It uses exactly three tools: `Ticket_Query_Tool`, `Ticket_Update_Tool`, and `Ticket_Finish_Tool`. Direct episodes require Update -> Finish; indirect episodes require Query -> Update -> Finish. No SFT is used: the experiment order is frozen AgentFlow baseline first, then LoRA turn-level GSPO.

The task uses binary reward: the hidden verifier returns 1 only when the requested field, finish submission, step limit, and no-collateral-mutation checks all pass. Hidden `initial_state` and `goal_spec` are bound to an episode backend and never enter Planner prompts or Memory. Every concurrent solver owns a separate backend and reset replaces Memory.

## Data

Run from `project/`:

```bash
python try_ticket_agent/scripts/generate_blueprints.py --seed 42 --smoke 32 --train 2500 --validation 256 --test 512 --output-dir try_ticket_agent/data/blueprints
cp try_ticket_agent/config_synthesis.example.yaml try_ticket_agent/config_synthesis.yaml
DEEPSEEK_API_KEY=... python try_ticket_agent/scripts/synthesize_dataset.py --config try_ticket_agent/config_synthesis.yaml
python try_ticket_agent/scripts/validate_dataset.py --dataset try_ticket_agent/data/generated
```

Synthesis is blueprint -> LLM rewrite -> deterministic validator -> LLM judge used only during data synthesis -> registered-tool reference execution. Missing credentials stop the run; there is no template fallback. Progress, rejected records, usage, summaries, and SHA-256 manifests support resume and audit.

Local synthesis is CPU/lightweight orchestration around an OpenAI-compatible API. It requires `openai`, `PyYAML`, `DEEPSEEK_API_KEY`, network access to the configured base URL, and a copied `config_synthesis.yaml`; it does not require a local GPU and may incur API charges.

## Baseline and training

Start a frozen Qwen3-0.6B OpenAI-compatible vLLM server on GPU 0. Run baseline and training from `project/`:

```bash
bash try_ticket_agent/baseline/run_agentflow_baseline.sh
bash try_ticket_agent/flowgrpo_general_2x40g/run_train_general_2x40g.sh
QUESTION_BATCH_SIZE=2 GROUP_SIZE=2 ROLLOUT_CONCURRENCY=4 PLANNER_BATCH_SIZE=4 MAX_TRAIN_ITEMS=2 EPOCHS=1 bash try_ticket_agent/flowgrpo_general_2x40g/run_train_general_2x40g.sh
```

Training defaults to GPU 1, BF16, LoRA 64/128, learning rate `2e-6`, two policy epochs, and asymmetric clip `0.001/0.003`. Infrastructure failures are removed before reward mean/std. Rewards are normalized within each query's valid trajectory group; one trajectory advantage is broadcast to all Planner turns. Each turn then receives a per-turn GSPO ratio computed from the exact generated prompt/response token IDs in FP32 ratio math. Equal-reward groups produce zero advantage and `skip_no_advantage`.

Metrics include reward/advantage groups, valid and infrastructure-failed trajectories, nonzero trajectories and turns, response token count, ratio/clip/KL statistics, timings, and checkpoints. Core Planner/Executor modes are opt-in; legacy defaults remain unchanged.

## Controlled evaluation

Use the same frozen model, accepted test set, three tools, `max_steps=3`, structured Planner/Executor, workflow output, and deterministic verifier:

```bash
EVAL_MODE=baseline ADAPTER_PATH=false bash try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh
EVAL_MODE=adapter ADAPTER_PATH=try_ticket_agent/flowgrpo_general_2x40g/outputs/train_general_2x40g/final_adapter bash try_ticket_agent/flowgrpo_general_2x40g/run_eval_learnable_general_2x40g.sh
```

Evaluation reports overall and direct/indirect success, invalid-action rate, and infrastructure-failure rate separately. A local run cannot establish remote GPU success; verify finite loss/ratio/KL, LoRA parameter updates, and adapter save/reload on the target 2x40G environment.
