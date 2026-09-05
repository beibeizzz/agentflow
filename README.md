# AgentFlow RL v3

AgentFlow RL alpha is a veRL + vLLM implementation of four verifiable AgentFlow
task environments:

- GSM8K calculator reasoning;
- an isolated synthetic Ticket workflow;
- DeepResearch over HotpotQA and 2Wiki with BM25 retrieval and citation checks;
- TACO-Verified Easy/Medium coding with Docker-isolated execution.

GSM8K and Ticket retain the reviewed data, environments, terminal evaluators,
and GSPO settings while adopting the complete shared role loop and Base
Generator action. DeepResearch and Coding train separate `Qwen3-4B` Planner
LoRA checkpoints while sharing frozen
`Qwen3-8B` role modules. See [alpha_deployment.md](docs/alpha_deployment.md) for
data preparation, exact training parameters, and the two-GPU runbook.

The role and prompt changes define a new rollout distribution. Baseline,
training, and evaluation results for this alpha must come from the same commit.

The experimental order first measures the frozen AgentFlow baseline and then
trains the Planner next-step LoRA with turn-level GSPO directly from the selected
base or post-trained checkpoint.

## Preserved research semantics

- Query Analyzer, Executor, Verifier, Generator, and Base Generator are frozen.
- Deterministic tools and task environments expose typed observations through
  the shared append-only Memory.
- Only the Planner uses LoRA (`r=64`, `alpha=128`).
- One rollout session is one complete multi-step trajectory.
- A trajectory reward enters query-local mean/std exactly once.
- Population standard deviation is used; zero-variance groups get zero
  advantage.
- The trajectory advantage is broadcast to its real Planner turns.
- One veRL row is one Planner turn, so native `loss_mode=gspo` remains
  turn-level rather than whole-trajectory GSPO.
- GSM8K/Ticket use two PPO epochs; DeepResearch/Coding use one PPO epoch.
- Ticket reward is binary and all formal splits are 50:50 direct:indirect.
- Verifier controls role-loop termination; deterministic terminal evaluators
  alone produce reward.

GSM8K/Ticket use temperature `1.2` and GSPO clips `0.001/0.003`.
DeepResearch/Coding use temperature `1.0` and GSPO clips `0.0003/0.0004`.
Every task uses top-p `1.0`, disabled top-k, and repetition penalty `1.0`.

## Main structure

```text
configs/                       veRL partial configs and AgentLoop registry
data/                          source JSON/JSONL and provenance
scripts/                       data, baseline, smoke, train, and eval entrypoints
src/agentflow_rl/tasks/        prompts, tools, environments, deterministic checks
src/agentflow_rl/synthesis/    isolated Ticket synthesis pipeline
src/agentflow_rl/verl/         AgentLoops, token ports, advantage, Trainer, entrypoint
tests/                         pure, parity, and fake-server integration tests
```

See [architecture.md](docs/architecture.md) for the exact control/data flow and
[migration.md](docs/migration.md) for the v2-to-v3 mapping.
The cross-task invariant audit is recorded in
[framework_consistency_review.md](docs/framework_consistency_review.md).
Local and remote acceptance evidence is tracked in
[alpha_verification.md](docs/alpha_verification.md).
The Chinese project guide starts at
[docs/project_guide/README.md](docs/project_guide/README.md) and covers the
repository structure, architecture, four task flows, runtime stack, data and
sandbox design, two-GPU execution, post-training theory, and Agentic RL trends.

## Remote installation

Linux, CUDA, Python 3.11, and Java 21 are recommended. Use a clean environment;
the pinned veRL extra installs Ray, FSDP support, TransferQueue dependencies,
and a compatible vLLM range.

Place the repository itself on the expanded data volume. All model, corpus,
index, rollout, and checkpoint paths are repository-relative, and the remote
audit requires the project root and `DATA_ROOT` to share one filesystem.

```bash
conda create -n agentflow-v3 python=3.11 -y
conda activate agentflow-v3
pip install -r requirements.txt
pip install -e ".[test,data,research]"
DATA_ROOT=/data bash scripts/remote/audit_environment.sh
```

Prepare the external DeepResearch and Coding sources and indexes with
[alpha_deployment.md](docs/alpha_deployment.md#data-preparation), then run
`python scripts/prepare_verl_data.py --task all`.

veRL is pinned to tag `v0.8.0`, commit
`7aed6b230776f963fa09509c10d9c3a767d1102c`. Do not silently upgrade it: the
custom Trainer relies on the synchronous PPO/TransferQueue/AgentLoop API at
that commit.

If the frozen server uses a separate environment, install
`requirements-vllm.txt` there. `MODEL_PATH` defaults to
`model/Qwen/Qwen3-8B` and serves the name `Qwen3-8B`. Legacy GSM8K/Ticket runs
can set both `MODEL_PATH=model/Qwen/Qwen3-0.6B` and
`SERVED_MODEL_NAME=Qwen3-0.6B`.

## Two-GPU experiment workflow

The alpha reference layout uses two A800 80 GB GPUs:

- GPU0: external frozen OpenAI-compatible vLLM server;
- GPU1: veRL FSDP LoRA actor and colocated Planner vLLM rollout.

Use the 0.6B frozen-role service for GSM8K/Ticket and the 8B frozen-role service
for DeepResearch/Coding. The exact two-phase command sequence and backend gates
are documented in [alpha_deployment.md](docs/alpha_deployment.md#two-gpu-execution).

Start the selected frozen model in terminal 1:

```bash
bash scripts/serve_frozen_vllm.sh
```

Prepare data once, then run the baseline before training:

```bash
python scripts/prepare_verl_data.py --task all
bash scripts/run_ticket_baseline.sh
bash scripts/run_gsm8k_baseline.sh
```

Run a two-step smoke before the formal job:

```bash
bash scripts/run_ticket_smoke.sh
bash scripts/run_gsm8k_smoke.sh
```

Run Planner-only LoRA GSPO:

```bash
bash scripts/run_ticket_train.sh
bash scripts/run_gsm8k_train.sh
```

Training configs use veRL `resume_mode=auto`. Baseline, smoke, and default eval
disable resume to prevent stale checkpoints from changing the measured policy.
To evaluate a specific veRL checkpoint, point eval at its `global_step_*`
directory:

```bash
bash scripts/run_ticket_eval.sh \
  --override trainer.resume_mode=resume_path \
  --override trainer.resume_from_path=outputs/ticket/train/global_step_20
```

The same form applies to GSM8K. An external PEFT adapter can instead be passed
with `--adapter-path`, which sets the Planner LoRA rank/alpha automatically.

## Metrics and outputs

Each shell entrypoint enables veRL's native `console` and `file` loggers. The
JSONL file is written to:

```text
outputs/<task>/<mode>/metrics.jsonl
```

It contains veRL-native loss, learning-rate, grad-norm, throughput, timing,
response-length, validation, and checkpoint-step fields plus:

- reward/advantage mean and population std;
- query-group, trajectory, valid/invalid trajectory, and real-turn counts;
- zero-variance/skipped-group counts and fractions;
- task success, step count, validity, Ticket direct/indirect, and GSM8K
  verifier-stop metrics.

Formal training and smoke configs also write turn-level rollout generation
dumps under `outputs/<task>/<mode>/rollouts/`. Baseline/eval and periodic train
validation write all Planner-turn prompts/responses under the corresponding
`validation/` directory. Because each later Planner prompt contains prior tool
observations or GSM8K judge memory, these dumps retain the trajectory evidence
needed for error analysis without a separate runtime logger.

## Local verification

CPU tests use fake Planner and frozen-model servers while preserving exact
token IDs and log-probabilities.

```bash
python -m pytest -q
python -m compileall -q src scripts
```

WSL2 single-GPU validation runs real Qwen3-0.6B inference, vLLM rollout, veRL
AgentLoop training, weight synchronization, and checkpoint persistence:

```bash
bash scripts/local/setup_wsl_env.sh
bash scripts/local/serve_gsm8k_frozen_single_gpu.sh  # terminal 1
bash scripts/local/run_gsm8k_single_gpu_smoke.sh     # terminal 2
```

See [alpha_deployment.md](docs/alpha_deployment.md#wsl2-single-gpu-smoke) for
the tested environment and output contract. Formal 4B/8B preflight uses the
remote Linux CUDA host. Locally generated `data/verl/` files are deterministic
build artifacts and can be regenerated from the tracked sources.

Data sources and hashes are recorded in [data/README.md](data/README.md).
Licensing and reviewed-source provenance are recorded in `NOTICE`,
`THIRD_PARTY_NOTICES.md`, and `LICENSES/`.
