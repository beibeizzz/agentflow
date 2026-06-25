# Data Sources and Generation Notes

This repository contains two experiment families: GSM8K AgentFlow experiments and a synthetic sandbox ticket-agent task. The repository is intended to be reproducible without depending on real business systems or private customer data.

## Upstream codebase

The AgentFlow framework code is based on:

- Repository: https://github.com/lupantech/AgentFlow
- License: MIT License
- Copyright: Copyright (c) 2025 the AgentFlow Team

Local changes add GSM8K experiment scripts, a synthetic ticket-agent environment, ticket tools, deterministic verification, and turn-level GSPO/GRPO training utilities.

## GSM8K data

The GSM8K experiment files under `project/try_gsm8k_0522/data/` are derived from the GSM8K math word-problem dataset and local preprocessing/rewrite pipelines used by this project.

Important notes:

- Check the original GSM8K dataset license/terms before redistributing derived files.
- The project includes scripts for preparing and scoring GSM8K-style JSON data.
- Some derived files are rewritten or filtered for AgentFlow calculator-style workflows.

Relevant paths:

```text
project/try_gsm8k_0522/data/
project/try_gsm8k_0522/prepare_gsm8k_json.py
project/try_gsm8k_0522/rewrite_calculator_prompts/
project/try_gsm8k_0522/score_gsm8k.py
```

## Synthetic ticket-agent data

The ticket-agent task is synthetic. It does not use real customer, order, or ticket records.

Each generated episode contains:

- a public `user_request`;
- an isolated `episode_id`;
- `curriculum_mode` and `lookup_mode` metadata;
- an `initial_state` with synthetic tickets;
- a hidden `goal_spec` used only for deterministic verification;
- `max_steps`.

The data pipeline has two layers:

1. deterministic blueprints, generated locally from code and seed;
2. optional natural-language synthesis using an external LLM API.

Recreate and validate ticket data from `project/`:

```bash
python try_ticket_agent/scripts/generate_blueprints.py --seed 42 --smoke 32 --train 2500 --validation 256 --test 512 --output-dir try_ticket_agent/data/blueprints
cp try_ticket_agent/config_synthesis.example.yaml try_ticket_agent/config_synthesis.yaml
DEEPSEEK_API_KEY=... python try_ticket_agent/scripts/synthesize_dataset.py --config try_ticket_agent/config_synthesis.yaml
python try_ticket_agent/scripts/validate_dataset.py --blueprints try_ticket_agent/data/blueprints
python try_ticket_agent/scripts/validate_dataset.py --dataset try_ticket_agent/data/generated
```

Relevant paths:

```text
project/try_ticket_agent/data/README.md
project/try_ticket_agent/data_synthesis/
project/try_ticket_agent/scripts/generate_blueprints.py
project/try_ticket_agent/scripts/synthesize_dataset.py
project/try_ticket_agent/scripts/validate_dataset.py
```

The synthesis judge is used only during data synthesis. Training and evaluation consume frozen JSONL data and do not synthesize online.

## Models and external services

Model weights are not part of the repository. Experiments commonly use Qwen3-0.6B served locally or remotely through vLLM-compatible endpoints.

Before publishing artifacts, check the license/terms for:

- Qwen model weights and tokenizer files;
- vLLM;
- Transformers, PEFT, PyTorch, and related training dependencies;
- any external API provider used for synthesis or evaluation;
- GSM8K and any derived dataset files.

Model checkpoints, adapters, rollout logs, metrics, and local smoke-test outputs should be treated as experiment artifacts and should not be committed unless intentionally published with their own license/metadata.
