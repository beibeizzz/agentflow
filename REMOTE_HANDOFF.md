# Remote Formal-Run Handoff

This package contains the effective AgentFlow alpha source, four-task configs,
small tracked GSM8K/Ticket datasets, tests, Docker sandbox, data preparation
tools, deployment documentation, and lightweight local verification evidence.

## Target layout

Place the extracted project root on the expanded data volume. Keep these model
paths relative to the project root:

```text
model/Qwen/Qwen3-0.6B
model/Qwen/Qwen3-4B
model/Qwen/Qwen3-8B
```

The formal topology is:

```text
GPU 0: frozen-role vLLM service
GPU 1: veRL FSDP LoRA actor and colocated Planner vLLM
```

## Acceptance order

1. Create the Python 3.11 environment and install `requirements.txt` plus the
   project extras `test,data,research`.
2. Run `DATA_ROOT=<expanded-volume> bash scripts/remote/audit_environment.sh`.
3. Prepare the labeled DeepResearch subsets, the two research corpora, and the
   TACO-Verified Easy/Medium split using `docs/alpha_deployment.md`.
4. Build `data/indexes/hotpotqa`, `data/indexes/2wiki`, and the Coding Docker
   image.
5. Run both research backend gates and `scripts/remote/check_code_sandbox.py`.
6. Run `python scripts/prepare_verl_data.py --task all`.
7. Start the Qwen3-0.6B frozen service on GPU 0; run GSM8K and Ticket baseline,
   smoke, training, and evaluation from a second terminal.
8. Start the Qwen3-8B frozen service on GPU 0; run DeepResearch and Coding
   baseline and 32-prompt preflight from a second terminal.
9. Start DeepResearch and Coding formal training after both preflight output
   checks pass.

## Mandatory preflight evidence

Each new-task preflight must contain:

- at least one query group with reward variance;
- positive trainable Planner-turn count;
- finite actor loss and gradient norm;
- a post-tool Planner prompt in the rollout dump;
- a persisted actor checkpoint.

Formal configuration keeps prompt batch `4`, rollout group `6`, initial
Planner-turn mini-batch `8`, learning rate `1e-6`, KL coefficient `0`, five
AgentFlow rounds, 4096 input tokens, and 1024 output tokens per role.

## Transfer integrity

`PACKAGE_MANIFEST.sha256` lists every packaged file and its SHA-256 value.
Verify the archive hash before extraction and the manifest after extraction.
The `verification_evidence` directory contains metrics and rollout JSONL from
real local Qwen3-0.6B vLLM/veRL runs. Checkpoints remain local build artifacts.

Detailed commands are in `docs/alpha_deployment.md`. Completed local checks and
known remote gates are in `docs/alpha_verification.md`.
