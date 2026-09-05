# Alpha Verification Record

Updated on 2026-09-05. The real GPU smoke evidence was produced on 2026-09-04.
The deployment target is one Linux host with two A800
80 GB GPUs, 22 logical CPU cores, 110 GB RAM, and the repository placed on the
expanded data volume.

## Completed local verification

| Check | Result | Evidence |
|---|---:|---|
| Python unit and fake-server integration tests | 145 passed | `python -m pytest -q` |
| veRL native config validation | 17 passed | `scripts/remote/validate_all_configs.sh` |
| Python compilation, Linux Bash syntax, and LF checkout policy | passed | `compileall`, WSL `bash -n`, `.gitattributes` |
| Real Qwen3-0.6B vLLM + veRL AgentLoop smoke | exit 0 | `outputs/gsm8k/local_single_gpu_smoke/resume_20260904_b` |
| Real GSPO actor-update probe | exit 0 | `outputs/gsm8k/local_single_gpu_smoke/actor_update_v2_20260904` |

The real smoke produced two complete sessions and three Planner turns. veRL
recomputed three old-log-probability rows, persisted a checkpoint, synchronized
weights into the rollout engine, and recorded rollout/actor probability
correlation `0.9990957`. One calculator action failed validation, the failure
entered Memory, and the next Planner turn corrected the expression.

Both sessions received reward `1`, so the query group had zero reward variance
and the actor update was skipped by the configured advantage rule. Formal
preflight uses 32 prompts with group size 8. Its output checker requires a group
with reward variance, a positive trainable-turn count, finite loss and gradient
norm, an actor checkpoint, and a post-tool Planner prompt.

The actor-update probe used the same real Qwen3-0.6B actor/frozen service and a
55-token stress ceiling on one exact-arithmetic question. Six sessions produced
three rewards of `1` and three rewards of `0`: reward mean/std were `0.5/0.5`,
advantage std was `1.0`, all six Planner turns entered the actor update,
`actor/grad_norm` was `2.0630`, and `actor_update_skipped` was `0`. The run then
saved its checkpoint and synchronized updated LoRA weights into vLLM. This
stress ceiling is local verification input; formal task configs retain the
specified 2048-token output ceiling.

## DeepResearch execution contract

1. HotpotQA FullWiki uses the official introductory-Wikipedia Lucene index.
2. 2Wiki uses a separate Lucene index built from the complete supplied 2Wiki
   contexts.
3. The backend gate checks document count, search, read, sampled gold titles,
   sentence bounds, and source sentence text.
4. Hotpot distractor preflight uses per-example local context; baseline, later
   curriculum stages, and final evaluation use their benchmark-specific index.
5. Search and read results enter append-only Memory; role-specific projections
   select analysis, executed evidence, judgements, and final-generation context
   under the 8192-token input ceiling.
6. Terminal answer and supporting-fact F1 provide the trajectory reward.
7. Citation-grounding diagnostics verify that generated title/sentence pairs
   appeared in executed Read results.

Run the two backend checks before GPU allocation:

```bash
python scripts/remote/check_research_backend.py \
  --index data/indexes/hotpotqa --min-documents 1000000 \
  --examples data/deepresearch/hotpot_fullwiki.jsonl
python scripts/remote/check_research_backend.py \
  --index data/indexes/2wiki --min-documents 1000 \
  --examples data/deepresearch/2wiki.jsonl
```

## Coding execution contract

1. TACO-Verified Easy/Medium rows are split by normalized-question hash.
2. Each problem receives deterministic public and hidden test subsets.
3. Public test output enters Memory; hidden tests execute only during terminal
   reward computation.
4. Docker receives the request through stdin and mounts no host path.
5. Candidate code runs as UID/GID 65534 with network disabled, a read-only root,
   dropped capabilities, process/CPU/memory limits, and a 1 MiB output limit.
6. The backend gate runs a `Solution` class, a standard-input numeric case,
   request-isolation inspection, and output-limit inspection.

Run the gate before Coding preflight:

```bash
bash scripts/build_code_sandbox.sh
python scripts/remote/check_code_sandbox.py
```

## Required remote acceptance sequence

Run the environment audit, then keep the frozen service in a dedicated first
terminal:

```bash
DATA_ROOT=/data bash scripts/remote/audit_environment.sh
bash scripts/serve_frozen_vllm.sh
```

Run acceptance jobs in a second terminal:

```bash
bash scripts/run_deepresearch_baseline.sh
bash scripts/run_deepresearch_preflight.sh
bash scripts/run_coding_baseline.sh
bash scripts/run_coding_preflight.sh
```

The local WSL installation contains the real vLLM/veRL smoke stack. Docker and
the two formal Lucene indexes are remote acceptance dependencies. Formal
training starts after both backend gates and both 32-prompt preflights pass.
