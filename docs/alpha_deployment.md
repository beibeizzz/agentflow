# AgentFlow Alpha deployment

## Scope

`project_alpha` preserves the GSM8K and Ticket experiments and adds two isolated
Planner-training environments:

- DeepResearch: HotpotQA and 2Wiki multi-hop retrieval with terminal answer and
  supporting-fact verification;
- Coding: TACO-Verified Easy/Medium with trajectory-visible public tests and
  terminal hidden tests.

The trainable Planner uses `Qwen3-4B` with LoRA. Query Analyzer, Executor,
Verifier, Generator, and Base Generator share one frozen `Qwen3-8B` vLLM
server. Every role uses no-think mode. Each role can generate up to 1024 tokens,
each role prompt targets a 4096-token ceiling, and each trajectory has at most
five Planner turns.

## Training contract

Formal jobs use prompt batch 4, group 6, Planner-turn mini-batch 8, one PPO
epoch, learning rate `1e-6`, KL coefficient 0, and GSPO asymmetric clipping
`3e-4/4e-4`. A completed trajectory contributes its terminal reward once to
query-local group normalization. The resulting advantage is attached to every
real Planner turn. Actor loss then treats every Planner action as one sequence.

The native veRL field `ppo_mini_batch_size` remains 4 so its pre-rollout config
validator can validate the prompt batch. `agentflow.turn_mini_batch_size=8`
sets the upper bound for expanded Planner-turn actor updates. Each step chooses
the largest value at or below that bound which evenly divides the actual turn
count. The GSPO sequence mean uses that effective optimizer mini-batch as its
denominator. The single-GPU micro-batch and log-prob micro-batch are both 1.

Preflight jobs use 32 prompts and group 8 for eight prompt batches. Successful
preflight requires finite loss and gradient norms, non-empty turn dumps,
multiple rewards inside at least some query groups, valid tool execution, and a
successful checkpoint save.

## WSL2 single-GPU smoke

The local smoke runs the complete training path on one GPU with Qwen3-0.6B for
every role. It uses one prompt, two rollout sessions, up to two Planner turns,
and a 256-token output limit. The external frozen-role server reserves 20% of
GPU memory; the colocated Planner vLLM engine uses 60% while the FSDP LoRA actor
shares the same device.

Create the layered WSL environment from the existing vLLM environment and the
pinned veRL source checkout:

```bash
cd /mnt/c/Users/north/Desktop/agentflow/project_alpha
bash scripts/local/setup_wsl_env.sh
```

Start the frozen-role service in terminal 1:

```bash
bash scripts/local/serve_gsm8k_frozen_single_gpu.sh
```

Run the complete smoke in terminal 2:

```bash
bash scripts/local/run_gsm8k_single_gpu_smoke.sh
```

The runner can also start and clean up its own frozen service when port 8001 is
available. Override `AGENTFLOW_LOCAL_MODEL_PATH`, `VENV_ROOT`, `BASE_VENV`,
`VERL_SOURCE`, and `PROJECT_ROOT` for another local layout.

The validated Windows host used WSL2 Ubuntu 24.04, Python 3.11, PyTorch 2.11,
vLLM 0.20.1, veRL 0.8.0, and one RTX 5070 12 GB. WSL uses shared-memory model
weight transfer. Outputs are written under a UTC run directory in
`outputs/gsm8k/local_single_gpu_smoke/`; a successful run produces
`metrics.jsonl`, `rollouts/1.jsonl`, and `global_step_1/`. A smoke group with
equal terminal rewards is excluded from actor update and records
`actor_update_skipped=1`. The run still validates rollout, old-policy scoring,
weight synchronization, metric persistence, and checkpoint persistence. A
separate native-GSPO contract test verifies finite nonzero gradients with mixed
advantages.

## Data preparation

Install the pinned training stack plus the data and retrieval extras in a clean
Python 3.11 Linux environment:

```bash
pip install -e ".[test,data,research]"
```

Export source splits with the generic Hugging Face adapter:

```bash
python scripts/export_hf_dataset.py --dataset <dataset-id> --name <config> \
  --split <split> --output data/raw/<name>.jsonl
```

Create disjoint labeled validation and final-evaluation subsets from each
benchmark's labeled development split. The stable hash keeps the selection
independent of source row order:

```bash
python scripts/split_labeled_eval.py \
  --input data/raw/hotpot_labeled_dev.json \
  --validation-output data/raw/hotpot_validation.jsonl \
  --test-output data/raw/hotpot_test.jsonl \
  --validation-size 64 --test-size 500 --seed 42
python scripts/split_labeled_eval.py \
  --input data/raw/2wiki_labeled_dev.json \
  --validation-output data/raw/2wiki_validation.jsonl \
  --test-output data/raw/2wiki_test.jsonl \
  --validation-size 64 --test-size 500 --seed 42
```

Prepare DeepResearch examples. Each argument maps one source schema and one
curriculum/output split:

```bash
python scripts/prepare_deepresearch_data.py \
  --source hotpotqa:hotpot_distractor=data/raw/hotpot_distractor.jsonl \
  --source 2wiki:2wiki=data/raw/2wiki_train.jsonl \
  --source hotpotqa:hotpot_fullwiki=data/raw/hotpot_fullwiki_train.jsonl \
  --source hotpotqa:hotpot_validation=data/raw/hotpot_validation.jsonl \
  --source hotpotqa:hotpot_test=data/raw/hotpot_test.jsonl \
  --source 2wiki:2wiki_validation=data/raw/2wiki_validation.jsonl \
  --source 2wiki:2wiki_test=data/raw/2wiki_test.jsonl \
  --limit hotpot_distractor=512 --limit 2wiki=512 \
  --limit hotpot_fullwiki=512
```

The generated context corpus supports local and distractor smoke runs. Formal
FullWiki runs use the official HotpotQA 2017 introductory-paragraph corpus so
citation sentence IDs retain benchmark semantics. The converter reads the
extracted directory of `.bz2` shards directly and also accepts JSON/JSONL:
The official archive MD5 is `01edf64cd120ecc03a2745352779514c`; verify it
against the [HotpotQA corpus documentation](https://hotpotqa.github.io/wiki-readme.html)
before indexing.

```bash
python scripts/prepare_research_corpus.py \
  --input data/raw/hotpotqa_wiki_intro \
  --output data/deepresearch/hotpot_corpus/documents.jsonl
bash scripts/build_research_index.sh \
  data/deepresearch/hotpot_corpus data/indexes/hotpotqa
bash scripts/build_research_index.sh \
  data/deepresearch/context_corpus/2wiki data/indexes/2wiki
```

HotpotQA uses the official million-document FullWiki corpus. The 2Wiki index is
the deduplicated union of the complete contexts carried by every supplied
2Wiki source split. Separate indexes preserve each benchmark's sentence-ID
semantics. The remote backend gate checks sampled gold titles, sentence bounds,
and source sentence text before allocating GPU time.

Prepare TACO-Verified Easy/Medium from its verified source split. The script
deduplicates by normalized question and creates deterministic 80/10/10
train/validation/test partitions with seed 42:

```bash
python scripts/prepare_coding_data.py \
  --source data/raw/taco_verified_train.jsonl \
  --train-limit 1024 --validation-limit 64 --test-limit 500
bash scripts/build_code_sandbox.sh
```

These limits define the initial reproducible subbenchmarks for the two-A800
host. Each manifest records both available and selected rows. Omit or increase
the limits only after the 32-prompt preflight establishes measured trajectory
throughput and projected wall time.

The 0.6B tasks save and validate every 25 steps. DeepResearch and Coding save
and validate every 64 steps. This cadence bounds checkpoint volume on the
340 GB data disk while retaining intermediate recovery points.

The coding pipeline keeps Easy/Medium tasks, filters image/interactive/special
judge tasks, rejects duplicates with a normalized question fingerprint,
retains a separate question-plus-tests fingerprint for data audit, and splits
each problem's tests deterministically into public and hidden partitions. The
runner supports standard-input and call-based tasks, including `Solution`
classes, wrapped expected outputs, and numeric output tolerance. The default
resource gate keeps at most 64 tests and 2 MB of serialized tests per problem;
each public or hidden suite receives one 10-second total execution budget.

Convert all standardized files to veRL Parquet:

```bash
python scripts/prepare_verl_data.py --task deepresearch
python scripts/prepare_verl_data.py --task coding
```

Run the concrete executor backend gates before allocating GPU training time:

```bash
export JAVA_TOOL_OPTIONS="-Xms1g -Xmx8g"
python scripts/remote/check_research_backend.py \
  --index data/indexes/hotpotqa --min-documents 1000000 \
  --examples data/deepresearch/hotpot_fullwiki.jsonl
python scripts/remote/check_research_backend.py \
  --index data/indexes/2wiki --min-documents 1000 \
  --examples data/deepresearch/2wiki.jsonl
python scripts/remote/check_code_sandbox.py
```

DeepResearch uses two AgentLoop workers so the process creates at most two
Pyserini/Lucene JVM clients. Coding uses four AgentLoop workers to bound Docker
container concurrency on the 22-core host. The Planner group still contains six
trajectories per prompt; worker count controls execution concurrency only.

## Two-GPU execution

GPU 0 hosts the frozen-role service. GPU 1 hosts the Planner actor, colocated
vLLM rollout engine, and actor updates. Run the four tasks in two service phases.

For GSM8K and Ticket, start the 0.6B frozen-role service on GPU 0:

```bash
DATA_ROOT=/data bash scripts/remote/audit_environment.sh
MODEL_PATH=model/Qwen/Qwen3-0.6B \
SERVED_MODEL_NAME=Qwen3-0.6B \
bash scripts/serve_frozen_vllm.sh
```

Run the original tasks from a second terminal:

```bash
bash scripts/run_gsm8k_baseline.sh
bash scripts/run_gsm8k_smoke.sh
bash scripts/run_gsm8k_train.sh
bash scripts/run_gsm8k_eval.sh

bash scripts/run_ticket_baseline.sh
bash scripts/run_ticket_smoke.sh
bash scripts/run_ticket_train.sh
bash scripts/run_ticket_eval.sh
```

Stop that foreground service, then start the shared 8B frozen-role service:

```bash
bash scripts/serve_frozen_vllm.sh
```

Run the new tasks from the second terminal:

```bash
bash scripts/run_deepresearch_baseline.sh
bash scripts/run_deepresearch_preflight.sh
bash scripts/run_deepresearch_train.sh
bash scripts/run_deepresearch_eval.sh

bash scripts/run_coding_baseline.sh
bash scripts/run_coding_preflight.sh
bash scripts/run_coding_train.sh
bash scripts/run_coding_eval.sh
```

DeepResearch training executes HotpotQA distractor, 2Wiki global retrieval, and
HotpotQA FullWiki in sequence. Each stage initializes from the previous LoRA
adapter and writes a separate checkpoint directory. The stage runner exports
the latest FSDP checkpoint into a standard PEFT adapter containing only LoRA
weights before loading the next stage. Each stage starts a fresh optimizer while
preserving learned Planner weights. Coding writes an independent adapter under
`outputs/coding/train`.

## Remote environment gate

The target machine provides two A800 80 GB GPUs, 22 CPU cores, 110 GB RAM, and a
data disk expanded toward 340 GB for models, Wiki corpus, Lucene index,
checkpoints, and rollouts. `audit_environment.sh` enforces two 80 GB `sm_80`
GPUs, 22 logical CPUs, 100 GiB host memory, 200 GiB free on `DATA_ROOT`, Docker,
Java 21, Pyserini/Lucene import, the pinned veRL commit, vLLM, and all 17 veRL
config contracts.

Place the complete repository under `DATA_ROOT`; relative `model/`, `data/`, and
`outputs/` paths then remain on the expanded volume. The audit requires the
project root and `DATA_ROOT` to share one filesystem.

Complete both preflight jobs before formal training. Their rollout dumps must
show valid tool events and reward variance; metrics must show finite loss and
gradient norm, a positive trainable-turn count, and a saved checkpoint.
