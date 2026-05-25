# GSM8K AgentFlow Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the local `Qwen3-0.6B-Instruct` vLLM-backed AgentFlow pipeline on GSM8K, first on a small smoke subset and then on all 1319 test examples.

**Architecture:** Keep the evaluation self-contained under `try_gsm8k_0522`. Convert GSM8K parquet data into JSON once, run each problem independently with resumable per-problem result files, and score results with deterministic numeric answer extraction instead of an LLM judge.

**Tech Stack:** Python 3.11, AgentFlow inner runtime package, vLLM OpenAI-compatible API at `http://localhost:8000/v1`, stdlib `unittest`, and `pyarrow` for reading parquet.

---

### Task 1: Deterministic GSM8K Scoring Helpers

**Files:**
- Create: `try_gsm8k_0522/gsm8k_utils.py`
- Create: `try_gsm8k_0522/tests/test_gsm8k_utils.py`

- [ ] Write tests for extracting gold answers from `#### ...`.
- [ ] Write tests for extracting the last numeric prediction from free-form model output.
- [ ] Write tests for comparing equivalent numeric strings such as `1,000`, `1000.0`, and `7/2`.
- [ ] Implement helper functions with no AgentFlow dependency.
- [ ] Run `python -m unittest discover -s try_gsm8k_0522/tests -v`.

### Task 2: Data Preparation

**Files:**
- Create: `try_gsm8k_0522/prepare_gsm8k_json.py`
- Output: `try_gsm8k_0522/data/gsm8k_test.json`
- Output: `try_gsm8k_0522/data/gsm8k_smoke_20.json`

- [ ] Read `data/gsm8k/main/test-00000-of-00001.parquet`.
- [ ] Write JSON rows with `pid`, `question`, `query`, `answer`, and `gold_answer`.
- [ ] Support `--smoke-size 20`.
- [ ] Validate expected row counts before writing.

### Task 3: AgentFlow Runner

**Files:**
- Create: `try_gsm8k_0522/run_gsm8k_agentflow.py`
- Output: `try_gsm8k_0522/results/smoke/output_*.json`
- Output: `try_gsm8k_0522/results/full/output_*.json`

- [ ] Bootstrap the inner AgentFlow runtime as top-level `agentflow`.
- [ ] Check `/v1/models` for `Qwen3-0.6B-Instruct` before running.
- [ ] Construct the solver with `enabled_tools=["Base_Generator_Tool"]`, `tool_engine=["self"]`, and all model roles set to `trainable`.
- [ ] Run examples one at a time, write each result atomically, and skip existing result files by default.
- [ ] Include CLI flags for data path, output dir, limit, max steps, max time, model name, and base URL.

### Task 4: Scoring Script

**Files:**
- Create: `try_gsm8k_0522/score_gsm8k.py`
- Output: `try_gsm8k_0522/summary/smoke_summary.json`
- Output: `try_gsm8k_0522/summary/full_summary.json`

- [ ] Load prepared data and per-problem outputs.
- [ ] Score `direct_output` using deterministic numeric comparison.
- [ ] Save aggregate accuracy and per-problem details.
- [ ] Report missing result files separately from wrong answers.

### Task 5: Entry Scripts and README

**Files:**
- Create: `try_gsm8k_0522/run_smoke.sh`
- Create: `try_gsm8k_0522/run_full.sh`
- Create: `try_gsm8k_0522/README.md`

- [ ] `run_smoke.sh` prepares data, runs 20 examples, and scores smoke results.
- [ ] `run_full.sh` prepares data, runs all 1319 examples, and scores full results.
- [ ] README documents prerequisites, commands, outputs, and known limitations.

### Task 6: Verification

**Commands:**
- `python -m unittest discover -s try_gsm8k_0522/tests -v`
- `bash try_gsm8k_0522/run_smoke.sh`
- `bash try_gsm8k_0522/run_full.sh`

- [ ] Confirm the helper tests pass.
- [ ] Confirm smoke produces 20 result JSON files and a summary.
- [ ] Confirm full produces 1319 result JSON files and a summary, or document any failed/missing cases.
