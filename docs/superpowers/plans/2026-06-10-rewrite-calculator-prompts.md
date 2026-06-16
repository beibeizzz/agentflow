# Calculator Prompt Rewriter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable DeepSeek-backed pipeline that rewrites GSM8K questions into strict `Known facts` and `Question` prompts for the calculator-only AgentFlow workflow.

**Architecture:** A deterministic validator rejects malformed or leaking rewrites before a separate DeepSeek judge reviews semantic equivalence and three-step calculator suitability. A pipeline coordinates up to three rewrite attempts, preserves accepted source records with updated `question` and `query`, and writes append-only progress/rejection logs plus a final dataset and summary.

**Tech Stack:** Python standard library, OpenAI Python SDK, DeepSeek OpenAI-compatible Chat Completions API, PyYAML for optional configuration, `unittest`.

---

### Task 1: Schemas and deterministic validation

**Files:**
- Create: `project/try_gsm8k_0522/rewrite_calculator_prompts/__init__.py`
- Create: `project/try_gsm8k_0522/rewrite_calculator_prompts/schemas.py`
- Create: `project/try_gsm8k_0522/rewrite_calculator_prompts/validators.py`
- Test: `project/try_gsm8k_0522/rewrite_calculator_prompts/tests/test_validators.py`

- [ ] **Step 1: Write failing tests**

Test strict two-section parsing, equation and answer leakage rejection, and acceptance of natural-language relationships such as `half`, `twice`, and `3 pens for $2`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
C:\all_software\anaconda3\envs\all-in-rag\python.exe -B -m unittest discover -s project\try_gsm8k_0522\rewrite_calculator_prompts\tests -p "test_validators.py"
```

Expected: import failure because production modules do not exist.

- [ ] **Step 3: Implement schemas and validator**

Define `RewriteCandidate`, `JudgeDecision`, and `ValidationResult`. Implement exact section parsing, forbidden heading/instruction checks, arithmetic/equation leakage checks, answer and intermediate-result leakage checks, and conservative numeric consistency checks.

- [ ] **Step 4: Run tests and verify GREEN**

Expected: all validator tests pass.

### Task 2: Prompt construction and DeepSeek client

**Files:**
- Create: `project/try_gsm8k_0522/rewrite_calculator_prompts/prompts.py`
- Create: `project/try_gsm8k_0522/rewrite_calculator_prompts/deepseek_client.py`
- Test: `project/try_gsm8k_0522/rewrite_calculator_prompts/tests/test_deepseek_client.py`

- [ ] **Step 1: Write failing tests**

Test that rewrite prompts include hidden source solution context but demand JSON containing only `rewritten_question`; test judge prompts request a strict decision schema; test empty content and transient failures retry.

- [ ] **Step 2: Run tests and verify RED**

Expected: missing module failure.

- [ ] **Step 3: Implement prompt builders and client**

Use `deepseek-v4-flash` with thinking disabled for rewriting and `deepseek-v4-pro` with thinking enabled for judging. Read only `DEEPSEEK_API_KEY`, request JSON mode, validate `finish_reason`, parse JSON, collect token usage, and retry empty/transient responses with exponential delay.

- [ ] **Step 4: Run tests and verify GREEN**

Expected: prompt and client tests pass without network access.

### Task 3: Rewrite pipeline and persistence

**Files:**
- Create: `project/try_gsm8k_0522/rewrite_calculator_prompts/pipeline.py`
- Test: `project/try_gsm8k_0522/rewrite_calculator_prompts/tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Test acceptance, local-validation retry, judge retry, rejection after three attempts, synchronized `question`/`query`, append-only progress, and resume behavior that avoids repeat API calls.

- [ ] **Step 2: Run tests and verify RED**

Expected: missing pipeline module.

- [ ] **Step 3: Implement the pipeline**

Inject a client interface for tests. Preserve source fields, replace `question`, rebuild `query`, store no reasoning content, write accepted/rejected progress records atomically where appropriate, and produce summary counters and token usage.

- [ ] **Step 4: Run tests and verify GREEN**

Expected: pipeline tests pass.

### Task 4: CLI, configuration, and documentation

**Files:**
- Create: `project/try_gsm8k_0522/rewrite_calculator_prompts/rewrite_dataset.py`
- Create: `project/try_gsm8k_0522/rewrite_calculator_prompts/config.example.yaml`
- Create: `project/try_gsm8k_0522/rewrite_calculator_prompts/README.md`
- Test: `project/try_gsm8k_0522/rewrite_calculator_prompts/tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Test defaults, `--start`, `--limit`, `--max-attempts`, output paths, missing API key failure, and dry configuration loading.

- [ ] **Step 2: Run tests and verify RED**

Expected: CLI module missing.

- [ ] **Step 3: Implement CLI and docs**

Provide defaults for `data/gsm8k_train_learnable.json`, output files, models, temperature, retries, and concurrency. Document PowerShell/Bash API-key setup, smoke execution, resume behavior, generated files, and cost-conscious limits.

- [ ] **Step 4: Run tests and verify GREEN**

Expected: CLI tests pass.

### Task 5: Full verification

**Files:**
- Verify all files under `project/try_gsm8k_0522/rewrite_calculator_prompts/`

- [ ] **Step 1: Run the new test suite**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\all_software\anaconda3\envs\all-in-rag\python.exe -B -m unittest discover -s project\try_gsm8k_0522\rewrite_calculator_prompts\tests -p "test_*.py"
```

Expected: all tests pass with no network calls.

- [ ] **Step 2: Run syntax checks**

Parse every new Python file with `ast.parse`; expected zero errors.

- [ ] **Step 3: Run CLI help**

```powershell
C:\all_software\anaconda3\envs\all-in-rag\python.exe -B project\try_gsm8k_0522\rewrite_calculator_prompts\rewrite_dataset.py --help
```

Expected: help text lists input, output, model, retry, start, limit, and resume options.

- [ ] **Step 4: Inspect repository changes**

Confirm no source dataset was modified and no API key, generated output, cache, or bytecode file was added.
