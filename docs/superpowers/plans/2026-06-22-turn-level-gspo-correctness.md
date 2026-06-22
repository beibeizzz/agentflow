# Turn-Level GSPO Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve turn-level GSPO and trajectory-level advantages while using exact sampled tokens, asymmetric GSPO clipping, and FP32 ratio math.

**Architecture:** Extend rollout records with optional token IDs, add an exact-ID log-probability path beside the existing text path, and make the GSPO objective select the exact path whenever IDs exist. Replace the general trainer's symmetric clip parameter across Python, YAML, shell, logs, and documentation without altering advantage grouping or optimizer scheduling.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, unittest/pytest, YAML, Bash.

---

### Task 1: Exact rollout token identity

**Files:**
- Modify: `project/try_gsm8k_0522/tests/test_flowgrpo_light.py`
- Modify: `project/try_gsm8k_0522/tests/test_flowgrpo_light_agentflow.py`
- Modify: `project/try_gsm8k_0522/flowgrpo_light/policy.py`
- Modify: `project/try_gsm8k_0522/flowgrpo_light/rollout.py`
- Modify: `project/try_gsm8k_0522/flowgrpo_light/agentflow_rollout.py`

- [ ] **Step 1: Write failing tests for sampled IDs and propagation**

Add tests that construct a fake padded generation batch and assert `GeneratedResponse` contains unpadded prompt IDs plus response IDs through the first EOS. Extend AgentFlow adapter tests to assert both ID lists are copied to `PlannerSample`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest project/try_gsm8k_0522/tests/test_flowgrpo_light.py project/try_gsm8k_0522/tests/test_flowgrpo_light_agentflow.py -q`

Expected: failures because token-ID fields do not exist.

- [ ] **Step 3: Implement minimal ID capture and propagation**

Add optional list fields to both dataclasses. In `generate_many()`, derive prompt IDs with `attention_mask.bool()`, slice generated IDs after input width, retain the first EOS, and remove only tokens after termination. Copy IDs in both rollout adapters.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2 and expect all selected tests to pass.

### Task 2: Exact-token likelihood API

**Files:**
- Modify: `project/try_gsm8k_0522/tests/test_flowgrpo_light.py`
- Modify: `project/try_gsm8k_0522/flowgrpo_light/policy.py`

- [ ] **Step 1: Write a failing response-mask test**

Add a test calling the wished-for `sequence_logprob_token_ids_many()` with unequal prompt/response lengths. Assert that response tokens including EOS are scored and prompt/padding positions are excluded.

- [ ] **Step 2: Run the focused test and verify RED**

Run the exact pytest node and expect `AttributeError` for the missing method.

- [ ] **Step 3: Implement the shared exact-ID encoder and likelihood method**

Build `input_ids`, `attention_mask`, and shifted response masks from supplied IDs. Reuse the existing selected-token cross-entropy aggregation. Keep `sequence_logprob_many()` as a text-tokenization wrapper.

- [ ] **Step 4: Run policy tests and verify GREEN**

Run all `TestFlowGRPOLight` tests and expect them to pass.

### Task 3: Asymmetric FP32 GSPO objective

**Files:**
- Create: `project/try_gsm8k_0522/tests/test_gspo_objective.py`
- Modify: `project/try_gsm8k_0522/flowgrpo_light/grpo_objective.py`

- [ ] **Step 1: Write failing objective tests**

Cover asymmetric upper/lower clipping for positive and negative advantages, FP32 ratio/loss from BF16 log probabilities, exact-ID method selection, and text fallback for legacy samples.

- [ ] **Step 2: Run the new test module and verify RED**

Run: `python -m pytest project/try_gsm8k_0522/tests/test_gspo_objective.py -q`

Expected: failures for missing low/high arguments and exact-ID selection.

- [ ] **Step 3: Implement the minimal objective changes**

Represent each loss item with its optional ID lists. Accept `clip_range_low` and `clip_range_high`, cast normalized log probabilities to FP32 before subtraction, and dispatch each micro-batch to the exact-ID or legacy text likelihood API without changing turn expansion, policy epochs, or optimizer steps.

- [ ] **Step 4: Run objective tests and verify GREEN**

Run the command from Step 2 and expect all tests to pass.

### Task 4: General trainer configuration and observability

**Files:**
- Modify: `project/try_gsm8k_0522/tests/test_flowgrpo_light_40g_eval.py` or create a focused config test beside existing GSM8K tests
- Modify: `project/try_gsm8k_0522/flowgrpo_light/train_light_grpo_general.py`
- Modify: `project/try_gsm8k_0522/flowgrpo_general_2x40g/config_train_general_2x40g.yaml`
- Modify: `project/try_gsm8k_0522/flowgrpo_general_2x40g/run_train_general_2x40g.sh`
- Modify: `project/try_gsm8k_0522/flowgrpo_general_2x40g/README (2).md`

- [ ] **Step 1: Write failing CLI/config assertions**

Assert the parser accepts `--clip-range-low` and `--clip-range-high`, rejects non-positive values during configuration, and repository text contains the new names without the legacy symmetric general-path setting.

- [ ] **Step 2: Run focused tests and verify RED**

Run the selected config tests and expect parser/config assertion failures.

- [ ] **Step 3: Wire both clip bounds end-to-end**

Replace parser arguments, resolved config values, objective call arguments, metrics/summary keys, shell environment variables, YAML values, and README examples. Use the requested defaults `0.001` and `0.003`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2 and expect all selected tests to pass.

### Task 5: Regression verification

**Files:**
- Verify only

- [ ] **Step 1: Run the complete GSM8K test suite**

Run: `python -m pytest project/try_gsm8k_0522/tests -q`

Expected: zero failures.

- [ ] **Step 2: Run syntax and repository consistency checks**

Run `python -m compileall -q project/try_gsm8k_0522/flowgrpo_light` and search the general path for legacy `clip_range`/`CLIP_RANGE`. Confirm only low/high names remain.

- [ ] **Step 3: Review the final diff against the specification**

Confirm no edits changed `flatten_rollout_groups`, reward normalization, policy epoch count, optimizer-step scheduling, or VeRL files.
