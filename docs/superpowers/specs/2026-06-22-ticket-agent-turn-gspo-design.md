# Ticket AgentFlow and Turn-Level GSPO Design

## Objective

Add a synthetic ticket-processing task that first runs as a standard AgentFlow baseline and then trains only `Planner.generate_next_step` with the repository's existing turn-level GSPO implementation. The task targets Qwen3-0.6B, uses no SFT, has no real business dependencies, and mirrors `project/try_gsm8k_0522` in experiment layout and execution order.

The first version limits the workflow to three tools and two or three planner turns so binary reward remains viable for a 0.6B model.

## Experiment Sequence

1. **Ticket AgentFlow baseline:** base Qwen3-0.6B performs query analysis and planner next-step generation through the core AgentFlow Solver. Tools and verification are deterministic.
2. **Ticket AgentFlow + GSPO:** the query analyzer is frozen Qwen3-0.6B and planner next-step generation is replaced by a trainable Qwen3-0.6B LoRA policy. The same Solver, Memory, tools, backend, verifier, data, and step limit are used.
3. **Evaluation:** baseline and LoRA adapter run at planner temperature zero on identical held-out episodes.

There is no SFT stage, SFT dataset, or oracle planner target.

## Scope

The first version includes core AgentFlow integration, a general structured-action Planner/Executor mode, three registered Ticket BaseTools, worker-local state, deterministic hidden verification, binary reward, LLM-assisted request synthesis, concurrent rollouts, exact sampled token IDs, and current turn-level GSPO.

It excludes notes, ambiguous matching, clarification, no-op/adversarial tasks, multiple updates, an LLM training/evaluation judge, trainable non-Planner components, trajectory-level ratios, and model-generated code execution in the Ticket path.

The data-synthesis judge is an offline dataset filter only. It is not a reward model and is not called during baseline, training, or evaluation.

## Core AgentFlow Extension

The Ticket task follows `try_gsm8k_0522`: construct a core Solver, register standard tools, and replace `solver.planner.llm_engine` with the trainable policy only during GSPO rollout.

### Structured Planner Action

Add `StructuredToolAction` to `agentflow.models.formatters`:

```json
{
  "tool_name": "Ticket_Update_Tool",
  "arguments": {
    "ticket_id": "T-1042",
    "field": "status",
    "value": "resolved"
  }
}
```

`Planner` gains explicit `action_mode="structured"`. It prompts for one exact action object using available tool metadata. Extraction accepts a `StructuredToolAction` or strict JSON object and returns the tool name plus canonical JSON arguments for the Solver loop. Existing Planner and Calculator behavior remain the default.

### Deterministic Structured Executor

`Executor` gains explicit `execution_mode="structured"`. In this mode it does not call an LLM, treats canonical argument JSON as the internal command, finds the cached tool instance, and calls `tool.execute(**arguments)`. Generated Python, `exec`, and `eval` are never used. The existing Executor mode remains the default.

### Workflow Output

`Solver` gains `output_types="workflow"`. This runs query analysis, planner actions, tools, Memory, and Verifier but skips LLM final/direct generation. `construct_solver` passes the new Planner and Executor modes. Existing `base`, `final`, and `direct` behavior remains unchanged.

## Ticket Tools and Backend

Add standard AgentFlow tools under:

```text
project/agentflow/agentflow/tools/ticket_common/
project/agentflow/agentflow/tools/ticket_query/
project/agentflow/agentflow/tools/ticket_update/
project/agentflow/agentflow/tools/ticket_finish/
```

The tool names are `Ticket_Query_Tool`, `Ticket_Update_Tool`, and `Ticket_Finish_Tool`. Each inherits `BaseTool` and is discovered by the existing `Initializer`.

The three instances accept a bindable `TicketBackend`. Before binding, calls return `BACKEND_NOT_BOUND`. A Ticket solver factory constructs a normal Solver, creates one backend, and binds that backend to all three cached instances.

Each rollout worker owns a distinct Solver, Memory, backend, and tool set. Each episode resets the backend from a deep copy and binds its hidden goal only to deterministic verification. No mutable module-level state is allowed.

`Ticket_Query_Tool` accepts exactly `lookup_by,value`; lookup is `ticket_id`, `customer_id`, or `order_id`. `Ticket_Update_Tool` accepts exactly `ticket_id,field,value` and atomically updates one of `status`, `assigned_team`, or `priority`. `Ticket_Finish_Tool` accepts exactly `ticket_id,outcome` and records a submission without mutating ticket fields. Expected errors are JSON-serializable results.

## Runtime Architecture

### Baseline

```text
User request
  -> Core Planner.analyze_query (base Qwen3-0.6B)
  -> Core Planner.generate_next_step (base Qwen3-0.6B)
  -> Core Executor(structured)
  -> Registered Ticket BaseTools
  -> Worker-local TicketBackend
  -> Core Memory
  -> Deterministic TicketVerifier
  -> Core Solver(workflow)
```

### GSPO

```text
User request
  -> Core Planner.analyze_query (frozen Qwen3-0.6B)
  -> Core Planner.generate_next_step (trainable LoRA policy)
  -> the same Executor, tools, backend, Memory, verifier, and Solver
  -> binary terminal reward
  -> existing turn-level GSPO update
```

The deterministic TicketVerifier implements the core Verifier interface. It stops after a finish submission and supplies structured feedback after other steps. A wrong finish terminates but receives reward zero.

## Ticket State and Curriculum

Each episode contains six to ten synthetic tickets with `ticket_id`, `customer_id`, `order_id`, immutable `subject`, and mutable `status`, `assigned_team`, and `priority`.

Every episode requires one legal field update on one ticket and a correct finish within three planner turns:

- 80% direct-ID tasks include the target `ticket_id`; optimal path is Update then Finish.
- 20% indirect tasks include a unique `customer_id` or `order_id` but not the target ticket ID; optimal path is Query, Update, then Finish.

Names, fuzzy matching, multiple candidates, notes, no-op behavior, adversarial content, and multiple updates are deferred.

## LLM-Assisted Data Synthesis

Data synthesis follows the engineering pattern in `try_gsm8k_0522/rewrite_calculator_prompts` with task-specific modules:

```text
deterministic episode blueprint
  -> canonical request
  -> rewrite API
  -> deterministic validator
  -> judge API
  -> reference execution through AgentFlow Ticket tools
  -> hidden verifier
  -> accepted dataset and manifest
```

The blueprint owns all state, identifiers, legal transition, target field/value, lookup mode, and finish outcome. The rewrite model returns only `user_request` and cannot define the goal.

The deterministic validator rejects missing required values, introduced identifiers or values, indirect target-ticket leakage, tool/JSON/goal/trace hints, second mutations, excessive length, invalid characters, and non-unique targets. The judge checks semantic equivalence. Each candidate has at most three semantic attempts with feedback; transport retry is separate.

The pipeline supports fake SDK injection, bounded concurrency, resume, progress/rejected JSONL, token usage, and atomic final writes. Configuration follows:

```yaml
base_url: https://api.deepseek.com
rewrite_model: deepseek-v4-flash
judge_model: deepseek-v4-pro
rewrite_temperature: 0.3
max_attempts: 3
transport_attempts: 4
concurrency: 2
resume: true
```

API output is not assumed byte-reproducible across time. Blueprints, accepted requests, API metadata, progress records, and final hashes are frozen before experiments. Training never synthesizes online.

Splits and identifiers are assigned before API calls and cannot share episode IDs, initial-state hashes, or normalized request-goal signatures. Reference actions execute through actual registered Ticket tools and the hidden verifier but are never stored as Planner targets or exposed to the model.

## Hidden Verification and Binary Reward

Success requires the exact target value, correct ticket and finish outcome, no collateral mutation, no invalid Planner response, no tool error, and completion within three turns.

```text
reward = 1.0 if verification.success else 0.0
```

There is no shaped reward. Recovery after an invalid Planner action still receives zero so positive trajectory advantage cannot be broadcast to a known-invalid turn.

Unavailable model services, generation exceptions, missing token IDs, and internal backend failures are infrastructure errors marked `valid_for_training=False` and excluded before normalization.

## Shared AgentFlow GSPO Rollout

`flowgrpo_light/agentflow_rollout.py` gains an optional task rollout adapter and episode-reset callback. Without callbacks, current GSM8K behavior is unchanged. Ticket callbacks reset backend/Memory, run the core Solver, and convert verification to the shared rollout result.

The same `BatchedAgentFlowPlannerEngine` replaces `solver.planner.llm_engine`. Query analysis remains frozen. No further change is expected in `policy.py`, `rollout.py`, or `grpo_objective.py` beyond the completed exact-token and turn-level work.

## Turn-Level GSPO Semantics

For each episode, sample `group_size` trajectories and compute reward mean/std once across valid trajectories in that episode group:

```text
advantage_i = (reward_i - group_mean) / (group_std + epsilon)
```

Broadcast trajectory advantage to its Planner turns. Turns do not re-enter reward normalization and a multi-turn trajectory is not multiple rollouts.

Each Planner turn remains one independent sequence using exact sampled IDs:

```text
ratio_turn = exp(mean_response_logp_new - mean_response_logp_old)
clipped_ratio_turn = clamp(ratio_turn, 1 - 0.001, 1 + 0.003)
```

Ratio arithmetic remains FP32. A three-turn trajectory contributes three turn losses and a two-turn trajectory contributes two. Analyzer, tool, verifier, and deterministic output tokens are excluded.

## Repository Changes

Core modifications are limited to:

```text
project/agentflow/agentflow/models/formatters.py
project/agentflow/agentflow/models/planner.py
project/agentflow/agentflow/models/executor.py
project/agentflow/agentflow/solver.py
project/agentflow/agentflow/tools/ticket_common/
project/agentflow/agentflow/tools/ticket_query/
project/agentflow/agentflow/tools/ticket_update/
project/agentflow/agentflow/tools/ticket_finish/
```

Shared GSPO integration modifies only `flowgrpo_light/agentflow_rollout.py` and its regression tests. All experiment-specific code is under `project/try_ticket_agent`. Existing GSM8K scripts, configs, defaults, and layout are not renamed or reorganized.

## Remote Two-40G Configuration

The experiment mirrors `flowgrpo_general_2x40g` names, YAML keys, overrides, and launch style. Initial GSPO defaults are:

```yaml
model_path: /home/north/vllm_test/models/Qwen/Qwen3-0.6B
question_batch_size: 4
group_size: 8
rollout_concurrency: 32
planner_batch_size: 32
planner_batch_timeout_s: 0.10
max_steps: 3
dtype: bfloat16
gradient_checkpointing: true
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.0
learning_rate: 2e-6
weight_decay: 0.0
max_grad_norm: 1.0
clip_range_low: 0.001
clip_range_high: 0.003
policy_epochs: 2
planner_max_new_tokens: 256
planner_temperature: 1.2
planner_top_p: 0.95
think_mode: off
query_analysis_think_mode: on
reward_mode: binary
```

Baseline uses the base Qwen vLLM endpoint. During GSPO, frozen Qwen uses GPU 0 and the trainable policy uses GPU 1. YAML and shell defaults must agree; environment variables are explicit overrides.

## TDD and Acceptance

Implementation order is:

1. structured formatter, Planner, Executor, and workflow Solver;
2. registered Ticket tools and backend binding;
3. Ticket solver factory, verifier, and AgentFlow baseline;
4. deterministic blueprints and reference execution;
5. fake-client-tested rewrite/judge pipeline;
6. shared rollout adapter and worker isolation;
7. turn-level GSPO integration;
8. baseline/adapter evaluation and remote configs;
9. regression and remote smoke.

Tests must prove strict JSON handling; no LLM or generated-code execution in structured Executor; unchanged Calculator/legacy behavior; correct tool errors and atomicity; shared backend within a Solver and isolation across workers; no goal leakage; deterministic 80/20 blueprints and split isolation; API retry/resume/concurrency/usage/rejection; reference execution through AgentFlow; valid two/three-turn baseline; 16-worker isolation; exact token IDs; one normalization per trajectory group; broadcast advantage; per-turn FP32 ratio and 0.001/0.003 clip; safe zero-variance skipping; GSM8K regression; and remote adapter save/reload.

## Metrics

Report binary success; JSON/action/tool validity; correct ticket/field/finish; direct versus indirect success; average turns and truncation; invalid/tool/infrastructure errors; reward mean/std; zero-variance groups; nonzero-advantage trajectories/turns; turn ratio/clip/KL; token counts; and controlled baseline-versus-adapter results.

Dataset generation, smoke success, finite loss, or loss decrease must not be described as model improvement.

## Completion Criteria

The task is complete when the baseline runs through core AgentFlow and registered BaseTools; GSPO replaces only Planner next-step generation; structured mode is opt-in with legacy regressions passing; no Ticket path executes generated code; no SFT exists; synthesized data passes deterministic validation, judge filtering, reference execution, hashing, and leakage checks; state isolation passes; exact token and turn-level GSPO semantics are tested; and baseline/train/eval scripts reproduce the existing remote operational pattern.
