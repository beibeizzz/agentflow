# Ticket Agent Turn-Level GSPO Design

## Objective

Add a deterministic, concurrent ticket-processing sandbox suitable for training a Qwen3-0.6B planner without SFT. The task reuses the repository's completed turn-level GSPO implementation and mirrors the remote experiment layout and configuration style of `project/try_gsm8k_0522/flowgrpo_general_2x40g`.

The first version deliberately limits task complexity so binary reward remains usable with a 0.6B model. It evaluates whether the planner can complete a two- or three-turn workflow with structured tool calls while preserving exact token identity during GSPO updates.

## Scope

The first version includes:

- a frozen Query Analyzer;
- a trainable Qwen3-0.6B LoRA planner;
- three deterministic tools: `Ticket_Query`, `Ticket_Update`, and `Ticket_Finish`;
- worker-local sandbox state;
- a deterministic hidden verifier and binary terminal reward;
- two- or three-turn planner trajectories;
- concurrent rollout collection;
- turn-level GSPO using exact sampled prompt and response token IDs;
- deterministic dataset generation and validation;
- local unit and integration tests plus remote real-model smoke tests.

The first version excludes:

- SFT data construction or SFT training;
- notes or free-text mutation;
- ambiguous matching and clarification tasks;
- no-op and adversarial tasks;
- multi-ticket or multi-field updates;
- LLM-as-a-judge;
- trainable analyzers, executors, verifiers, or final generators;
- trajectory-level importance ratios or trajectory-level clipping;
- execution of model-generated code.

## Isolation and Repository Layout

All task-specific implementation lives under a new sibling directory:

```text
project/try_ticket_agent/
├── README.md
├── flowgrpo_general_2x40g/
│   ├── config_train_general_2x40g.yaml
│   ├── config_eval_learnable_general_2x40g.yaml
│   ├── run_train_general_2x40g.sh
│   ├── run_eval_learnable_general_2x40g.sh
│   └── outputs/
├── ticket_env/
├── agent/
├── data/
├── scripts/
└── tests/
```

Ticket tools are task-local and are not added to `project/agentflow/agentflow/tools`. They therefore do not enter the existing global tool discovery and caching path. Existing files and entry points under `project/try_gsm8k_0522` are not moved, renamed, or structurally reorganized.

The ticket training entry point imports the existing turn-level GSPO primitives from `try_gsm8k_0522/flowgrpo_light`. Shared code may receive only minimal backward-compatible changes required to accept the task-specific rollout provider. Such changes must retain GSM8K behavior and pass its existing tests.

## Runtime Architecture

```text
User request
  -> frozen Query Analyzer
  -> trainable Qwen3-0.6B Planner
  -> exact JSON action
  -> task-local deterministic dispatcher
  -> Ticket_Query / Ticket_Update / Ticket_Finish
  -> trajectory-local SandboxState
  -> deterministic hidden verifier
  -> binary terminal reward
  -> existing turn-level GSPO update
```

The Query Analyzer receives the user request and returns a short structured summary. It is frozen, its tokens are not trained, and its output does not expose hidden state or goals. The planner is the only trainable component.

There is no LLM final-response generator. After `Ticket_Finish`, deterministic code produces the response and terminates the trajectory.

## Ticket State

The sandbox uses strict dataclasses or Pydantic schemas. Each ticket contains only the fields needed by the first version:

- `ticket_id`;
- `customer_id`;
- `order_id`;
- `status`;
- `assigned_team`;
- `priority`;
- a short immutable subject used only as context.

Allowed mutable fields are `status`, `assigned_team`, and `priority`. State transition rules and enum validation are centralized in the environment rather than duplicated in tools.

Each trajectory receives a deep copy of the initial state, action log, step counter, and finish submission. No mutable backend, action log, or ticket object is shared across rollout workers.

## Tool Contract

The planner emits exactly one JSON object per turn. Markdown wrappers, prose, arrays, nested tool calls, unknown top-level fields, and executable code are rejected.

### Ticket_Query

```json
{"tool":"Ticket_Query","arguments":{"lookup_by":"ticket_id","value":"T-1042"}}
```

`lookup_by` is one of `ticket_id`, `customer_id`, or `order_id`. Exactly one lookup is performed. First-version data guarantees a unique result. The tool returns a compact complete ticket record or a structured error.

### Ticket_Update

```json
{"tool":"Ticket_Update","arguments":{"ticket_id":"T-1042","field":"status","value":"resolved"}}
```

The tool changes exactly one allowed field. It validates ticket existence, enum membership, legal status transitions, and closed-ticket restrictions before applying an atomic update.

### Ticket_Finish

```json
{"tool":"Ticket_Finish","arguments":{"ticket_id":"T-1042","outcome":"completed"}}
```

The tool records the submission without mutating ticket fields and terminates the episode. The hidden verifier decides whether the submission and final state are correct.

All expected tool and validation failures return JSON-serializable structured errors. They do not escape as unhandled rollout exceptions.

## Episodes and Curriculum

An episode contains approximately six to ten synthetic tickets and one required mutation on one ticket. It has a maximum of three planner turns.

The initial training distribution is:

- 80% direct-ID tasks: the request contains a unique `ticket_id`; the optimal path is normally `Ticket_Update -> Ticket_Finish`;
- 20% indirect exact-lookup tasks: the request contains a unique `customer_id` or `order_id`; the optimal path is `Ticket_Query -> Ticket_Update -> Ticket_Finish`.

Every request requires only one update to one of `status`, `assigned_team`, or `priority`. Names, fuzzy search, multiple candidates, multiple updates, notes, no-op behavior, and adversarial text are deferred.

The generator is deterministic for a generator version, seed, split, and episode index. It constructs and executes a deterministic reference action sequence to reject invalid generated episodes. Reference actions are used only for dataset validation; no SFT dataset or training target is produced and reference actions are never included in planner prompts or memory.

Training, validation, and test splits must not share episode IDs, initial-state hashes, or normalized request-and-goal signatures.

The indirect-lookup fraction may be raised in a later experiment only after measured validation performance justifies it. This is a dataset/configuration change, not an objective-function change.

## Hidden Verification and Reward

The hidden goal specifies the target ticket, required final field value, required finish submission, and a prohibition on all other mutations. It is available only to deterministic validation and reward code.

Success requires all of the following:

- the required target field has the exact expected value;
- `Ticket_Finish` names the correct ticket and outcome;
- no non-target field changed;
- no other ticket changed;
- the trajectory terminated within three planner turns.

Training and evaluation both use binary terminal reward:

```text
reward = 1.0 if hidden_verifier.success else 0.0
```

There is no shaped reward. Reports must include exact episode success and the fraction of rollout groups with zero reward variance so reward sparsity is visible.

## Turn-Level GSPO Semantics

For each episode, the rollout runner samples `group_size` complete trajectories. Reward normalization occurs exactly once over that episode's trajectory group:

```text
advantage_i = (reward_i - group_mean) / (group_std + epsilon)
```

Each trajectory receives one advantage. That advantage is broadcast to every trainable planner turn in the trajectory. Turns are not inserted back into reward mean/std computation and a multi-turn trajectory is not treated as several rollouts.

Every planner turn remains an independent GSPO sequence. The implementation uses the exact prompt and response token IDs captured during sampling and computes:

```text
ratio_turn = exp(mean_response_logp_new - mean_response_logp_old)
clipped_ratio_turn = clamp(ratio_turn, 1 - 0.001, 1 + 0.003)
```

Ratio and clipping arithmetic remains FP32. The loss does not concatenate a trajectory's turns, compute one trajectory mean log-prob, or apply one trajectory clip. Query Analyzer, observations, tool outputs, verifier output, and deterministic final-response tokens are not response tokens in the policy loss.

The existing turn-flattened optimization behavior is retained: after trajectory-group normalization and advantage broadcast, each valid planner turn contributes a training sequence. Consequently, a three-turn trajectory contributes three turn losses and a two-turn trajectory contributes two. This is intentional for compatibility with the completed turn-level GSPO implementation.

## Invalid Actions and Infrastructure Failures

Invalid JSON, unknown tools, invalid arguments, and illegal transitions are model behavior. Their sampled planner token IDs are retained. A structured error observation is returned if a step remains, allowing correction; the final trajectory receives binary reward according to the hidden verifier, normally zero.

The following are infrastructure failures rather than model failures:

- frozen Analyzer service unavailable;
- planner sampling service failure;
- missing or inconsistent sampled token IDs;
- unrecoverable internal sandbox or dispatcher failure.

Such trajectories are marked `valid_for_training=False` and excluded from the optimizer batch. They must be counted and reported separately rather than converted into reward-zero samples.

## Remote Two-40G Configuration

The new experiment directory mirrors the naming, YAML fields, environment-variable overrides, and launch style of `try_gsm8k_0522/flowgrpo_general_2x40g`. The intended deployment keeps the frozen Qwen service on one GPU and the trainable LoRA policy on the other, matching the existing remote workflow.

Initial training defaults are:

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

The YAML and shell-script defaults must agree. Environment variables may override them explicitly. This avoids the current GSM8K script's ambiguous differences between YAML values and shell defaults while preserving the same operational interface.

No claim is made that these throughput and sampling values are optimal for ticket prompts. The remote smoke test validates memory use and throughput before a full run.

## Tests and Acceptance Criteria

Implementation follows TDD. Required tests cover:

1. Tool behavior: valid calls, missing fields, unknown values, illegal transitions, closed tickets, atomic update, and JSON-serializable results.
2. Sandbox behavior: deep-copy reset, immutable initial-state snapshot, state diff, action log, step count, and finish reset.
3. Parser and dispatcher safety: rejection of prose-wrapped JSON, multiple objects, unknown fields, arrays, Python, markdown mixtures, nested calls, and any `exec`, `eval`, `subprocess`, or `os.system` execution path.
4. Generator and splits: deterministic seeds, 80/20 lookup distribution, unique lookup result, all reference actions verified, and no cross-split duplication.
5. Hidden verifier: exact success, missing update, wrong ticket, extra mutation, wrong finish, missing finish, and step-limit failure.
6. Rollout: two- and three-turn success, correction after structured errors when a turn remains, truncation, terminal finish, and exact sampled token-ID preservation.
7. GSPO semantics: normalization by episode trajectory group exactly once, trajectory advantage broadcast to turns, independent per-turn ratios, FP32 ratio math, and asymmetric clipping at 0.001/0.003.
8. Concurrency: at least 16 simultaneous trajectories with deliberately repeated ticket IDs and no cross-worker state contamination.
9. Local GSPO unit smoke: synthetic token/log-prob inputs produce finite loss and expected parameter gradients without a real model service.
10. Remote real-model smoke: `question_batch_size=2`, `group_size=2`, one optimizer update, LoRA parameter change, finite metrics, adapter save, and adapter reload.
11. Regression: existing `try_gsm8k_0522` tests pass and all existing script paths remain valid.

Local development uses the `all-in-rag` Conda environment where applicable and may ignore the two previously identified missing optional modules. GPU/vLLM behavior is verified only in the remote two-40G environment.

## Metrics

At minimum, training and evaluation report:

- binary episode success rate;
- JSON-valid and tool-call-valid rates;
- correct-ticket and correct-field rates;
- finish correctness;
- average planner turns and truncation rate;
- invalid-action and infrastructure-failure rates;
- reward mean/std;
- zero-variance rollout-group fraction;
- count of nonzero-advantage trajectories and planner turns;
- turn ratio mean/min/max, clip fraction, and approximate KL;
- planner response-token counts.

No dataset validation result, successful smoke test, finite loss, or observed loss decrease may be described as a model-performance improvement.

## Completion Criteria

The first version is complete when:

- all task-specific code is isolated under `project/try_ticket_agent` except minimal tested backward-compatible GSPO integration changes;
- no SFT files or execution path exist;
- the three tools and deterministic dispatcher satisfy their contracts;
- worker-local state passes concurrency isolation tests;
- hidden goals never appear in planner-visible data;
- binary reward and turn-level GSPO grouping semantics are covered by tests;
- exact sampled token IDs are used for every trained planner turn;
- new train/eval configs and scripts mirror the existing remote operational interface and have consistent defaults;
- local non-GPU tests and existing GSM8K regressions pass;
- documented remote smoke commands are available, with remote-only checks clearly marked until executed.
