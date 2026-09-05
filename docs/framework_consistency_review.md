# AgentFlow framework consistency review

Reviewed on 2026-09-05 against the original AgentFlow role loop, the alpha
AgentLoop implementations, task tools, terminal evaluators, GSPO adapter, and
remote configuration.

## 1. Framework invariant

Every task follows one control contract:

```text
Frozen Query Analyzer
  -> trainable Planner action
  -> Frozen Executor
  -> task tool/environment or Frozen Base Generator
  -> append-only shared Memory
  -> Frozen Verifier: STOP / CONTINUE
  -> Frozen Generator
  -> deterministic terminal evaluator
```

`AgentFlowLoopBase` owns frozen-role calls, Planner token capture, role-specific
Memory projection, deadlines, blocking-backend offload, and veRL output
construction. A task extension owns its data schema, action schema, prompts,
tool environment, step budget, and terminal evaluator. The veRL Trainer and
GSPO adapter remain task-independent.

## 2. Four-task mapping

| Task | Planner action space | Stateful environment | Loop authority | Terminal signal |
|---|---|---|---|---|
| GSM8K | Calculator, Base Generator | executed arithmetic evidence | frozen Verifier | numeric match, 0/1 |
| Ticket | Query, Update, Finish, Base Generator | isolated ticket state and submission | frozen Verifier | goal, action order, finish, side-effect and action checks, 0/1 |
| DeepResearch | Search, Read, Base Generator | benchmark-specific retrieval index | frozen Verifier | answer/supporting-fact joint F1 |
| Coding | Write, Run Tests, Base Generator | current code revision and public-test state | frozen Verifier | hidden-test pass rate |

Ticket Finish records a business submission inside the task state. Search and
Coding end through the common Verifier decision. Time and step limits provide
deterministic safety boundaries for every loop.

## 3. Role responsibility and visibility

| Role | Responsibility | Memory view |
|---|---|---|
| Query Analyzer | identify the task structure and evidence needs | original query and task artifact |
| Planner | choose one sub-goal and one tool | analysis, executed history within budget, latest state/result/judgement |
| Executor | validate and concretize the selected action | proposed action, analysis, latest state/result/judgement, bounded recent history |
| Base Generator | provide supporting reasoning for one sub-goal | requested sub-goal, latest state and bounded recent history |
| Verifier | judge evidence sufficiency and select STOP/CONTINUE | analysis, executed evidence, current state and prior judgements |
| Generator | compose the terminal answer | query, module outputs, evidence, judgements and latest task artifact |

The full Memory remains append-only in rollout records. Each prompt receives a
deterministic projection. Projection priority is latest tagged state, stable
episode identity, then recent history. Coding stores complete code revisions
once in Memory and binds public-test results to both revision and SHA-256.
DeepResearch gives executed Read evidence priority over Search snippets and
generated notes.

Retrieved documents and test output are treated as environment data in role
prompts. This establishes a clear trust boundary for prompt injection and
test-output injection.

## 4. Reward and gradient path

Only the terminal evaluator writes the training reward. Query Analyzer,
Executor, Verifier, Generator, Base Generator, and environment outputs remain
outside `response_mask`.

For one query, `rollout.n` complete sessions form the normalization group. Each
valid session contributes its terminal reward once. Population-normalized
trajectory advantage is then attached to every real Planner action in that
session. Each Planner action enters veRL as one sequence row and receives one
turn-level GSPO sequence ratio. Infrastructure-invalid sessions are excluded;
model and tool-choice failures remain valid low-reward samples.

The episode deadline is an end-to-end task budget, so a `TIME_LIMIT` trajectory
remains a valid zero-reward sample. Connection failures, unavailable backends,
and runtime infrastructure errors set `valid_for_training=False` and leave the
normalization group.

DeepResearch citation grounding is recorded as a diagnostic metric. The
benchmark answer/supporting-fact joint F1 remains its terminal reward. Coding
public tests provide trajectory feedback; hidden tests produce terminal reward.

## 5. Configuration boundary

GSM8K and Ticket use Qwen3-0.6B for Planner and frozen roles. DeepResearch and
Coding use separate Qwen3-4B Planner checkpoints with a shared Qwen3-8B frozen
role service. All task configs explicitly define role output limits, prompt
reserve, and Planner/Executor/Verifier/Generator/Base Generator Memory budgets.

DeepResearch and Coding preserve the confirmed formal settings: prompt batch
4, group 6, Planner-turn mini-batch upper bound 8, learning rate `1e-6`, KL 0,
five turns, 8192-token inputs, and 2048-token outputs. The frozen vLLM server
uses a 10240-token model context to cover the full input/output envelope.

## 6. Resolved consistency issues

1. GSM8K and Ticket now execute the complete Analyzer/Planner/Executor/Memory/Verifier/Generator chain.
2. Every task now uses the shared `MemoryStore` and role-specific projections.
3. Search and Coding use Verifier STOP as the common loop decision.
4. Coding invalidates stale public-test state after every code write and records revision-bound evidence.
5. Coding Verifier and Generator receive one complete current-code copy through Memory.
6. DeepResearch records citation grounding against executed Read events while preserving terminal reward semantics.
7. Latest task state receives first priority when a role Memory budget is tight.
8. GSM8K and Ticket role limits are explicit in every baseline/train/eval/smoke config.
9. Frozen vLLM context covers the configured 8192 input plus 2048 output envelope.
10. Current architecture and deployment documents describe all four tasks and the two-A800 topology.
11. Ticket terminal evaluation enforces direct/indirect action order and indirect lookup grounding.

## 7. Remote acceptance gates

The remote host supplies the remaining execution evidence:

1. environment audit for two A800 80 GB GPUs, pinned veRL, vLLM, Java, Pyserini and Docker;
2. HotpotQA and 2Wiki index coverage checks for sampled supporting facts;
3. Docker sandbox isolation and function/stdio execution checks;
4. frozen-role service health and model-name checks for each service phase;
5. 32-prompt, group-8 DeepResearch and Coding preflights with reward variance;
6. finite loss and gradient norm, positive trainable-turn count, checkpoint save and weight synchronization;
7. baseline, training and evaluation generated from one commit for each task.

These gates validate Linux/CUDA backend integration and empirical model
behavior. The local suite validates deterministic task logic, fake-server role
flows, prompt propagation, reward grouping, GSPO adapter contracts, config
contracts, and failure handling.
