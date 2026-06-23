# Ticket Agent Hardening Design

## Goal

Close the post-merge review gaps before remote experiments without changing the accepted task contract: three deterministic tools, two-to-three Planner turns, no SFT, binary trajectory reward, query-group advantage normalization, and per-turn exact-token GSPO.

## Scope

The change is limited to Ticket synthesis resume safety, shared AgentFlow rollout error classification, Ticket evaluation output isolation, Ticket training metrics/config validation, synthesis validator precision, tests, and documentation. It must preserve legacy GSM8K defaults and the current GSPO objective.

## 1. Strict synthesis resume identity

Each blueprint receives a stable SHA-256 fingerprint over its complete canonical serialized payload. Every progress record stores this fingerprint plus a synthesis progress schema version. On resume, the pipeline loads only records whose episode IDs are present in the current input and requires every stored fingerprint and schema version to match.

Any mismatch fails before API calls or output rewriting with an actionable error instructing the operator to use `resume: false` or a new output directory. Extra stale progress records also fail instead of contaminating the accepted dataset. A matching resume continues to make zero API calls for completed records.

## 2. Rollout infrastructure failure containment

The batch worker places question extraction, row-aware reset, sample clearing, solve, and result adaptation inside one exception boundary. Any exception produces one `RolloutResult` with reward zero and `valid_for_training=False`; it never escapes through `Future.result()` to abort the remaining query group. Samples captured before a failure remain attached when available.

This behavior applies to both new task hooks and the legacy default path. Model-valid failures returned by the deterministic verifier remain `valid_for_training=True` and participate in binary reward normalization.

## 3. Controlled evaluation isolation

When no explicit output directory override is supplied, evaluation derives a mode-specific directory: `eval_baseline` or `eval_adapter`. The configured adapter directory is treated as the adapter-mode default only. Baseline and adapter runs therefore cannot overwrite each other's details or summary files.

The evaluation CLI and launcher expose an explicit output override for remote use. Existing model, dataset, temperature, max-step, structured workflow, and verifier parity remains unchanged.

## 4. Training observability and config enforcement

Training validates `reward_mode == "binary"` before model or runner construction. Each step records:

- reward count, mean, population standard deviation, minimum, and maximum;
- valid and infrastructure-failed trajectories;
- nonzero trajectories and turns, response token count, and zero-variance groups;
- loss, ratio mean/min/max, clip fraction, approximate KL, policy epochs, and update status at top level;
- CUDA availability, device, allocated/reserved memory, and peak allocated/reserved memory;
- rollout, training, and total step durations.

The final summary includes adapter path, update/skip counts, clip ranges, policy epochs, steps, rows, and elapsed time. Nested `train_stats` remains for backward-compatible detail.

## 5. Natural-language validation precision

Identifier, target-leak, required-value, multiple-mutation, tool-hint, and finish-intent checks remain deterministic. Enum rejection changes from "the word immediately after a field label" to explicit assignment-shaped phrases only. Natural wording such as “status of the ticket” does not create an unsupported enum, while “set status to pending” still does.

The documentation describes the second DeepSeek call as an LLM judge used only during data synthesis, not as an offline judge.

## 6. Local synthesis

Local synthesis is supported because it is CPU/lightweight orchestration around an OpenAI-compatible API. Requirements are `openai`, `PyYAML`, a valid `DEEPSEEK_API_KEY`, network access to the configured base URL, and a copied `config_synthesis.yaml`. The pipeline does not require a local GPU. It may incur API charges and must never run implicitly during tests.

Recommended local sequence:

```powershell
cd project
Copy-Item try_ticket_agent/config_synthesis.example.yaml try_ticket_agent/config_synthesis.yaml
python try_ticket_agent/scripts/synthesize_dataset.py --config try_ticket_agent/config_synthesis.yaml
python try_ticket_agent/scripts/validate_dataset.py --dataset try_ticket_agent/data/generated
```

## Testing

Strict TDD adds focused red tests for stale/mismatched/extra progress, matching zero-call resume, reset/getter containment, mode-specific evaluation paths, binary-mode validation, metric shape, GPU snapshots, natural validator phrases, and documentation wording. After focused tests pass, run all Ticket tests and the existing GSPO/exact-token/AgentFlow batch suites. Real API calls and remote GPU training remain explicit operator steps.

## Acceptance Criteria

1. Resume cannot silently reuse or emit records from a different blueprint set.
2. A reset or question extraction exception cannot abort a rollout batch.
3. Baseline and adapter evaluation default outputs are disjoint.
4. Required remote diagnostic fields are present and binary mode is enforced.
5. Valid natural ticket wording is not rejected by enum parsing.
6. Local synthesis prerequisites and commands are accurate, and tests make no network calls.
