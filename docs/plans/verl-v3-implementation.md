# veRL v3 strict TDD implementation plan

Status: implemented through CP6 on 2026-07-16. Local verification: 80 tests
passed, source/scripts compiled, and no v2 TRL/LangGraph imports remain. A real
CUDA veRL smoke remains an explicit remote-environment verification step.

## CP0 — repository and dependency contract

1. Freeze veRL v0.8.0 commit and remote Linux requirements.
2. Replace TRL/LangGraph package metadata with veRL/Ray/TransferQueue.
3. Add contract tests for version pins, sampling, LoRA, GSPO, clip values, and
   explicit GSM8K Executor mode.
4. Gate: contract tests pass without importing GPU-only veRL modules.

## CP1 — trajectory/turn protocol and advantage

1. Define serializable Planner-turn and trajectory metadata.
2. Implement final-session selection from `{uid}_{session}_{turn}` keys.
3. Implement valid-only population mean/std normalization.
4. Broadcast one trajectory advantage to all real turns.
5. Record zero-variance, invalid, reward, and advantage metrics.
6. Gate: pure unit tests cover mixed rewards, zero variance, invalid sessions,
   one-valid-session skip, and multi-turn non-duplication.

## CP2 — veRL Planner and frozen-model ports

1. Wrap `LLMServerClient.generate()` as an async Planner port preserving exact
   prompt/response token IDs and rollout log-probs.
2. Add an async OpenAI-compatible frozen subagent client.
3. Preserve think-mode, system prompt, timeout, retry, and temperature behavior.
4. Gate: fake-server tests prove token identity and frozen/trainable separation.

## CP3 — task AgentLoops

1. Port Ticket to a veRL coroutine with a fresh environment per session.
2. Port GSM8K with restored prompts, judge feedback, and Executor dual mode.
3. Return one `AgentLoopOutput` per Planner turn and set reward only on the final
   output.
4. Gate: direct/indirect Ticket and one-/three-turn GSM8K parity tests pass.

## CP4 — veRL trainer adapter

1. Subclass `main_ppo_sync.PPOTrainer`.
2. Override advantage computation with CP1 population semantics.
3. Override actor-update batch metadata so all actual turns form one logical
   mini-batch and `ppo_epochs=2` means exactly two updates.
4. Add TaskRunner/entrypoint using veRL worker/resource/checkpoint machinery.
5. Gate: fake TransferQueue/worker integration proves ordering, two-update
   lifecycle, and no turn re-entry into reward normalization.

## CP5 — data, Hydra configs, and scripts

1. Convert existing GSM8K JSON and Ticket JSONL to veRL parquet rows.
2. Add task-specific AgentLoop registration YAML.
3. Add baseline/train/eval/smoke Hydra configs for both tasks.
4. Add two-GPU launch scripts and frozen-server script.
5. Gate: dry-run config validation and deterministic data hashes pass.

## CP6 — repository cleanup and verification

1. Remove v2-only TRL, LangGraph, Accelerate, collector, and trainer files.
2. Retain prompts, environments, tools, synthesis, provenance, and parity data.
3. Update README, architecture, migration, license, and remote setup docs.
4. Run unit, integration, parity, compile, diff, and optional Linux veRL smoke.
5. Gate: no TRL/LangGraph runtime imports; all local non-GPU tests pass; remote
   smoke command is reproducible from a clean environment.
