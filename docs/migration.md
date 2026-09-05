# Migration from v2 to v3

The review source remains the synchronized `main` checkout recorded in
`NOTICE`. v3 preserves the experiment rather than the v2 framework.

The current alpha extends that migration with a shared role-specific Memory
contract, a full Ticket role loop, and Base Generator actions for Ticket and
GSM8K. These changes require a fresh alpha baseline and define the current
four-task runtime described in `docs/architecture.md`.

| v2 component | v3 replacement |
|---|---|
| LangGraph task graphs | veRL `AgentLoopBase` coroutines |
| local `TransformersPlanner` | veRL `LLMServerClient.generate()` |
| thread-pool trajectory collector | AgentLoop workers plus `rollout.n` |
| custom rollout records | `list[AgentLoopOutput]`, one per Planner turn |
| TRL custom Trainer | `main_ppo_sync.PPOTrainer` subclass |
| custom GSPO loss | veRL native `loss_mode=gspo` |
| Accelerate/DDP | Ray resource pools plus FSDP |
| custom checkpoint loop | veRL checkpoint engine and dataloader state |
| custom metrics JSONL | veRL native `file` logger plus AgentFlow metrics |

## Preserved behavior

- baseline first, then Planner-only LoRA GSPO;
- no SFT;
- restored Query Analyzer, Planner, Verifier, Generator prompts;
- Verifier judge feedback enters the next GSM8K Planner prompt;
- explicit GSM8K legacy/deterministic Executor modes;
- binary Ticket and numeric-match GSM8K rewards;
- query-local trajectory normalization before turn flattening;
- population std, valid-only groups, zero-variance zero advantage;
- two updates over one rollout batch;
- temperature 1.2 and asymmetric clips 0.001/0.003;
- formal Ticket direct:indirect ratio 50:50.

## v3 corrections and simplifications

- Top-p is fixed at 1.0, removing the old nucleus re-normalization mismatch.
- Planner prompt/response token IDs come directly from the rollout engine path;
  decoded text is used only for task parsing.
- Actor old/current likelihood is computed by veRL with temperature 1.2.
- A multi-step trajectory can produce multiple queue rows, but only its final
  row enters reward mean/std.
- The actor mini-batch is dynamic because the number of Planner turns varies.
- Baseline/eval/smoke disable automatic resume by default.
- veRL's native file logger restores local `metrics.jsonl` without a parallel
  custom logging lifecycle.

## Removed v2-only code

The independent v3 repository removes the old `training`, `models`, `rollout`
collector, evaluation runner, experiment facade, task adapters, and LangGraph
graph/node modules. It retains only task-local pure logic, synthesis, strict
actions/events, and the veRL integration.

## Remote environment update

Use a clean Linux CUDA environment. v3 requires:

- Python 3.10 or 3.11 recommended;
- veRL v0.8.0 at commit
  `7aed6b230776f963fa09509c10d9c3a767d1102c`;
- Ray `>=2.41.0`, TransferQueue `0.1.6`, torchdata, PEFT, Transformers, and
  tensordict from veRL's dependency set;
- vLLM `>=0.8.5,<=0.12.0` for the Planner rollout;
- an OpenAI-compatible frozen vLLM server on GPU0.

Do not reuse the v2 TRL/LangGraph environment as-is. Install `requirements.txt`
in a clean environment, then install this repository editable and regenerate
`data/verl/`.
