# Data provenance

This repository contains only public benchmark data and deterministic synthetic
Ticket episodes. Runtime code has no real ticketing-system or business-data
dependency.

## Ticket

The formal v3 source splits were generated with
`scripts/synthesize_ticket.py` and synthesis schema `2.0.0`. They are newly
materialized blueprints (not rebalanced copies) and each split is exactly 50:50
direct:indirect.

| file | rows | SHA-256 |
|---|---:|---|
| `ticket/smoke.jsonl` | 16 | `9560bc8881d5b03b8069d15c3be9b697434f2286f324191f4c668f35cd07af2c` |
| `ticket/train.jsonl` | 2000 | `f9aac62fe95a043a14594e288fe1ccd95b2bf6f125dbdb5076eb780fb44e3fa4` |
| `ticket/validation.jsonl` | 256 | `b7a18ae513d4f7390f737ffda2328405b86abc393acceafc6e32a42d01cf1814` |
| `ticket/test.jsonl` | 512 | `7cb9a22c5a21cf05c456b559f4e66bf8159632fb93c240510a516bcc89065eca` |

The historical AgentFlow-generated data from
`main/project/try_ticket_agent/data/generated` is copied byte-for-byte under
`ticket/parity_80_20/` only for runtime parity. Its accepted rows are
approximately 80:20 and must not be used as the formal v3 training distribution.

| parity source | rows | SHA-256 |
|---|---:|---|
| `smoke.jsonl` | 26 | `73a9708d80bcc6bbcd6fb9a9a315ef7e5de2ea7bff650734dc72c0fe64647e0c` |
| `train.jsonl` | 1932 | `b2a4e7daf9d176bd05b64184426541fcc65705d7f33505031497d59f04369e2d` |
| `validation.jsonl` | 192 | `4d23b86c260c19e2f85461c7cd7163a4729ba4486e21c82046510722463425da` |
| `test.jsonl` | 390 | `dbc583f80ce2fca81f74a6bf5388ae0773554c37f3febd991bb3270100a16e42` |

## GSM8K

GSM8K is distributed under the MIT license. The selected files are copied from
`main/project/try_gsm8k_0522/data`:

| destination | source | rows | SHA-256 |
|---|---|---:|---|
| `gsm8k/train.json` | `gsm8k_train_calculator_structured.json` | 1327 | `cee82c56735960c318ee1c89b4aee65192f681e9f4bfb9c55bb23ea56c854224` |
| `gsm8k/test.json` | `gsm8k_test_eval_rest.json` | 319 | `5b3edc64cf843913a903c3e1d76de39220dd57ac6f00c0125b26aefa30534916` |
| `gsm8k/smoke.json` | `gsm8k_smoke_20.json` | 20 | `a65230ba7240331d26ba00aa0d9ad408741a0291dfa61360a980646207df83ad` |

`gsm8k_train_learnable.json` (1827 rows,
`ced9de224e9d92aa...`) is recorded as the other historical candidate but is
not silently merged with the selected calculator-structured training set.

## DeepResearch

The research task uses HotpotQA distractor/fullwiki questions and
2WikiMultiHopQA. `scripts/prepare_deepresearch_data.py` standardizes questions,
answers, and title-plus-sentence supporting facts, audits cross-split duplicate
questions, and applies optional deterministic limits by episode hash. Its
`manifest.json` records source paths, available rows, selected rows, and limits.

Formal HotpotQA global retrieval uses the official introductory-paragraph
Wikipedia corpus. The official archive MD5 is
`01edf64cd120ecc03a2745352779514c`. The corpus converter preserves paragraph
sentence order, the Lucene gate requires at least one million documents, and
sampled gold titles, sentence IDs, and source sentence text are checked before
GPU execution. Formal 2Wiki retrieval uses a separate Lucene index built from
the deduplicated union of all supplied 2Wiki contexts. Local Hotpot distractor
trajectories use only each example's attached documents.

Each benchmark receives a stable-hash 64-row validation subset and a disjoint
500-row final-evaluation subset from its labeled development data. The official
unlabeled leaderboard test data remains reserved for server-side evaluation.

## Coding

The coding task uses the verified TACO source split and retains Easy/Medium
problems with standard-input or call-based tests. The preparation script removes
unsupported tasks and duplicate normalized questions, creates a seeded 80/10/10
problem split, partitions each problem's tests into public and hidden subsets,
and applies optional deterministic split limits. `manifest.json` records source
paths, available and selected rows, filter counts, limits, and split seed.

Planner, Executor, and Verifier prompts receive the problem, current code, and
public-test feedback. Hidden tests remain inside the terminal Docker evaluator
and only contribute the final pass-rate reward.

## veRL build artifacts

`scripts/prepare_verl_data.py` converts the tracked JSON/JSONL files into the
schema consumed by veRL `RLHFDataset`:

```text
data_source, prompt, ability, reward_model, extra_info, agent_name
```

`extra_info` retains each task's complete isolated environment state. Generated
files live under `data/verl/<task>/*.parquet`; they are
ignored build artifacts and accompanied by `data/verl/manifest.json` containing
row counts and SHA-256 values. No synthesis or benchmark content is changed by
this conversion.

## Configuration provenance

Historical Ticket remote runs used question/group/concurrency/planner values
`4/8/32/32`. Historical GSM8K YAML recorded `4/8/32/32`, while the effective
later shell/metrics configuration used `4/6/24/24`; v2 selects the latter and
records the disagreement rather than presenting a fabricated single old value.
The v3 YAML files are the sole default source. Shell entry points select topology
only and do not redefine these values.
