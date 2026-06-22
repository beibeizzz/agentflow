# Turn-Level GSPO Correctness Design

## Goal

Keep the existing turn-level GSPO objective and trajectory-level advantage behavior, while making the minimum changes required for a correct sequence-level importance ratio:

- use asymmetric GSPO clipping;
- compute ratios from the exact sampled token sequences;
- compute log-ratios and ratio statistics in FP32.

## Reviewed Data Flow

The `flowgrpo_general_2x40g` launcher calls `flowgrpo_light/train_light_grpo_general.py`. AgentFlow planner calls are intercepted by `flowgrpo_light/agentflow_rollout.py`, which stores each trainable planner call as a `PlannerSample`. `flatten_rollout_groups()` computes one normalized advantage per valid trajectory, and `grpo_objective.build_loss_items()` broadcasts that advantage to every planner turn in the trajectory.

The optimizer currently treats each planner turn as one GSPO sequence. This design preserves that unit. It does not flatten turns into the reward mean/std calculation and does not change optimizer-step or trajectory weighting behavior.

## Required Changes

### Exact rollout token identities

`GeneratedResponse` and `PlannerSample` will gain optional `prompt_token_ids` and `response_token_ids` fields.

`PlannerPolicy.generate_many()` will capture token IDs directly from the tensors used by and returned from `model.generate()`:

- prompt IDs exclude left-padding positions according to the input attention mask;
- response IDs start immediately after the padded input width;
- generated padding after termination is excluded;
- the first generated EOS is retained as part of the sampled response.

AgentFlow and light rollout adapters will copy these IDs into `PlannerSample` without changing the decoded text used by the solver.

### Exact-token log probabilities

`PlannerPolicy` will expose a batched log-probability method accepting prompt-ID and response-ID lists. It will construct padded model inputs, mask only sampled response positions, and sum the selected response-token log probabilities.

The existing text-based log-probability API remains available for compatibility. The GSPO objective will prefer exact IDs and fall back to text only for manually constructed or legacy samples that do not contain IDs.

### Asymmetric clipping

The single `clip_range` input will be replaced on the general training path by:

```yaml
clip_range_low: 0.0003
clip_range_high: 0.0004
```

The clipped ratio interval is `[1 - clip_range_low, 1 + clip_range_high]`. Both values must be positive. The general launcher will use `CLIP_RANGE_LOW` and `CLIP_RANGE_HIGH`; the old `CLIP_RANGE` option will not be accepted silently.

### FP32 ratio math

Normalized current and old log probabilities will be converted to FP32 before subtraction and exponentiation. Ratio metrics and clipping decisions therefore use FP32 even when the model runs in BF16. Gradients still propagate to the model through the FP32 cast.

## Explicit Non-Goals

This change will not modify:

- turn-level GSPO optimization;
- trajectory-level reward normalization or advantage broadcasting;
- invalid-rollout filtering;
- `policy_epochs` or optimizer-step frequency;
- rollout group size, temperature, learning rate, or KL regularization;
- per-turn versus per-trajectory loss weighting;
- the separate VeRL Flow-GRPO path.

## Compatibility and Failure Handling

Text-only `GeneratedResponse` and `PlannerSample` construction remains valid through optional token-ID fields. Training fails early for non-positive low/high clip values. The general launcher and documentation stop advertising the old symmetric parameter so stale GSPO settings are visible instead of silently mapped.

## Test Design

Tests will be written before production changes and will verify:

1. batched generation stores unpadded prompt IDs and response IDs through the first EOS while excluding later padding;
2. exact-token log-probability masking scores response tokens, including EOS, and excludes prompt/padding tokens;
3. the GSPO objective uses token-ID log probabilities when IDs are present and retains the legacy text fallback;
4. asymmetric clipping handles upper and lower boundaries correctly for positive and negative advantages;
5. ratio math returns FP32 outputs when normalized inputs are BF16;
6. CLI, YAML, shell launcher, metrics, and summary output carry both clip bounds;
7. existing trajectory-level advantage grouping and turn-level sample expansion remain unchanged.

## Acceptance Criteria

- The relevant new tests demonstrate expected failures before implementation and pass afterward.
- All existing `try_gsm8k_0522` tests pass.
- No production path under `flowgrpo_general_2x40g` refers to the old symmetric `clip_range` or `CLIP_RANGE`.
- A code search confirms reward normalization and advantage grouping were not changed.
