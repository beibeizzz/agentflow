from __future__ import annotations

import inspect
import json


def main() -> int:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
    from verl.trainer.main_ppo_sync import PPOTrainer, tq

    signatures = {
        "AgentLoopBase.__init__": str(inspect.signature(AgentLoopBase.__init__)),
        "PPOTrainer._compute_advantage": str(inspect.signature(PPOTrainer._compute_advantage)),
        "PPOTrainer._compute_old_log_prob": str(inspect.signature(PPOTrainer._compute_old_log_prob)),
        "PPOTrainer._update_actor": str(inspect.signature(PPOTrainer._update_actor)),
        "tq.kv_batch_put": str(inspect.signature(tq.kv_batch_put)),
    }
    required_output_fields = {
        "prompt_ids",
        "response_ids",
        "response_mask",
        "response_logprobs",
        "reward_score",
        "extra_fields",
    }
    output_fields = set(AgentLoopOutput.model_fields)
    missing = sorted(required_output_fields - output_fields)
    print(
        json.dumps(
            {
                "signatures": signatures,
                "agent_loop_output_fields": sorted(output_fields),
                "missing_required_fields": missing,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
