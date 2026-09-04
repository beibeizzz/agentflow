from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
VERL_COMMIT = "7aed6b230776f963fa09509c10d9c3a767d1102c"


def test_package_uses_pinned_verl_without_trl_or_langgraph() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    combined = f"{pyproject}\n{requirements}".lower()

    assert 'name = "agentflow-verl"' in pyproject
    assert VERL_COMMIT in combined
    assert "transferqueue==0.1.6" in combined
    assert "trl==" not in combined
    assert "langgraph==" not in combined


def test_all_experiment_configs_preserve_turn_gspo_contract() -> None:
    for task in ("ticket", "gsm8k"):
        for mode in ("baseline", "train", "eval", "smoke"):
            path = ROOT / "configs" / task / f"{mode}.yaml"
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            actor = config["actor_rollout_ref"]["actor"]
            rollout = config["actor_rollout_ref"]["rollout"]
            model = config["actor_rollout_ref"]["model"]

            assert config["algorithm"]["adv_estimator"] == "agentflow_grpo"
            assert config["algorithm"]["use_kl_in_reward"] is False
            assert actor["policy_loss"]["loss_mode"] == "gspo"
            assert actor["clip_ratio_low"] == 0.001
            assert actor["clip_ratio_high"] == 0.003
            assert actor["ppo_epochs"] == 2
            assert actor["use_kl_loss"] is False
            expected_rank = 64 if mode in {"train", "smoke"} else 0
            assert model["lora_rank"] == expected_rank
            assert model["lora_alpha"] == 128
            assert rollout["temperature"] == 1.2
            assert rollout["top_p"] == 1.0
            assert rollout["top_k"] == -1
            assert rollout["repetition_penalty"] == 1.0
            assert rollout["n"] >= 2
            assert rollout["agent"]["default_agent_loop"] == f"agentflow_{task}"
            assert "file" in config["trainer"]["logger"]


def test_gsm8k_executor_mode_is_explicit_and_formal_runs_use_legacy_llm() -> None:
    expected = {
        "baseline": "legacy_llm",
        "train": "legacy_llm",
        "eval": "legacy_llm",
        "smoke": "deterministic",
    }
    for mode, executor_mode in expected.items():
        config = yaml.safe_load(
            (ROOT / "configs" / "gsm8k" / f"{mode}.yaml").read_text(encoding="utf-8")
        )
        assert config["agentflow"]["gsm8k"]["executor_mode"] == executor_mode


def test_agent_loop_registry_targets_all_task_coroutines() -> None:
    registry = yaml.safe_load(
        (ROOT / "configs" / "agent_loops.yaml").read_text(encoding="utf-8")
    )
    targets = {item["name"]: item["_target_"] for item in registry}
    assert targets == {
        "agentflow_ticket": "agentflow_rl.verl.agent_loops.ticket.TicketAgentLoop",
        "agentflow_gsm8k": "agentflow_rl.verl.agent_loops.gsm8k.GSM8KAgentLoop",
        "agentflow_deepresearch": "agentflow_rl.verl.agent_loops.deepresearch.DeepResearchAgentLoop",
        "agentflow_coding": "agentflow_rl.verl.agent_loops.coding.CodingAgentLoop",
    }


def test_new_task_configs_preserve_confirmed_training_contract() -> None:
    for task in ("deepresearch", "coding"):
        for mode in ("baseline", "preflight", "train", "eval"):
            config = yaml.safe_load(
                (ROOT / "configs" / task / f"{mode}.yaml").read_text(encoding="utf-8")
            )
            actor = config["actor_rollout_ref"]["actor"]
            rollout = config["actor_rollout_ref"]["rollout"]
            data = config["data"]
            agentflow = config["agentflow"]
            assert config["algorithm"] == {
                "adv_estimator": "agentflow_grpo",
                "use_kl_in_reward": False,
            }
            assert actor["ppo_epochs"] == 1
            assert actor["ppo_mini_batch_size"] == 4
            assert actor["ppo_micro_batch_size_per_gpu"] == 1
            assert actor["clip_ratio_low"] == 0.0003
            assert actor["clip_ratio_high"] == 0.0004
            assert actor["use_kl_loss"] is False
            assert actor["optim"]["lr"] == 0.000001
            assert rollout["n"] == (8 if mode == "preflight" else 6)
            assert rollout["log_prob_micro_batch_size_per_gpu"] == 1
            assert data["train_batch_size"] == 4
            assert data["max_prompt_length"] == 4096
            assert data["max_response_length"] == 1024
            assert agentflow["max_steps"] == 5
            assert agentflow["turn_mini_batch_size"] == 8
            assert agentflow["role_max_tokens"] == 1024
            assert agentflow["frozen_model"] == "Qwen3-8B"


def test_remote_scripts_use_verl_entrypoint_and_two_gpu_split() -> None:
    entrypoint = (ROOT / "src" / "agentflow_rl" / "verl" / "main.py").read_text(
        encoding="utf-8"
    )
    assert "AgentFlowPPOTrainer" in entrypoint
    assert "run_ppo" in entrypoint

    server = (ROOT / "scripts" / "serve_frozen_vllm.sh").read_text(encoding="utf-8")
    ticket_train = (ROOT / "scripts" / "run_ticket_train.sh").read_text(encoding="utf-8")
    gsm8k_train = (ROOT / "scripts" / "run_gsm8k_train.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=0" in server
    assert "CUDA_VISIBLE_DEVICES=1" in ticket_train
    assert "CUDA_VISIBLE_DEVICES=1" in gsm8k_train
    assert "python -m agentflow_rl.verl.main" in ticket_train
    assert "python -m agentflow_rl.verl.main" in gsm8k_train


def test_local_single_gpu_smoke_uses_small_real_model_budget() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "gsm8k" / "local_single_gpu_smoke.yaml").read_text(encoding="utf-8")
    )
    actor = config["actor_rollout_ref"]["actor"]
    rollout = config["actor_rollout_ref"]["rollout"]
    assert config["actor_rollout_ref"]["model"]["path"].endswith("Qwen3-0.6B}")
    assert config["data"]["train_batch_size"] == 1
    assert config["data"]["max_response_length"] == 256
    assert config["data"]["dataloader_num_workers"] == 0
    assert config["agentflow"]["role_max_tokens"] == 256
    assert config["agentflow"]["max_steps"] == 2
    assert config["agentflow"]["turn_mini_batch_size"] == 8
    assert actor["ppo_mini_batch_size"] == 1
    assert rollout["n"] == 2
    assert rollout["enforce_eager"] is True
    assert rollout["gpu_memory_utilization"] == 0.60
    assert rollout["max_model_len"] == 1280
    assert rollout["max_num_seqs"] == 2
    assert rollout["custom"]["force_shm_weight_transfer"] is True
    assert config["transfer_queue"]["backend"]["SimpleStorage"]["num_data_storage_units"] == 1
