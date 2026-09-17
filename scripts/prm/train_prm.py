from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from agentflow_rl.prm import (
    load_process_labels,
    regression_metrics,
    split_labels_by_prompt,
    write_process_labels,
)
from agentflow_rl.rewards.transition_view import encode_process_transitions, PROCESS_VIEW_REVISION
from agentflow_rl.rewards.rubric import PROCESS_RUBRIC_REVISION
from agentflow_rl.prm.training import validate_label_protocol


def _batch_plan(
    *,
    train_examples: int,
    world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, int]:
    if min(
        train_examples,
        world_size,
        per_device_batch_size,
        gradient_accumulation_steps,
    ) <= 0:
        raise ValueError("PRM batch-plan inputs must be positive")
    samples_per_rank = math.ceil(train_examples / world_size)
    batches_per_rank = math.ceil(samples_per_rank / per_device_batch_size)
    return {
        "world_size": world_size,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": (
            per_device_batch_size * gradient_accumulation_steps * world_size
        ),
        "train_examples": train_examples,
        "optimizer_steps_per_epoch": math.ceil(
            batches_per_rank / gradient_accumulation_steps
        ),
    }


def main() -> int:
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    parser = argparse.ArgumentParser(description="Train the Qwen3-0.6B AgentFlow PRM")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values bound a stress preflight; -1 runs the configured epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Physical sequence batch per DDP process/GPU.",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--group-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=None,
        help="Fail when torchrun WORLD_SIZE differs from the frozen experiment value.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")

    labels = load_process_labels(args.labels)
    validate_label_protocol(labels)
    train_labels, dev_labels = split_labels_by_prompt(
        labels, dev_fraction=0.10, seed=str(args.seed)
    )
    if not train_labels or not dev_labels:
        raise RuntimeError("PRM training requires non-empty grouped train and dev splits")
    args.output.mkdir(parents=True, exist_ok=True)
    # torchrun launches one process per GPU. Only rank zero writes shared
    # split artifacts; every rank builds the same deterministic in-memory split.
    process_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.expected_world_size is not None and world_size != args.expected_world_size:
        raise RuntimeError(
            f"expected WORLD_SIZE={args.expected_world_size}, got {world_size}; "
            "launch the frozen PRM configuration with torchrun"
        )
    batch_plan = _batch_plan(
        train_examples=len(train_labels),
        world_size=world_size,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
    )
    if process_rank == 0:
        write_process_labels(train_labels, args.output / "train_labels.jsonl")
        write_process_labels(dev_labels, args.output / "dev_labels.jsonl")
        (args.output / "training_batch_manifest.json").write_text(
            json.dumps(
                {
                    **batch_plan,
                    "epochs": args.epochs,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "use_cache": False,
                    "attention_implementation": "flash_attention_2",
                    "max_length": args.max_length,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # PRM classification processes each transition in one full forward pass.
    # Generation cache has no reuse here and conflicts with checkpointed layers.
    model.config.use_cache = False
    model.config.agentflow_process_view_revision = PROCESS_VIEW_REVISION
    model.config.agentflow_process_rubric_revision = PROCESS_RUBRIC_REVISION

    def make_dataset(values):
        rows = []
        for item in values:
            encoded = encode_process_transitions(
                [item.transition], tokenizer, max_length=args.max_length,
            )
            rows.append({**{key: value[0] for key, value in encoded.items()}, "labels": item.target})
        return Dataset.from_list(rows)

    class SigmoidRegressionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            targets = inputs.pop("labels").float()
            outputs = model(**inputs)
            predictions = torch.sigmoid(outputs.logits.float().view(-1))
            loss = torch.nn.functional.mse_loss(predictions, targets.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(evaluation):
        logits, targets = evaluation
        predictions = 1.0 / (1.0 + np.exp(-np.asarray(logits).reshape(-1)))
        return regression_metrics(predictions, np.asarray(targets).reshape(-1))

    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        group_by_length=args.group_by_length,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        seed=args.seed,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
    )
    trainer = SigmoidRegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=make_dataset(train_labels),
        eval_dataset=make_dataset(dev_labels),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    local_runtime = {
        "rank": process_rank,
        "device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
        "max_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
        "max_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        ),
    }
    rank_runtime = [local_runtime]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank_runtime = [None] * world_size
        torch.distributed.all_gather_object(rank_runtime, local_runtime)
    metrics = trainer.evaluate()
    trainer.save_model(str(args.output / "model"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(args.output / "model"))
        (args.output / "eval_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output / "training_runtime_metrics.json").write_text(
            json.dumps(
                {
                    "trainer": train_result.metrics,
                    "devices": rank_runtime,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
