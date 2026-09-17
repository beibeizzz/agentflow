from __future__ import annotations

# Synchronous fit ordering adapted from veRL 7aed6b2 (Apache-2.0),
# Copyright 2024 Bytedance Ltd. and/or its affiliates.

from types import SimpleNamespace
from typing import Any


def _runtime() -> Any:  # pragma: no cover - pinned veRL runtime
    from omegaconf import OmegaConf
    from tqdm import tqdm
    from verl.utils.debug import marked_timer
    from verl.utils.tracking import Tracking, ValidationGenerationsLogger
    from verl.utils.transferqueue_utils import tq

    return SimpleNamespace(
        Tracking=Tracking,
        ValidationGenerationsLogger=ValidationGenerationsLogger,
        to_container=OmegaConf.to_container,
        progress=tqdm,
        timer=marked_timer,
        tq=tq,
    )


def fit_data_epochs(trainer: Any) -> None:
    """Run configured data epochs while preserving AgentFlow counters.

    A formal run completes after every full prompt batch in the configured data
    epochs has been collected. DAPO replenishment consumes later batches from the
    same stream, so successful actor updates remain an observed result. A small
    successful-update cap is available only for bounded preflight diagnostics.
    """
    runtime = _runtime()
    config = trainer.config
    total_epochs = int(config.trainer.total_epochs)
    batches_per_epoch = len(trainer.train_dataloader)
    expected_collection_batches = batches_per_epoch * total_epochs
    collection_limit = config.get("agentflow", {}).get("max_collection_steps", None)
    diagnostic_target = config.get("agentflow", {}).get(
        "diagnostic_successful_update_limit", None
    )
    if total_epochs <= 0 or batches_per_epoch <= 0:
        raise ValueError("training data epochs and dataloader length must be positive")
    if collection_limit is not None and int(collection_limit) <= 0:
        raise ValueError("collection limit must be positive")
    if diagnostic_target is not None and int(diagnostic_target) <= 0:
        raise ValueError("diagnostic successful-update limit must be positive")
    if (
        collection_limit is not None
        and int(collection_limit) < expected_collection_batches
        and diagnostic_target is None
    ):
        raise ValueError(
            "collection limit is smaller than the batches required by the configured data epochs"
        )
    if trainer._dump_executor._shutdown:
        trainer._init_dump_executor()

    progress = None
    try:
        trainer.logger = runtime.Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=config.trainer.logger,
            config=runtime.to_container(config, resolve=True),
        )
        trainer.validation_generations_logger = runtime.ValidationGenerationsLogger(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
        )
        trainer._load_checkpoint()
        trainer.checkpoint_manager.update_weights()
        if config.trainer.get("val_before_train", True):
            val_metrics = trainer._validate()
            assert val_metrics, f"{val_metrics=}"
            trainer.logger.log(data=val_metrics, step=trainer.global_steps)
            if config.trainer.get("val_only", False):
                return

        if trainer._collection_batches() >= expected_collection_batches:
            return
        if (
            diagnostic_target is not None
            and trainer._successful_updates() >= int(diagnostic_target)
        ):
            return

        progress = runtime.progress(
            total=expected_collection_batches,
            initial=trainer._collection_batches(),
            desc="Collected prompt batches",
        )
        trainer.global_steps += 1
        trainer.prev_step_profile = False
        trainer.curr_step_profile = (
            trainer.global_steps in config.global_profiler.steps
            if config.global_profiler.steps is not None
            else False
        )
        trainer.next_step_profile = False

        # StatefulDataLoader restores its own position from data.pt. The single
        # stream lets DAPO consume replenishment batches without replaying them in
        # the outer loop.
        batch_stream = iter(
            (epoch, batch_dict)
            for epoch in range(total_epochs)
            for batch_dict in trainer.train_dataloader
        )

        while True:
            try:
                epoch, batch_dict = next(batch_stream)
            except StopIteration:
                break
            if (
                collection_limit is not None
                and trainer._collection_batches() >= int(collection_limit)
            ):
                raise RuntimeError(
                    f"collection limit {collection_limit} exhausted after "
                    f"{trainer._collection_batches()} prompt batches"
                )

            def next_prompt_batch():
                try:
                    return next(batch_stream)[1]
                except StopIteration:
                    return None

            trainer._agentflow_replenishment_provider = next_prompt_batch
            before_updates = trainer._successful_updates()
            before_collections = trainer._collection_batches()
            metrics: dict[str, Any] = {}
            timing_raw: dict[str, Any] = {}
            trainer._start_profiling()
            with runtime.timer("step", timing_raw):
                batch = trainer.step(batch_dict, metrics, timing_raw)
                after_updates = trainer._successful_updates()
                after_collections = trainer._collection_batches()
                if after_updates - before_updates not in (0, 1):
                    raise RuntimeError("one DAPO selection may complete zero or one actor update")
                progressed = after_updates > before_updates
                completed_data = after_collections >= expected_collection_batches
                reached_diagnostic_target = (
                    diagnostic_target is not None
                    and after_updates >= int(diagnostic_target)
                )
                is_last_step = completed_data or reached_diagnostic_target
                if (
                    is_last_step
                    or (
                        progressed
                        and config.trainer.save_freq > 0
                        and after_updates % config.trainer.save_freq == 0
                    )
                ):
                    with runtime.timer("save_checkpoint", timing_raw, color="green"):
                        trainer._save_checkpoint()
                with runtime.timer("update_weights", timing_raw, color="red"):
                    trainer.checkpoint_manager.update_weights()
            trainer._stop_profiling()

            if (
                is_last_step
                or (
                    progressed
                    and config.trainer.test_freq > 0
                    and after_updates % config.trainer.test_freq == 0
                )
            ) and config.trainer.test_freq != -1:
                with runtime.timer("testing", timing_raw, color="green"):
                    metrics.update(trainer._validate())
            metrics.update(
                {
                    "agentflow/collection_batch_count": float(after_collections),
                    "agentflow/expected_collection_batches": float(
                        expected_collection_batches
                    ),
                    "agentflow/data_epoch_complete": float(completed_data),
                }
            )
            trainer._compute_metrics(
                batch,
                metrics,
                timing_raw,
                global_steps=trainer.global_steps,
                epoch=epoch,
            )
            rollout_dir = config.trainer.get("rollout_data_dir", None)
            if rollout_dir:
                trainer._log_rollout_data(batch, timing_raw, rollout_dir)
            runtime.tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
            trainer.replay_buffer.remove(batch.partition_id, batch.keys)
            trainer.logger.log(data=metrics, step=trainer.global_steps)
            progress.update(after_collections - before_collections)
            trainer.global_steps += 1
            if is_last_step:
                return

        if trainer._collection_batches() < expected_collection_batches:
            raise RuntimeError(
                "dataloader ended before all configured data-epoch batches were collected"
            )
    finally:
        trainer._shutdown_dump_executor()
        if progress is not None:
            progress.close()


__all__ = ["fit_data_epochs"]
