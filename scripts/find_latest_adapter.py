from __future__ import annotations

import argparse
import re
from pathlib import Path


STEP_RE = re.compile(r"global_step_(\d+)")


def checkpoint_step(path: Path) -> int:
    for part in reversed(path.parts):
        match = STEP_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return -1


def latest_actor_checkpoint(root: Path) -> Path:
    candidates = [path.parent for path in root.rglob("lora_train_meta.json")]
    if not candidates:
        raise FileNotFoundError(f"no veRL LoRA checkpoint under {root}")
    return max(
        candidates,
        key=lambda path: (checkpoint_step(path), path.stat().st_mtime_ns),
    )


def export_lora_adapter(actor_path: Path) -> Path:
    """Export only PEFT weights from a pinned veRL FSDP checkpoint."""
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    target = actor_path / "lora_adapter"
    config_path = target / "adapter_config.json"
    weights_path = target / "adapter_model.safetensors"
    if config_path.is_file() and weights_path.is_file():
        return target

    if target.exists():
        resolved_target = target.resolve()
        if resolved_target.parent != actor_path.resolve():
            raise RuntimeError(f"unexpected export path: {resolved_target}")
        raise RuntimeError(
            f"incomplete adapter export exists at {resolved_target}; inspect it before retrying"
        )
    target.mkdir(parents=False)

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(actor_path),
        target_dir=str(target.parent),
        hf_model_config_path=str(actor_path / "huggingface"),
    )
    merger = FSDPModelMerger(config)
    world_size = merger._get_world_size()
    rank_zero = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(rank_zero, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(
        mesh, mesh_dim_names
    )
    del rank_zero
    state_dict = merger._load_and_merge_state_dicts(
        world_size, total_shards, mesh_shape, mesh_dim_names
    )
    exported = Path(merger.save_lora_adapter(state_dict) or "")
    if exported.resolve() != target.resolve():
        raise RuntimeError(f"unexpected veRL adapter export path: {exported}")
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"incomplete LoRA adapter export under {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print or export the latest veRL PEFT adapter directory"
    )
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    latest = export_lora_adapter(latest_actor_checkpoint(args.root))
    print(latest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
