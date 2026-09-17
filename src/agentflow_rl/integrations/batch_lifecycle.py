from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any, Sequence


class ReplayPaddingGuard:
    """Prevent a delayed ReplayBuffer poll from reviving reclaimed TQ keys."""

    def __init__(self, replay_buffer: Any) -> None:
        self.replay_buffer = replay_buffer
        self._lock = threading.Lock()
        self._tombstones: dict[str, set[str]] = {}
        original_add = getattr(replay_buffer, "add", None)
        if callable(original_add):

            def guarded_add(partition_id: str, items: dict[str, Any]) -> Any:
                with self._lock:
                    blocked = self._tombstones.get(str(partition_id), set())
                    visible = {
                        key: value
                        for key, value in items.items()
                        if str(key) not in blocked
                    }
                if visible:
                    return original_add(partition_id, visible)
                return None

            replay_buffer.add = guarded_add

    def tombstone(self, partition_id: str, keys: Sequence[str]) -> None:
        with self._lock:
            self._tombstones.setdefault(str(partition_id), set()).update(
                str(key) for key in keys
            )


def replay_padding_guard(replay_buffer: Any) -> ReplayPaddingGuard:
    guard = getattr(replay_buffer, "_agentflow_padding_guard", None)
    if guard is None:
        guard = ReplayPaddingGuard(replay_buffer)
        replay_buffer._agentflow_padding_guard = guard
    return guard


def is_padding_tag(tag: dict[str, Any] | None) -> bool:
    return bool(tag and tag.get("is_padding", False))


def padding_keys(batch: Any) -> tuple[str, ...]:
    if len(batch.keys) != len(batch.tags):
        raise ValueError("batch keys and tags must align")
    return tuple(
        str(key)
        for key, tag in zip(batch.keys, batch.tags, strict=True)
        if is_padding_tag(tag)
    )


def real_batch_view(batch: Any) -> Any:
    """Return the semantic trajectory view selected by the authoritative row tag."""
    real_keys = [
        key
        for key, tag in zip(batch.keys, batch.tags, strict=True)
        if not is_padding_tag(tag)
    ]
    if len(real_keys) == len(batch.keys):
        return batch
    return batch.select_keys(real_keys)


def real_metadata_view(
    metadata: Sequence[dict[str, Any] | None],
    tags: Sequence[dict[str, Any] | None],
) -> list[dict[str, Any] | None]:
    if len(metadata) != len(tags):
        raise ValueError("metadata and padding tags must align")
    return [
        item
        for item, tag in zip(metadata, tags, strict=True)
        if not is_padding_tag(tag)
    ]


def aligned_execution_size(real_count: int, multiple: int) -> int:
    if real_count <= 0 or multiple <= 0:
        raise ValueError("real row count and execution multiple must be positive")
    return ((real_count + multiple - 1) // multiple) * multiple


@dataclass
class PaddingLifecycle:
    """Own synthetic TQ keys created for one worker call and reclaim them once."""

    stage: str
    partition_id: str
    keys: tuple[str, ...]
    transfer_queue: Any
    replay_buffer: Any
    owner: str
    cleaned: int = 0
    cleanup_failed: int = 0
    _closed: bool = False

    @classmethod
    def register(
        cls,
        *,
        stage: str,
        before: Any,
        execution: Any,
        transfer_queue: Any,
        replay_buffer: Any,
    ) -> "PaddingLifecycle":
        if before.partition_id != execution.partition_id:
            raise ValueError("padding execution view changed partition")
        if len({str(key) for key in execution.keys}) != len(execution.keys):
            raise ValueError("padding execution keys must be unique")
        original = {str(key) for key in before.keys}
        execution_keys = {str(key) for key in execution.keys}
        created = tuple(str(key) for key in execution.keys if str(key) not in original)
        tag_by_key = {
            str(key): tag
            for key, tag in zip(execution.keys, execution.tags, strict=True)
        }
        if any(not is_padding_tag(tag_by_key[key]) for key in created):
            raise ValueError("every new execution key must carry is_padding=True")
        if any(str(key) not in execution_keys for key in before.keys):
            raise ValueError("padding execution view dropped a real key")
        return cls(
            stage=stage,
            partition_id=str(execution.partition_id),
            keys=created,
            transfer_queue=transfer_queue,
            replay_buffer=replay_buffer,
            owner=f"agentflow-{stage}-{uuid.uuid4().hex}",
        )

    @property
    def created(self) -> int:
        return len(self.keys)

    @property
    def live(self) -> int:
        return self.created - self.cleaned

    def cleanup(self) -> None:
        if self._closed:
            return
        if not self.keys:
            self._closed = True
            return
        try:
            replay_padding_guard(self.replay_buffer).tombstone(
                self.partition_id, self.keys
            )
            result = self.transfer_queue.kv_clear(
                keys=list(self.keys), partition_id=self.partition_id
            )
            _resolve_completion(result)
            result = self.replay_buffer.remove(self.partition_id, list(self.keys))
            _resolve_completion(result)
        except Exception:
            self.cleanup_failed += len(self.keys) - self.cleaned
            raise
        self.cleaned = len(self.keys)
        self._closed = True

    def metrics(self) -> dict[str, float]:
        prefix = f"agentflow/{self.stage}_padding"
        return {
            f"{prefix}_created": float(self.created),
            f"{prefix}_cleaned": float(self.cleaned),
            f"{prefix}_live": float(self.live),
            f"{prefix}_cleanup_failed": float(self.cleanup_failed),
        }


def _resolve_completion(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    get = getattr(value, "get", None)
    if callable(get):
        return get()
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    return value


def close_padding_lifecycle(
    lifecycle: PaddingLifecycle,
    metrics: dict[str, Any],
    *,
    active_error: BaseException | None = None,
) -> None:
    try:
        lifecycle.cleanup()
    except Exception as cleanup_error:
        if active_error is not None:
            active_error.add_note(
                f"{lifecycle.stage} padding cleanup also failed: {cleanup_error!r}"
            )
        else:
            raise
    finally:
        metrics.update(lifecycle.metrics())


__all__ = [
    "PaddingLifecycle",
    "ReplayPaddingGuard",
    "aligned_execution_size",
    "close_padding_lifecycle",
    "is_padding_tag",
    "padding_keys",
    "real_batch_view",
    "real_metadata_view",
    "replay_padding_guard",
]
