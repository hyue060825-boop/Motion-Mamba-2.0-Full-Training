"""Separate state containers for attention K/V and recurrent Mamba state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from minimal_attention_kernel import ChunkSlidingWindowKVCache, PendingChunkKV


def _detach_optional(value: Any) -> Any:
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if detach is None:
        raise TypeError("recurrent state must provide detach()")
    return detach()


def _clone_optional(value: Any) -> Any:
    if value is None:
        return None
    clone = getattr(value, "clone", None)
    if clone is None:
        raise TypeError("recurrent state must provide clone()")
    return clone()


@dataclass
class SequenceKernelState:
    """Committed memory; the two memory mechanisms remain independent."""

    attention_cache: ChunkSlidingWindowKVCache
    mamba_state: Optional[Any]
    source_kernel_id: int

    def detach(self) -> "SequenceKernelState":
        self.attention_cache.detach()
        self.mamba_state = _detach_optional(self.mamba_state)
        return self

    def clone(self) -> "SequenceKernelState":
        return SequenceKernelState(
            attention_cache=self.attention_cache.clone(),
            mamba_state=_clone_optional(self.mamba_state),
            source_kernel_id=self.source_kernel_id,
        )


@dataclass(frozen=True)
class PendingSequenceKernelState:
    """Candidate next memory produced by a read-only chunk evaluation."""

    attention_kv: Optional[PendingChunkKV]
    mamba_state: Optional[Any]
    source_kernel_id: int

    def detach(self) -> "PendingSequenceKernelState":
        return PendingSequenceKernelState(
            attention_kv=None if self.attention_kv is None else self.attention_kv.detach(),
            mamba_state=_detach_optional(self.mamba_state),
            source_kernel_id=self.source_kernel_id,
        )
