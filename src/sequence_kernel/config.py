"""Configuration for the generic minimal/hybrid sequence kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from minimal_attention_kernel import MinimalAttentionKernelConfig


@dataclass(frozen=True)
class SequenceKernelConfig:
    """Select the Stage-1 attention path or its explicit Mamba hybrid."""

    mode: Literal["minimal", "hybrid"]
    attention: MinimalAttentionKernelConfig
    fusion: Literal["gated_add"] = "gated_add"
    fusion_gate_logit_init: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"minimal", "hybrid"}:
            raise ValueError(f"mode must be 'minimal' or 'hybrid', got {self.mode!r}")
        if self.fusion != "gated_add":
            raise ValueError(
                "the v1 hybrid baseline supports only explicit 'gated_add' fusion"
            )

    @property
    def d_model(self) -> int:
        return self.attention.d_model
