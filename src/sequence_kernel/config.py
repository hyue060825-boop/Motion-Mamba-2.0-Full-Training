"""Configuration for the generic minimal/hybrid sequence kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from minimal_attention_kernel import MinimalAttentionKernelConfig

from .token_state import TokenStateVariant


InteractionVariant = TokenStateVariant | Literal["h0_parallel", "h1_innovation", "h2_delta_rule"]


@dataclass(frozen=True)
class SequenceKernelConfig:
    """Select the Stage-1 attention path or its explicit Mamba hybrid."""

    mode: Literal["minimal", "hybrid"]
    attention: MinimalAttentionKernelConfig
    fusion: Literal["gated_add"] = "gated_add"
    fusion_gate_logit_init: float = 0.0
    interaction: InteractionVariant = "naive_gated_add"
    state_summary_dim: int = 128
    use_state_delta: bool = True
    state_conditioning_scale_init: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"minimal", "hybrid"}:
            raise ValueError(f"mode must be 'minimal' or 'hybrid', got {self.mode!r}")
        if self.fusion != "gated_add":
            raise ValueError(
                "the v1 hybrid baseline supports only explicit 'gated_add' fusion"
            )
        if self.interaction not in {
            "naive_gated_add",
            "state_query",
            "state_kv_modulation",
            "h0_parallel",
            "h1_innovation",
            "h2_delta_rule",
        }:
            raise ValueError(f"invalid token/state interaction {self.interaction!r}")
        if self.mode == "minimal" and self.interaction != "naive_gated_add":
            raise ValueError("minimal mode cannot enable token/state interaction")
        if self.state_summary_dim <= 0:
            raise ValueError("state_summary_dim must be positive")

    @property
    def d_model(self) -> int:
        return self.attention.d_model
