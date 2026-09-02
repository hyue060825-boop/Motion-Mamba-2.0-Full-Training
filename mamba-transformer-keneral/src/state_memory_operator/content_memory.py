"""Fixed-capacity persistent content memory for the D-stage ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ContentMemoryPending:
    slots: torch.Tensor


class PersistentContentMemory(nn.Module):
    """Decode the existing projected state summary and delta into K slots."""

    def __init__(self, state_dim: int, d_model: int, slots: int = 1) -> None:
        super().__init__()
        if state_dim <= 0 or d_model <= 0 or slots <= 0:
            raise ValueError("state_dim, d_model, and slots must be positive")
        self.state_dim = state_dim
        self.d_model = d_model
        self.slots = slots
        self.content_decoder = nn.Linear(2 * state_dim, slots * d_model)
        self.write_gate = nn.Linear(2 * state_dim + slots * d_model, slots)
        self.read_gate = nn.Parameter(torch.tensor(-12.0))
        nn.init.zeros_(self.content_decoder.weight)
        nn.init.zeros_(self.content_decoder.bias)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, -12.0)

    def initial(self, batch_size: int, *, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.slots, self.d_model, device=device, dtype=dtype)

    def propose(self, state_summary: torch.Tensor, state_delta: torch.Tensor, previous: torch.Tensor) -> ContentMemoryPending:
        self._validate(state_summary, state_delta, previous)
        context = torch.cat((state_summary, state_delta), dim=-1)
        proposal = self.content_decoder(context).view(-1, self.slots, self.d_model)
        gate_input = torch.cat((context, proposal.flatten(1)), dim=-1)
        gate = torch.sigmoid(self.write_gate(gate_input)).unsqueeze(-1)
        return ContentMemoryPending((1.0 - gate) * previous + gate * proposal)

    def read(self, slots: torch.Tensor, *, enabled: bool = True) -> torch.Tensor:
        if slots.ndim != 3 or slots.shape[-2:] != (self.slots, self.d_model):
            raise ValueError(f"slots must have shape [B,{self.slots},{self.d_model}]")
        scale = torch.sigmoid(self.read_gate) if enabled else self.read_gate.new_zeros(())
        return slots * scale.to(slots.dtype)

    def reset(self, slots: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        if reset_mask is None:
            return torch.zeros_like(slots)
        if reset_mask.shape != (slots.shape[0],) or reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must be bool with shape [batch]")
        return torch.where(reset_mask[:, None, None], torch.zeros_like(slots), slots)

    def permute_rows(self, slots: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        if permutation.shape != (slots.shape[0],) or permutation.dtype != torch.long:
            raise ValueError("permutation must be long with shape [batch]")
        return slots.index_select(0, permutation.to(slots.device))

    def _validate(self, summary: torch.Tensor, delta: torch.Tensor, slots: torch.Tensor) -> None:
        if summary.ndim != 2 or summary.shape[1] != self.state_dim:
            raise ValueError(f"state_summary must have shape [B,{self.state_dim}]")
        if delta.shape != summary.shape:
            raise ValueError("state_delta must match state_summary shape")
        if slots.ndim != 3 or slots.shape != (summary.shape[0], self.slots, self.d_model):
            raise ValueError("persistent slots have incompatible shape")


__all__ = ["ContentMemoryPending", "PersistentContentMemory"]
