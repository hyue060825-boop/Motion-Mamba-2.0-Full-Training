"""Thin integration layer over the closed attention and existing Mamba paths."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

from minimal_attention_kernel import MinimalAttentionKernel

from .config import SequenceKernelConfig
from .state import PendingSequenceKernelState, SequenceKernelState


ResetFn = Callable[[Any, Optional[torch.Tensor]], Any]


def _reset_tensor_state(state: Any, reset_mask: Optional[torch.Tensor]) -> Any:
    """Fallback used by lightweight injected wrappers in CPU interface tests."""
    if not isinstance(state, torch.Tensor):
        raise TypeError(
            "an injected non-tensor recurrent state requires mamba_reset_fn"
        )
    if reset_mask is None:
        return torch.zeros_like(state)
    if reset_mask.shape != (state.shape[0],) or reset_mask.dtype != torch.bool:
        raise ValueError(
            f"reset_mask must be bool with shape {(state.shape[0],)}, "
            f"got shape={tuple(reset_mask.shape)} dtype={reset_mask.dtype}"
        )
    shape = [state.shape[0]] + [1] * (state.ndim - 1)
    return torch.where(reset_mask.to(state.device).view(*shape), torch.zeros_like(state), state)


class SequenceKernel(nn.Module):
    """Generic sequence operator with exact minimal and opt-in hybrid modes.

    Minimal mode registers only :class:`MinimalAttentionKernel`. Hybrid mode
    additionally wraps the repository's existing ``Mamba3ChunkWrapper`` and
    fuses both outputs as ``attention + sigmoid(gate) * mamba``.

    ``forward_chunk(commit=False)`` is read-only with respect to committed
    memory. Call :meth:`commit` explicitly, or pass ``commit=True``.
    """

    def __init__(
        self,
        config: SequenceKernelConfig,
        *,
        mamba3: Optional[nn.Module] = None,
        mamba_wrapper: Optional[Any] = None,
        mamba_reset_fn: Optional[ResetFn] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.attention = MinimalAttentionKernel(config.attention)

        if config.mode == "minimal":
            if mamba3 is not None or mamba_wrapper is not None or mamba_reset_fn is not None:
                raise ValueError("minimal mode accepts no Mamba objects")
            self._mamba_wrapper = None
            self._mamba_reset_fn = None
            return

        if mamba3 is None:
            raise ValueError("hybrid mode requires the existing Mamba3 module")
        if not isinstance(mamba3, nn.Module):
            raise TypeError("mamba3 must be a torch.nn.Module")

        # The import is intentionally deferred: constructing/importing minimal
        # mode must not import either state_memory_operator or mamba_ssm.
        if mamba_wrapper is None:
            from state_memory_operator.mamba3_chunk_wrapper import (
                Mamba3ChunkWrapper,
                apply_reset_mask,
            )

            mamba_wrapper = Mamba3ChunkWrapper(mamba3)
            mamba_reset_fn = apply_reset_mask
        elif mamba_reset_fn is None:
            mamba_reset_fn = _reset_tensor_state

        wrapped_module = getattr(mamba_wrapper, "mamba3", mamba3)
        if wrapped_module is not mamba3:
            raise ValueError("mamba_wrapper must wrap the supplied mamba3 module")

        self.mamba_path = mamba3
        self._mamba_wrapper = mamba_wrapper
        self._mamba_reset_fn = mamba_reset_fn
        self.fusion_gate_logit = nn.Parameter(
            torch.tensor(float(config.fusion_gate_logit_init))
        )

    @property
    def mode(self) -> str:
        return self.config.mode

    def init_state(
        self,
        *,
        batch_size: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> SequenceKernelState:
        attention_cache = self.attention.init_cache(
            batch_size=batch_size, device=device, dtype=dtype
        )
        mamba_state = None
        if self.mode == "hybrid":
            init_state = getattr(self._mamba_wrapper, "init_state", None)
            if init_state is not None:
                mamba_state = init_state(
                    batch_size,
                    device=attention_cache.device,
                    dtype=attention_cache.dtype,
                )
        return SequenceKernelState(
            attention_cache=attention_cache,
            mamba_state=mamba_state,
            source_kernel_id=id(self),
        )

    def _validate_state(self, state: SequenceKernelState, batch_size: int) -> None:
        if state.source_kernel_id != id(self):
            raise ValueError("state was created by a different sequence kernel")
        if state.attention_cache.batch_size != batch_size:
            raise ValueError(
                f"state batch size is {state.attention_cache.batch_size}, got input batch {batch_size}"
            )
        if self.mode == "minimal" and state.mamba_state is not None:
            raise ValueError("minimal state must not contain recurrent Mamba state")

    def forward_chunk(
        self,
        x: torch.Tensor,
        *,
        state: SequenceKernelState,
        do_read: bool = True,
        commit: bool = False,
        chunk_id: Optional[int] = None,
        current_mask: Optional[torch.Tensor] = None,
        detach_on_commit: bool = True,
        enable_attention_path: bool = True,
        enable_mamba_path: Optional[bool] = None,
    ) -> tuple[torch.Tensor, PendingSequenceKernelState, dict]:
        """Evaluate selected paths and return their uncommitted next memory."""
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,L,D], got {tuple(x.shape)}")
        self._validate_state(state, x.shape[0])

        if enable_mamba_path is None:
            enable_mamba_path = self.mode == "hybrid"
        if self.mode == "minimal" and enable_mamba_path:
            raise ValueError("Mamba path is unavailable in minimal mode")
        if not enable_attention_path and not enable_mamba_path:
            raise ValueError("at least one sequence path must be enabled")
        if current_mask is not None:
            expected_mask_shape = (x.shape[0], x.shape[1])
            if current_mask.shape != expected_mask_shape or current_mask.dtype != torch.bool:
                raise ValueError(
                    f"current_mask must be bool with shape {expected_mask_shape}, "
                    f"got shape={tuple(current_mask.shape)} dtype={current_mask.dtype}"
                )
            if enable_mamba_path and not bool(current_mask.all()):
                raise NotImplementedError(
                    "the existing dense Mamba chunk path cannot skip padded per-row "
                    "timesteps; use unpadded hybrid chunks"
                )

        attention_output = None
        attention_pending = None
        attention_aux = None
        if enable_attention_path:
            attention_output, attention_pending, attention_aux = self.attention.forward_chunk(
                x,
                cache=state.attention_cache,
                do_read=do_read,
                commit=False,
                current_mask=current_mask,
            )

        mamba_output = None
        next_mamba_state = None
        if enable_mamba_path:
            mamba_output, next_mamba_state = self._mamba_wrapper.forward_chunk(
                x,
                state.mamba_state if do_read else None,
            )

        if attention_output is not None and mamba_output is not None:
            gate = torch.sigmoid(self.fusion_gate_logit).to(mamba_output.dtype)
            output = attention_output + gate * mamba_output
        elif attention_output is not None:
            output = attention_output
        else:
            output = mamba_output

        pending = PendingSequenceKernelState(
            attention_kv=attention_pending,
            mamba_state=next_mamba_state,
            source_kernel_id=id(self),
        )
        if commit:
            self.commit(
                state,
                pending,
                chunk_id=chunk_id,
                detach=detach_on_commit,
            )

        aux = {
            "mode": self.mode,
            "attention_path_enabled": enable_attention_path,
            "mamba_path_enabled": bool(enable_mamba_path),
            "fusion": "gated_add" if attention_output is not None and mamba_output is not None else None,
            "fusion_gate": (
                float(torch.sigmoid(self.fusion_gate_logit.detach()).cpu())
                if self.mode == "hybrid"
                else None
            ),
            "attention": attention_aux,
            "state_committed": commit,
        }
        return output, pending, aux

    def commit(
        self,
        state: SequenceKernelState,
        pending: PendingSequenceKernelState,
        *,
        chunk_id: Optional[int] = None,
        detach: bool = True,
    ) -> SequenceKernelState:
        """Accept exactly one candidate attention chunk and recurrent state."""
        self._validate_state(state, state.attention_cache.batch_size)
        if pending.source_kernel_id != id(self):
            raise ValueError("pending state was produced by a different sequence kernel")
        if pending.attention_kv is not None:
            if chunk_id is None:
                raise ValueError("committing attention K/V requires chunk_id")
            self.attention.commit(
                state.attention_cache,
                pending.attention_kv,
                chunk_id=chunk_id,
                detach=detach,
            )
        if pending.mamba_state is not None:
            state.mamba_state = (
                pending.mamba_state.detach() if detach else pending.mamba_state
            )
        return state

    def reset_state(
        self,
        state: SequenceKernelState,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> SequenceKernelState:
        """Reset attention history and recurrent state without conflating them."""
        self._validate_state(state, state.attention_cache.batch_size)
        self.attention.reset_cache(state.attention_cache, reset_mask)
        if self.mode == "hybrid" and state.mamba_state is not None:
            recurrent_reset_mask = reset_mask
            if recurrent_reset_mask is None:
                recurrent_reset_mask = torch.ones(
                    state.attention_cache.batch_size,
                    dtype=torch.bool,
                    device=state.attention_cache.device,
                )
            state.mamba_state = self._mamba_reset_fn(
                state.mamba_state, recurrent_reset_mask
            )
        return state
