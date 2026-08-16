"""Thin integration layer over the closed attention and existing Mamba paths."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

from minimal_attention_kernel import MinimalAttentionKernel

from .config import SequenceKernelConfig
from .delta_rule import DeltaRuleStateCorrection
from .innovation import OutputInteraction
from .state import PendingSequenceKernelState, SequenceKernelState
from .token_state import TokenStateInteraction


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
        state_projector: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.attention = MinimalAttentionKernel(config.attention)

        if config.mode == "minimal":
            if (
                mamba3 is not None
                or mamba_wrapper is not None
                or mamba_reset_fn is not None
                or state_projector is not None
            ):
                raise ValueError("minimal mode accepts no Mamba objects")
            self._mamba_wrapper = None
            self._mamba_reset_fn = None
            return

        if mamba3 is None:
            raise ValueError("hybrid mode requires the existing Mamba3 module")
        if not isinstance(mamba3, nn.Module):
            raise TypeError("mamba3 must be a torch.nn.Module")
        if config.interaction in {"naive_gated_add", "h0_parallel", "h1_innovation", "h2_delta_rule"} and state_projector is not None:
            raise ValueError("naive hybrid does not use a state_projector")

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
        if config.interaction == "naive_gated_add":
            self.fusion_gate_logit = nn.Parameter(
                torch.tensor(float(config.fusion_gate_logit_init))
            )
            self.state_projector = None
            self.token_state_interaction = None
            self.output_interaction = None
            self.delta_rule = None
        elif config.interaction in {"h0_parallel", "h1_innovation"}:
            self.fusion_gate_logit = None
            self.state_projector = None
            self.token_state_interaction = None
            self.output_interaction = OutputInteraction(
                config.interaction,
                config.d_model,
                scale_init=config.state_conditioning_scale_init,
            )
            self.delta_rule = None
        elif config.interaction == "h2_delta_rule":
            self.fusion_gate_logit = None
            self.state_projector = None
            self.token_state_interaction = None
            self.output_interaction = None
            self.delta_rule = DeltaRuleStateCorrection(
                config.d_model,
                nheads=mamba3.nheads,
                headdim=mamba3.headdim,
                d_state=mamba3.d_state,
                scale_init=config.state_conditioning_scale_init,
            )
        else:
            if state_projector is None:
                from state_memory_operator.state_projector import StateProjector

                nheads = getattr(mamba3, "nheads", None)
                headdim = getattr(mamba3, "headdim", None)
                if nheads is None or headdim is None:
                    raise ValueError(
                        "C1 interactions require Mamba nheads/headdim or an "
                        "explicit state_projector"
                    )
                state_projector = StateProjector(
                    nheads=nheads,
                    headdim=headdim,
                    d_summary=config.state_summary_dim,
                )
            if not isinstance(state_projector, nn.Module):
                raise TypeError("state_projector must be a torch.nn.Module")
            self.state_projector = state_projector
            self.token_state_interaction = TokenStateInteraction(
                variant=config.interaction,
                d_model=config.d_model,
                num_heads=config.attention.num_heads,
                d_state_summary=config.state_summary_dim,
                use_state_delta=config.use_state_delta,
                scale_init=config.state_conditioning_scale_init,
            )
            self.output_interaction = None
            self.delta_rule = None

    @property
    def uses_token_state_interaction(self) -> bool:
        return self.mode == "hybrid" and self.config.interaction in {
            "state_query", "state_kv_modulation"
        }

    @property
    def uses_output_interaction(self) -> bool:
        return self.mode == "hybrid" and self.output_interaction is not None

    @property
    def uses_delta_rule(self) -> bool:
        return self.mode == "hybrid" and self.delta_rule is not None

    @staticmethod
    def _raw_summary_state(mamba_state: Any) -> torch.Tensor:
        if isinstance(mamba_state, torch.Tensor):
            return mamba_state
        ssm_state = getattr(mamba_state, "ssm_state", None)
        if isinstance(ssm_state, torch.Tensor):
            return ssm_state
        raise TypeError("state projector requires a tensor or a state with .ssm_state")

    def _project_state(self, mamba_state: Any) -> torch.Tensor:
        return self.state_projector(self._raw_summary_state(mamba_state))

    def _mamba_parameter(self) -> torch.Tensor:
        parameter = getattr(getattr(self.mamba_path, "in_proj", None), "weight", None)
        if isinstance(parameter, torch.Tensor):
            return parameter
        return next(self.mamba_path.parameters())

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
                mamba_parameter = self._mamba_parameter()
                mamba_state = init_state(
                    batch_size,
                    device=mamba_parameter.device,
                    dtype=mamba_parameter.dtype,
                )
        state_summary = None
        state_delta = None
        if self.uses_token_state_interaction:
            if mamba_state is None:
                raise RuntimeError("C1 interaction requires an initialized Mamba state")
            state_summary = self._project_state(mamba_state)
            state_delta = torch.zeros_like(state_summary)
        return SequenceKernelState(
            attention_cache=attention_cache,
            mamba_state=mamba_state,
            source_kernel_id=id(self),
            state_summary=state_summary,
            state_delta=state_delta,
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
        if self.uses_token_state_interaction and (
            state.state_summary is None or state.state_delta is None
        ):
            raise ValueError("C1 state requires committed state_summary and state_delta")

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
        query_bias = None
        history_chunk_scales = None
        interaction_aux = None
        if enable_attention_path and self.uses_token_state_interaction:
            summary = state.state_summary
            delta = state.state_delta
            if not do_read:
                summary = torch.zeros_like(summary)
                delta = torch.zeros_like(delta)
            if self.config.interaction == "state_query":
                query_bias = self.token_state_interaction.query_bias(summary, delta).to(
                    dtype=x.dtype
                )
                interaction_aux = {
                    "query_bias_norm": query_bias.detach().float().norm(dim=-1).mean(),
                }
            else:
                history_chunk_scales, raw_scores = (
                    self.token_state_interaction.chunk_key_scales(
                        summary,
                        delta,
                        state.attention_cache,
                    )
                )
                interaction_aux = {
                    "chunk_key_scales": history_chunk_scales.detach(),
                    "chunk_key_scores": raw_scores.detach(),
                }
        if enable_attention_path:
            attention_output, attention_pending, attention_aux = self.attention.forward_chunk(
                x,
                cache=state.attention_cache,
                do_read=do_read,
                commit=False,
                current_mask=current_mask,
                query_bias=query_bias,
                history_chunk_scales=history_chunk_scales,
            )

        mamba_output = None
        next_mamba_state = None
        if enable_mamba_path:
            mamba_parameter = self._mamba_parameter()
            mamba_input = x.to(
                device=mamba_parameter.device, dtype=mamba_parameter.dtype
            )
            mamba_output, next_mamba_state = self._mamba_wrapper.forward_chunk(
                mamba_input,
                state.mamba_state if do_read else None,
            )
            if self.uses_delta_rule:
                next_mamba_state, interaction_aux = self.delta_rule.correct(
                    next_mamba_state,
                    attention_output if attention_output is not None else x,
                )

        next_state_summary = None
        next_state_delta = None
        if self.uses_token_state_interaction and next_mamba_state is not None:
            next_state_summary = self._project_state(next_mamba_state)
            next_state_delta = next_state_summary - state.state_summary

        if attention_output is not None and mamba_output is not None:
            if self.uses_output_interaction:
                output, interaction_aux = self.output_interaction(
                    attention_output, mamba_output
                )
            elif self.uses_token_state_interaction:
                # Contribution-1 uses Mamba only to build the next durable
                # address state. Content remains the attention value path.
                output = attention_output
            elif self.uses_delta_rule:
                output = attention_output + (
                    self.delta_rule.output_scale * mamba_output.float()
                ).to(attention_output.dtype)
            else:
                gate = torch.sigmoid(self.fusion_gate_logit).to(mamba_output.dtype)
                output = attention_output + (gate * mamba_output).to(
                    attention_output.dtype
                )
        elif attention_output is not None:
            output = attention_output
        else:
            output = mamba_output

        pending = PendingSequenceKernelState(
            attention_kv=attention_pending,
            mamba_state=next_mamba_state,
            source_kernel_id=id(self),
            state_summary=next_state_summary,
            state_delta=next_state_delta,
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
            "fusion": (
                (
                    "gated_add"
                    if self.config.interaction == "naive_gated_add"
                    else self.config.interaction
                )
                if attention_output is not None and mamba_output is not None
                else None
            ),
            "fusion_gate": (
                float(torch.sigmoid(self.fusion_gate_logit.detach()).cpu())
                if self.mode == "hybrid"
                and self.config.interaction == "naive_gated_add"
                else None
            ),
            "state_conditioning_scale": (
                float(
                    (
                        self.output_interaction.scale
                        if self.uses_output_interaction
                        else (
                            self.delta_rule.beta
                            if self.uses_delta_rule
                            else self.token_state_interaction.effective_scale
                        )
                    ).detach().cpu()
                )
                if self.uses_token_state_interaction or self.uses_output_interaction or self.uses_delta_rule
                else None
            ),
            "committed_state_norm": (
                None
                if state.state_summary is None
                else state.state_summary.detach().float().norm(dim=-1).mean().item()
            ),
            "committed_delta_norm": (
                None
                if state.state_delta is None
                else state.state_delta.detach().float().norm(dim=-1).mean().item()
            ),
            "interaction": interaction_aux,
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
        if pending.state_summary is not None:
            state.state_summary = (
                pending.state_summary.detach() if detach else pending.state_summary
            )
        if pending.state_delta is not None:
            state.state_delta = (
                pending.state_delta.detach() if detach else pending.state_delta
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
            if self.uses_token_state_interaction:
                projected = self._project_state(state.mamba_state)
                state.state_summary = projected
                state.state_delta = torch.where(
                    recurrent_reset_mask[:, None],
                    torch.zeros_like(projected),
                    state.state_delta,
                )
        return state
