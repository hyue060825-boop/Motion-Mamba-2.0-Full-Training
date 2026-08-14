"""ARDY diffusion lifecycle adapter: noisy reads, one accepted commit."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .ardy_dit_block import ArdyDiTSequenceBlock


@dataclass(frozen=True)
class DiffusionMemoryTrace:
    completed_windows: int
    accepted_commits: int
    last_sampling_timestep: Optional[int]


class ArdyDiffusionMemoryController:
    """Attach commit semantics to ARDY without replacing its sampler loop.

    The original bound ``denoising_step`` remains responsible for diffusion.
    This controller observes its result. Every noisy call is naturally
    read-only because the DiT blocks never auto-commit. Only a call whose
    sampling index is zero triggers one extra clean block evaluation with an
    accepted-commit context.
    """

    def __init__(self, ardy_model: Any) -> None:
        self.ardy_model = ardy_model
        self.raw_denoiser = ardy_model.denoiser.model
        self.root_block = self.raw_denoiser.root_model
        self.body_block = self.raw_denoiser.body_model
        if not isinstance(self.root_block, ArdyDiTSequenceBlock) or not isinstance(
            self.body_block, ArdyDiTSequenceBlock
        ):
            raise TypeError("ARDY root/body models must be ArdyDiTSequenceBlock")
        if self.root_block.sequence_mode != self.body_block.sequence_mode:
            raise ValueError("root and body sequence modes must match")
        self._original_denoising_step = ardy_model.denoising_step
        self._signature = inspect.signature(self._original_denoising_step)
        self._installed = False
        self._next_chunk_id = 0
        self._completed_windows = 0
        self._accepted_commits = 0
        self._last_sampling_timestep: Optional[int] = None

    @property
    def trace(self) -> DiffusionMemoryTrace:
        return DiffusionMemoryTrace(
            completed_windows=self._completed_windows,
            accepted_commits=self._accepted_commits,
            last_sampling_timestep=self._last_sampling_timestep,
        )

    @property
    def commit_enabled(self) -> bool:
        return self.root_block.has_cross_window_memory

    def reset(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        self.root_block.reset_memory(reset_mask)
        self.body_block.reset_memory(reset_mask)
        if reset_mask is None:
            self._next_chunk_id = 0
            self._completed_windows = 0
            self._accepted_commits = 0
            self._last_sampling_timestep = None

    def install(self) -> "ArdyDiffusionMemoryController":
        if self._installed:
            return self

        def controlled_denoising_step(*args, **kwargs):
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if self.commit_enabled:
                # ARDY's CFG wrapper expands B to 2B/3B internally. Keep the
                # persistent cache at the accepted user batch B so guidance
                # branches can read it independently and the final clean B
                # result can commit atomically.
                base_batch = bound.arguments["x"].shape[0]
                self.root_block._ensure_state(base_batch)
                self.body_block._ensure_state(base_batch)
            result = self._original_denoising_step(*args, **kwargs)
            timestep = bound.arguments["t"]
            self._last_sampling_timestep = int(timestep.max().item())
            if self.commit_enabled and bool((timestep == 0).all()):
                self._commit_accepted_result(result, bound.arguments)
            return result

        self.ardy_model.denoising_step = controlled_denoising_step
        self._installed = True
        return self

    def uninstall(self) -> None:
        if self._installed:
            self.ardy_model.denoising_step = self._original_denoising_step
            self._installed = False

    def _commit_accepted_result(self, accepted_x: torch.Tensor, call: dict) -> None:
        generation_mask = call["generation_token_mask"]
        chunk_id = self._next_chunk_id
        root_state_before = (
            None
            if self.root_block._sequence_state is None
            else self.root_block._sequence_state.clone()
        )
        body_state_before = (
            None
            if self.body_block._sequence_state is None
            else self.body_block._sequence_state.clone()
        )
        root_commits_before = self.root_block.accepted_commits
        body_commits_before = self.body_block.accepted_commits
        self.root_block.prepare_accepted_commit(generation_mask, chunk_id=chunk_id)
        self.body_block.prepare_accepted_commit(generation_mask, chunk_id=chunk_id)
        try:
            # Reuse ARDY's exact projections, root/body coupling, constraints,
            # masks, and motion conversion. The only special behavior lives in
            # the two blocks, which commit only selected generation tokens.
            with torch.inference_mode():
                self.raw_denoiser(
                    accepted_x,
                    call["history_len"],
                    call["generation_len"],
                    call["future_len"],
                    call["history_mask"],
                    call["generation_mask"],
                    call["future_mask"],
                    call["history_token_mask"],
                    generation_mask,
                    call["future_token_mask"],
                    call["text_feat"],
                    call["text_pad_mask"],
                    torch.zeros_like(call["t"]),
                    call["first_heading_angle"],
                    call["motion_mask"],
                    call["observed_motion"],
                )
            if self.root_block.accepted_commits != root_commits_before + 1:
                raise RuntimeError(
                    "accepted ARDY window must commit root memory exactly once"
                )
            if self.body_block.accepted_commits != body_commits_before + 1:
                raise RuntimeError(
                    "accepted ARDY window must commit body memory exactly once"
                )
        except Exception:
            # Root and body are one accepted ARDY window. Roll both back if
            # either block/projection fails so no half-committed window leaks.
            self.root_block._sequence_state = root_state_before
            self.body_block._sequence_state = body_state_before
            self.root_block.accepted_commits = root_commits_before
            self.body_block.accepted_commits = body_commits_before
            raise
        finally:
            # A failed raw pass must never leave a future noisy call armed to
            # commit accidentally.
            self.root_block.cancel_accepted_commit()
            self.body_block.cancel_accepted_commit()

        self._next_chunk_id += 1
        self._completed_windows += 1
        self._accepted_commits += 1


class ArdySequenceKernelClient:
    """Generation client that resets ephemeral kernel state per request."""

    def __init__(
        self,
        ardy_model: Any,
        controller: Optional[ArdyDiffusionMemoryController],
    ) -> None:
        self.model = ardy_model
        self.controller = controller

    def reset_memory(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        if self.controller is not None:
            self.controller.reset(reset_mask)

    def __call__(self, *args, **kwargs):
        self.reset_memory()
        return self.model(*args, **kwargs)

    def autoregressive_step(self, *args, **kwargs):
        self.reset_memory()
        return self.model.autoregressive_step(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.model, name)
