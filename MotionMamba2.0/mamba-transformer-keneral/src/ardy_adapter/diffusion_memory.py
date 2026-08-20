"""ARDY diffusion lifecycle adapter: noisy reads, one accepted commit."""

from __future__ import annotations

import inspect
import types
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
        # A clean t=0 prediction is only eligible for memory after ARDY has
        # recentered the completed window. Committing inside denoising_step
        # stores the pre-recenter coordinate frame and causes boundary jumps.
        self._pending_commit: Optional[tuple[torch.Tensor, dict[str, Any]]] = None
        self._original_recenter_history = getattr(ardy_model, "_recenter_history", None)

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
            self._pending_commit = None

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
                # Defer until _recenter_history has transformed this window
                # into the coordinate frame consumed by the next window.
                if self._original_recenter_history is None:
                    # Lightweight test doubles do not implement ARDY's
                    # recenter lifecycle; preserve the original immediate
                    # commit semantics for those callers.
                    self._commit_accepted_result(result, bound.arguments)
                else:
                    self._pending_commit = (result.detach(), dict(bound.arguments))
            return result

        self.ardy_model.denoising_step = controlled_denoising_step

        def controlled_recenter_history(model_self, history_sequence, center_frame_index, requantize):
            recentered = self._original_recenter_history(
                history_sequence, center_frame_index, requantize
            )
            if self.commit_enabled and self._pending_commit is not None:
                recentered_sequence = recentered[0]
                accepted_x, call = self._pending_commit
                self._commit_recentered_result(recentered_sequence, accepted_x, call)
                self._pending_commit = None
            return recentered

        if self._original_recenter_history is not None:
            self.ardy_model._recenter_history = types.MethodType(
                controlled_recenter_history, self.ardy_model
            )
        self._installed = True
        return self

    def uninstall(self) -> None:
        if self._installed:
            self.ardy_model.denoising_step = self._original_denoising_step
            if self._original_recenter_history is not None:
                self.ardy_model._recenter_history = self._original_recenter_history
            self._installed = False

    def _commit_recentered_result(
        self,
        recentered_sequence: torch.Tensor,
        accepted_x: torch.Tensor,
        call: dict[str, Any],
    ) -> None:
        """Commit generated tokens after ARDY's recenter operation.

        The denoising call uses a local history+generation window. The
        recentered full history has the same newly generated tokens at its
        tail; crop that tail back to the local window before reusing ARDY's
        exact denoiser projections and masks for the accepted commit.
        """
        generation_tokens = int(call["generation_token_mask"].sum(dim=1)[0].item())
        if recentered_sequence.shape[1] < generation_tokens:
            raise RuntimeError("recentered history is shorter than current generation")
        # Recentered history is in the coordinate frame for the next window;
        # commit only the newly generated chunk. Replay it as a standalone
        # generation-only denoiser call so all frame/token masks have the same
        # 40-frame shape (the original call may have an 80/120-frame future).
        local_sequence = recentered_sequence[:, -generation_tokens:]
        batch = local_sequence.shape[0]
        generation_frames = int(call["generation_len"].max().item())
        device = local_sequence.device
        history_len = torch.zeros(batch, dtype=torch.long, device=device)
        generation_len = torch.full_like(history_len, generation_frames)
        future_len = torch.zeros_like(history_len)
        history_mask = torch.zeros(batch, generation_frames, dtype=torch.bool, device=device)
        generation_mask = torch.ones_like(history_mask)
        future_mask = torch.zeros_like(history_mask)
        motion_mask = torch.zeros(
            batch, generation_frames, call["motion_mask"].shape[-1],
            dtype=call["motion_mask"].dtype, device=device,
        )
        observed_motion = torch.zeros_like(motion_mask)
        history_token_mask, generation_token_mask, future_token_mask = (
            self.raw_denoiser.hybrid.convert_frame_mask_to_token_mask(
                history_mask, generation_mask, future_mask, motion_mask
            )
        )
        local_call = dict(call)
        local_call.update(
            x=local_sequence,
            history_len=history_len,
            generation_len=generation_len,
            future_len=future_len,
            history_mask=history_mask,
            generation_mask=generation_mask,
            future_mask=future_mask,
            history_token_mask=history_token_mask,
            generation_token_mask=generation_token_mask,
            future_token_mask=future_token_mask,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
        )
        self._commit_accepted_result(local_sequence, local_call)

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
