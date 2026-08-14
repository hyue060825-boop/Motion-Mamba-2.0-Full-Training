"""CPU contract tests for the Stage-2 generic minimal/hybrid interface."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from minimal_attention_kernel import MinimalAttentionKernelConfig
from sequence_kernel import SequenceKernel, SequenceKernelConfig


def _attention_config() -> MinimalAttentionKernelConfig:
    return MinimalAttentionKernelConfig(
        d_model=32,
        num_heads=2,
        window_size_chunks=2,
        use_attention_sinks=True,
        num_sink_tokens=2,
        attention_backend="sdpa",
    )


class FakeMamba(nn.Module):
    """Small differentiable recurrent module used only for CPU interface tests."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)


class FakeMambaWrapper:
    def __init__(self, mamba3: FakeMamba) -> None:
        self.mamba3 = mamba3

    def init_state(self, batch_size, device=None, dtype=None):
        parameter = self.mamba3.in_proj.weight
        return torch.zeros(
            batch_size,
            parameter.shape[0],
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def forward_chunk(self, x, state=None, reset_mask=None):
        if state is None:
            state = torch.zeros_like(x[:, 0])
        if reset_mask is not None:
            state = torch.where(reset_mask[:, None], torch.zeros_like(state), state)
        outputs = []
        hidden = state
        for token in x.unbind(dim=1):
            hidden = torch.tanh(self.mamba3.in_proj(token) + 0.5 * hidden)
            outputs.append(self.mamba3.out_proj(hidden))
        return torch.stack(outputs, dim=1), hidden


def _make_hybrid(seed: int = 0) -> SequenceKernel:
    torch.manual_seed(seed)
    mamba = FakeMamba(32)
    return SequenceKernel(
        SequenceKernelConfig(mode="hybrid", attention=_attention_config()),
        mamba3=mamba,
        mamba_wrapper=FakeMambaWrapper(mamba),
    )


def test_minimal_is_exact_stage1_path_and_has_zero_mamba_modules():
    torch.manual_seed(1)
    kernel = SequenceKernel(
        SequenceKernelConfig(mode="minimal", attention=_attention_config())
    )
    state = kernel.init_state(batch_size=2)
    x = torch.randn(2, 5, 32)

    expected, _, _ = kernel.attention.forward_chunk(
        x, cache=state.attention_cache.clone(), commit=False
    )
    actual, pending, aux = kernel.forward_chunk(x, state=state)

    assert torch.equal(actual, expected)
    assert pending.mamba_state is None
    assert aux["mamba_path_enabled"] is False
    module_names = dict(kernel.named_modules())
    assert "mamba_path" not in module_names
    assert all("mamba" not in type(module).__name__.lower() for module in module_names.values())


def test_clean_minimal_import_never_loads_mamba_packages():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    code = """
import sys
from minimal_attention_kernel import MinimalAttentionKernelConfig
from sequence_kernel import SequenceKernel, SequenceKernelConfig
c = MinimalAttentionKernelConfig(d_model=32, num_heads=2, window_size_chunks=1,
    attention_backend='sdpa', num_sink_tokens=2)
k = SequenceKernel(SequenceKernelConfig(mode='minimal', attention=c))
assert 'mamba_ssm' not in sys.modules
assert 'state_memory_operator' not in sys.modules
assert not any('mamba' in type(m).__name__.lower() for m in k.modules())
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = src_dir
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_hybrid_forward_backward_shape_finite_and_explicit_fusion():
    kernel = _make_hybrid().train()
    state = kernel.init_state(batch_size=2)
    x = torch.randn(2, 6, 32, requires_grad=True)

    hybrid, pending, aux = kernel.forward_chunk(x, state=state)
    attention, _, _ = kernel.forward_chunk(
        x, state=state, enable_attention_path=True, enable_mamba_path=False
    )
    mamba, _, _ = kernel.forward_chunk(
        x, state=state, enable_attention_path=False, enable_mamba_path=True
    )
    expected = attention + torch.sigmoid(kernel.fusion_gate_logit) * mamba

    assert hybrid.shape == x.shape
    assert torch.allclose(hybrid, expected)
    assert torch.isfinite(hybrid).all()
    assert pending.attention_kv is not None
    assert pending.mamba_state is not None
    assert aux["fusion"] == "gated_add"

    hybrid.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert kernel.fusion_gate_logit.grad is not None
    assert kernel.mamba_path.in_proj.weight.grad is not None
    assert kernel.attention.q_proj.weight.grad is not None


def test_hybrid_state_and_cache_handoff_are_commit_controlled():
    kernel = _make_hybrid().eval()
    state = kernel.init_state(batch_size=2)
    initial_mamba = state.mamba_state.clone()
    x = torch.randn(2, 4, 32)

    pending = None
    for _ in range(7):
        _, pending, _ = kernel.forward_chunk(x, state=state, commit=False)
        assert state.attention_cache.num_chunks == 0
        assert torch.equal(state.mamba_state, initial_mamba)

    kernel.commit(state, pending, chunk_id=10)
    assert state.attention_cache.chunk_ids == (10,)
    assert not torch.equal(state.mamba_state, initial_mamba)

    _, pending2, _ = kernel.forward_chunk(x, state=state, commit=True, chunk_id=11)
    assert state.attention_cache.chunk_ids == (10, 11)
    assert torch.equal(state.mamba_state, pending2.mamba_state.detach())


def test_controlled_ablations_show_attention_history_and_mamba_state_effects():
    kernel = _make_hybrid(seed=4).eval()
    fresh = kernel.init_state(batch_size=1)
    with_history = fresh.clone()
    history = torch.randn(1, 5, 32) * 3.0
    _, history_pending, _ = kernel.forward_chunk(history, state=with_history)
    kernel.commit(with_history, history_pending, chunk_id=0)

    current = torch.randn(1, 4, 32)
    attention_fresh, _, _ = kernel.forward_chunk(
        current,
        state=fresh,
        enable_attention_path=True,
        enable_mamba_path=False,
    )
    attention_history, _, _ = kernel.forward_chunk(
        current,
        state=with_history,
        enable_attention_path=True,
        enable_mamba_path=False,
    )
    assert not torch.allclose(attention_fresh, attention_history)

    zero_recurrent = fresh.clone()
    nonzero_recurrent = fresh.clone()
    nonzero_recurrent.mamba_state = torch.full_like(nonzero_recurrent.mamba_state, 2.0)
    mamba_zero, _, _ = kernel.forward_chunk(
        current,
        state=zero_recurrent,
        enable_attention_path=False,
        enable_mamba_path=True,
    )
    mamba_nonzero, _, _ = kernel.forward_chunk(
        current,
        state=nonzero_recurrent,
        enable_attention_path=False,
        enable_mamba_path=True,
    )
    assert not torch.allclose(mamba_zero, mamba_nonzero)


def test_hybrid_per_row_and_full_reset_cover_both_memory_types():
    kernel = _make_hybrid(seed=7).eval()
    state = kernel.init_state(batch_size=2)
    x = torch.randn(2, 4, 32)
    _, pending, _ = kernel.forward_chunk(x, state=state)
    kernel.commit(state, pending, chunk_id=0)
    row1_before = state.mamba_state[1].clone()

    kernel.reset_state(state, torch.tensor([True, False]))
    assert not bool(state.attention_cache.chunks[0].token_valid[0].any())
    assert bool(state.attention_cache.chunks[0].token_valid[1].any())
    assert torch.count_nonzero(state.mamba_state[0]) == 0
    assert torch.equal(state.mamba_state[1], row1_before)

    kernel.reset_state(state)
    assert state.attention_cache.num_chunks == 0
    assert torch.count_nonzero(state.mamba_state) == 0


def test_hybrid_module_tree_contains_both_paths_and_mamba_only_is_supported():
    kernel = _make_hybrid()
    modules = dict(kernel.named_modules())
    assert "attention" in modules
    assert "mamba_path" in modules

    state = kernel.init_state(batch_size=2)
    x = torch.randn(2, 3, 32)
    output, pending, aux = kernel.forward_chunk(
        x,
        state=state,
        enable_attention_path=False,
        enable_mamba_path=True,
    )
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert pending.attention_kv is None
    assert pending.mamba_state is not None
    assert aux["attention_path_enabled"] is False


def test_invalid_mode_and_minimal_mamba_injection_are_rejected():
    with pytest.raises(ValueError, match="mode"):
        SequenceKernelConfig(mode="unknown", attention=_attention_config())
    mamba = FakeMamba(32)
    with pytest.raises(ValueError, match="accepts no Mamba"):
        SequenceKernel(
            SequenceKernelConfig(mode="minimal", attention=_attention_config()),
            mamba3=mamba,
        )


def test_hybrid_rejects_masked_recurrent_timesteps_instead_of_silently_stepping():
    kernel = _make_hybrid()
    state = kernel.init_state(batch_size=2)
    x = torch.randn(2, 4, 32)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    with pytest.raises(NotImplementedError, match="cannot skip padded"):
        kernel.forward_chunk(x, state=state, current_mask=mask)
