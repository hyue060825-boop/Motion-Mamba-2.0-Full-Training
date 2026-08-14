"""Real Mamba-3 CUDA acceptance test for the Stage-2 hybrid kernel."""

from __future__ import annotations

import torch

from conftest import requires_cuda


def _make_real_hybrid():
    from mamba_ssm.modules.mamba3 import Mamba3
    from minimal_attention_kernel import MinimalAttentionKernelConfig
    from sequence_kernel import SequenceKernel, SequenceKernelConfig

    mamba = Mamba3(
        d_model=64,
        d_state=32,
        expand=2,
        headdim=32,
        ngroups=1,
        is_mimo=False,
        chunk_size=16,
        layer_idx=0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    attention = MinimalAttentionKernelConfig(
        d_model=64,
        num_heads=4,
        window_size_chunks=2,
        use_attention_sinks=True,
        num_sink_tokens=2,
        attention_backend="sdpa",
    )
    return SequenceKernel(
        SequenceKernelConfig(mode="hybrid", attention=attention),
        mamba3=mamba,
    ).to(device="cuda", dtype=torch.bfloat16)


@requires_cuda
def test_real_mamba_hybrid_forward_backward_state_cache_reset_and_ablations():
    torch.manual_seed(23)
    kernel = _make_real_hybrid().train()
    state = kernel.init_state(
        batch_size=2, device=torch.device("cuda"), dtype=torch.bfloat16
    )

    history = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    _, history_pending, _ = kernel.forward_chunk(history, state=state)
    kernel.commit(state, history_pending, chunk_id=0)
    assert state.attention_cache.chunk_ids == (0,)
    assert state.mamba_state.ssm_state.shape == (2, 4, 32, 32)

    current = torch.randn(
        2, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    output, pending, aux = kernel.forward_chunk(current, state=state)
    assert output.shape == current.shape
    assert torch.isfinite(output.float()).all()
    assert aux["fusion"] == "gated_add"
    assert pending.attention_kv is not None and pending.mamba_state is not None

    output.float().square().mean().backward()
    assert current.grad is not None and torch.isfinite(current.grad.float()).all()
    assert kernel.attention.q_proj.weight.grad is not None
    assert kernel.mamba_path.in_proj.weight.grad is not None
    assert kernel.fusion_gate_logit.grad is not None

    kernel.eval()
    with torch.no_grad():
        fresh = kernel.init_state(
            batch_size=2, device=torch.device("cuda"), dtype=torch.bfloat16
        )
        attention_fresh, _, _ = kernel.forward_chunk(
            current.detach(),
            state=fresh,
            enable_attention_path=True,
            enable_mamba_path=False,
        )
        attention_history, _, _ = kernel.forward_chunk(
            current.detach(),
            state=state,
            enable_attention_path=True,
            enable_mamba_path=False,
        )
        assert not torch.allclose(attention_fresh, attention_history)

        mamba_fresh, _, _ = kernel.forward_chunk(
            current.detach(),
            state=fresh,
            enable_attention_path=False,
            enable_mamba_path=True,
        )
        mamba_history, _, _ = kernel.forward_chunk(
            current.detach(),
            state=state,
            enable_attention_path=False,
            enable_mamba_path=True,
        )
        assert not torch.allclose(mamba_fresh, mamba_history)

        row1_before = state.mamba_state.ssm_state[1].clone()
        kernel.reset_state(state, torch.tensor([True, False], device="cuda"))
        assert not bool(state.attention_cache.chunks[0].token_valid[0].any())
        assert bool(state.attention_cache.chunks[0].token_valid[1].any())
        assert torch.count_nonzero(state.mamba_state.ssm_state[0]) == 0
        assert torch.equal(state.mamba_state.ssm_state[1], row1_before)
