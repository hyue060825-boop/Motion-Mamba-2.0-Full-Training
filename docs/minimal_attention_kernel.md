# Minimal Attention Kernel

Last updated: 2026-08-14.

## Repository audit

The repository is an installable `src/`-layout Python package. Its existing
runtime is Mamba-specific:

- `src/state_memory_operator/` exposes `Mamba3ChunkWrapper`, recurrent state
  projection, State-QKV retrieval, and FIFO/participative Event Memory.
- Importing `state_memory_operator` eagerly imports
  `mamba3_chunk_wrapper.py`, which imports `mamba_ssm` and its Triton kernels.
  The minimal path therefore must not be re-exported through that package.
- `src/ardy_adapter/` is downstream integration code and is intentionally not
  used by this kernel.
- Configuration uses ordinary dataclasses (`StateMemoryOperatorConfig`).
- Stateful callers explicitly create state/memory, pass it to
  `forward_chunk`, and receive state plus an `aux` dictionary. The new API
  follows this convention while keeping read and commit separate.
- Existing tests are mainly CUDA Mamba tests plus CPU tests for Event Memory.
  `tests/conftest.py` only adds the repository `src/` directory when CUDA is
  available, so the standalone tests add it without initializing Mamba.
- There was no reusable self-attention implementation in this repository.

Local backend audit (2026-08-14):

- `/home/plote/miniconda3/envs/beingh/bin/python` has PyTorch
  `2.10.0+cu128` and FlashAttention `2.8.3`.
- FlashAttention exports `flash_attn_func`, whose separate Q/K/V interface
  supports `Q length != K/V length`.
- No FlashAttention 4 Python package/interface was found. FA4 is not claimed.
- CUDA was not visible in the local execution session, so CPU correctness is
  verified with PyTorch SDPA; FlashAttention tests skip explicitly unless a
  compatible CUDA device and dtype are available.

The existing Mamba path remains unchanged and coexists with the new package.

## Architecture

```text
Current Chunk
    |
    v
Attention
    ^
    |
Sliding Window KV
+
Attention Sinks
```

More precisely:

```text
                  Persistent
              Attention Sinks
                    |
                    v
Previous Chunks -> Sliding-Window KV
                    |
                    | cached K,V
                    v
Current Chunk ---> Attention ---> Output
       |                |
       +--- current K,V--+
                        |
                   explicit commit
                        |
                        v
                Sliding-Window KV
```

For a current chunk `X_t`, queries come only from the current chunk. Keys and
values are concatenated in this order:

```text
learned static sink K/V + recent committed chunk K/V + current chunk K/V
```

Current K/V always participate in the current self-attention computation.
`commit=False` only means that they are not persisted for later calls.

Within a chunk, attention is bidirectional by default. Optional causal mode
masks only future positions inside the current chunk; all sink/history entries
remain visible. Across chunks, history is one-way because only already
committed chunks can be read.

## Public API

```python
from minimal_attention_kernel import (
    MinimalAttentionKernel,
    MinimalAttentionKernelConfig,
)

kernel = MinimalAttentionKernel(config)
cache = kernel.init_cache(batch_size=B, device=x.device, dtype=x.dtype)

out, pending, aux = kernel.forward_chunk(x, cache=cache, commit=False)
kernel.commit(cache, pending, chunk_id=0)
```

`forward_chunk(..., commit=True, chunk_id=...)` is a convenience form, but
the explicit two-call form is preferred when a caller must decide whether a
chunk is final/accepted. The cache is an explicit mutable object; it is never
hidden in module-global or module-instance state.

## Cache representation and sliding window

One `ChunkSlidingWindowKVCache` belongs to one attention layer/operator.
Downstream multi-layer models create one cache per layer. Every projected K/V
tensor has shape:

```text
[batch, heads, chunk_tokens, head_dim]
```

Normal chunks are stored as distinct `CachedChunk` entries so variable chunk
lengths and chunk-level FIFO eviction are preserved. With
`window_size_chunks=W`, only the latest `W` normal chunks remain. A zero
window is valid and stores no normal history.

Sinks are learned static tokens owned by the kernel. Their projected K/V are
computed on demand for the current weights and are not stored as episode data
inside the cache. They are therefore always present when enabled, never
evicted, and remain after full or per-row reset.

## Read versus commit

Read and commit are separate operations:

1. `forward_chunk` projects current Q/K/V, reads sinks and committed history,
   and returns output plus a `PendingChunkKV` handle.
2. The cache is unchanged when `commit=False`.
3. `kernel.commit(cache, pending, chunk_id)` validates ownership/shapes and
   appends one whole chunk, then evicts old normal chunks by FIFO.

This permits any caller to perform repeated read-only passes and commit only
an accepted representation. The kernel has no diffusion-specific knowledge.

## Reset semantics

`kernel.reset_cache(cache)` drops every normal chunk. A direct `cache.reset()`
is also valid when weights have not changed. `kernel.reset_cache(cache,
reset_mask)` clears
history only for selected batch rows. Per-row reset is represented by a
valid-row mask on each cached chunk; invalid rows have zeroed K/V and are
masked out during attention, so no previous-episode data can be read. Entries
that become invalid for every row are removed. Learned static sinks remain.

## Attention backends

- `sdpa`: PyTorch `scaled_dot_product_attention`, always the correctness
  reference.
- `flash2`: FlashAttention 2 `flash_attn_func` when installed and compatible.
  If the package/kernel is unavailable, the implementation falls back to SDPA
  when configured to allow fallback; diagnostics report the backend actually
  used. Explicit `allow_backend_fallback=False` raises instead.
- `flash4`: reserved configuration name which currently raises
  `NotImplementedError`. No FA4 support is claimed.

Both backends use the same Q/K/V concatenation and mask semantics. Arbitrary
per-row history masks (needed by asynchronous batch reset) use SDPA because
FlashAttention's dense API does not accept that mask.

## Scope

This package intentionally contains:

```text
Attention + chunk-level sliding-window KV cache + learned attention sinks
```

It contains no ARDY adapter, DiT, diffusion loop or timestep conditioning,
motion/root/body representation, Mamba import, hybrid fusion, Deep Forcing,
participative eviction, simulation, or robotics code.

## Known limitations

- This is one attention operator, not a Transformer/DiT stack. A downstream
  multi-layer model must instantiate one kernel and one cache per layer.
- The implementation concatenates retained chunks for each read. It uses
  FlashAttention for the attention computation when possible, but does not yet
  use a fused paged-KV-cache append kernel.
- Per-row asynchronous reset introduces an arbitrary validity mask and therefore
  falls back to SDPA for that read when the dense FlashAttention backend is
  requested.
- Commit detaches K/V by default, which is appropriate for inference and
  prevents accidental unbounded graphs. Recurrent training can request
  `detach=False`, accepting the associated graph-memory cost.
- Cached values are projected activations. They must be reset/recomputed after
  an optimizer step that changes projection weights; carrying them across
  parameter updates would make them stale. The cache records the K/V projection
  parameter mutation counters and rejects stale reads/commits until the caller
  performs a full `kernel.reset_cache(cache)`.
- Actual FlashAttention latency/parity and peak GPU allocation remain to be
  executed on a session with a working CUDA driver.

## Local verification record

Executed on 2026-08-14 with
`/home/plote/miniconda3/envs/beingh/bin/python` (PyTorch `2.10.0+cu128`):

```bash
PYTHONPATH=src python -m pytest tests/test_minimal_attention_kernel.py -q
# 25 passed, 1 skipped

PYTHONPATH=src python examples/minimal_attention_kernel.py --backend sdpa
# 8 chunks completed; W=3 stayed bounded at 3 chunks / 18 history tokens.

PYTHONPATH=src python examples/minimal_attention_kernel.py --backend flash2 --chunks 2
# completed through the documented CPU -> SDPA fallback.
```

The local skip is retained because that workstation session has no usable
CUDA driver. It is an environment-specific skip, not the final project status:
the exact A800 FA2 acceptance later passed and is recorded in the Stage-01
closeout.

Small CPU smoke benchmark (`B=2`, `D=128`, `H=4`, chunk length 32, `W=4`,
four sinks, 20 measured iterations after five warmups):

```text
requested  used  median latency  throughput       cache K/V  peak GPU
SDPA       SDPA  0.296 ms        216,332 token/s  256 KiB    n/a
FA2        SDPA  0.293 ms        218,723 token/s  256 KiB    n/a
```

The second row is a fallback sanity check, not a FlashAttention performance
number. CPU timings are preliminary and machine/load dependent.
