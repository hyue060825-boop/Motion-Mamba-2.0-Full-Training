# Mamba Integration Contract

Last updated: 2026-07-20.

Audience: collaborators connecting the Mamba-side operator to downstream
training, simulation, or server-side experiments.

For ARDY-specific replacement details, also read:

```text
docs/ardy_mamba_replacement_plan.md
```

## 1. Responsibility Split

The Mamba side owns the **sequence operator and memory interface**.
Downstream motion / ARDY / G1 code owns **data, task loss, simulator runtime,
and evaluation**.

Boundary:

```text
downstream motion system
  -> prepares chunks, conditions, masks, episode resets
  -> calls Mamba operator
  <- receives chunk outputs, next recurrent state, memory diagnostics
```

The Mamba-side code should not own simulator stepping, robot deployment,
dataset preprocessing, or benchmark reporting. The downstream side should not
write into Mamba internal state tensors directly.

## 2. Current Technical Target

Active target: **State-Transition Memory Operator**.

Instead of using a plain transformer-style token KV cache, the external memory
is built from Mamba recurrent state transitions:

```text
raw Mamba recurrent state
  -> StateProjector
  -> s_before, s_after, delta
  -> state-based Q / K / V
  -> fixed-capacity Event Memory
  -> memory-conditioned next chunk
```

Stage 1 is **SISO Mamba-3 only**. MIMO, CUTE step decode, learned memory
lifecycle, compressed memory, and downstream motion integration are later.

## 3. Stage Order

Do not skip ahead.

1. Implement `Mamba3ChunkWrapper.forward_chunk(x, state) -> (y, next_state)`.
2. Pass Phase-0 tests:
   - full-sequence vs chunked consistency;
   - state reset;
   - determinism.
3. Implement `StateProjector` on `ssm_state`.
4. Build synthetic tasks:
   - associative recall;
   - state-transition recall.
5. Train pure Mamba-3 baseline and record accuracy-vs-horizon decay.
6. Add oracle Event Memory.
7. Only if oracle memory helps, add learned State-QKV retrieval.

## 4. Public API

Downstream code should depend on a small API, not on upstream Mamba internals.

### 4.1 Chunk Wrapper

```python
outputs, next_state = model.forward_chunk(
    x=chunk,
    state=prev_state,
    memory_condition=memory_condition,
    reset_mask=reset_mask,
)
```

Expected shapes:

```text
chunk:            [batch, chunk_len, d_model]
outputs:          [batch, chunk_len, d_model] or task-specific hidden dim
prev_state:       MambaState or None
next_state:       MambaState
memory_condition: [batch, d_memory] or None
reset_mask:       [batch] bool, True means reset this episode state
```

Stage-1 implementation requirements:

- Use raw `mamba3_siso_combined(..., Input_States=...,
  return_final_states=True)`.
- Do not use `Mamba3.forward()` for non-zero chunk continuation.
- Do not use `Mamba3.step()` in the current local environment.

### 4.2 State Object

The state object hides upstream Mamba internals from downstream code. At
minimum it must carry:

```text
ssm_state: [batch, nheads, headdim, d_state], fp32
```

Optional fields can be added if the wrapper needs them. Downstream code should
treat the state as opaque except for reset and detach policy.

### 4.3 Memory-Augmented Forward

After Event Memory exists, expose:

```python
outputs, next_state, next_memory, aux = model.forward_chunk_with_memory(
    x=chunk,
    state=prev_state,
    memory=prev_memory,
    reset_mask=reset_mask,
    update_memory=True,
)
```

Expected `aux` keys:

```text
state_before_summary
state_after_summary
state_delta
retrieval_hit_at_k
retrieval_rank
retrieval_entropy
memory_usage_norm
injection_delta_norm
peak_gpu_memory
```

Downstream experiments should save `aux`; otherwise we cannot tell whether
memory works or is being ignored.

## 5. Downstream Inputs

The downstream side provides:

```text
chunk tensor x
condition tensor, if used
target tensor and task loss
episode_id or reset_mask
sequence position / chunk id
event labels for synthetic or motion-event retrieval probes
evaluation metrics
```

For motion tasks, useful event labels include:

```text
contact onset / release
foot support switch
root velocity transition
action phase transition
repeated pose motif
object-interaction transition
```

## 6. Training Modes

Expose these modes as configs:

```text
baseline_mamba3:
  pure Mamba-3, no external memory

oracle_event_memory:
  ground-truth event id retrieves memory value

state_qkv_memory:
  learned State-QKV retrieval

injection_ablation:
  output injection / input injection / state correction
```

Recommended curriculum:

1. Train pure Mamba-3 baseline.
2. Freeze backbone; train projector / memory / injection modules.
3. Joint finetune with smaller backbone LR.

## 7. Local-vs-Server Boundary

Local RTX 3070 Laptop 8 GB is enough for the current Mamba-side stage:

```text
SISO
bf16
D <= 768 initially
chunk_len in {8, 16, 32, 64, 128}
memory capacity <= 512 initially
synthetic Phase 0-5 experiments
```

Use chunked execution for long horizons such as `T=8192` or `T=32768`. Do not
keep an unbounded full-sequence computation graph unless explicitly testing
memory limits.

Use a server for:

```text
MIMO / CUTE path
ARDY or motion downstream large training
HumanML3D / KITML / BABEL full-scale training
large model-width or batch-size sweeps
actual local OOM
```

## 8. Minimal Integration Checklist

Before downstream work depends on this operator, verify:

- `forward_chunk` returns the same outputs as full-sequence execution within
  tolerance.
- State reset prevents leakage across episodes.
- Carrying state without reset changes the second episode, proving state is
  active.
- Determinism holds under the same seed and input.
- Baseline training loss decreases on a small synthetic task.
- State projector features predict basic event attributes better than random.
- Oracle memory improves long-horizon performance before learned QKV is built.

## 9. Known Environment Notes

Known-good local path for SISO:

```text
env: env_isaaclab
torch: 2.7.0+cu128
temporary Triton overlay: /tmp/mamba_op_triton
triton in overlay: 3.5.1
```

Known blocked paths:

```text
MIMO: TileLang import failure in current local environment
Mamba3.step(): CUTE/quack requires torch.float4_e2m1fn_x2, missing in Torch 2.7
```

Therefore the current implementation should call the raw SISO chunk kernel
directly.
