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

## 3. Stage Order And Interface Maturity

Do not skip ahead.

The Mamba side has three interface levels:

```text
Level 0: planned interface, not implemented yet
Level 1: chunk wrapper implemented and Phase-0 verified
Level 2: memory-augmented operator implemented and verified
```

Current state is **Level 0**.

## 4. Current Mamba-Side Progress

Current status:

```text
done: Mamba-3 state/chunk API audit
done: SISO kernel smoke and basic forward/backward memory checks
current: no committed Mamba wrapper implementation in this repo yet
next: implement Mamba3ChunkWrapper.forward_chunk
next: run Phase-0 full-vs-chunk / reset / determinism tests
not yet: StateProjector
not yet: Event Memory
not yet: learned State-QKV
not yet: ARDY drop-in Mamba block
```

What collaborators can rely on now:

- the intended API shape;
- the replacement point for ARDY;
- the order of implementation and validation.

What collaborators should not assume yet:

- that `Mamba3ChunkWrapper` already exists;
- that `ArdyMambaBlock` already exists;
- that Event Memory or State-QKV retrieval already exists.

## 5. Level 0 Interface: Planned, Not Yet Implemented

This is the interface we are about to implement. It is documented so downstream
collaborators can prepare adapters, but it should not be treated as available
code until Level 1 is marked done.

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

Implementation requirements:

- Use raw `mamba3_siso_combined(..., Input_States=...,
  return_final_states=True)`.
- Do not use `Mamba3.forward()` for non-zero chunk continuation.
- Do not make downstream code depend on single-token decode internals.
- Expose recurrent state through an opaque `MambaState`.
- Use `reset_mask` to reset episode state.

Planned state object:

```text
MambaState:
  ssm_state: [batch, nheads, headdim, d_state], fp32
```

Optional fields can be added if the wrapper needs them. Downstream code should
treat the state as opaque except for reset and detach policy.

## 6. Level 1 Interface: After Chunk Wrapper Verification

Level 1 is reached after:

```text
Mamba3ChunkWrapper.forward_chunk implemented
full-sequence vs chunked consistency passed
state reset test passed
determinism test passed
basic forward/backward smoke passed
```

At Level 1, the usable interface is:

```python
outputs, next_state, aux = model.forward_chunk(
    x=chunk,
    state=prev_state,
    reset_mask=reset_mask,
    return_aux=True,
)
```

Expected capability:

- run chunked Mamba sequence modeling;
- carry recurrent state across chunks;
- reset state between episodes;
- measure latency and peak memory;
- support pure Mamba baseline training on synthetic tasks.

Expected `aux` keys:

```text
state_shape
chunk_len
state_was_reset
peak_gpu_memory
```

At Level 1, ARDY should still not depend on Mamba for production generation.
It can start shape-adapter work and synthetic tests.

## 7. Level 2 Interface: After State Projector And Event Memory

Level 2 is reached after:

```text
StateProjector implemented
synthetic state probes pass
oracle Event Memory improves long-horizon tasks
learned State-QKV is implemented only after oracle memory helps
```

At Level 2, expose:

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

Expected capability:

- preserve compressed recurrent state across chunks;
- retrieve explicit long-term event memory;
- reduce long-horizon recall decay in synthetic tasks;
- provide diagnostics showing whether memory is used.

## 8. Final ARDY-Facing Interface

The final ARDY-facing block should preserve ARDY's transformer-block signature
and hide Mamba internals behind optional state/memory arguments:

```python
class ArdyMambaBlock(nn.Module):
    def forward(
        self,
        x,
        x_pad_mask,
        text_feat,
        text_feat_pad_mask,
        timesteps,
        first_heading_angle=None,
        token_index=None,
        state=None,
        memory=None,
        reset_mask=None,
        update_memory=False,
        return_aux=False,
    ):
        ...
```

Return modes:

```python
# drop-in ARDY mode
output = block(...)

# memory-aware mode
output, next_state, next_memory, aux = block(..., return_aux=True)
```

Final capability:

- replace ARDY `root_model` and `body_model`;
- keep ARDY diffusion, autoencoder, motion representation, and export code
  unchanged;
- run autoregressive diffusion with Mamba sequence mixing;
- optionally carry Mamba state / Event Memory across AR windows.

## 9. Downstream Inputs

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

## 10. Training Modes

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

## 11. Minimal Integration Checklist

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
