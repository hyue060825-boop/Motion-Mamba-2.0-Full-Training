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

## 3. Master Checklist And Interface Maturity

Do not skip ahead.

The Mamba side has three interface levels:

```text
Level 0: planned interface, not implemented yet
Level 1: chunk wrapper implemented and Phase-0 verified
Level 2: memory-augmented operator implemented and verified
```

Current state is **Level 0**.

Master checklist:

- [x] **Audit Mamba-3 state/chunk API.**
  Acceptance: identify the recurrent state tensor to expose, identify the
  kernel path for multi-token chunk continuation, and document why downstream
  code must not call Mamba internals directly.
- [x] **Verify SISO Mamba-3 kernel can run in isolation.**
  Acceptance: basic forward/backward smoke test runs outside ARDY.
- [ ] **Implement `Mamba3ChunkWrapper.forward_chunk`.**
  Acceptance: exposes the planned `forward_chunk` API and returns an opaque
  `MambaState`.
- [ ] **Pass Phase-0 wrapper tests.**
  Acceptance: full-vs-chunk consistency, reset, determinism, and basic
  forward/backward tests all pass.
- [ ] **Implement `StateProjector`.**
  Acceptance: projects `ssm_state` into stable per-chunk state summaries
  `s_before`, `s_after`, and `delta`.
- [ ] **Pass state probe.**
  Acceptance: state summaries predict basic synthetic event attributes above
  random baseline.
- [ ] **Implement fixed-capacity oracle Event Memory.**
  Acceptance: oracle retrieval improves long-horizon synthetic tasks.
- [ ] **Implement learned State-QKV retrieval.**
  Acceptance: learned retrieval beats random and approaches oracle behavior.
- [ ] **Implement ARDY-compatible `ArdyMambaBlock`.**
  Acceptance: same call signature as ARDY `TransformerEncoderBlock`; output
  shapes match `root_model` and `body_model` requirements.
- [ ] **Pass ARDY drop-in smoke tests.**
  Acceptance: single denoising step, one autoregressive window, short sample,
  and G1 export path run with the Mamba block.
- [ ] **Add cross-window state/memory for ARDY.**
  Acceptance: memory is read during denoising and committed only after the
  final clean/generated window.
- [ ] **Run final comparison.**
  Acceptance: compare ARDY baseline, Mamba-denoiser ARDY, and
  memory-augmented Mamba on agreed latency, memory, and motion metrics.

## 4. Current Mamba-Side Progress

Current status checklist:

- [x] Mamba-3 state/chunk API audit is complete.
- [x] SISO kernel smoke and basic forward/backward memory checks are complete.
- [ ] No committed `Mamba3ChunkWrapper` implementation exists yet.
- [ ] No Phase-0 wrapper test suite has passed yet.
- [ ] No `StateProjector` implementation exists yet.
- [ ] No Event Memory implementation exists yet.
- [ ] No learned State-QKV implementation exists yet.
- [ ] No ARDY-compatible `ArdyMambaBlock` exists yet.

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

Level 0 checklist:

- [x] Interface shape documented.
- [x] ARDY replacement target documented.
- [ ] Interface implemented in code.
- [ ] Interface verified by tests.
- [ ] Interface safe for downstream dependency.

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

Implementation requirements checklist:

- [ ] Use raw `mamba3_siso_combined(..., Input_States=...,
  return_final_states=True)`.
- [ ] Do not use `Mamba3.forward()` for non-zero chunk continuation.
- [ ] Do not make downstream code depend on single-token decode internals.
- [ ] Expose recurrent state through an opaque `MambaState`.
- [ ] Use `reset_mask` to reset episode state.

Planned state object:

```text
MambaState:
  ssm_state: [batch, nheads, headdim, d_state], fp32
```

Optional fields can be added if the wrapper needs them. Downstream code should
treat the state as opaque except for reset and detach policy.

## 6. Level 1 Interface: After Chunk Wrapper Verification

Level 1 is reached after:

- [ ] `Mamba3ChunkWrapper.forward_chunk` implemented.
- [ ] Full-sequence vs chunked consistency test passed.
- [ ] State reset test passed.
- [ ] Determinism test passed.
- [ ] Basic forward/backward smoke test passed.

At Level 1, the usable interface is:

```python
outputs, next_state, aux = model.forward_chunk(
    x=chunk,
    state=prev_state,
    reset_mask=reset_mask,
    return_aux=True,
)
```

Expected capability after Level 1:

- [ ] Run chunked Mamba sequence modeling.
- [ ] Carry recurrent state across chunks.
- [ ] Reset state between episodes.
- [ ] Measure latency and peak memory.
- [ ] Support pure Mamba baseline training on synthetic tasks.

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

- [ ] `StateProjector` implemented.
- [ ] Synthetic state probes passed.
- [ ] Oracle Event Memory improves long-horizon tasks.
- [ ] Learned State-QKV implemented only after oracle memory helps.

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

Expected capability after Level 2:

- [ ] Preserve compressed recurrent state across chunks.
- [ ] Retrieve explicit long-term event memory.
- [ ] Reduce long-horizon recall decay in synthetic tasks.
- [ ] Provide diagnostics showing whether memory is used.

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

Final capability checklist:

- [ ] Replace ARDY `root_model` and `body_model`.
- [ ] Keep ARDY diffusion, autoencoder, motion representation, and export code
  unchanged;
- [ ] Run autoregressive diffusion with Mamba sequence mixing.
- [ ] Optionally carry Mamba state / Event Memory across AR windows.

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

- [ ] Train pure Mamba-3 baseline.
- [ ] Freeze backbone; train projector / memory / injection modules.
- [ ] Joint finetune with smaller backbone LR.

## 11. Minimal Integration Checklist

Before downstream work depends on this operator, verify:

- [ ] `forward_chunk` returns the same outputs as full-sequence execution within
  tolerance.
- [ ] State reset prevents leakage across episodes.
- [ ] Carrying state without reset changes the second episode, proving state is
  active.
- [ ] Determinism holds under the same seed and input.
- [ ] Baseline training loss decreases on a small synthetic task.
- [ ] State projector features predict basic event attributes better than
  random.
- [ ] Oracle memory improves long-horizon performance before learned QKV is
  built.
