# ARDY to Mamba Replacement Plan

Last updated: 2026-07-20.

Purpose: define exactly how the Mamba-side work connects to ARDY-style
humanoid motion generation/control.

This document is for collaborators using
`https://github.com/neptune-T/mamba_transformer_keneral` as the server-side
handoff repo. It is not an experiment log.

## 1. Paper-Level Goal

Our humanoid-control paper has three contribution tracks:

1. **Real-time sequence modeling.** Replace transformer-style sequence mixing
   with Mamba-3 / linear-time kernels where possible. The goal is
   hardware-friendly inference with linear sequence complexity.
2. **Long-horizon autoregressive diffusion.** Generate/control motion by
   diffusion inside each chunk and autoregression across chunks:

   ```text
   chunk 1: cond -> diffusion -> output_1
   chunk 2: memory/cache + cond + output_1 -> diffusion -> output_2
   chunk k: memory/cache + cond + output_{k-1} -> diffusion -> output_k
   ```

   The key problem is error accumulation across chunks. Our Mamba-side answer
   is state-transition memory first, then compressed memory / sink-style memory
   after the minimal mechanism works.
3. **Humanoid benchmark and G1 evaluation.** Compare on HumanML3D, KITML,
   BABEL, ARDY-style motion generation, and eventually G1 simulation / real
   robot evaluation.

## 2. What ARDY Already Provides

ARDY already provides:

- autoregressive diffusion over motion windows;
- online text prompting;
- kinematic constraints through `motion_mask` and `observed_motion`;
- G1 skeleton assets and MuJoCo qpos export;
- interactive generation through `Ardy.autoregressive_step()`;
- non-interactive generation through `scripts/generate.py`;
- motion representation, autoencoder, hybrid root/body token conversion, and
  post-processing.

We should not rewrite those first.

The Mamba replacement should integrate at the model interface level: keep
ARDY's generation loop and motion I/O stable, and replace only the denoiser
sequence mixer first.

## 3. Actual Replacement Point

The first Mamba replacement target is the denoiser backbone, not the whole
ARDY system.

ARDY's denoiser is two-stage:

```text
AutoLatentTwostageDenoiser.forward(...)
  -> root_model: TransformerEncoderBlock
       predicts global root motion
  -> convert predicted global root to local root
  -> body_model: TransformerEncoderBlock
       predicts latent body motion conditioned on local root
  -> concatenate root prediction + body prediction
```

The current ARDY transformer block:

```text
TransformerEncoderBlock.forward(
    x,
    x_pad_mask,
    text_feat,
    text_feat_pad_mask,
    timesteps,
    first_heading_angle,
    token_index,
) -> output
```

Inside it:

```text
motion tokens x
  -> input projection
text tokens + diffusion timestep + optional heading
  -> prefix tokens
prefix tokens + motion tokens
  -> positional encoding
  -> SDPATransformerEncoder
  -> keep only motion-token outputs
  -> output projection
```

Mamba should first replace this block while preserving the same call signature.

## 4. What Mamba Replaces vs Does Not Replace

Mamba replaces:

```text
root_model sequence mixer
body_model sequence mixer
later: cross-window history/memory mechanism
```

Mamba does not replace in the first integration:

```text
Ardy.autoregressive_step
Ardy._generate_window
Diffusion / DDIMSampler
AutoLatentClassifierFreeGuidedModel CFG wrapper
HybridMotionConverter
motion_rep
autoencoder / FSQ VAE
G1 skeleton and MuJoCo export
post_process_motion
interactive demo UI
```

This keeps the ARDY runtime and downstream APIs stable while we test whether
Mamba improves speed / memory / horizon.

## 5. Drop-In Block API

This interface is the target, not current implemented code. The Mamba block
should expose the ARDY transformer block interface:

```python
class ArdyMambaBlock(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: torch.Tensor,
        first_heading_angle: torch.Tensor | None = None,
        token_index: torch.Tensor | None = None,
        state: "MambaState | None" = None,
        reset_mask: torch.Tensor | None = None,
        memory_condition: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        ...
```

Required tensor convention:

```text
x:                  [batch, num_tokens, input_dim]
x_pad_mask:         [batch, num_tokens], True = valid token
text_feat:          [batch, max_text_len, llm_dim]
text_feat_pad_mask: [batch, max_text_len], True = valid token
timesteps:          [batch]
first_heading_angle:[batch] or None
token_index:        [batch, num_tokens] or None
output:             [batch, num_tokens, output_dim]
```

For strict ARDY compatibility, `state`, `reset_mask`, `memory_condition`, and
`return_aux` can be ignored in the first drop-in version. They become active
when cross-window memory is added.

## 6. What We Do Now

**Corrected 2026-07-25** -- this section had gone stale since 2026-07-20 and
understated actual progress. Current status checklist:

- [x] Mamba-3 SISO kernel can run in isolation.
- [x] `Mamba3ChunkWrapper.forward_chunk` implemented and Phase-0-verified.
- [x] `StateProjector`, Event Memory (FIFO + Participative), and learned
  State-QKV retrieval are ALL implemented and validated -- see
  `src/state_memory_operator/` and `docs/mamba_integration_contract.md`
  Sections 3/6/7 for the full status.
- [ ] No ARDY Mamba replacement code is implemented yet -- `ArdyMambaBlock`
  does not exist. This is the actual next step now, not the chunk wrapper.

Current interface status:

- [x] `forward_chunk`: implemented, real, tested
  (`state_memory_operator.StateTransitionMemoryOperator.forward_chunk`).
- [x] `forward_chunk_with_memory`: implemented, real, tested (same class).
- [ ] `ArdyMambaBlock`: planned API, not implemented yet -- this is the
  actual immediate work now.

The immediate work is still not to touch ARDY's own code -- but it's no
longer "make the chunk interface real," since that's done. It's building
`ArdyMambaBlock` around the now-real `state_memory_operator` package.

## 7. The Chunk Interface Is Verified -- What That Unlocks

`Mamba3ChunkWrapper.forward_chunk` passed Phase-0 tests; the interface is
usable for non-ARDY synthetic tests (and has been used for exactly that --
see the research repo's whole Stage-1 program):

```python
outputs, next_state, aux = operator.forward_chunk(
    x=chunk,
    state=prev_state,
    reset_mask=reset_mask,
)
```

Implemented and verified:

- [x] Chunked SISO Mamba inference/training.
- [x] State carry across chunks.
- [x] Episode reset -- and made PER-ROW, which the original plan didn't
  call for but real motion training (asynchronous per-sequence resets in a
  batch) needs.
- [x] Peak-memory measurement.
- [ ] Latency measurement against a transformer baseline -- NOT done; do
  this as part of Milestone B/C below, since it needs an actual ARDY
  -shaped comparison to mean anything.
- [x] Synthetic baseline training.

Also implemented, going beyond what this section originally scoped for
"after one step is verified" (validated in order: oracle memory shown to
help BEFORE learned QKV work started, per Section 12's stage order):

- [x] Learned State-QKV retrieval (Hit@1 ~99.8% on the research repo's
  validated synthetic task).
- [x] Long-horizon memory correction -- FIFO vs. Participative-eviction
  comparison validated (2026-07-25): FIFO shows a hard capacity cliff,
  Participative eliminates it out to 3.6x raw capacity.

Still NOT implemented (unchanged from the original plan -- these are
genuinely ARDY-side, not Mamba-side, work):

- [ ] ARDY root/body replacement.
- [ ] Cross-window Event Memory wired into ARDY's actual generation loop
  (the Mamba-side primitives for this -- `write_mask`, `update_memory`,
  per-row `reset_mask` -- exist and are unit-tested, but nothing calls them
  from ARDY yet).

## 8. First ARDY Implementation Mode

First version: **stateless denoiser-block replacement**.

Keep the ARDY windowing and diffusion loop unchanged. Replace:

```python
self.root_model = TransformerEncoderBlock(...)
self.body_model = TransformerEncoderBlock(...)
```

with:

```python
self.root_model = ArdyMambaBlock(...)
self.body_model = ArdyMambaBlock(...)
```

The first Mamba block should:

- [ ] Build the same prefix tokens as `TransformerEncoderBlock`.
- [ ] Concatenate prefix + motion tokens.
- [ ] Run a SISO Mamba-3 sequence mixer.
- [ ] Return only the motion-token outputs.
- [ ] Project to `output_dim`.

This version answers:

```text
Can Mamba replace ARDY's transformer denoiser block without breaking generation?
Does it reduce latency or memory for the same ARDY window?
Can it train on the same denoising target?
```

It does not yet solve long-horizon memory.

## 9. After ARDY Drop-In Block Is Verified

After `ArdyMambaBlock` exists and passes shape / denoising / one-window tests,
the ARDY-facing interface is:

```python
output = block(
    x,
    x_pad_mask,
    text_feat,
    text_feat_pad_mask,
    timesteps,
    first_heading_angle,
    token_index,
)
```

This can implement after verification:

- [ ] Drop-in replacement of `root_model`.
- [ ] Drop-in replacement of `body_model`.
- [ ] Single denoising step.
- [ ] One autoregressive generation window.
- [ ] Short ARDY generation sample.
- [ ] Latency / memory comparison against transformer block.

It still cannot implement:

- [ ] Cross-window memory.
- [ ] Event retrieval.
- [ ] State-based long-horizon correction.

## 10. Final Work After Memory Is Done

After StateProjector + Event Memory + State-QKV pass the required gates, the
final interface is:

```python
output, next_state, next_memory, aux = block(
    x,
    x_pad_mask,
    text_feat,
    text_feat_pad_mask,
    timesteps,
    first_heading_angle,
    token_index,
    state=prev_state,
    memory=prev_memory,
    reset_mask=reset_mask,
    update_memory=commit_clean_window,
    return_aux=True,
)
```

This can implement after final memory work:

- [ ] Mamba state carry across autoregressive windows.
- [ ] Event Memory retrieval across windows.
- [ ] Memory read during diffusion.
- [ ] Memory commit after final clean/generated window.
- [ ] Long-horizon drift / recall correction.
- [ ] Diagnostics for retrieval and memory usage.

## 11. Long-Horizon Upgrade

ARDY currently manages long generation by cropping history to a trained window.
In `scripts/generate.py`, `--history_frames` caps visible history; in
`Ardy.__call__`, each autoregressive step calls `_generate_window(...)` with
`history_start_frame` and `history_end_frame`.

This is practical for real-time interaction, but it means the denoiser only
sees a bounded history window.

The Mamba upgrade should add cross-window memory:

```python
outputs, next_state, next_memory, aux = model.forward_chunk_with_memory(
    x=chunk,
    state=prev_state,
    memory=prev_memory,
    reset_mask=reset_mask,
    update_memory=True,
)
```

For ARDY integration, the memory should be updated only after a clean generated
window is accepted, not at every noisy denoising intermediate. The diffusion
loop should treat memory as read-only during denoising, then commit memory from
the final clean/generated chunk.

This preserves the rule:

```text
inner chunk = diffusion
inter chunk = autoregressive memory update
```

## 12. State and Memory Semantics

Mamba recurrent state is continuous compressed history. Event Memory is
explicit addressable history.

Stage order:

```text
SISO Mamba block
  -> Mamba3ChunkWrapper.forward_chunk
  -> StateProjector(ssm_state)
  -> oracle Event Memory
  -> learned State-QKV
  -> compressed/sink memory after oracle memory works
```

Do not train learned QKV or compressed memory before the oracle-memory gate
shows a clear gain on long-horizon synthetic tasks.

## 13. Current Mamba-Side Progress

**Corrected 2026-07-25.** Current status:

```text
done: Mamba-3 state/chunk API audit
done: SISO kernel smoke and basic forward/backward memory checks
done: Mamba3ChunkWrapper.forward_chunk (Phase-0 tests passing)
done: StateProjector
done: Event Memory (FIFO + Participative, unified, validated:
      FIFO hard capacity cliff vs. Participative flat accuracy)
done: learned State-QKV retrieval (~99.8% Hit@1 on validated synthetic task)
not yet: ArdyMambaBlock
not yet: ARDY root/body replacement
not yet: cross-window Event Memory integration INTO ARDY
```

Code lives in `src/state_memory_operator/` in this repo, importable as
`from state_memory_operator import StateTransitionMemoryOperator,
StateMemoryOperatorConfig` (see `examples/quickstart.py`,
`tests/test_smoke.py`).

What this means for ARDY collaborators:

- The interface contract is ready AND the Mamba-side operator it describes
  is now real, tested code -- not just a plan.
- The first ARDY replacement target is clear (Section 3).
- The drop-in `ArdyMambaBlock` is NOT implemented yet -- this is the actual
  next step, not a formality; nothing in this repo has touched ARDY.
- ARDY-side data, diffusion, motion representation, and export code should
  remain unchanged until `ArdyMambaBlock` passes its own drop-in tests
  (Milestone B below).
- Content/event-label design for motion (Section 9) is unstarted --
  everything validated so far used discrete synthetic-vocabulary tasks, not
  continuous motion latents. Don't assume the memory mechanism transfers to
  motion content without that design work.

## 14. ARDY Integration Milestones

### Milestone A: Read-only ARDY baseline

Run ARDY generation without modifying ARDY:

```text
scripts/generate.py
scripts/interactive_demo/generation.py
```

Checklist:

- [ ] Run unmodified ARDY generation.
- [ ] Record model name.
- [ ] Record duration.
- [ ] Record diffusion steps.
- [ ] Record history frames.
- [ ] Record generation time.
- [ ] Record peak GPU memory.
- [ ] Record output `.npz` / `.csv` path.

### Milestone B: Drop-in Mamba denoiser block

Implement `ArdyMambaBlock` with the same forward signature as
`TransformerEncoderBlock`.

Validation checklist:

- [ ] Shape parity: root output and body output match ARDY expected shapes.
- [ ] Mask parity: valid/padded tokens do not crash.
- [ ] Single denoising step runs.
- [ ] One autoregressive window runs.
- [ ] Short generation sample writes `.npz`.
- [ ] G1 sample writes MuJoCo `.csv`.

### Milestone C: Basic training

Train the Mamba-denoiser version on the same ARDY-style target.

Checklist:

- [ ] Record loss curve.
- [ ] Record peak GPU memory.
- [ ] Record tokens/sec or windows/sec.
- [ ] Record short qualitative generation.
- [ ] Compare to transformer denoiser under same settings.

### Milestone D: Cross-window state/memory

Add state handoff and Event Memory across AR windows.

Validation checklist:

- [ ] Reset prevents episode leakage.
- [ ] State carry changes later windows.
- [ ] Oracle memory improves long-horizon synthetic recall.
- [ ] Motion drift reduces at longer horizons.

## 15. Contribution Mapping

Contribution 1, real time:

```text
ARDY transformer denoiser block
  -> Mamba-3 SISO block
  -> lower attention memory growth
  -> latency / peak-memory benchmark
```

Contribution 2, long horizon:

```text
ARDY cropped history window
  -> Mamba recurrent state across chunks
  -> Event Memory / State-QKV
  -> later compressed memory + sink-style memory
```

Contribution 3, humanoid evaluation:

```text
Keep ARDY G1 motion representation and MuJoCo export.
Compare baseline ARDY vs Mamba-denoiser ARDY vs memory-augmented Mamba.
Later connect generated G1 qpos/motion to G1 simulation or real-world stack.
```

## 16. Non-Goals For First Server Handoff

Do not start with:

- [ ] Rewriting ARDY diffusion sampler.
- [ ] Rewriting the autoencoder.
- [ ] Rewriting `motion_rep`.
- [ ] Rewriting MuJoCo export.
- [ ] Training MIMO/CUTE path.
- [ ] Full HumanML3D/KITML/BABEL benchmark.
- [ ] Real robot deployment.

The clean first target is the root/body denoiser block replacement.
