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

Local ARDY reference inspected:

```text
/home/plote/mambamotion/basecode/ardy
```

Key ARDY files:

```text
ardy/model/ardy_model.py
ardy/model/auto_latent_twostage_denoiser.py
ardy/model/backbone.py
scripts/generate.py
scripts/interactive_demo/generation.py
ardy/exports/mujoco.py
```

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

The Mamba block should expose the ARDY transformer block interface:

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

## 6. First Implementation Mode

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

1. build the same prefix tokens as `TransformerEncoderBlock`;
2. concatenate prefix + motion tokens;
3. run a SISO Mamba-3 sequence mixer;
4. return only the motion-token outputs;
5. project to `output_dim`.

This version answers:

```text
Can Mamba replace ARDY's transformer denoiser block without breaking generation?
Does it reduce latency or memory for the same ARDY window?
Can it train on the same denoising target?
```

It does not yet solve long-horizon memory.

## 7. Long-Horizon Upgrade

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

## 8. State and Memory Semantics

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

## 9. ARDY Integration Milestones

### Milestone A: Read-only ARDY baseline

Run ARDY generation without modifying ARDY:

```text
scripts/generate.py
scripts/interactive_demo/generation.py
```

Record:

```text
model name
duration
diffusion steps
history_frames
generation time
peak GPU memory
output .npz / .csv path
```

### Milestone B: Drop-in Mamba denoiser block

Implement `ArdyMambaBlock` with the same forward signature as
`TransformerEncoderBlock`.

Validation:

```text
shape parity: root output and body output match ARDY expected shapes
mask parity: valid/padded tokens do not crash
single denoising step runs
one autoregressive window runs
short generate.py sample writes .npz
G1 sample writes MuJoCo .csv
```

### Milestone C: Basic training

Train the Mamba-denoiser version on the same ARDY-style target.

Record:

```text
loss curve
peak GPU memory
tokens/sec or windows/sec
short qualitative generation
comparison to transformer denoiser under same settings
```

### Milestone D: Cross-window state/memory

Add state handoff and Event Memory across AR windows.

Validation:

```text
reset prevents episode leakage
state carry changes later windows
oracle memory improves long-horizon synthetic recall
motion drift reduces at longer horizons
```

## 10. Contribution Mapping

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

## 11. Non-Goals For First Server Handoff

Do not start with:

```text
rewriting ARDY diffusion sampler
rewriting the autoencoder
rewriting motion_rep
rewriting MuJoCo export
training MIMO/CUTE path
full HumanML3D/KITML/BABEL benchmark
real robot deployment
```

The clean first target is the root/body denoiser block replacement.
