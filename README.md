# Motion Mamba 2.0: ARDY-Mamba Experimental Branch

This branch is the ARDY-Mamba experimental fork reconstructed from local logs,
memory records, and the locally validated Mamba operator code after the server
became unavailable.

It is not a byte-for-byte snapshot of the server checkout. It is a cleaned,
self-contained ARDY fork containing the pieces that were validated locally and
recorded in the project memory:

- slim ARDY basecode;
- `ardy.model.ardy_mamba_block.ArdyMambaBlock`, a drop-in root/body denoiser
  block that replaces ARDY's Transformer sequence mixer with Mamba-3;
- `ardy.model.mamba3_chunk_wrapper.Mamba3ChunkWrapper`, the minimal Mamba-3
  SISO chunk wrapper needed by `ArdyMambaBlock`;
- `ardy/scripts/mamba/` distillation, quantitative evaluation, prompt-bank,
  and viser-generation helpers.

The default ARDY path remains transformer-based. To instantiate Mamba blocks
inside `AutoLatentTwostageDenoiser`, set:

```bash
export ARDY_SEQUENCE_MODEL=mamba
```

The training/eval scripts usually load the teacher ARDY model first and swap in
`ArdyMambaBlock` explicitly, matching the verified Milestone C workflow.

Do not commit checkpoints, generated videos, W&B runs, caches, or private
server paths.

Related branches:

- `main`: clean ARDY basecode.
- `mamba-transformer-keneral`: standalone Mamba-side operator code.
- `ardymamba`: this experimental ARDY/Mamba fork.
