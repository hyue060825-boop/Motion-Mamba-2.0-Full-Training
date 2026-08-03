# Motion Mamba 2.0

Autoregressive diffusion Mamba for real-time humanoid motion control.

This is the clean collaboration and paper-code repository. Keep this repository
focused on reusable code, documented interfaces, and reproducible scripts. Do
not commit local checkpoints, generated videos, W&B runs, caches, or private
server paths.

## Current Layout

```text
mamba-transformer-keneral/
```

Standalone Mamba-side code copied into this repository as source, not as a git
submodule. It contains:

- Mamba-3 sequence/state wrappers.
- State-memory operator experiments.
- ARDY-compatible `ArdyMambaBlock`.
- Distillation, quantitative evaluation, and ARDY viser helper scripts.

## Planned Layout

```text
ardy/                         # main ARDY basecode, to be cleaned and added
mamba-transformer-keneral/    # Mamba operator and ARDY adapter code
docs/                         # paper-code notes, setup, repo map
```

The older direct ARDY Transformer-to-Mamba modification from the server should
be inspected before import. If it is useful, add it as a cleaned code directory
or merge the reusable pieces into `mamba-transformer-keneral`; do not import
large experiment outputs.
