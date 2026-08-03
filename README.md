# Motion Mamba 2.0

Autoregressive diffusion Mamba for real-time humanoid motion control.

This is the clean collaboration and paper-code repository. Keep this repository
focused on reusable code, documented interfaces, and reproducible scripts. Do
not commit local checkpoints, generated videos, W&B runs, caches, or private
server paths.

## Current Layout

```text
ardy/
```

`ardy/`

Main ARDY basecode used as the current humanoid motion generation/control
baseline and visualization stack.

## Branch Layout

Use branches to keep the paper-code repository readable:

- `main`: clean ARDY basecode.
- `mamba-transformer-keneral`: standalone Mamba-side operator code.
- `ardymamba`: planned branch for the cleaned experimental ARDY/Mamba fork
  imported from the server.

## Planned Layout

```text
ardy/                         # main ARDY basecode
docs/                         # paper-code notes, setup, repo map
```

The older direct ARDY Transformer-to-Mamba modification from the server should
be inspected before import. If it is useful, put the cleaned version on the
`ardymamba` branch. Do not import large experiment outputs.
