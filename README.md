# Mamba Transformer Kernel

This repository is the clean integration surface for the Mamba-side operator
work used by downstream motion / simulation / server-side experiments.

It is intentionally not a research log repo. Planning notes, daily logs, and
experiment memory live elsewhere. This repo should contain only code,
interfaces, tests, and short documentation needed by collaborators who connect
the Mamba operator to training or deployment systems.

## Current Scope

The Mamba side owns:

- Mamba-3 SISO chunk operator wrapper.
- Explicit recurrent state handoff across chunks.
- State reset / detach policy.
- State projection from Mamba recurrent state.
- Event Memory and future State-QKV retrieval modules.
- Operator diagnostics such as retrieval metrics, injection norm, latency, and
  peak GPU memory.

Downstream systems own:

- Motion / sequence data loading.
- Conditions, targets, and task losses.
- Episode boundaries and reset masks.
- Simulator or robot runtime.
- Benchmark and evaluation scripts.

## Start Here

Read the integration contract:

```text
docs/mamba_integration_contract.md
```

Read the ARDY replacement plan:

```text
docs/ardy_mamba_replacement_plan.md
```

Progress is tracked with `[ ]` / `[x]` checklists in the docs. A planned API is
not considered usable until its checklist and validation criteria are marked
complete.

The first implementation target is:

```python
outputs, next_state = model.forward_chunk(
    x=chunk,
    state=prev_state,
    memory_condition=memory_condition,
    reset_mask=reset_mask,
)
```

This is the planned next interface, not a completed implementation yet.

Current stage is SISO Mamba-3 only. MIMO/CUTE and downstream motion training
are later-stage work.
