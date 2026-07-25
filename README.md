# Mamba Transformer Kernel

This repository is the clean integration surface for the Mamba-side operator
work used by downstream motion / simulation / server-side experiments.

It is intentionally not a research log repo. Planning notes, daily logs, and
experiment memory live elsewhere. This repo should contain only code,
interfaces, tests, and short documentation needed by collaborators who connect
the Mamba operator to training or deployment systems.

## Current Scope

The Mamba side owns, and now IMPLEMENTS (`src/state_memory_operator/`):

- Mamba-3 SISO chunk operator wrapper.
- Explicit recurrent state handoff across chunks, with per-row episode reset.
- State projection from Mamba recurrent state.
- Fixed-capacity Event Memory (FIFO and Participative-eviction backends,
  unified in one class) with per-row async reset and automatic compaction.
- State-based Q/K retrieval + content-based V (validated: state indexes
  reliably but does not carry fine-grained content -- see
  `state_qkv.py`'s docstring).
- Operator diagnostics (`aux` dict): retrieval entropy/weights/sims,
  injection norm, memory usage norm, peak GPU memory.

Downstream systems own:

- Motion / sequence data loading.
- Conditions, targets, and task losses.
- Episode boundaries and reset masks.
- Simulator or robot runtime.
- Benchmark and evaluation scripts.

## Start Here

Install (editable) and run the tests:

```bash
pip install -e .
pytest tests/test_smoke.py -v
```

Run the quickstart example:

```bash
python examples/quickstart.py
```

Minimal usage:

```python
from mamba_ssm.modules.mamba3 import Mamba3
from state_memory_operator import StateMemoryOperatorConfig, StateTransitionMemoryOperator

mamba3 = Mamba3(d_model=128, d_state=64, expand=2, headdim=64, ngroups=1,
                 is_mimo=False, chunk_size=16, layer_idx=0,
                 device="cuda", dtype=torch.bfloat16)
config = StateMemoryOperatorConfig(d_model=128, memory_backend="participative")
operator = StateTransitionMemoryOperator(mamba3, config).cuda()

op_state = operator.init_operator_state(batch_size, device="cuda")
memory = operator.init_memory(batch_size, device="cuda")

out, op_state, memory, aux = operator.forward_chunk_with_memory(
    x, op_state, memory, read_content=content, write_content=content,
)
```

Read the integration contract and ARDY replacement plan for what's validated,
what isn't, and where this connects to ARDY:

```text
docs/mamba_integration_contract.md
docs/ardy_mamba_replacement_plan.md
```

Progress is tracked with `[ ]` / `[x]` checklists in those docs -- keep them
honest: update a box only when its stated acceptance criterion has actually
been verified, and correct it immediately if you find it's gone stale (this
happened once already, 2026-07-20 -> 2026-07-25, and cost real confusion).

**Current stage: Level 2 (chunk wrapper + state projector + Event Memory +
learned State-QKV all implemented and validated) on synthetic
associative-recall tasks only.** `ArdyMambaBlock` and all ARDY-side
integration work are NOT started -- see `ardy_mamba_replacement_plan.md`
Section 6 for the actual next step. MIMO/CUTE remain later-stage work.
