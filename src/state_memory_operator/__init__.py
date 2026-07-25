"""State-Transition Memory Operator: clean, ready-to-use package.

    from state_memory_operator import (
        StateMemoryOperatorConfig,
        StateTransitionMemoryOperator,
        OperatorState,
        EventMemory,
        Mamba3ChunkWrapper,
        Mamba3State,
    )

See `examples/quickstart.py` for end-to-end usage and
`docs/mamba_integration_contract.md` for what's validated vs. not yet.
"""

from .config import StateMemoryOperatorConfig
from .event_memory import EventMemory
from .mamba3_chunk_wrapper import Mamba3ChunkWrapper, Mamba3State, apply_reset_mask
from .operator import OperatorState, StateTransitionMemoryOperator
from .state_projector import StateProjector, StateTransition, project_transition
from .state_qkv import ContentQueryEncoder, ContentValueEncoder, StateQKVEncoder

__all__ = [
    "StateMemoryOperatorConfig",
    "StateTransitionMemoryOperator",
    "OperatorState",
    "EventMemory",
    "Mamba3ChunkWrapper",
    "Mamba3State",
    "apply_reset_mask",
    "StateProjector",
    "StateTransition",
    "project_transition",
    "StateQKVEncoder",
    "ContentValueEncoder",
    "ContentQueryEncoder",
]
