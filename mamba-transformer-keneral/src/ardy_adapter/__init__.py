"""ARDY-specific adapters bridging `state_memory_operator` to ARDY's
denoiser blocks. Depends on `ardy` being importable -- kept as a separate
package so `state_memory_operator` itself stays ARDY-agnostic."""

from .ardy_mamba_block import ArdyMambaBlock

__all__ = ["ArdyMambaBlock"]
