# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for vllm-mlx inference.

Provides two engine implementations:
- SimpleEngine: Direct model calls for maximum single-user throughput
- BatchedEngine: Continuous batching for multiple concurrent users

Also re-exports core engine components for backwards compatibility.
"""

from .base import BaseEngine, GenerationOutput
from .simple import SimpleEngine
from .batched import BatchedEngine
from .hybrid import HybridEngine

# Re-export from parent engine.py for backwards compatibility
from ..engine_core import EngineCore, AsyncEngineCore, EngineConfig

__all__ = [
    "BaseEngine",
    "GenerationOutput",
    "SimpleEngine",
    "BatchedEngine",
    "HybridEngine",
    # Core engine components
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]
