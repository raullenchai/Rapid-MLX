# SPDX-License-Identifier: Apache-2.0
"""Metal kernel sources and Python bindings (R15 Phase 4).

Metal sources live in ``.metal`` files alongside the Python binding
modules. The Python module compiles the source via
``mx.fast.metal_kernel`` at first call and caches the compiled handle
at module scope. On compile failure each binding logs a warning and
returns ``None`` so callers can fall back to the pure-MLX reference
path without an unhandled exception.
"""

from __future__ import annotations
