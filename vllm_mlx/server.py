# SPDX-License-Identifier: Apache-2.0
"""Deprecated module shim: run ``python -m rapid_mlx.server`` instead.

Keeps ``python -m vllm_mlx.server`` (the pre-rename invocation still
found in older scripts and docs) working for one deprecation window.
Importing :mod:`vllm_mlx` already emits the package-level
``DeprecationWarning``; this module adds the module-specific pointer.
"""

import warnings

from rapid_mlx.server import main

warnings.warn(
    "python -m vllm_mlx.server is deprecated; use "
    "'python -m rapid_mlx.server' instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
