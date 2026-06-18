# SPDX-License-Identifier: Apache-2.0
"""
Rapid-MLX Doctor — environment-health check.

``rapid-mlx doctor`` is a fast (≤ 5 s) self-diagnostic that answers one
question: *is my install/env broken?*  It probes hardware, Python, packages,
HuggingFace cache, network, shell integration, and optional tooling.  It
never loads a model, boots a server, or runs benchmarks.

Model-validation tiers that used to live here (``smoke / check / full /
benchmark``) moved to ``rapid-mlx bench --tier ...`` as of v0.7.22.

Entry point: ``rapid-mlx doctor [--verbose]``.

Exit codes:
  0 — everything ok or only warnings
  1 — one or more ✗ issues
"""

from .env_health import Check, CheckStatus, Report, Section, run_all

# Deprecated compatibility re-exports. The internal consumer
# (``vllm_mlx.bench.tiers.*``) was removed; these are kept solely so
# external PyPI users with ``from vllm_mlx.doctor import DoctorRunner``
# (or any of the others) don't break across the upgrade. Prefer
# importing from ``vllm_mlx.doctor.runner`` directly. May be dropped in
# a future major-version bump.
from .runner import (  # noqa: F401  # public surface, deprecated
    CheckResult,
    DoctorRunner,
    Status,
    TierResult,
    python_executable,
    run_subprocess,
)

__all__ = [
    # New env-health surface (the public face of `rapid-mlx doctor`).
    "Check",
    "CheckStatus",
    "Report",
    "Section",
    "run_all",
    # Deprecated — kept for back-compat with external imports. See note
    # above the ``.runner`` re-export block.
    "CheckResult",
    "DoctorRunner",
    "Status",
    "TierResult",
    "python_executable",
    "run_subprocess",
]
