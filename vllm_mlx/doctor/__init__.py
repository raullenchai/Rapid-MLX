# SPDX-License-Identifier: Apache-2.0
"""
Rapid-MLX Doctor — comprehensive regression harness.

Three tiers + a benchmark sweep:
  - smoke      (~2 min, no model)         — pytest, ruff, CLI sanity
  - check      (~15 min, qwen3.5-35b)     — server + perf + agents + baseline diff
  - full       (~2-3 hr, 2 models)        — check across qwen3.5-35b + qwen3.6-35b
  - benchmark  (overnight, all models)    — cross-model × cross-engine scorecard

Entry point: ``rapid-mlx doctor {smoke,check,full,benchmark}``

Exit codes:
  0  all checks pass
  1  performance regression detected (vs baseline)
  2  functional test failure
"""

from .runner import DoctorRunner, TierResult

__all__ = ["DoctorRunner", "TierResult"]
