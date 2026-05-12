# SPDX-License-Identifier: Apache-2.0
"""DFlash speculative-decoding integration (issue #264).

DFlash is a block-diffusion drafter (z-lab) integrated into mlx-vlm's
``generate_step``. Rapid-MLX wires it into ``BatchedEngine`` for B=1
generation; B>1 transparently falls back to AR until phase-2 batched
support lands.

Eligibility (enforced by ``eligibility.check``):
  - alias has ``supports_dflash=True`` in ``aliases.json``
  - alias is NOT MoE (PoC: 0.76-0.82× regression on Qwen3.6-35B-A3B)
  - main-model precision ≥ 8-bit (PoC: 4-bit accept rate collapses)
  - drafter HF path is reachable (no gating block)

Public API:
  - ``DFlashUnavailable``: raised by ``check`` when an alias fails any gate
  - ``check(profile)``: returns ``None`` on success, raises with message
  - ``load_runtime(drafter_repo)``: lazy import of mlx-vlm drafter loader
"""

from .eligibility import DFlashUnavailable, check
from .runtime import load_runtime

__all__ = ["DFlashUnavailable", "check", "load_runtime"]
