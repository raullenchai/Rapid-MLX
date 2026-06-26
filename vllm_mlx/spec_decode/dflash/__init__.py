# SPDX-License-Identifier: Apache-2.0
"""DFlash block-diffusion speculative decoding for Qwen3.5 / Qwen3.6.

This package wires the **block-diffusion drafter** family (arxiv 2410.04097
"Block Diffusion-based Speculative Decoding") into rapid-mlx's standardized
``--spec-decode`` interface. It sits next to :mod:`vllm_mlx.spec_decode.mtp`
as the second model-side speculative-decode backend the CLI knows about.

Background
----------

DFlash differs from MTP in two ways:

1. **Drafter is a SEPARATE small model** (~0.5 B params) rather than a
   head built into the target — the same target weights can be paired
   with different drafter checkpoints, and the drafter loads through
   its own HF path.
2. **Drafter emits a BLOCK of tokens per forward** (default 16) using
   block diffusion / parallel denoising, not a single token. The
   verifier confirms the whole block in 1 target forward pass with
   arbitrary ``position_ids``. The longest accepted prefix wins; a
   rejection at position ``k`` collapses to a single corrective token
   at position ``k`` (lossless to the no-spec-decode path).

Public surface
--------------

* :class:`DFlashAcceptCounter` — process-local attempts/accepts/
  tokens-saved counter. Surfaced via
  ``rapid_mlx_spec_decode_dflash_*_total`` in
  :mod:`vllm_mlx.routes.metrics`.
* :func:`detect_dflash_eligibility` — returns
  :class:`DFlashEligibility` for a (model_type, alias_name) pair.
* :func:`get_dflash_drafter_path` — resolves the drafter HF path for
  an alias from the side-registry (kept OUT of ``aliases.json`` because
  the closed-key schema reserves :attr:`AliasProfile.dflash_draft_model`
  for the existing mlx-vlm DFlash bridge — see
  :mod:`vllm_mlx.speculative.dflash`).

Architecture note (vs the mlx-vlm bridge)
-----------------------------------------

There are **two** DFlash code paths in rapid-mlx and they target different
runtimes:

* :mod:`vllm_mlx.speculative.dflash` — the original PoC bridge that
  calls into ``mlx-vlm 0.5.0+``'s built-in drafter loader. Lives next to
  the SuffixDecoding / prompt-lookup speculative paths. Driven by the
  ``--enable-dflash`` flag. **This is the only production-working DFlash
  path in 0.9.x** (paper-bench-verified ≥3.5× on 27B int4+bf16-KV).
* :mod:`vllm_mlx.spec_decode.dflash` (THIS module) — the R15-P1 #313
  vendored scaffolding for the standardized
  ``--spec-decode {none,mtp,dflash}`` interface. The accept counter, the
  drafter registry, and the eligibility detector are stable and used by
  metrics + the CLI banner. The drafter/verifier/generator integration
  with BatchedEngine is **DEFERRED to 0.10** — the original adapter
  (``drafter.py:275``) called ``DFlashDraftModel.draft_block(prefix,
  position)`` against an mlx-vlm 0.5.0 signature that requires
  ``(last_bonus, hidden, cache, block_size, sampler, token_dtype)``.

For 0.9, ``--spec-decode dflash`` on the CLI redirects to
``--enable-dflash`` (see :mod:`vllm_mlx.cli`) so operators still get a
working DFlash path through the unified flag surface.

Both paths share the lossless contract: byte-identical output to the
no-spec-decode path (rejection always emits the verify model's pred at
the divergent position, not the drafter's).
"""

from __future__ import annotations

from .accept_counter import (
    DFlashAcceptCounter,
    DFlashAcceptSnapshot,
    get_global_counter,
    reset_global_counter_for_tests,
)
from .detect import DFlashEligibility, detect_dflash_eligibility
from .drafter_registry import (
    clear_drafter_registry_for_tests,
    get_dflash_drafter_path,
    register_dflash_drafter,
)

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DFlashAcceptCounter",
    "DFlashAcceptSnapshot",
    "DFlashEligibility",
    "clear_drafter_registry_for_tests",
    "detect_dflash_eligibility",
    "get_dflash_drafter_path",
    "get_global_counter",
    "register_dflash_drafter",
    "reset_global_counter_for_tests",
]

# Default block size — the value the paper bench uses (16). Surfaced
# as a module constant so the CLI banner, the metrics gauge, and the
# verifier can all read from a single SSOT. Operators that want a
# smaller "prototype block size 4" can override at SchedulerConfig
# construction; the verifier validates positive int >= 1.
DEFAULT_BLOCK_SIZE = 16
