# SPDX-License-Identifier: Apache-2.0
"""Vendored MTP (Multi-Token Prediction) speculative decoding for Qwen3.5/3.6.

This package vendors mlx-lm PR #990 — "Native MTP speculative decoding
(Qwen3.5/3.6 reference implementation)" — at upstream commit
``50c164fb82bf50d89ec4eb30fc5dc33820b4540f``. The PR is still ``OPEN``
upstream as of vendoring, and our floor mlx-lm 0.31.3 does not yet
include any of these changes (verified by inspecting
``mlx_lm.models.qwen3_5.TextModel.__call__`` — it lacks
``return_hidden``, ``n_confirmed``, ``mtp_forward``, ``MTPModule``,
``mtp_num_hidden_layers``).

Public surface
--------------

The public API the CLI / scheduler hooks into is intentionally tiny:

* :class:`MTPAcceptCounter` — process-local counter of attempts /
  accepts / tokens-saved. Surfaced via
  ``rapid_mlx_spec_decode_*_total`` and
  ``rapid_mlx_spec_decode_accept_ratio`` in
  :mod:`vllm_mlx.routes.metrics`.
* :func:`detect_mtp_eligibility` — returns
  :class:`MTPEligibility` for a given HF-style ``config.json`` dict.
  Only Qwen3.5 / Qwen3.6 with ``mtp_num_hidden_layers >= 1`` are
  eligible. Other architectures (Qwen3.0/3.1, Llama, Mistral) return
  ``MTPEligibility.NONE``.
* :func:`mtp_generate_step` — the chain MTP generation loop vendored
  from mlx-lm PR #990 (``mlx_lm/generate.py::mtp_generate_step``).
  Requires the loaded model to expose ``mtp_forward(hidden,
  next_token_ids, mtp_cache)``, ``return_hidden=True`` in
  ``__call__``, ``n_confirmed`` in ``__call__``, and
  ``make_mtp_cache()``. The model-side injection helper that adds
  those methods to an in-memory ``mlx_lm.models.qwen3_5.TextModel``
  lives at :func:`vllm_mlx.spec_decode.mtp.qwen3_5_inject.inject_mtp_support`.
* :func:`patch_arrays_cache_rollback_state` — adds a ``rollback_state``
  slot to mlx-lm's ``ArraysCache``. Required by both the model-side
  GatedDeltaNet ``_process_chunk`` split (saves the conv/SSM snapshot
  at ``n_confirmed``) and the generator-side ``_rollback_draft``
  (restores the snapshot on draft rejection). Idempotent — calling it
  twice is a no-op.

Architecture note (chain vs tree)
---------------------------------

Qwen3.5 / Qwen3.6 ship with ``mtp_num_hidden_layers: 1`` in
``config.json``, so the MTP head produces **one** draft token per step
(predicts token ``t+2`` from the hidden state at ``t`` and the embedding
of token ``t+1``). The R15-P1 task description called this a "4-step
tree" with positions like ``[10, 11, 11, 12, 12, 13, 13]``, but the
upstream Qwen3.5/3.6 release only ships a single MTP head. Vendoring
the upstream PR as-is gives us chain-style MTP (1 draft / verify
backbone pass, up to 2 tokens emitted). The tree-decode infrastructure
in :mod:`vllm_mlx.positioned_kv_cache` is still used: PR #990's
acceptance/rejection loop calls ``cache.trim(1)`` (attention layers) and
``cache.rollback_state`` (GatedDeltaNet layers), both of which the
:func:`positioned_update_and_fetch` helper from R15 #301 will
transparently delegate to on the monotonic-decode fast path. A future
``--spec-decode mtp-tree`` could plug into the same scaffolding once
upstream ships ``mtp_num_hidden_layers >= 2`` checkpoints.
"""

from __future__ import annotations

from .accept_counter import MTPAcceptCounter, get_global_counter
from .cache_patch import patch_arrays_cache_rollback_state
from .detect import MTPEligibility, detect_mtp_eligibility

__all__ = [
    "MTPAcceptCounter",
    "MTPEligibility",
    "detect_mtp_eligibility",
    "get_global_counter",
    "patch_arrays_cache_rollback_state",
]


# Re-export of the heavy generator + model-side injection is deferred so
# this top-level import remains cheap for the CLI surface (which only
# needs the detection + eligibility helpers at parse time). Callers that
# actually run the loop should import them directly:
#
#     from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step
#     from vllm_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support
