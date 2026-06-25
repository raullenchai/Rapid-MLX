# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding bundle (R15-P1, task #302).

The legacy n-gram/suffix lookup speculation lives under
``vllm_mlx.speculative`` (prompt-lookup, suffix-decoding, dflash). This
package hosts the **vendored** model-side speculative-decode paths that
are too tightly coupled to a specific architecture to live under the
neutral ``speculative`` namespace.

Currently bundled:

* :mod:`vllm_mlx.spec_decode.mtp` — Native Multi-Token Prediction
  speculative decoding for Qwen3.5 / Qwen3.6 family models, vendored
  from mlx-lm PR #990 (`Native MTP speculative decoding
  (Qwen3.5/3.6 reference implementation)
  <https://github.com/ml-explore/mlx-lm/pull/990>`_, head
  ``50c164fb82bf50d89ec4eb30fc5dc33820b4540f``). The PR is still
  ``OPEN`` upstream as of the vendoring date — we ship the code under
  this namespace rather than waiting on mlx-lm merge so the rest of
  R15-P1 (DFlash-MLX, DDTree-MLX) can build on top of the same accept-
  rate counter / lossless-contract scaffolding.

Each backend exposes a thin public API the CLI / scheduler can call to
choose between ``--spec-decode none|mtp`` at boot — the model-side
patches that make a particular family eligible stay private to the
sub-package.
"""

from __future__ import annotations

__all__ = ["mtp"]
