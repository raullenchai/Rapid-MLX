# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding bundle (R15-P1, task #302).

Suffix-decoding lives under ``rapid_mlx.speculative``. This package hosts the
**vendored** model-side speculative-decode paths that
are too tightly coupled to a specific architecture to live under the
neutral ``speculative`` namespace.

Currently bundled:

* :mod:`rapid_mlx.spec_decode.mtp` — Native Multi-Token Prediction
  speculative decoding for Qwen3.5 / Qwen3.6 family models, vendored
  from mlx-lm PR #990 (`Native MTP speculative decoding
  (Qwen3.5/3.6 reference implementation)
  <https://github.com/ml-explore/mlx-lm/pull/990>`_, head
  ``50c164fb82bf50d89ec4eb30fc5dc33820b4540f``). The PR is still
  ``OPEN`` upstream as of the vendoring date — we ship the code under
  this namespace rather than waiting on mlx-lm merge so the rest of
  R15-P1 (DFlash-MLX, DDTree-MLX) can build on top of the same accept-
  rate counter / lossless-contract scaffolding.
* :mod:`rapid_mlx.spec_decode.config` and
  :mod:`rapid_mlx.spec_decode.registry` — vLLM-style
  ``--speculative-config`` parsing plus the small method registry used
  to migrate DFlash / MTP / SuffixDecoding behind one frontend surface.

Each backend exposes a thin public API the CLI / scheduler can call after
parsing ``--speculative-config`` at boot. The model-side patches that make
a particular family eligible stay private to the sub-package.
"""

from __future__ import annotations

__all__ = ["config", "mtp", "registry"]
