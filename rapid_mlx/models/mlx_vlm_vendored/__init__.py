# SPDX-License-Identifier: Apache-2.0
"""Vendored cache/vision primitives from mlx-vlm, pinned at 0.7.1.

The native serialized MLLM lane (``MLLMBatchGenerator``) leans on a set of
mlx-vlm cache and vision-preprocessing primitives. Vendoring them brings
those code paths under this repo's review, tests, and mypy budget while the
upstream dependency stays pinned for everything else it provides (model
loading, processors, templating, the speculative-decode runtime).

Design note: ``docs/engineering/design/2026-09-18-vendor-mllm-primitives.md``.
Upstream: https://github.com/Blaizzy/mlx-vlm/tree/v0.7.1

Provenance contract: every file here is a **verbatim** copy of the upstream
``mlx-vlm==0.7.1`` source except *import redirects*, each documented in the
file's own header. A future behavior change must re-qualify against the
pinned tag: ``diff`` against upstream may show only the documented redirect
hunks.

Inventory (sha256 over the vendored bytes):

- ``cache.py`` — byte-identical to ``mlx_vlm/models/cache.py`` @ v0.7.1
  (``b736c299bc576f4bdf8d6edf3e1ac6fa3019ca888a2f3cba115d4252f84d5b29``).
  Redirects callers that imported
  ``from mlx_vlm.models.cache import ArraysCache, KVCache`` — the module is
  fully self-contained (mlx + stdlib only), so no redirects inside.

``kv_quant.py`` is NOT vendored here (step 2a): its ``from_legacy()`` lazily
imports ``.turboquant`` (7k lines, itself importing ``.models.cache``), so it
moves to step 2b with the APC family, where the turboquant dependency gets an
explicit home (vendored slice or a documented redirect to the pinned
upstream).
"""
