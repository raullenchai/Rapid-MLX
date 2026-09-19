# SPDX-License-Identifier: Apache-2.0
"""Vendored cache primitives from mlx-vlm, pinned at 0.7.1.

The native serialized MLLM lane (``MLLMBatchGenerator``) leans on mlx-vlm
cache primitives. Vendoring this first slice brings those code paths under
this repo's review and tests while the Rapid-owned compatibility seam remains
under the repository's mypy budget. The upstream dependency stays pinned for
everything else it provides (model loading, processors, templating, vision
preprocessing, and the speculative-decode runtime).

Design note: ``docs/engineering/design/2026-09-18-vendor-mllm-primitives.md``.
Upstream: https://github.com/Blaizzy/mlx-vlm/tree/v0.7.1

Provenance contract: every file here is a copy of the upstream
``mlx-vlm==0.7.1`` source except *import redirects* and *in-source
``VENDOR-DEVIATION`` hunks*, each documented in the file's own body and in
the inventory below. A future behavior change must re-qualify against the
pinned tag: ``diff`` against upstream may show only the documented hunks.

Inventory (digests are of the UPSTREAM file, not the vendored bytes — the
vendored copy differs by exactly the deviations listed):

- ``cache.py`` — identical to ``mlx_vlm/models/cache.py`` @ v0.7.1
  (upstream sha256
  ``b736c299bc576f4bdf8d6edf3e1ac6fa3019ca888a2f3cba115d4252f84d5b29``)
  **except three in-source ``VENDOR-DEVIATION(upstream-bugfix)`` hunks**,
  each reproducible against the pinned upstream:
  1. ``BatchRotatingKVCache.merge`` called the zero-arg in-place
     ``_temporal_order`` with an argument, raising TypeError on every
     merge with content (latent upstream; mlx-lm 0.31.3 carries the same
     defect). It also selected the allocation tail of an unrotated,
     preallocated decode cache, merging unused zeros instead of its live
     prefix. Fixed to call the zero-arg form and copy the first ``length``
     temporal entries.
  2. ``BatchPoolingCache.make_mask``'s scalar-offset branch added
     ``offset`` twice to the absolute query positions, admitting pooled
     tokens earlier than the causal contract allows. Fixed to match the
     ``mx.array`` branch semantics.
  3. ``BatchRotatingKVCache.meta_state`` serializes ``rotated`` via
     ``str()`` but the setter read it with ``bool()``, so the string
     ``"False"`` restored as True and corrupted every unrotated cache
     round-tripped through ``from_state`` (the lane's memory-cache restore
     path; mlx-lm 0.31.3 carries the same defect). Fixed to parse the
     serialized spelling.
  A diff against the pinned tag must show only these hunks (plus this
  note in the inventory).

``kv_quant.py`` is NOT vendored here (step 2a): its ``from_legacy()`` lazily
imports ``.turboquant`` (7k lines, itself importing ``.models.cache``), so it
moves to step 2b with the APC family, where the turboquant dependency gets an
explicit home (vendored slice or a documented redirect to the pinned
upstream).
"""
