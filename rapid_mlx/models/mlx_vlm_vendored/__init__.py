# SPDX-License-Identifier: Apache-2.0
"""Vendored APC engine and cache primitives from mlx-vlm, pinned at 0.7.1.

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

- ``apc_storage.py`` — byte-identical to ``mlx_vlm/apc_storage.py`` @ v0.7.1
  (upstream sha256
  ``e58b3a5aa5fa875764194712e38a7752006aeb31db5abe132057c667759e7010``).
- ``_stream_cleanup.py`` — byte-identical to ``mlx_vlm/_stream_cleanup.py``
  @ v0.7.1 (upstream sha256
  ``00bf5797510f088cfe4cea7798dce0f2902af99a956888748ff0ddc11de21c5d``).
- ``vision_cache.py`` — identical to ``mlx_vlm/vision_cache.py`` @ v0.7.1
  (upstream sha256
  ``5db081a4ef9ee07bb1102c6a87b5fb4a0895821bb2c05d19e110ba6f700e4561``)
  **except in-source ``VENDOR-DEVIATION(upstream-bugfix)`` hunks** covering
  ``_make_key`` (plus the ``import os`` it needs), reproducible against the
  pinned upstream:
  1. Upstream's key derivation was ambiguous and could serve one image
     set's cached features to another: list sources joined the recursively
     derived keys with a bare ``"|"`` (``["a|b", "c"]`` vs
     ``["a", "b|c"]`` collide), and even length-prefixing collides across
     nesting boundaries (``["1:a", "b"]`` vs ``[["a"], "b"]``). Fixed with
     a count-delimited, type-tagged encoding (``s``=str, ``l``=list with a
     child count, ``p``=content hash) that is injective. Upstream also
     documented Path
     sources but only accepted ``str`` (a ``Path`` fell into the
     ``obj:{id}`` fallback); ``os.PathLike`` is now normalized via
     ``os.fsdecode`` — ``os.fspath`` alone can return ``bytes`` for
     byte-valued paths, which fell into the image-content hash branch and
     collided with a raw image payload equal to the path bytes.
  2. Unsupported source types fell back to ``obj:{id(...)}``; Python may
     hand that id to an unrelated object after collection — a silent
     stale-feature hit. Fixed to raise ``TypeError`` (the str/PathLike and
     bytes-like branches cover every real caller; the lane passes
     pre-hashed string keys).
  3. ``tobytes()`` sources were hashed on their raw bytes alone, so two
     images with identical byte sequences but different mode or size (a
     2x1 ``"L"`` vs a 1x1 ``"RGB"``) collided and one image's cached
     features were served for the other. Fixed to hash stable
     type/mode/size metadata together with the raw bytes (bytes-like
     sources carry no such metadata and stay content-addressed). Palette
     images (``"P"``) hash to palette indices in ``tobytes()``, so
     same-sized images with identical indices but different palettes
     rendered different content yet still collided; the effective
     ``getpalette()`` bytes and palette transparency metadata are folded
     into the digest as well.
  4. ``put()`` with ``max_size <= 0`` evaluated
     ``len(self._cache) >= self.max_size`` against an empty mapping and
     called ``popitem()`` on it, raising KeyError. Fixed so zero (or
     negative) ``max_size`` disables storage instead of crashing.
  The lane's ``VisionFeatureCache`` import resolves here.
- ``kv_quant.py`` — identical to ``mlx_vlm/kv_quant.py`` @ v0.7.1 (upstream
  sha256
  ``2936878096435dd2540e7a029b986259e5b7101c972a5be7168495a58e7fbfa3``)
  **except one import redirect**: ``from_legacy()``'s lazy
  ``from .turboquant import ...`` resolves the pinned upstream
  ``mlx_vlm.turboquant`` instead. TurboQuant itself (7k lines with its own
  ``.models.cache`` dependency) is NOT vendored — it can never fit a
  reviewable diff, so it stays a pinned-dependency redirect for the whole
  transition.
- ``apc_coordinator.py`` — identical to ``mlx_vlm/apc_coordinator.py`` @
  v0.7.1 (upstream sha256
  ``8c3939a15b8bee2c4ac8f1144a1048c3463d6cf935537d63eb21403b6f63773b``)
  **except documented redirects**: its module-level
  ``from .apc_adapters import ...`` resolves the vendored sibling; its lazy
  ``from .apc import ...`` engine calls and the ``fresh_cache`` fallback
  ``make_prompt_cache`` remain on upstream mlx-vlm until producers flip
  (step 3). Every site carries a ``VENDOR-DEVIATION`` comment. Additionally,
  return-value locals
  are explicitly annotated where the redirected engine calls type-resolve to
  ``Any`` (the repo's mypy ``no-any-return`` discipline; no behavior
  change).
- ``apc_adapters.py`` — identical to ``mlx_vlm/apc_adapters.py`` @ v0.7.1
  (upstream sha256
  ``9ce11d3c420d983281faf229cfc06ae4a8dce28469dd1536a05d26e4e980c101``)
  **except documented ``VENDOR-DEVIATION`` hunks**, all part of one
  transition mechanism (type-namespace duality — see the design note):
  1. Module-level ``_cache_namespaces`` / ``_cache_namespace_of`` helpers:
     every type table, capability registration, contract probe and
     constructor covers the vendored AND upstream cache namespaces, so the
     adapters behave identically whichever namespace produced a cache. A
     stripped installation without mlx-vlm yields one namespace (no None
     entries); ownership is decided by base-class identity so third-party
     subclasses resolve to their own namespace.
  2. Constructors route through ``_cache_namespace_of`` so cloned/merged
     results keep the producer's cache types (upstream-typed inputs yield
     upstream-typed results — byte-identical behavior today; correct
     typing once producers emit vendored caches in step 3). Bare tuples
     are namespace-agnostic: ``clone_cache_entry`` clones a tuple before
     the owning-namespace lookup (each element resolves its own), and
     ``merge_cache_entries`` derives the container namespace from the
     first tuple element — a stripped install cannot resolve a namespace
     for a tuple itself, which silently dropped composite caches that
     ``apc_exact_eligible`` declares supported. The lazy
     ``_apc_type_tables`` / ``_clone_rules`` builders also publish their
     globals only after both namespaces are processed — as one immutable
     assignment for the two type tables — so a concurrent first caller
     can never observe a partially built (or half-published) table.
  3. Redirects: ``_apc_array_helpers``' lazy ``.apc`` import and the
     ``build_prefix_cache_plan`` fallback ``make_prompt_cache`` remain on
     upstream mlx-vlm until producers flip (step 3); the turboquant
     registration resolves the
     pinned upstream ``mlx_vlm.turboquant`` (not vendored — see above).
  The lane's ``clone_cache_entry`` / ``Capability`` / ``resolve_capability``
  imports now resolve here; the four test modules that stub
  ``clone_cache_entry`` were re-pointed at this module in the same commit.

- ``apc.py`` — identical to ``mlx_vlm/apc.py`` @ v0.7.1 (upstream sha256
  ``5b2b940852f11f34f7b4daf627bc31fc701f8abffc72d40189bc3e5ac57f878c``)
  except documented deviations, each carrying a ``# VENDOR-DEVIATION``
  sentinel:

  - 14 one-line import redirects: the top-level relative imports bind the
    vendored siblings; the lazy ``.models*`` (11) and ``.turboquant`` (3)
    sites resolve upstream. (2b-3 folded the four cache-typed sites — the
    ``_dense_checkpoint*`` pair and both exact-snapshot ladders — into the
    dual-namespace helpers below.)
  - dual-namespace recognition helpers (``_cache_ns_*``, 2b-3) so the
    engine's exact-type tables, snapshot/restore constructors, and
    ``_resolve_checkpoint_class`` accept both cache namespaces; exact
    snapshots record ``_ns="v"`` for vendored-typed entries (absent key =
    upstream, so pre-2b-3 shards stay readable). The step-3 mechanical
    revert drops the helpers.
  - three ``upstream-bugfix`` hunks (2b-3), each repro-tested in
    ``tests/test_mlx_vlm_vendored_apc_engine.py``: ``_rebuild_index``
    disk-bytes accounting (unreadable shards inflated ``_disk_bytes``),
    ``_finish_write`` in-flight entry ownership (partially overlapping
    writers could erase another writer's entry), and
    ``_save_layer_major_shard`` temp-file cleanup on write failure.

- ``inputs.py`` — verbatim from ``mlx_vlm/utils.py`` @ v0.7.1 lines
  1714-2543 (``load_image`` .. ``prepare_inputs``, an unbroken region;
  region sha256
  ``ac610b0e2c157de878b17ec9f5ebaa8bf2c75000e44c09d84b5c17dbaf7c7b5f``),
  except the import block: exactly the names the region references
  (ruff F821 closure), with ``mlx_vlm.models.base`` (``
  BaseImageProcessor``) still resolving upstream and the logger pinned to
  the upstream ``mlx_vlm.utils`` name (``mlx_vlm.apc`` precedent), plus
  four documented hunks (all repro-tested in
  ``tests/test_mlx_vlm_vendored_inputs.py``):

  - ``VENDOR-DEVIATION(dual-namespace)`` in ``processor_video_sampling``:
    a processor hook may return the upstream ``VideoSampling`` dataclass
    while the vendored class is in effect, so matching-shape objects
    normalize by their known fields instead of class identity.
  - ``VENDOR-DEVIATION(upstream-bugfix)`` in ``load_video``: the cv2
    capture handle is released via try/finally (upstream leaked it on
    failed opens, frame-sampler errors, index validation, and read
    failures).
  - ``VENDOR-DEVIATION(upstream-bugfix)`` in ``prepare_inputs``: bytes
    video paths are ``os.fsdecode``-d (upstream str()-ed them into
    ``"b'/tmp/a.mp4'"``-style garbage filenames).
  - ``VENDOR-DEVIATION(upstream-bugfix)`` in ``load_audio``: the streamed
    URL response is closed via ``with`` (upstream left it open).

  The
  lane's ``prepare_inputs`` call sites (``multimodal_processor.py``,
  ``mllm_batch_generator.py``) resolve this module; per-function upstream
  parity is probed in ``tests/test_mlx_vlm_vendored_inputs.py`` (all
  functions except ``processor_video_sampling``, ``load_video``,
  ``load_audio``, and ``prepare_inputs``, which carry the hunks above).
  A diff against the pinned tag must show only the header/import block
  and the documented hunks.
"""
