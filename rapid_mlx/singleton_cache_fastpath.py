"""Singleton KV-cache fast path for B=1 decode (bench-tuning, #1861 family).

Why
---
mlx-lm's ``BatchGenerator`` always converts per-request caches into batched
forms (``KVCache.merge([c])`` -> ``BatchKVCache``) even when the batch holds a
single row. The batched representation costs real decode throughput at B=1:
per-row ``left_padding``/``offset`` bookkeeping and — decisively — a batched
rank-4 array mask instead of the plain causal mask, which keeps
``mx.fast.scaled_dot_product_attention`` off its native causal fast path.
The 2026-08-11 cross-runtime bench measured the gap on qwen3.5-4b:
``stream_generate`` (singleton cache) 172.5 tok/s @128 / 146.6 @16k vs
``BatchGenerator`` 167.7 / 138.2 — and oMLX, which keeps singleton caches
while the row count is one, holds 175 / 149 through a full server.

What
----
``install_singleton_cache_fastpath()`` patches ``mlx_lm.generate`` so that:

* ``_merge_caches([single])`` passes the per-layer caches through unchanged
  (no batched conversion) when they qualify;
* ``_extend_cache`` promotes singleton layers to their batched forms via
  ``cls.merge([self])`` the moment a second row joins, then extends as usual
  (returning a NEW list — promotion replaces layer objects);
* layers admitted to the pass-through gain an INSTANCE-scoped batch-API
  surface — the minimal set ``GenerationBatch`` touches while the row
  count remains one: ``filter`` (pass-through for one row), ``extract``
  (independent copy of row 0), ``extend`` (promote-first guard). The
  surface is bound per-object, never onto the classes: patching
  ``KVCache`` itself would flip ``hasattr(cache, "extend")``-style
  capability gates for EVERY plain cache in the process (the MLLM batch
  path guards exactly like that and would silently skip cache merging —
  codex MAJOR on #1874).

Only exact ``KVCache`` / ``RotatingKVCache`` layers (and layers that already
carry native batch surfaces, e.g. hybrid ``ArraysCache``) pass through.
Subclasses are rejected even when they LOOK batch-capable: once instances
of the base classes can carry the singleton surface, a duck-type probe
cannot tell a native batch cache from an inherited shim, and
``_QuantizableKVCache`` (#1197/#1862) must keep its own ``merge`` — the
hook that installs the quantized batch cache; short-circuiting it would
silently serve bf16 under ``--kv-cache-dtype int4``.

``extract`` mirrors the batched-path contract (``BatchKVCache.extract``):
an independent, offset-trimmed ``mx.contiguous`` copy. The scheduler
extracts caches from LIVE rows too (prompt-cache save callback, disk-KV
checkpoints) while the row keeps decoding into the original buffers, so
a shallow array-sharing clone is an aliasing hazard, not a payload. The
zero-row ``filter`` reset is defensive only: mlx-lm's drain path clears
the layer list without calling ``filter([])``.

Admission is copy-on-admit for data-carrying layers: stock batched
merging always copied per-request caches into fresh batch buffers, and
prefix caches RELY on that — supersequence/LCP loans share the stored
entry's arrays with only the offset rewound. Keeping the loaned object
live would let in-place decode writes corrupt the stored prefix (the
l1-smoke llama3-3b golden regression). Fresh (empty) layers admit
unchanged, so the steady-state B=1 win is untouched.
"""

from __future__ import annotations

import copy
import logging
import threading
import types
from typing import Any

from . import _mlx_compat as _mlx_compat

_mlx_compat.install()

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import KVCache, RotatingKVCache  # noqa: E402

logger = logging.getLogger(__name__)

# Exact types only — subclasses (e.g. the quantized _QuantizableKVCache)
# carry semantics their merge() must apply and never pass through.
_SINGLETON_EXACT_TYPES = (KVCache, RotatingKVCache)


def _is_singleton_passthrough_layer(cache_obj: Any) -> bool:
    sub = getattr(cache_obj, "caches", None)
    if isinstance(sub, (list, tuple)):
        # The composite itself must carry the batch surface (CacheList
        # does natively) — child qualification alone would admit a
        # wrapper whose own filter/extract/extend calls AttributeError
        # at batch time (codex r2 on #1874).
        if not (
            hasattr(cache_obj, "filter")
            and hasattr(cache_obj, "extract")
            and hasattr(cache_obj, "extend")
        ):
            return False
        return all(_is_singleton_passthrough_layer(c) for c in sub)
    if type(cache_obj) in _SINGLETON_EXACT_TYPES:
        return True
    if isinstance(cache_obj, _SINGLETON_EXACT_TYPES):
        # Subclasses carry merge() semantics that must apply (quantized
        # batch cache install, #1197/#1862). They must never reach the
        # duck-type fallback below: an inherited singleton surface would
        # make them look natively batch-capable (pr_validate codex
        # BLOCKING on #1874).
        return False
    # Layers with native batch surfaces (hybrid ArraysCache/MambaCache)
    # already store B=1-leading state and need no conversion.
    return (
        hasattr(cache_obj, "filter")
        and hasattr(cache_obj, "extract")
        and hasattr(cache_obj, "extend")
        and type(cache_obj).__module__.startswith("mlx_lm.")
    )


def _promote_layer(cache_obj: Any) -> Any:
    """Convert a singleton layer to its batched form (idempotent)."""
    # Lazy import keeps this module usable without loading the optional replay
    # implementation until a cache actually presents its private marker.
    if getattr(cache_obj, "_rapid_shape_stable_kv", False):
        from .compiled_decode import ShapeStableKVCache

        if type(cache_obj) is not ShapeStableKVCache:
            return cache_obj
        # Compiled replay owns fixed-shape singleton slabs. Its conversion
        # drains any submitted graph before exposing an ordinary KVCache to
        # the stock B>1 merge transaction.
        eager = cache_obj.to_kv_cache()
        return type(eager).merge([eager])
    sub = getattr(cache_obj, "caches", None)
    if isinstance(sub, (list, tuple)):
        converted = [_promote_layer(c) for c in sub]
        if all(a is b for a, b in zip(sub, converted)):
            return cache_obj
        # Clone the wrapper and swap the children — reconstructing via
        # ``type(obj)(*converted)`` assumes a splat constructor and would
        # crash wrappers holding extra state (codex r3 on #1874).
        # copy.copy (not __new__ + __dict__) so wrappers using __slots__
        # or custom copy hooks clone correctly (codex r5).
        clone = copy.copy(cache_obj)
        clone.caches = type(sub)(converted) if isinstance(sub, tuple) else converted
        return clone
    if type(cache_obj) in _SINGLETON_EXACT_TYPES:
        return type(cache_obj).merge([cache_obj])
    return cache_obj


# -- minimal batch surface for singleton caches ------------------------------


def _singleton_filter(self, batch_indices):
    try:
        n = len(batch_indices)
    except TypeError:
        n = int(getattr(batch_indices, "shape", (0,))[0] or 0)
    if n == 0:
        self.keys = None
        self.values = None
        self.offset = 0
        if hasattr(self, "_idx"):
            self._idx = 0
        return
    if n == 1:
        # The kept index must actually be row 0 — silently accepting
        # filter([1]) would associate another request with this row's
        # KV state (codex r2 on #1874).
        kept = int(batch_indices[0])
        if kept != 0:
            raise IndexError(
                f"{type(self).__name__} singleton cache only has row 0; "
                f"filter kept row {kept}"
            )
        return
    raise NotImplementedError(
        f"{type(self).__name__}.filter is singleton pass-through only; "
        "promote to a batched cache before keeping multiple rows."
    )


_SURFACE_NAMES = ("filter", "extract", "extend")


def _copy_exact_layer(cache_obj: Any) -> Any:
    """Independent copy of an exact-type layer (no shared array objects).

    Shared body of ``extract`` (payload leaving the batch) and
    copy-on-admit (loaned state entering the batch). KVCache copies are
    offset-trimmed like ``BatchKVCache.extract``; RotatingKVCache is
    deep-copied whole so its circular layout (offset/_idx/rotation)
    continues decoding exactly as before. The bound singleton surface is
    never carried over.
    """
    if isinstance(cache_obj, RotatingKVCache):
        clone = copy.deepcopy(cache_obj)
        for name in _SURFACE_NAMES:
            clone.__dict__.pop(name, None)
        return clone
    clone = type(cache_obj).__new__(type(cache_obj))
    clone.__dict__.update(
        (k, v) for k, v in cache_obj.__dict__.items() if k not in _SURFACE_NAMES
    )
    if cache_obj.keys is None:
        return clone
    clone.keys = mx.contiguous(cache_obj.keys[..., : cache_obj.offset, :])
    clone.values = mx.contiguous(cache_obj.values[..., : cache_obj.offset, :])
    clone.offset = cache_obj.offset
    return clone


def _detach_layer(cache_obj: Any) -> Any:
    """Copy-on-admit: decouple a data-carrying layer from whoever loaned it.

    Stock batched merging always copied per-request caches into fresh
    batch buffers, and the prefix caches RELY on that: the memory-aware
    cache's supersequence/LCP loans (``_trim_cache_offset``) share the
    STORED entry's array objects with only the offset rewound. The
    singleton pass-through keeps the admitted object live, and
    ``KVCache.update_and_fetch`` writes in place — without this copy,
    decode scribbles over the stored prefix entry and every later hit
    on it returns corrupted state (l1-smoke llama3-3b golden regression
    on #1874: fluent-but-wrong answers). Empty layers (fresh requests)
    are untouched — the copy costs exactly what the retired batched
    merge paid, once per admission with preloaded state.
    """
    sub = getattr(cache_obj, "caches", None)
    if isinstance(sub, (list, tuple)):
        converted = [_detach_layer(c) for c in sub]
        if all(a is b for a, b in zip(sub, converted)):
            return cache_obj
        # copy.copy (not __new__ + __dict__) so wrappers using __slots__
        # or custom copy hooks clone correctly (codex r5 on #1874).
        clone = copy.copy(cache_obj)
        clone.caches = type(sub)(converted) if isinstance(sub, tuple) else converted
        return clone
    if type(cache_obj) in _SINGLETON_EXACT_TYPES:
        if getattr(cache_obj, "keys", None) is None:
            return cache_obj
        return _copy_exact_layer(cache_obj)
    # Native batch-surface layers (hybrid ArraysCache/MambaCache): every
    # CURRENT loan path deepcopies or refuses them (the non-trimmable
    # gate), but that is the prefix cache's contract, not this lane's —
    # deepcopy unconditionally so a future shallow loan cannot alias.
    # Fresh layers are near-empty; the copy is noise (codex r5).
    return copy.deepcopy(cache_obj)


def _singleton_extract(self, idx: int):
    """Independent copy of row 0, mirroring ``BatchKVCache.extract``.

    The scheduler extracts from LIVE rows (prompt-cache save, disk-KV
    checkpoints) while decode keeps writing into the original buffers,
    so the payload must not share array objects with the layer. The
    copy is offset-trimmed exactly like the batched extract; the bound
    singleton surface is NOT carried over — extract yields a plain
    cache, same as the batched path.
    """
    if int(idx) != 0:
        raise IndexError(f"{type(self).__name__} singleton cache only has row 0")
    return _copy_exact_layer(self)


def _singleton_extend(self, other):
    raise NotImplementedError(
        f"{type(self).__name__}.extend requires batched promotion first "
        "(singleton_cache_fastpath._promote_layer)"
    )


def _bind_singleton_surface(cache_obj: Any) -> None:
    """Attach the batch surface to THIS object only (never the class).

    Class-level patching would flip ``hasattr``-based capability gates
    for every plain cache in the process (MLLM batch merging guards that
    way). Skips names the class provides natively so a future mlx-lm
    that grows real batch methods on KVCache wins over the shims.
    """
    sub = getattr(cache_obj, "caches", None)
    if isinstance(sub, (list, tuple)):
        for c in sub:
            _bind_singleton_surface(c)
        return
    if type(cache_obj) not in _SINGLETON_EXACT_TYPES:
        return  # native-surface layers already expose the batch API
    for name, fn in (
        ("filter", _singleton_filter),
        ("extract", _singleton_extract),
        ("extend", _singleton_extend),
    ):
        if not hasattr(type(cache_obj), name) and name not in cache_obj.__dict__:
            setattr(cache_obj, name, types.MethodType(fn, cache_obj))


_install_lock = threading.Lock()


def install_singleton_cache_fastpath() -> bool:
    """Idempotent module-level install. Returns False when the running
    mlx-lm lacks the expected seams (fall back to stock batched merging)."""
    # ``import mlx_lm.generate as gen`` binds the package ATTRIBUTE, which
    # mlx_lm/__init__ shadows with the generate() function — importlib
    # returns the real module (same trap as the `from mlx_lm import
    # generate` gotcha).
    import importlib

    gen = importlib.import_module("mlx_lm.generate")

    # Serialize check + patch: concurrent generator creation (multi-model
    # residency loads on separate executor threads) must never capture a
    # half-installed seam as orig_merge (codex r6 on #1874).
    with _install_lock:
        return _install_locked(gen)


def _install_locked(gen) -> bool:
    if getattr(gen, "_rapid_singleton_cache_fastpath", False):
        return True
    orig_merge = getattr(gen, "_merge_caches", None)
    orig_extend = getattr(gen, "_extend_cache", None)
    if orig_merge is None or orig_extend is None:
        logger.warning(
            "[singleton-fastpath] mlx-lm lacks _merge_caches/_extend_cache "
            "seams; keeping stock batched merging"
        )
        return False

    def _merge_caches_singleton(caches):
        if len(caches) == 1 and all(
            _is_singleton_passthrough_layer(c) for c in caches[0]
        ):
            admitted = []
            for c in caches[0]:
                c = _detach_layer(c)
                _bind_singleton_surface(c)
                admitted.append(c)
            return admitted
        # Promote any singleton layers a previous pass left behind before
        # the stock merge sees them (its BatchKVCache.merge expects raw
        # KVCache inputs, which promoted layers no longer are).
        return orig_merge(caches)

    def _extend_cache_promote(cache_a, cache_b):
        if not cache_a:
            # cache_b normally arrives through the patched merge (which
            # detaches loaned state and binds the singleton surface), but
            # handle raw per-request layers handed straight to the extend
            # seam defensively (codex r3 on #1874) — including the
            # copy-on-admit, since a loaned layer is hazardous wherever
            # it enters the batch. All-or-nothing, mirroring
            # _merge_caches_singleton: a partially bound list would mix
            # singleton and raw layers (codex r6). Non-qualifying lists
            # return unchanged — exactly stock _extend_cache's behavior.
            if all(_is_singleton_passthrough_layer(c) for c in cache_b):
                admitted = []
                for c in cache_b:
                    c = _detach_layer(c)
                    _bind_singleton_surface(c)
                    admitted.append(c)
                return admitted
            return cache_b
        if not cache_b:
            return cache_a
        if len(cache_a) != len(cache_b):
            # Stock _extend_cache zips just like this and would silently
            # drop trailing layers; same-model joins always agree on
            # layer count, so a mismatch is corruption — fail loudly.
            raise ValueError(
                "cache layer count mismatch on extend: "
                f"{len(cache_a)} != {len(cache_b)}"
            )
        out = []
        for ca, cb in zip(cache_a, cache_b):
            pa, pb = _promote_layer(ca), _promote_layer(cb)
            pa.extend(pb)
            out.append(pa)
        return out

    gen._merge_caches = _merge_caches_singleton
    gen._extend_cache = _extend_cache_promote
    gen._rapid_singleton_cache_fastpath = True
    logger.info("[singleton-fastpath] installed (B=1 keeps singleton KV caches)")
    return True
