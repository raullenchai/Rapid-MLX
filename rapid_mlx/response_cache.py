# SPDX-License-Identifier: Apache-2.0
"""Opt-in prompt-deterministic RESPONSE CACHE (exact-match short-circuit).

Completely repeated *deterministic* requests short-circuit the whole GPU
pipeline and return the previously-computed completion verbatim — zero
decode. This is distinct from the KV / prefix cache (which reuses prefix
*state* to speed prefill): the response cache returns the *entire stored
completion*, doing no engine work at all.

Opt-in, default OFF. Enabled by ``--response-cache-entries N`` (N > 0);
``N == 0`` (the default) makes the cache fully inert — no store, no
lookup, counters stay at zero, zero behavior change. Mirrors the
``--hybrid-cache-entries`` opt-in knob.

Determinism gate
----------------
Only *greedy* requests are cached and short-circuited. A request is
eligible when ``temperature == 0`` OR ``top_k == 1`` (see
:func:`is_deterministic`). If ``temperature > 0`` and the caller did not
pin a ``seed``, they expect sampling variety, so returning a stale
identical response would be WRONG — those requests are skipped (a miss,
which is always correct). A pinned ``seed`` with ``temperature > 0`` is
NOT treated as deterministic in this MVP: MLX's batched sampler advances
shared PRNG state whose per-request split depends on scheduling
neighbours, so "same seed" does not guarantee "same tokens" across runs
here. Conservative by design — widen later with evidence, not hope.

Correctness note
----------------
At ``temperature == 0`` MLX greedy decode is NOT bit-stable across runs:
batched SDPA numerics diverge between ``q_len == 1`` and ``q_len >= 2``
under quantized weights, so a *fresh recompute* of the same prompt may
differ by a token. This does NOT break the response cache. The cache
RETURNS A STORED VALID COMPLETION — it never recomputes. The contract is
exactly OpenAI's prompt-caching contract: *"an identical deterministic
request MAY return a previously-computed valid response."* A fresh
recompute possibly differing by a token is therefore irrelevant to
correctness — the cache serves a valid prior response by design.

Concurrency
-----------
The server is async and multi-request. The LRU store and the counters
are guarded by a single ``threading.Lock`` around an ``OrderedDict``
(mirrors ``memory_cache.py``). No background thread is introduced.
Because ``get``/``put`` hold the lock only for O(1) dict ops (never
across ``await``), lock contention is negligible.

Metrics
-------
Two process-local counters, read by ``routes/metrics.py`` via
:func:`snapshot`:

* ``rapid_mlx_response_cache_hits_total``
* ``rapid_mlx_response_cache_misses_total``

Counters never decrease for the process lifetime; they reset to zero on
restart (the normal Prometheus convention). Tests use
:func:`reset_response_cache_for_tests`.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any


class ResponseCache:
    """Bounded LRU cache of fully-assembled deterministic responses.

    Capacity ``0`` means disabled: :meth:`get` always misses (without
    ticking the miss counter — an inert cache records nothing) and
    :meth:`put` is a no-op. Capacity ``N > 0`` retains at most ``N``
    entries, evicting the least-recently-USED entry on overflow (a HIT
    refreshes recency, so eviction order is true LRU, not FIFO).

    Stored values are opaque to this class — the chat route stores the
    serialized response body plus the small amount of metadata it needs
    to rebuild a fresh :class:`Response` (with a new ``id`` / ``created``)
    on a hit.

    Epoch versioning
    ----------------
    A stored completion is only valid for the exact model artifact that
    produced it, but the cache key spans only the model *id* (not the
    weights), so a hot reload of changed weights under the same id must
    invalidate the whole cache. :meth:`reconfigure` — the load-time entry
    point — bumps ``_epoch`` while it clears the store, all under ONE lock
    acquisition. Every :meth:`get`/:meth:`put` carries the epoch the
    caller observed at request start: an operation whose epoch no longer
    matches the current one is rejected. This closes two holes that
    clear-on-load alone could not:

    * split-lock TOCTOU — capacity-set and clear were two separate lock
      acquisitions; now they are one atomic critical section.
    * in-flight-put poisoning — an old-model generation still running when
      the reload happens would otherwise ``put`` its stale completion into
      the freshly-cleared cache; its epoch is now stale, so the ``put`` is
      dropped.
    """

    def __init__(self, capacity: int = 0) -> None:
        if capacity < 0:
            raise ValueError("ResponseCache capacity must be >= 0")
        self._capacity = int(capacity)
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        # Counters are process-local and monotonic (see module docstring).
        self._hits = 0
        self._misses = 0
        # Bumped on every load-time reconfigure; gates get/put against the
        # epoch the caller observed at request start (see class docstring).
        self._epoch = 0

    @property
    def enabled(self) -> bool:
        """True when the cache is active (capacity > 0)."""
        return self._capacity > 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def current_epoch(self) -> int:
        """Return the current epoch (read under the lock).

        The chat route captures this ONCE at request start and threads the
        same value into its later :meth:`get` and :meth:`put` so a request
        that began under the previous model cannot read or poison the
        cache after a reload bumps the epoch.
        """
        with self._lock:
            return self._epoch

    def configure(self, capacity: int) -> None:
        """(Re)set the LRU capacity WITHOUT bumping the epoch or clearing.

        Kept for callers that only want a capacity change (e.g. a runtime
        resize) with the existing entries intact. The load path uses
        :meth:`reconfigure` instead, which additionally clears + bumps the
        epoch for cross-model invalidation. Shrinking capacity evicts the
        coldest entries so ``len <= capacity`` holds immediately. Counters
        are intentionally NOT reset here — they are process-lifetime
        monotonic. ``configure(0)`` disables the cache and clears the
        store.
        """
        if capacity < 0:
            raise ValueError("ResponseCache capacity must be >= 0")
        with self._lock:
            self._capacity = int(capacity)
            if self._capacity == 0:
                self._store.clear()
                return
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def reconfigure(self, capacity: int) -> None:
        """Atomic load-time (re)configuration: set capacity, clear, bump epoch.

        This is the model-load invalidation point. All three effects
        happen under a SINGLE lock acquisition so there is no window in
        which the new capacity is live but the old entries have not yet
        been dropped (the split-lock TOCTOU that ``configure`` + ``clear``
        had). Bumping ``_epoch`` additionally invalidates any in-flight
        request that observed the previous epoch — its later ``put`` is
        dropped, so an old-model generation completing after the reload
        cannot poison the freshly-cleared cache.
        """
        if capacity < 0:
            raise ValueError("ResponseCache capacity must be >= 0")
        with self._lock:
            self._capacity = int(capacity)
            self._store.clear()
            self._epoch += 1

    def get(self, key: str, epoch: int) -> Any | None:
        """Return the stored value for ``key`` and mark it MRU, or None.

        A hit ticks the hit counter AND moves the entry to the most-
        recently-used end (this is what makes eviction LRU rather than
        FIFO). A miss ticks the miss counter. When the cache is disabled
        (capacity 0) this is a no-op that returns None and ticks NOTHING
        — an inert cache must have zero observable effect, including on
        metrics.

        ``epoch`` is the value the caller observed at request start via
        :meth:`current_epoch`. If it no longer matches the current epoch
        (a reload happened mid-request) the lookup is rejected: it returns
        None and ticks NOTHING — a stale-epoch request is not a real
        lookup outcome, so it must not consume a new-model entry nor
        distort the hit/miss counters.
        """
        with self._lock:
            # Disabled/stale checks live INSIDE the lock so a concurrent
            # reconfigure() cannot flip capacity or epoch between the check
            # and the store access while we still tick a miss.
            if self._capacity == 0 or epoch != self._epoch:
                return None
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any, epoch: int) -> None:
        """Insert ``value`` under ``key`` as the most-recently-used entry.

        No-op when disabled (capacity 0). On overflow, evicts the
        least-recently-used entry (``last=False``). Re-inserting an
        existing key refreshes both its value and its recency.

        ``epoch`` is the value the caller observed at request start via
        :meth:`current_epoch`. If it no longer matches the current epoch
        the store is dropped: an old-model generation that finishes after
        a reload must not poison the freshly-cleared cache.
        """
        with self._lock:
            # Disabled/stale checks live INSIDE the lock (see get()).
            if self._capacity == 0 or epoch != self._epoch:
                return
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def clear(self) -> None:
        """Drop all cached entries (does not touch counters or epoch)."""
        with self._lock:
            self._store.clear()

    def snapshot(self) -> dict[str, int]:
        """Consistent snapshot of the counters for ``/metrics``."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._store),
                "capacity": self._capacity,
            }


# ── Determinism gate ──────────────────────────────────────────────────


def is_deterministic(sampling_kwargs: dict[str, Any]) -> bool:
    """Return True when a request is greedy enough to cache/short-circuit.

    Safe MVP rule (see module docstring): eligible when the effective
    sampling is greedy — ``temperature == 0`` OR ``top_k == 1``. Any
    other shape (``temperature > 0`` without a definitively deterministic
    decode, or missing/None temperature) is treated as non-deterministic
    and skipped. Missing keys default to "not greedy" so we never cache
    an ambiguous request.

    ``sampling_kwargs`` is the resolved kwargs dict the engine actually
    consumes (``chat_kwargs`` on the chat route) — i.e. the values AFTER
    the request → CLI → alias → generation_config cascade — so the gate
    sees exactly what will drive decoding, not the raw request fields.
    """
    temperature = sampling_kwargs.get("temperature")
    top_k = sampling_kwargs.get("top_k")
    # top_k == 1 forces a single candidate → argmax → greedy regardless
    # of temperature. temperature == 0 is greedy by definition.
    if top_k == 1:
        return True
    if temperature is not None and temperature == 0:
        return True
    return False


# ── Cache key ─────────────────────────────────────────────────────────

#: Sentinel returned by :func:`make_cache_key` when the request carries a
#: value that cannot be canonicalized to a STABLE string. Any such request
#: is treated as UNCACHEABLE: the caller must skip both lookup and store
#: (see the chat route). This is deliberately NOT ``None`` so it can never
#: be confused with a legitimately absent key and is identity-checkable
#: with ``is``.
UNCACHEABLE = object()


class _UncanonicalizableError(Exception):
    """Raised inside :func:`_json_default` for a value we cannot map to a
    deterministic representation. Caught by :func:`make_cache_key`, which
    converts it into the :data:`UNCACHEABLE` sentinel."""


def _json_default(obj: Any) -> Any:
    """Canonicalizer for non-JSON-native key components — SUPPORTED types
    only.

    ``json.dumps`` calls this only for values it can't serialize natively.
    We canonicalize exactly two extra shapes:

    * Pydantic-like models (tools, response_format) via ``model_dump`` —
      a stable, field-ordered dict.
    * ``set`` / ``frozenset`` — sorted so element order can't perturb the
      key.

    For ANYTHING ELSE we raise :class:`_UncanonicalizableError` rather than
    falling back to ``repr(obj)``. A ``repr`` fallback embeds the object's
    memory address (``<Foo object at 0x…>``) for most types, so two
    otherwise-identical requests carrying fresh equivalent objects would
    produce DIFFERENT keys — silent cache MISSES that defeat the whole
    point of an exact-match deterministic cache. Raising here lets
    :func:`make_cache_key` mark the request uncacheable instead of emitting
    an unstable key (or a wrong hit). A ``model_dump`` that itself throws
    is likewise treated as uncanonicalizable — we never guess.
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception as exc:
            raise _UncanonicalizableError(
                f"model_dump() failed on {type(obj).__name__}"
            ) from exc
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    raise _UncanonicalizableError(
        f"unsupported cache-key component of type {type(obj).__name__}"
    )


def make_cache_key(
    *,
    model: str,
    prompt: Any,
    sampling_kwargs: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Any:
    """Build a stable sha256 over EVERY output-affecting input.

    Returns the 64-char hex digest, or the :data:`UNCACHEABLE` sentinel
    when any component cannot be canonicalized to a STABLE string (see
    :func:`_json_default`). The caller MUST check ``is UNCACHEABLE`` and,
    on a match, skip both lookup and store — never key an unstable request.

    A missing field would be a correctness bug (a wrong response served),
    so the key spans:

    * ``model`` — the resolved model id.
    * ``prompt`` — the render-determining generation input. The caller
      passes the SAME value generation consumes (the ``messages`` list on
      the chat route). Keying on the raw messages rather than a re-rendered
      prompt string avoids a SECOND, independent chat-template render whose
      output could drift from the engine's own internal render (e.g. if the
      template were time/state-dependent) — which would store a completion
      under a key the engine never generated from. The render-affecting
      knobs (``tools`` / ``enable_thinking`` / ``forced_assistant_prefix``)
      travel in ``sampling_kwargs`` below, so they are part of the key too.
    * ``sampling_kwargs`` — the resolved kwargs dict passed to the engine
      (``temperature``, ``top_p``, ``top_k``, ``min_p``, ``seed``,
      ``max_tokens``, ``stop``, ``presence_penalty``,
      ``frequency_penalty``, ``repetition_penalty``, ``enable_thinking``,
      ``tools``, ``forced_assistant_prefix``, …). Because this is the
      SAME dict the engine consumes, no sampling param can silently drop
      out of the key.
    * ``extra`` — output-shape-affecting request fields that do NOT flow
      through ``sampling_kwargs`` but change the response body: e.g.
      ``response_format`` (JSON coercion), ``logprobs`` / ``top_logprobs``
      (adds the logprobs field). A change in any of these yields a
      different key → a MISS → a correct recompute.

    ``sort_keys=True`` + a compact separator make the JSON canonical
    (dict-order-independent). ``default=_json_default`` canonicalizes
    pydantic / set components (and raises for unsupported types →
    UNCACHEABLE). ``ensure_ascii=False`` keeps CJK / emoji from bloating
    the pre-hash string (the hash is over UTF-8 bytes either way).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "sampling": sampling_kwargs,
        "extra": extra or {},
    }
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_json_default,
        )
    except _UncanonicalizableError:
        # A component can't be stably canonicalized — do NOT emit an
        # unstable key. The request is uncacheable; the caller bypasses
        # both store and lookup. This never raises out of make_cache_key.
        return UNCACHEABLE
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Module singleton ──────────────────────────────────────────────────

_response_cache = ResponseCache(capacity=0)


def get_response_cache() -> ResponseCache:
    """Return the process-wide response-cache singleton.

    Read by BOTH the chat route (store / lookup) and the metrics route
    (counter snapshot). Starts disabled (capacity 0); the serve boot path
    calls :func:`configure_response_cache` with the resolved
    ``--response-cache-entries`` value.
    """
    return _response_cache


def configure_response_cache(capacity: int) -> None:
    """(Re)configure the singleton at model load — atomic invalidation.

    Called once per ``load_model`` (server boot AND every hot reload). A
    stored completion is only valid for the exact model artifact that
    produced it, but the cache key spans just the model *id* + inputs — not
    the underlying weights. Reloading changed weights under the same id
    would otherwise serve completions from the PREVIOUS model. Delegates to
    :meth:`ResponseCache.reconfigure`, which — under one lock acquisition —
    sets the new capacity, clears the store, and bumps the epoch so any
    in-flight old-model request cannot read or poison the new-model cache.
    """
    _response_cache.reconfigure(capacity)


def force_disable_response_cache() -> None:
    """Rebind the singleton to a fresh, disabled cache — fail-closed reset.

    Used as the fail-safe on a model-load reconfigure failure. It does NOT
    call any method on the existing instance (which may be in the wedged
    state that caused the failure): it replaces the module singleton with a
    brand-new ``ResponseCache(capacity=0)``. A fresh object at capacity 0 is
    inert by construction — no store, no lookup — so the previous model's
    cache cannot survive the failure and serve stale cross-model output.
    """
    global _response_cache
    _response_cache = ResponseCache(capacity=0)


def reset_response_cache_for_tests() -> None:
    """Reset the singleton to a fresh, disabled cache (tests only)."""
    global _response_cache
    _response_cache = ResponseCache(capacity=0)
