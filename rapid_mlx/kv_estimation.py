# SPDX-License-Identifier: Apache-2.0
"""Architecture-aware per-token KV-cache footprint estimation.

The D-METAL-CAP admission gate (``scheduler._enforce_metal_cap_at_admission``)
projects how much KV-cache a new request would allocate and rejects it before
the allocation pushes Metal active past the operator cap. That projection is
driven by a *per-token* byte figure. The historical figure was a UNIFORM
formula::

    per_token = 2 (K+V) * num_hidden_layers * num_key_value_heads
                  * head_dim * dtype_bytes

which treats EVERY layer as a full-growth attention layer. That is wrong for
the hybrid architectures this engine ships:

* **Sliding-window layers** (GPT-OSS ~50% sliding; Gemma-4 local layers) hold a
  fixed rotating window of KV. Their footprint is bounded by the window size —
  it does NOT grow per generated token.
* **KV-sharing / borrower layers** (Gemma-4 / Gemma-3n ``num_kv_shared_layers``)
  compute no K/V of their own; they reuse an earlier producer layer's cache and
  allocate ZERO KV.
* **Recurrent / linear-attention layers** (Qwen3.5/3.6 GatedDeltaNet, Mamba)
  carry a fixed-size recurrent state. Zero per-token growth.

Counting all of those as full-growth over-estimates the per-token figure by up
to ~2x for Gemma-4 / GPT-OSS / Qwen3.5, which makes the admission gate reject
requests that would actually fit — a spurious rejection on memory-constrained
Macs.

This module computes an accurate footprint by classifying each layer:

``per_token_growth_bytes``
    ``2 * dtype_bytes * Σ (kv_heads_L * head_dim_L)`` over ONLY the
    full-attention (unbounded per-token-growing) layers, honoring per-layer /
    global dims when the architecture is hybrid. Multiplied by
    ``T = prompt_tokens + max_tokens`` in the projection.

``sliding_slot_bytes`` + ``sliding_window``
    The sliding-window layers grow SUBLINEARLY: up to a rotating window, then
    flat. ``sliding_slot_bytes`` is the per-SLOT bytes summed over the
    window-bounded sliding layers (``2 * dtype_bytes * Σ_sliding (kv_heads_L *
    head_dim_L)``); the scheduler multiplies it by the per-request slot count
    ``rotating_cache_slots(sliding_window, T)`` — an over-count-safe UPPER BOUND
    of the real ``mlx_lm`` ``RotatingKVCache`` allocation that grows with ``T``
    up to ``ceil((window + keep) / step) * step`` (``step=256``, ``keep=4``) and
    then caps. Modelling this per request keeps it >= the true allocation at
    every ``T`` (never OOM) yet <= the full window (a short request is not
    over-charged the whole buffer). See :func:`rotating_cache_slots`.

``fixed_baseline_bytes``
    Charged once per sequence, independent of the token budget: the conservative
    fixed recurrent state of each sizeable recurrent (GatedDeltaNet) layer. Zero
    for a dense model and for a hybrid without recurrent layers.

Borrower (KV-sharing) layers contribute 0 to all terms. Sliding-window and
recurrent layers contribute 0 to per-token growth; a sliding layer's footprint
is charged per request via the slot term, a recurrent layer's via the fixed
baseline (see safety point 2).

Safety contract (this feeds a codex-hardened, over-estimate-safe admission
path — see ``scheduler._resolve_kv_bytes_per_token``):

1. **Over-estimate-safe fallback.** A recognized hybrid must expose a per-layer
   ``layer_types`` list whose length matches ``num_hidden_layers``. When that is
   absent, or anything is ambiguous, the estimator returns the caller's uniform
   figure unchanged (``per_token_growth_bytes == uniform_per_token_bytes`` and
   ``fixed_baseline_bytes == sliding_slot_bytes == sliding_window == 0``) so dense
   models (Llama/Qwen dense) and unknown/stub configs stay BYTE-IDENTICAL to the
   historical behavior. Only a recognized hybrid gets the smaller accurate
   number. ``num_kv_shared_layers`` is NOT sufficient on its own — the per-layer
   map must confirm the borrower split (see point 3).
2. **Never under-count the counted layers (over-count-safe throughout).** The
   load-bearing guarantee for the D-METAL-CAP cliff is that the projected
   footprint is >= the true footprint of every counted layer, so every estimate
   errs UP. Full-attention layers are charged exactly per token; a sliding layer
   with no readable window is charged as full unbounded growth; an unrecognized
   layer-type string is charged as full growth (exact-match allowlists, no
   substring guessing). A sliding-window layer WITH a readable window is charged
   per request as ``slot_bytes × rotating_cache_slots(window, T)``, where
   ``rotating_cache_slots`` rounds ``min(T, window) + keep`` UP to the buffer's
   step granularity — a provable upper bound of the real ``RotatingKVCache``
   buffer at every ``T`` (it grows in ``step`` blocks up to the window and keeps
   ``keep`` sinks), so a request is never charged less than its true sliding
   allocation, yet a SHORT request is not charged the whole window (which would
   over-count and spuriously reject it). Recurrent layers genuinely have zero
   per-token growth; a GatedDeltaNet layer's fixed state is sized from its real
   config fields (linear head dims + conv kernel) and charged at fp32/element to
   stay conservative. Every other recurrent family (Mamba/Mamba2, RWKV, generic)
   is NOT sized — its exact cache layout is not modelled here — and falls back to
   full per-token growth rather than a possibly wrong fixed state, so it is never
   under-reserved. The fixed baseline is charged once per sequence and never
   scales with generated tokens.
3. **KV-sharing zeroes only borrowers with a verified same-type producer.**
   ``num_kv_shared_layers`` declares that the LAST N layers borrow (Gemma-4 /
   Gemma-3n contract). We reconstruct the EXACT producer→borrower index map the
   shipped model builds (``models/gemma4_vendored/language.py``: each borrower
   reuses the LAST same-type producer below the split) and zero a borrower ONLY
   when that same-type producer exists. A borrower whose type has no producer
   below the split is charged its own footprint, never zeroed (over-count, never
   under-count) — the shipped model would fail to build such a config, so no
   runnable request is under-reserved.

The estimator is a pure function: it reads only plain fields off a config
object (or its ``text_config``), performs no I/O, loads no weights, and is unit
testable without a live model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Per-layer attention-type classification uses EXACT (case-folded) matches, not
# substrings: a substring rule would misclassify an unfamiliar full-attention
# name that merely contains ``"linear"`` / ``"gated"`` / ``"local"`` as
# zero-growth or window-bounded and silently UNDER-count it (codex round 1
# BLOCKING #4). Any value not on these allowlists falls through to full-growth —
# the conservative, never-under-count default. Values are the ``layer_types``
# strings the engine actually ships (Gemma-4 / GPT-OSS ``sliding_attention`` +
# ``full_attention``; Qwen3.5/3.6 GatedDeltaNet ``linear_attention``; Mamba
# stacks) plus their common upstream aliases.
# GatedDeltaNet (Qwen3-Next / Qwen3.5 / Qwen3.6) ships its linear-attention
# layers under the canonical ``linear_attention`` type plus these upstream
# aliases. ALL of them must route to ``_gateddeltanet_state_bytes`` — a name
# that classifies as recurrent (zero per-token growth) but whose state we cannot
# size would fall back to full-growth and lose the reduction this module exists
# to deliver (codex round 5 BLOCKING). Keeping the sizer's dispatch keyed on the
# same set the classifier uses prevents that drift.
_GATEDDELTANET_TYPES: frozenset[str] = frozenset(
    {
        "linear_attention",
        "gated_delta_net",
        "gated_deltanet",
    }
)
_RECURRENT_TYPES: frozenset[str] = _GATEDDELTANET_TYPES | frozenset(
    {
        "mamba",
        "mamba2",
        "recurrent",
        "rwkv",
    }
)
_SLIDING_TYPES: frozenset[str] = frozenset(
    {
        "sliding_attention",
        "sliding_window_attention",
        "local_attention",
        "local_sliding_attention",
        "swa",
    }
)

# Rotating-window cache allocation granularity, sourced from the shipped
# ``mlx_lm.models.cache.RotatingKVCache`` implementation:
#
#   * ``RotatingKVCache.step = 256`` — the class attribute that sizes buffer
#     growth: the single-token decode path grows the KV buffer in ``step``-token
#     increments up to ``max_size`` and only then rotates in place
#     (``_update_in_place``: ``new_size = min(self.step, self.max_size - prev)``),
#     so the live buffer after ``t`` tokens is ``min(ceil(t/step)*step, max_size)``.
#   * ``keep`` — the number of retained sink positions. The construction path
#     varies across the sliding models this engine serves: the vendored Gemma-4
#     stack (``models/gemma4_vendored/language.py``) and DeepSeek-V4
#     (``models/deepseek_v4.py``) build ``RotatingKVCache(max_size=sliding_window,
#     keep=0)``, while ``mlx_lm.models.cache.make_prompt_cache`` (the generic
#     factory) passes ``keep=4``. We pin ``KEEP = 4`` — the LARGEST keep any
#     shipped path uses — so the slot estimate is a valid upper bound for every
#     construction (more retained sinks ⇒ more slots ⇒ a looser, still-safe
#     bound; a ``keep=0`` model is over-counted by at most a fraction of one step).
#
# See :func:`rotating_cache_slots` for how these ground the per-request slot
# count. Sourced from the shipped defaults; if a future mlx-lm changes them,
# update here to match the impl.
_ROTATING_CACHE_STEP = 256
_ROTATING_CACHE_KEEP = 4


def rotating_cache_slots(window: int, tokens: int) -> int:
    """Upper-bound slot count a ``RotatingKVCache`` allocates for ``tokens``.

    The real per-layer buffer of a sliding-window (rotating) cache is
    SUBLINEAR in the token count: the decode path grows it in
    ``_ROTATING_CACHE_STEP``-token blocks up to ``window`` (``max_size``) and
    then rotates at a fixed size, retaining ``_ROTATING_CACHE_KEEP`` sink
    positions. So the live buffer after ``t`` tokens is
    ``min(ceil(t/step)*step, max_size)`` (verified against
    ``mlx_lm.models.cache.RotatingKVCache._update_in_place``).

    This returns a provable UPPER BOUND of that buffer for any ``tokens``::

        slots(T) = ceil( (min(T, window) + KEEP) / STEP ) * STEP
                 = min( ceil((T + KEEP)/STEP)*STEP, ceil((window + KEEP)/STEP)*STEP )

    which is:

    * ``>=`` the real buffer at every ``T`` (rounds UP to the step block and
      adds the retained sinks), so the D-METAL-CAP projection never under-counts
      the sliding allocation and trips the OOM cliff;
    * ``<=`` the full-window buffer ``ceil((window + KEEP)/STEP)*STEP`` for every
      ``T`` (the ``min(T, window)`` clamp), so a SHORT request is charged only
      what it can actually grow to — not the whole window (codex round 11
      BLOCKING #1: a fixed whole-window baseline over-counts short requests and
      can spuriously reject a request the uniform estimator would have admitted);
    * monotonic non-decreasing in ``T`` and flat once ``T >= window``.

    Returns ``0`` when there is no window or no tokens (no sliding term).
    """
    # Why ONE global (step=256, keep=4) is valid for every sliding layer we
    # reduce: rapid-mlx constructs ALL sliding-window caches through mlx_lm's
    # ``RotatingKVCache`` (directly, or via ``make_prompt_cache`` / the vendored
    # Gemma-4 / DeepSeek-V4 stacks), which share the class-default ``step = 256``
    # and a ``keep`` of 0–4. That global invariant IS the per-architecture
    # verification codex asks for — pinned by the hermetic drift test
    # ``tests/test_kv_estimation.py::TestRotatingCacheConstantsMatchInstalledMlxLm``
    # against the INSTALLED mlx_lm (fails loudly if step/keep ever change). If a
    # future family ships a sliding cache that is NOT mlx_lm ``RotatingKVCache``
    # (different granularity/retention), that combination must be re-verified
    # before its ``layer_types`` entry is added to ``_SLIDING_TYPES`` — otherwise
    # this slot count could under-reserve it.
    if window <= 0 or tokens <= 0:
        return 0
    effective = min(tokens, window) + _ROTATING_CACHE_KEEP
    blocks = (effective + _ROTATING_CACHE_STEP - 1) // _ROTATING_CACHE_STEP
    return blocks * _ROTATING_CACHE_STEP


@dataclass(frozen=True)
class KVFootprintEstimate:
    """Architecture-aware KV footprint of a single sequence.

    The D-METAL-CAP projection reconstructs a request's peak KV footprint,
    per request with ``T = prompt_tokens + max_tokens``, as::

        fixed_baseline_bytes
          + per_token_growth_bytes * T
          + sliding_slot_bytes * rotating_cache_slots(sliding_window, T)

    The three terms model the three footprint SHAPES this engine's hybrids mix:

    * full-attention layers grow UNBOUNDED per token → ``per_token_growth_bytes``
      (multiplied by ``T``);
    * sliding-window layers grow SUBLINEARLY — up to a rotating window, then flat
      → ``sliding_slot_bytes`` (per-slot bytes) times the per-request slot count
      ``rotating_cache_slots(sliding_window, T)`` (grounded in the real
      ``RotatingKVCache`` allocation; see :func:`rotating_cache_slots`);
    * recurrent / linear-attention layers carry a FIXED state → part of
      ``fixed_baseline_bytes`` (token-independent, charged once).

    Modelling sliding layers per request — rather than as either a flat window
    baseline (over-counts short requests, spuriously rejecting them) or unbounded
    per-token growth (over-counts long requests) — makes the projection an
    over-count-safe UPPER BOUND at every ``T`` while keeping the GPT-OSS/Gemma-4
    reduction: a short request is charged only what it can grow to, a long one is
    capped at the full rotating buffer.

    Attributes:
        per_token_growth_bytes: Bytes of KV cache allocated per additional token,
            summed over the full-attention (unbounded-growth) layers only. The
            projection multiplies this by ``T``. A sliding layer whose window is
            unreadable is folded in here (charged as full growth) so it is never
            silently dropped.
        fixed_baseline_bytes: Token-independent bytes allocated once per
            sequence — the conservative fixed recurrent state of each sizeable
            recurrent (GatedDeltaNet) layer. Zero for a dense model and for a
            hybrid without recurrent layers. (Sliding-window buffers are NOT here;
            they are request-dependent — see ``sliding_slot_bytes``.)
        sliding_slot_bytes: Bytes for ONE rotating-cache slot summed across all
            window-bounded sliding-window layers —
            ``Σ_sliding (2 · dtype · kv_heads_L · head_dim_L)`` (each layer clamped
            to the uniform per-layer floor). The per-request projection multiplies
            this by ``rotating_cache_slots(sliding_window, T)``. Zero when the
            config has no window-bounded sliding layer.
        sliding_window: The rotating window size (``max_size``) shared by the
            sliding-window layers. ``0`` when there is no sliding term (dense /
            unknown, or the window was unreadable and those layers were folded
            into ``per_token_growth_bytes``).
    """

    per_token_growth_bytes: int
    fixed_baseline_bytes: int
    sliding_slot_bytes: int
    sliding_window: int


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` off a config that may be a dict or an attribute object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _pos_int(value: Any) -> int:
    """Return ``value`` when it is a strictly-positive, non-bool int, else 0.

    Mirrors the scheduler's ``isinstance(..., int)`` MagicMock guard: a
    ``MagicMock`` attribute is not an ``int`` (and ``bool`` is excluded because
    it is an ``int`` subclass that never denotes a real dimension).
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _valid_layer_types(value: Any, expected_len: int) -> list[str] | None:
    """Return ``value`` as a list of type strings iff it is well-formed.

    Well-formed = a list/tuple of exactly ``expected_len`` string entries. Any
    other shape (missing, MagicMock, wrong length, non-string entries) returns
    ``None`` so the caller falls back to the uniform formula rather than
    guessing a per-layer layout it cannot trust.
    """
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) != expected_len or expected_len <= 0:
        return None
    if not all(isinstance(entry, str) for entry in value):
        return None
    return [entry.lower() for entry in value]


def _pick_structural_config(model_config: Any, base_num_layers: int) -> Any | None:
    """Select the config object that carries the hybrid layer structure.

    Layer-structure fields live either directly on ``model_config`` (text-lane
    configs) or nested under ``model_config.text_config`` (multimodal configs).
    Returns the config whose ``layer_types`` list matches ``base_num_layers``,
    or ``None`` when neither carries one (the byte-identical dense/unknown
    fallback).

    A per-layer ``layer_types`` list is REQUIRED to engage the hybrid path.
    ``num_kv_shared_layers`` on its own is not enough: without the per-layer map
    we cannot verify which layers actually borrow (codex round 1 BLOCKING #3),
    so a config exposing only a share count stays on the uniform estimate.
    """
    candidates = [model_config]
    text_config = _cfg_get(model_config, "text_config")
    if text_config is not None:
        candidates.append(text_config)

    for cfg in candidates:
        if _valid_layer_types(_cfg_get(cfg, "layer_types"), base_num_layers):
            return cfg
    return None


def _classify_layer(layer_type: str) -> str:
    """Map a case-folded ``layer_types`` entry to a footprint class.

    Returns one of ``"recurrent"``, ``"sliding"``, ``"full"``. Matching is
    EXACT against the zero-growth / window-bounded allowlists; every other
    value — a known full-attention type or an unrecognized string — resolves to
    ``"full"``, the conservative (over-counting) default that can never
    under-count.
    """
    if layer_type in _RECURRENT_TYPES:
        return "recurrent"
    if layer_type in _SLIDING_TYPES:
        return "sliding"
    return "full"


def _uniform_sliding_window(struct: Any) -> int:
    """Return the sliding window when it is a single positive scalar, else ``0``.

    The per-request slot model (``sliding_slot_bytes × rotating_cache_slots(window,
    T)``) charges EVERY sliding layer with ONE window, so it is sound only when the
    config states a single window that all sliding layers share. Shipped hybrids
    (GPT-OSS, Gemma-4) expose ``sliding_window`` as exactly that positive scalar.

    Any NON-scalar shape — a per-layer window ``list`` / ``tuple``, or an
    unreadable value — returns ``0`` so the caller folds those sliding layers into
    full per-token growth (over-count-safe) instead. We deliberately do NOT try to
    collapse a per-layer list to one window: a partial / misaligned list (e.g. a
    positive window on one sliding index and a ``None`` on another) could borrow the
    known layer's window for a sliding layer whose real window is UNKNOWN and
    possibly larger, under-reserving KV (codex round 13 BLOCKING #3 and follow-up).
    Verifying full ``layer_types``-aligned per-index uniformity would be the only
    safe way to trust a list — and no shipped model expresses a list window — so
    the safe, minimal policy is scalar-only, list ⇒ full-growth fallback.
    """
    return _pos_int(_cfg_get(struct, "sliding_window"))


# mlx-lm materializes the recurrent conv/SSM state buffers with ``mx.zeros``,
# which defaults to float32 — so we size the recurrent per-sequence state at 4
# bytes/element regardless of the (possibly smaller) KV dtype. Over-charging a
# small fixed term is the safe direction and guarantees we never under-reserve
# it (codex round 2 BLOCKING #1).
_RECURRENT_STATE_BYTES_PER_ELEM = 4


def _gateddeltanet_state_bytes(struct: Any) -> int | None:
    """GatedDeltaNet (Qwen3-Next / Qwen3.5 / Qwen3.6) per-sequence state bytes.

    State = a per-value-head ``head_k_dim × head_v_dim`` matrix plus the
    causal-conv ring buffer over ``conv_dim = 2·key_dim + value_dim`` channels.
    Shapes match ``mlx_lm.models.qwen3_next.Qwen3NextGatedDeltaNet``. ALL of the
    state + conv dimensions must be present and positive — including the conv
    kernel — else return ``None`` so the caller charges full growth rather than
    silently omitting the conv buffer and under-reserving (codex round 3
    BLOCKING #1).
    """
    num_v = _pos_int(_cfg_get(struct, "linear_num_value_heads"))
    num_k = _pos_int(_cfg_get(struct, "linear_num_key_heads"))
    head_k = _pos_int(_cfg_get(struct, "linear_key_head_dim"))
    head_v = _pos_int(_cfg_get(struct, "linear_value_head_dim"))
    conv_kernel = _pos_int(_cfg_get(struct, "linear_conv_kernel_dim"))
    if not (num_v and num_k and head_k and head_v and conv_kernel):
        return None
    key_dim = num_k * head_k
    value_dim = num_v * head_v
    recurrent_elems = num_v * head_k * head_v
    conv_dim = 2 * key_dim + value_dim
    conv_elems = (conv_kernel - 1) * conv_dim
    return _RECURRENT_STATE_BYTES_PER_ELEM * (recurrent_elems + conv_elems)


def _recurrent_state_bytes(struct: Any, layer_type: str) -> int | None:
    """Conservative per-sequence FIXED state of one recurrent layer, in bytes.

    Sizes the state only for the recurrent family whose EXACT state + conv
    layout we have validated against the shipped implementation — GatedDeltaNet
    (Qwen3-Next / Qwen3.5 / Qwen3.6), keyed on its family-specific ``linear_*``
    fields. Every GatedDeltaNet alias the classifier treats as recurrent
    (``_GATEDDELTANET_TYPES``: ``linear_attention`` + the ``gated_delta_net`` /
    ``gated_deltanet`` upstream spellings) routes to the same sizer, so a config
    that names its linear layers with an alias still gets the reduction rather
    than silently reverting to full per-token growth (codex round 5 BLOCKING).

    Every OTHER recurrent family — Mamba / Mamba2, RWKV, generic ``recurrent`` —
    returns ``None`` so the caller charges the layer full per-token growth (the
    safe, never-under-count fallback). We deliberately do NOT attempt to size a
    Mamba state: the conv buffer channel count differs between Mamba1
    (``d_inner``) and Mamba2 (``d_inner`` + grouped ``B``/``C`` channels =
    ``d_inner + 2·n_groups·d_state``), so a single generic formula would
    UNDER-reserve Mamba2 (codex round 7 BLOCKING) — and this engine ships no
    Mamba model, so over-counting those layers as full growth costs nothing while
    keeping the OOM-safety guarantee intact. When a real Mamba family is shipped,
    add a family-specific sizer here mirroring its exact cache shapes.

    The returned term is a per-sequence constant (it does not grow with
    generated tokens), so it lands in the fixed baseline, never in per-token
    growth.
    """
    if layer_type in _GATEDDELTANET_TYPES:
        return _gateddeltanet_state_bytes(struct)
    # Mamba/Mamba2/RWKV/other recurrent: state layout not modelled → full growth.
    return None


def estimate_kv_footprint(
    model_config: Any,
    *,
    dtype_bytes: int,
    uniform_per_token_bytes: int,
    base_num_layers: int,
    base_kv_heads: int,
    base_head_dim: int,
) -> KVFootprintEstimate:
    """Compute the architecture-aware KV footprint for one sequence.

    Args:
        model_config: The live model config (attribute object or dict). Only
            plain fields are read; no I/O, no weight load.
        dtype_bytes: KV element size in bytes, as already resolved by the
            caller (``scheduler._infer_kv_dtype_bytes``).
        uniform_per_token_bytes: The caller's historical uniform per-token
            figure. Returned unchanged when the config is not a recognized
            hybrid, guaranteeing byte-identical behavior for dense/unknown
            models.
        base_num_layers: ``num_hidden_layers`` as read by the caller. Used to
            validate the ``layer_types`` length and to bound the borrower split.
        base_kv_heads: ``num_key_value_heads`` (or attention heads) as read by
            the caller — the local/sliding-layer KV-head count.
        base_head_dim: ``head_dim`` as read by the caller — the local/sliding
            head dimension.

    Returns:
        A :class:`KVFootprintEstimate`. For a non-hybrid config this is exactly
        ``(uniform_per_token_bytes, 0, 0, 0)`` — the uniform per-token figure with
        no fixed baseline and no sliding term (byte-identical to the historical
        ``per_tok × tokens`` projection).
    """
    uniform = KVFootprintEstimate(
        per_token_growth_bytes=uniform_per_token_bytes,
        fixed_baseline_bytes=0,
        sliding_slot_bytes=0,
        sliding_window=0,
    )

    # Guard the primitives. If the caller could not read a positive uniform
    # figure (MagicMock model / missing config), stay at the byte-identical
    # fallback — the scheduler never calls us in that case, but be defensive.
    if (
        _pos_int(dtype_bytes) == 0
        or _pos_int(base_num_layers) == 0
        or _pos_int(base_kv_heads) == 0
        or _pos_int(base_head_dim) == 0
        or uniform_per_token_bytes <= 0
    ):
        return uniform

    struct = _pick_structural_config(model_config, base_num_layers)
    if struct is None:
        # Not a recognized hybrid — dense (Llama/Qwen dense) or an
        # unknown/stub config. Byte-identical to the uniform formula.
        return uniform

    # Local (sliding / default) dims — prefer the structural config, fall back
    # to the caller's base values so a config that only splits full vs sliding
    # via ``layer_types`` (no distinct local fields) still works.
    local_kv_heads = (
        _pos_int(_cfg_get(struct, "num_key_value_heads"))
        or _pos_int(_cfg_get(struct, "num_attention_heads"))
        or base_kv_heads
    )
    local_head_dim = _pos_int(_cfg_get(struct, "head_dim")) or base_head_dim

    # Global (full-attention) dims. Gemma-4 full layers use a wider
    # ``global_head_dim`` / ``num_global_key_value_heads`` than the local
    # sliding layers; when absent (GPT-OSS and most hybrids) they fall back to
    # the local dims. Using the (larger-or-equal) global dim for full layers is
    # both accurate and never-under-count.
    global_kv_heads = (
        _pos_int(_cfg_get(struct, "num_global_key_value_heads")) or local_kv_heads
    )
    global_head_dim = _pos_int(_cfg_get(struct, "global_head_dim")) or local_head_dim

    # Sliding window size — the SINGLE window every sliding layer shares. Resolved
    # via ``_uniform_sliding_window``, which is scalar-only: it returns the window
    # only when ``sliding_window`` is one positive scalar (the shipped GPT-OSS /
    # Gemma-4 shape). Any per-layer window list/tuple, or an unreadable value,
    # yields 0 and those sliding layers are charged as full per-token growth below
    # (over-count-safe, never under-count).
    window = _uniform_sliding_window(struct)

    # ``layer_types`` is guaranteed present + length-checked by
    # ``_pick_structural_config``; the ``is None`` branch is unreachable but
    # kept as a defensive fallback (avoids an assert that ``python -O`` strips).
    layer_types = _valid_layer_types(_cfg_get(struct, "layer_types"), base_num_layers)
    if layer_types is None:
        return uniform

    # Borrower map: Gemma-4 / Gemma-3n reuse the LAST ``num_kv_shared_layers``
    # decoder layers' K/V from an earlier producer (split at
    # ``num_hidden_layers - num_kv_shared_layers``). We build the EXACT
    # producer→borrower index map the shipped model builds, mirroring
    # ``models/gemma4_vendored/language.py`` (``LanguageModel.__init__``)::
    #
    #     M = num_hidden_layers - num_kv_shared_layers          # split
    #     last_producer_by_type = {layer_types[i]: i for i in range(M)}  # last wins
    #     previous_kvs[j] = last_producer_by_type[layer_types[j]]  (j in [M, N))
    #
    # so a borrower ``j`` reuses the LAST producer BELOW the split whose
    # attention type matches its own, and allocates zero KV. We zero borrower
    # ``j`` ONLY when such a same-type producer exists — i.e. its mapped
    # producer's type equals its own (true by construction of the map). A
    # borrower whose type has NO producer below the split is NOT zeroed: it is
    # charged its own footprint (the shipped model would raise a KeyError at
    # build time for that config, so no runnable request is ever under-reserved —
    # over-count, never under-count). This exact per-index map supersedes the old
    # set-subset test, which could zero a borrower without validating the actual
    # producer it maps to (codex round 11 BLOCKING #2). If ``num_kv_shared_layers``
    # is absent / out of range, no layer borrows.
    num_shared = _cfg_get(struct, "num_kv_shared_layers")
    if isinstance(num_shared, int) and not isinstance(num_shared, bool):
        num_shared = num_shared if 0 < num_shared < base_num_layers else 0
    else:
        num_shared = 0
    borrowed_layers: set[int] = set()
    if num_shared > 0:
        split = base_num_layers - num_shared
        last_producer_by_type: dict[str, int] = {}
        for producer_idx in range(split):
            last_producer_by_type[layer_types[producer_idx]] = producer_idx
        for borrower_idx in range(split, base_num_layers):
            # KV sharing reuses an earlier producer's ATTENTION K/V. A recurrent /
            # linear-attention layer that happens to sit in the last-N borrower
            # positions has no attention KV to share — it keeps its OWN fixed
            # recurrent state — so it must NEVER be zeroed here; it falls through
            # to the recurrent branch below and is charged its state (or
            # full-growth fallback). Only an attention-class borrower can borrow
            # and allocate zero (codex round 13 BLOCKING #1: zeroing a recurrent
            # borrower would drop its state and under-count).
            if _classify_layer(layer_types[borrower_idx]) == "recurrent":
                continue
            mapped = last_producer_by_type.get(layer_types[borrower_idx])
            # ``mapped`` is keyed by type, so ``layer_types[mapped]`` is the
            # borrower's own type whenever the lookup hits — the exact
            # same-type-producer condition the shipped model relies on.
            if mapped is not None and layer_types[mapped] == layer_types[borrower_idx]:
                borrowed_layers.add(borrower_idx)

    # The uniform per-layer charge (base dims). It is the never-under-count
    # floor: no per-token-growing layer is ever charged LESS than this.
    uniform_per_layer = 2 * dtype_bytes * base_kv_heads * base_head_dim
    # Full-attention layers use the (wider) global dims when present, but never
    # below the uniform base-layer size — a config whose ``global_head_dim`` /
    # ``num_global_key_value_heads`` are smaller than (or unrelated to) the base
    # dims must not shrink a full layer's per-token growth below uniform and
    # open an under-count (codex round 4 BLOCKING #1).
    full_per_token = max(
        2 * dtype_bytes * global_kv_heads * global_head_dim, uniform_per_layer
    )
    # Per-slot (per-token-position) KV bytes of ONE sliding-window layer (local
    # dims), clamped to the uniform base-layer floor for the same reason full
    # layers are (codex round 7 BLOCKING #2): a malformed / non-authoritative
    # nested config exposing smaller local dims than the base must not drop a
    # layer's charge below the uniform per-layer rate and open an under-count.
    sliding_per_layer = max(
        2 * dtype_bytes * local_kv_heads * local_head_dim, uniform_per_layer
    )

    per_token_growth = 0
    fixed_baseline = 0
    # ``sliding_slot_bytes`` accumulates the per-SLOT bytes of every
    # window-bounded sliding layer. The scheduler multiplies it by the
    # per-request slot count ``rotating_cache_slots(window, T)`` — a SUBLINEAR
    # bound that grows with the request's token budget up to the full rotating
    # buffer (codex round 11 BLOCKING #1: a fixed whole-window baseline
    # over-counts short requests). Sliding layers therefore contribute nothing to
    # the token-independent ``fixed_baseline`` (which now holds recurrent state
    # only) — their footprint is genuinely request-dependent.
    sliding_slot_bytes = 0
    for idx in range(base_num_layers):
        if idx in borrowed_layers:
            # Borrower (KV-sharing) layer — reuses its mapped same-type
            # producer's cache, allocates nothing.
            continue
        layer_type = layer_types[idx]
        layer_class = _classify_layer(layer_type)
        if layer_class == "recurrent":
            # Recurrent / linear-attention layers keep a FIXED state that does
            # not grow per token. When we can size that state from the family's
            # real config fields we reserve it in the per-sequence baseline (so
            # concurrent sequences don't silently allocate past the cap — codex
            # round 2 BLOCKING #1) while keeping per-token growth at zero — the
            # load-bearing correctness win for Qwen3.5/3.6. When we CANNOT size
            # it, we fall back to charging the layer the uniform per-token
            # estimate (full growth), which never under-counts.
            state = _recurrent_state_bytes(struct, layer_type)
            if state is not None:
                fixed_baseline += state
            else:
                per_token_growth += uniform_per_layer
            continue
        if layer_class == "sliding":
            if window > 0:
                # Window-bounded: contributes one slot's bytes to the
                # per-request sliding term (charged as
                # ``sliding_slot_bytes × rotating_cache_slots(window, T)``).
                sliding_slot_bytes += sliding_per_layer
            else:
                # No readable window → cannot bound it → charge the uniform
                # per-layer growth (never under-count).
                per_token_growth += uniform_per_layer
        else:  # "full"
            per_token_growth += full_per_token

    # Advertise the window only when at least one window-bounded sliding layer
    # actually contributed a slot term — otherwise there is no sliding term and
    # the scheduler must not multiply anything by a stray window.
    effective_sliding_window = window if sliding_slot_bytes > 0 else 0

    return KVFootprintEstimate(
        per_token_growth_bytes=int(per_token_growth),
        fixed_baseline_bytes=int(fixed_baseline),
        sliding_slot_bytes=int(sliding_slot_bytes),
        sliding_window=int(effective_sliding_window),
    )
