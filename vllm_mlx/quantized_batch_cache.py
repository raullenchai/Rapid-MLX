"""Quantized, continuous-batching KV cache (dequant-on-read).

Background (#1197)
------------------
``--kv-cache-dtype int8/int4`` was only ever wired into the memory-aware
*prefix* cache (:class:`MemoryCacheConfig`). The **live** per-request KV cache
used by continuous batching is mlx-lm's :class:`~mlx_lm.models.cache.BatchKVCache`,
which is always bf16 — so a fresh server with ``--disable-prefix-cache`` got no
memory reduction from the requested dtype at all.

``QuantizedBatchKVCache`` is a drop-in replacement for ``BatchKVCache`` that
stores keys/values **quantized** (``mx.quantize`` along the head dimension). The
paths the *model* reads through — ``update_and_fetch`` and ``extract`` —
dequantize on the way out, so attention/SDPA see bf16 and nothing in the model
changes. This is the "dequant-on-read" design that vLLM and SGLang both fall
back to on backends without a fused quantized-attention kernel — it wins back
the resident-memory footprint (the point of the flag) without a custom kernel.

``state`` / ``meta_state`` are the *serialization* interface (save / restore /
mid-prefill snapshot), not a model-read path. Like mlx-lm's own
``QuantizedKVCache``, ``state`` returns the raw quantized triples
``[packed_uint32, scales, biases]`` plus ``(group_size, bits)`` in
``meta_state`` so the value round-trips losslessly. mlx-lm's only in-loop
consumer of ``.state`` is ``mx.eval([c.state ...])``, which is structure-
agnostic; rapid-mlx's own snapshot reconstruction
(``Scheduler._reconstruct_cache_from_states``) dequantizes the triples back to a
bf16 ``KVCache``.

Why this is a mechanical mirror of ``BatchKVCache``
---------------------------------------------------
``mx.quantize`` packs only the **last** axis (head_dim). The sequence axis
(``axis=2``) stays 1:1 with tokens in the packed representation, so every
token-level bookkeeping operation ``BatchKVCache`` performs — left-padding,
capacity growth, ``filter`` (row select + left-shift), ``extend``/``merge``
(right-justified pad + concat), ``trim`` and ``make_mask`` — applies identically
to each of the three stored tensors ``(packed_uint32, scales, biases)``. The
quantization axis and the sequence axis are orthogonal.

Threshold semantics
--------------------
The live cache quantizes from the first token. ``kv_min_quantize_tokens`` keeps
its existing meaning for the *retained prefix cache* only; it is intentionally
not applied to the live cache here (a fresh live cache cannot know its final
length at construction, and int8/int4 affine quantization of short sequences is
harmless). Callers that want the live cache left as bf16 simply do not opt in.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

# MUST install the MLX hardware-compat shim BEFORE any `from mlx_lm.*` import.
# `mlx_lm/__init__.py` re-exports from `mlx_lm.generate`, which captures
# `mx.new_thread_local_stream(mx.default_device())` at module-import time; on
# M5 single-stream GPUs that stream is unusable (#404). The shim is idempotent
# and a no-op on hardware where the original API works.
from . import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.cache import (  # noqa: E402
    KVCache,
    _BaseCache,
    create_causal_mask,
    dynamic_roll,
)


def _quantize(x: mx.array, group_size: int, bits: int) -> list[mx.array]:
    """Quantize along the last (head) dim -> [packed_uint32, scales, biases]."""
    return list(mx.quantize(x, group_size=group_size, bits=bits))


def _dequantize(triple: list[mx.array], group_size: int, bits: int) -> mx.array:
    return mx.dequantize(
        triple[0], triple[1], triple[2], group_size=group_size, bits=bits
    )


# mx.quantize (affine) only accepts these group sizes, and the quantized
# dimension must be divisible by the chosen one.
_SUPPORTED_GROUP_SIZES = (128, 64, 32)


def supported_group_size(head_dim: int, requested: int) -> int | None:
    """Largest supported group size ``<= requested`` that divides ``head_dim``.

    Returns ``None`` when no supported group size (32/64/128) divides
    ``head_dim`` — e.g. ``head_dim=80`` — in which case the caller must fall
    back to a bf16 cache instead of quantizing.
    """
    for gs in _SUPPORTED_GROUP_SIZES:
        if gs <= requested and head_dim % gs == 0:
            return gs
    return None


class QuantizedBatchKVCache(_BaseCache):
    """Batch-aware KV cache that stores quantized KV and returns bf16.

    Storage layout mirrors :class:`~mlx_lm.models.cache.QuantizedKVCache` — each
    of ``self.keys`` / ``self.values`` is a 3-element list
    ``[packed_uint32, scales, biases]`` (or ``None`` when empty) shaped
    ``(B, n_kv_heads, capacity, *)``. Sequence bookkeeping mirrors
    :class:`~mlx_lm.models.cache.BatchKVCache` (per-row ``offset`` /
    ``left_padding`` arrays plus a scalar ``_idx`` fill pointer).
    """

    step = 256

    def __init__(
        self,
        left_padding: list[int],
        group_size: int = 64,
        bits: int = 8,
    ):
        self.keys: list[mx.array] | None = None
        self.values: list[mx.array] | None = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-pad for pad in left_padding])
        self._idx = 0
        self._right_padding: mx.array | None = None
        self._q_group_size = group_size
        self._q_bits = bits

    # -- internal helpers -------------------------------------------------

    def _capacity(self) -> int:
        return 0 if self.keys is None else self.keys[0].shape[2]

    def _init_storage(
        self, b: int, n_heads: int, cap: int, head_dim: int, dtype
    ) -> list[mx.array]:
        el_per_int = 8 * mx.uint32.size // self._q_bits
        n_groups = head_dim // self._q_group_size
        return [
            mx.zeros((b, n_heads, cap, head_dim // el_per_int), dtype=mx.uint32),
            mx.zeros((b, n_heads, cap, n_groups), dtype=dtype),
            mx.zeros((b, n_heads, cap, n_groups), dtype=dtype),
        ]

    def _slice_seq(self, triple: list[mx.array], upto: int) -> list[mx.array]:
        return [m[..., :upto, :] for m in triple]

    def _resolve_group_size(self, k_head_dim: int, v_head_dim: int) -> int:
        """Coerce the group size to one that divides both head dims.

        A best-effort probe at install time normally picks a compatible group
        size, but this is the last line of defense (e.g. when the probe could
        not read head_dim). Raises a clear, actionable error rather than letting
        ``mx.quantize`` surface its opaque divisibility message.
        """
        gs = supported_group_size(k_head_dim, self._q_group_size)
        if gs is not None and v_head_dim != k_head_dim:
            gs = supported_group_size(v_head_dim, gs)
        if gs is None:
            raise ValueError(
                f"KV cache quantization requires head_dim (K={k_head_dim}, "
                f"V={v_head_dim}) divisible by a supported group_size "
                f"(32/64/128); none fits requested group_size={self._q_group_size}. "
                f"Re-run with --kv-cache-dtype bf16."
            )
        return gs

    # -- core -------------------------------------------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx

        if self.keys is None:
            # First write: lock in a group size compatible with the real head
            # dims (mirrors the install-time probe; covers the case it failed).
            self._q_group_size = self._resolve_group_size(k_head_dim, v_head_dim)

        qk = _quantize(keys, self._q_group_size, self._q_bits)
        qv = _quantize(values, self._q_group_size, self._q_bits)

        if self.keys is None or (prev + num_steps) > self._capacity():
            n_steps = (self.step + num_steps - 1) // self.step
            add = n_steps * self.step
            if self.keys is not None:
                # Drop any unused reserved tail before growing, mirroring
                # BatchKVCache so concatenation stays contiguous with _idx.
                if prev % self.step != 0:
                    self.keys = self._slice_seq(self.keys, prev)
                    self.values = self._slice_seq(self.values, prev)

                def _grow(triple):
                    return [
                        mx.concatenate(
                            [
                                m,
                                mx.zeros(
                                    (*m.shape[:2], add, m.shape[-1]), dtype=m.dtype
                                ),
                            ],
                            axis=2,
                        )
                        for m in triple
                    ]

                self.keys = _grow(self.keys)
                self.values = _grow(self.values)
            else:
                self.keys = self._init_storage(
                    B, n_kv_heads, add, k_head_dim, keys.dtype
                )
                self.values = self._init_storage(
                    B, n_kv_heads, add, v_head_dim, values.dtype
                )

        self.offset += num_steps
        self._idx += num_steps
        for i in range(3):
            self.keys[i][..., prev : self._idx, :] = qk[i]
            self.values[i][..., prev : self._idx, :] = qv[i]

        # dequant-on-read: return bf16 K/V so attention/SDPA stays unchanged.
        # This materializes the current layer's full history in bf16, but only
        # transiently — MLX frees it before the next layer's dequant runs, while
        # the resident cache stays quantized. Net decode PEAK memory is therefore
        # LOWER than a bf16 cache (all layers resident bf16), not higher:
        # measured int8 0.65x / int4 0.43x of bf16 peak at L=32,B=4,T=4k. Because
        # decode is memory-bandwidth-bound, the smaller cache also makes each
        # step FASTER (int8 ~2.3x) despite the dequant. A fused quantized SDPA
        # (route B) would drop the transient entirely but needs a custom kernel.
        return (
            _dequantize(
                self._slice_seq(self.keys, self._idx), self._q_group_size, self._q_bits
            ),
            _dequantize(
                self._slice_seq(self.values, self._idx),
                self._q_group_size,
                self._q_bits,
            ),
        )

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty QuantizedBatchKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            if self.keys is not None:
                self.keys = [
                    dynamic_roll(m, padding[:, None], axis=2) for m in self.keys
                ]
                self.values = [
                    dynamic_roll(m, padding[:, None], axis=2) for m in self.values
                ]
            # No storage yet (no write since prepare) — nothing to roll, but the
            # padding bookkeeping still has to settle so offset/left_padding stay
            # consistent with a subsequent first write.
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    # -- state ------------------------------------------------------------

    @property
    def state(self):
        if self.keys is None:
            return None, None, self.offset, self.left_padding
        if self._idx < self._capacity():
            k = self._slice_seq(self.keys, self._idx)
            v = self._slice_seq(self.values, self._idx)
        else:
            k, v = self.keys, self.values
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = 0 if self.keys is None else self.keys[0].shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self._q_group_size, self._q_bits)))

    @meta_state.setter
    def meta_state(self, v):
        self._q_group_size, self._q_bits = map(int, v)

    # -- trimming / masking ----------------------------------------------

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, n: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            n, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    # -- batch composition ------------------------------------------------

    def filter(self, batch_indices):
        """In-place keep only ``batch_indices`` rows (then left-shift padding)."""
        if self.keys is not None:
            self.keys = [m[batch_indices] for m in self.keys]
            self.values = [m[batch_indices] for m in self.values]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = [m[..., min_left_pad:, :] for m in self.keys]
                self.values = [m[..., min_left_pad:, :] for m in self.values]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other: QuantizedBatchKVCache):
        """In-place concatenate ``other``'s rows onto this cache."""
        # The concatenated triples only dequantize correctly if both caches share
        # (group_size, bits). In continuous batching they always do (one group
        # size is resolved per server), but an empty cache may still carry its
        # construction-time default — adopt the populated cache's params, and
        # reject a genuine mismatch instead of silently mis-dequantizing.
        if self.keys is None and other.keys is not None:
            self._q_group_size, self._q_bits = other._q_group_size, other._q_bits
        elif (
            self.keys is not None
            and other.keys is not None
            and (self._q_group_size, self._q_bits)
            != (other._q_group_size, other._q_bits)
        ):
            raise ValueError(
                "Cannot extend QuantizedBatchKVCache with mismatched quantization "
                f"params: {(self._q_group_size, self._q_bits)} vs "
                f"{(other._q_group_size, other._q_bits)}"
            )

        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        max_size = max(self._capacity(), other._capacity())

        ref = self if self.keys is not None else other
        H = ref.keys[0].shape[1]
        key_dims = [m.shape[-1] for m in ref.keys]
        val_dims = [m.shape[-1] for m in ref.values]
        key_dtypes = [m.dtype for m in ref.keys]
        val_dtypes = [m.dtype for m in ref.values]

        def pad(c):
            Bc = c.offset.shape[0]
            if c.keys is None:
                k = [
                    mx.zeros((Bc, H, 0, d), dtype=dt)
                    for d, dt in zip(key_dims, key_dtypes)
                ]
                v = [
                    mx.zeros((Bc, H, 0, d), dtype=dt)
                    for d, dt in zip(val_dims, val_dtypes)
                ]
                cur_len = 0
            else:
                k, v = list(c.keys), list(c.values)
                cur_len = c.keys[0].shape[2]
            left = max_idx - c._idx
            right = max_size - cur_len - left
            if right < 0:
                k = [m[..., :right, :] for m in k]
                v = [m[..., :right, :] for m in v]
                right = 0
            if left != 0 or right != 0:
                cfg = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = [mx.pad(m, cfg) for m in k]
                v = [mx.pad(m, cfg) for m in v]
            return k, v, c.offset, c.left_padding + left

        ks, vs, offs, lps = zip(pad(self), pad(other))
        self.keys = [mx.concatenate([a, b], axis=0) for a, b in zip(ks[0], ks[1])]
        self.values = [mx.concatenate([a, b], axis=0) for a, b in zip(vs[0], vs[1])]
        self.offset = mx.concatenate(list(offs))
        self.left_padding = mx.concatenate(list(lps))
        self._idx = max_idx

    def extract(self, idx: int) -> KVCache:
        """Extract row ``idx`` as a single-sequence bf16 :class:`KVCache`."""
        cache = KVCache()
        if self.keys is None:
            # ``merge`` returns an empty cache (keys is None) for a fresh live
            # batch; a request cancelled before its first write reaches here.
            # Return an empty KVCache rather than iterating ``None``.
            return cache
        padding = self.left_padding[idx].item()
        k = _dequantize(
            [m[idx : idx + 1, :, padding : self._idx] for m in self.keys],
            self._q_group_size,
            self._q_bits,
        )
        v = _dequantize(
            [m[idx : idx + 1, :, padding : self._idx] for m in self.values],
            self._q_group_size,
            self._q_bits,
        )
        cache.keys = mx.contiguous(k)
        cache.values = mx.contiguous(v)
        cache.offset = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches, group_size: int = 64, bits: int = 8):
        """Merge single-sequence bf16 caches into one quantized batch cache.

        Falls back to a plain bf16 :class:`BatchKVCache` when the real stored
        dims admit no supported group size — so an install-time probe that
        over-estimated compatibility degrades to bf16 instead of aborting the
        request on the first write.
        """
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No content yet (fresh live batch): return an empty quantized cache so
        # the subsequent update_and_fetch quantizes from the first token.
        if max_length == 0:
            return cls([0] * len(caches), group_size, bits)

        padding = [max_length - length for length in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)

        # Resolve the group size against the REAL dims; if none fits, degrade to
        # a bf16 BatchKVCache rather than raising on the first write.
        eff_gs = supported_group_size(Dk, group_size)
        if eff_gs is not None and Dv != Dk:
            eff_gs = supported_group_size(Dv, eff_gs)
        if eff_gs is None:
            from mlx_lm.generate import BatchKVCache

            return BatchKVCache.merge(caches)
        group_size = eff_gs

        # Keys and values can carry different dtypes (e.g. MLA caches), so resolve
        # each independently instead of casting values to the key dtype.
        dt_k = next(iter(c.keys.dtype for c in caches if c.keys is not None))
        dt_v = next(iter(c.values.dtype for c in caches if c.values is not None))

        obj = cls(padding, group_size, bits)
        obj._q_group_size = group_size

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt_k)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt_v)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + c.offset] = c.keys[..., : c.offset, :]
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]

        obj.keys = _quantize(keys, group_size, bits)
        obj.values = _quantize(values, group_size, bits)
        obj.offset += max_length
        obj._idx = max_length
        return obj

    # -- misc -------------------------------------------------------------

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return sum(m.nbytes for m in self.keys) + sum(m.nbytes for m in self.values)


class _QuantizableKVCache(KVCache):
    """Single-sequence bf16 cache whose ``merge`` yields a quantized batch.

    ``BatchGenerator`` builds a fresh single-sequence cache per request via
    ``make_prompt_cache`` and only decides the *batch* cache type when it merges
    them (:func:`mlx_lm.generate._merge_caches` calls ``caches[0][i].merge(...)``).
    Swapping the plain ``KVCache`` for this subclass is therefore the single,
    minimal hook that makes continuous batching build a quantized batch cache —
    without patching any mlx-lm internals.
    """

    def __init__(self, group_size: int = 64, bits: int = 8):
        super().__init__()
        self.q_group_size = group_size
        self.q_bits = bits

    @classmethod
    def merge(cls, caches):
        c0 = caches[0]
        return QuantizedBatchKVCache.merge(
            caches, group_size=c0.q_group_size, bits=c0.q_bits
        )


def install_quantized_batch_cache(
    batch_gen: Any, group_size: int = 64, bits: int = 8
) -> Any:
    """Wire ``batch_gen``'s continuous batching to a quantized live KV cache.

    ``mlx_lm.generate.BatchGenerator`` builds a fresh single-sequence cache per
    request via ``_make_new_cache`` and only decides the *batch* cache type when
    those are merged. Replacing each plain ``KVCache`` layer with
    :class:`_QuantizableKVCache` (whose ``merge`` yields a
    :class:`QuantizedBatchKVCache`) is therefore a single, minimal instance-level
    hook — no mlx-lm internals are patched.

    Only exact top-level ``KVCache`` layers are swapped. Everything else keeps
    its bf16 behavior:

    * ``RotatingKVCache`` (sliding-window / ``max_kv_size``) — quantized batched
      sliding-window is NYI upstream (``BatchRotatingKVCache`` raises NYI).
    * ``ArraysCache`` / ``MambaCache`` — linear/hybrid state, not KV.
    * ``CacheList`` (hybrid models such as DeepSeek-V3.2, LongCat, Baichuan-M1,
      Falcon-H1) — deliberately NOT recursed into. Dequant-on-read only holds
      when the model reads KV exclusively through ``update_and_fetch`` (which
      returns bf16). These models instead touch ``cache.keys`` directly in their
      forward pass — e.g. DeepSeek-V3.2 runs
      ``cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, ...))`` — which
      assumes a raw ``mx.array``, not a quantized ``[packed, scales, biases]``
      triple. Quantizing them is a follow-up requiring per-model handling.
    """
    orig_make_new_cache = batch_gen._make_new_cache

    def _quantized_make_new_cache():
        return [
            _QuantizableKVCache(group_size, bits) if type(c) is KVCache else c
            for c in orig_make_new_cache()
        ]

    batch_gen._make_new_cache = _quantized_make_new_cache
    return batch_gen


def _head_dim_from_args(args: Any) -> int | None:
    """Head dim carried directly by a single args/config object, or ``None``.

    Reads an explicit ``head_dim`` first, then falls back to
    ``hidden_size // num_attention_heads``. Does not descend into sub-configs;
    see :func:`_text_attention_args` for the multimodal-aware resolution.
    """
    if args is None:
        return None
    hd = getattr(args, "head_dim", None)
    if isinstance(hd, int) and hd > 0:
        return hd
    hs = getattr(args, "hidden_size", None)
    nh = getattr(args, "num_attention_heads", None)
    if isinstance(hs, int) and isinstance(nh, int) and nh > 0 and hs % nh == 0:
        return hs // nh
    return None


def _text_attention_args(model: Any) -> Any:
    """The args object carrying the *language* model's attention dims.

    Multimodal wrappers (VLMs such as ``Qwen3.5-4B-MLX-4bit``) keep the language
    model's ``head_dim``/``hidden_size`` on a nested ``language_model`` submodule
    (its own ``.args``) or an ``args.text_config`` sub-config — the top-level
    ``model.args`` has no attention dims. Probing only the top level returns
    ``None`` and disables live KV quantization even though the text tower
    quantizes fine (e.g. head_dim=256). Prefer the top-level args when they
    already carry a head dim (unchanged for text-only models); otherwise descend
    so VLMs are covered too. Returns the top-level args as a last resort so
    callers can still read ``v_head_dim`` from it exactly as before.
    """
    args = getattr(model, "args", None)
    if _head_dim_from_args(args) is not None:
        return args
    lm = getattr(model, "language_model", None)
    lm_args = getattr(lm, "args", None) if lm is not None else None
    if _head_dim_from_args(lm_args) is not None:
        return lm_args
    text_cfg = getattr(args, "text_config", None) if args is not None else None
    if _head_dim_from_args(text_cfg) is not None:
        return text_cfg
    return args


def probe_head_dim(model: Any) -> int | None:
    """Best-effort head dimension of a loaded mlx-lm model; ``None`` if unknown.

    Used at install time to pick a compatible group size (or disable live
    quantization when no supported group size divides the head dim). Descends
    into a multimodal wrapper's language submodule so VLMs are not spuriously
    reported as unprobeable (see :func:`_text_attention_args`).
    """
    return _head_dim_from_args(_text_attention_args(model))


def probe_kv_head_dims(model: Any) -> tuple[int | None, int | None]:
    """Best-effort ``(key, value)`` head dims of a loaded mlx-lm model.

    ``mx.quantize`` groups along the last (head) dimension, and a model may use a
    distinct value head dim (``v_head_dim``) from its key head dim. Both must be
    divisible by the chosen group size, so probe both. A dim that cannot be
    determined comes back as ``None``. ``v_head_dim`` is read from the same
    (possibly nested) args object that supplied the key head dim.
    """
    args = _text_attention_args(model)
    k = _head_dim_from_args(args)
    v = getattr(args, "v_head_dim", None) if args is not None else None
    if not (isinstance(v, int) and v > 0):
        v = k  # standard models: value head dim == key head dim
    return k, v


def resolve_kv_quantization(
    k_head_dim: int | None,
    v_head_dim: int | None,
    requested_group_size: int,
) -> tuple[int, bool]:
    """Decide the LIVE cache's group size and whether to install it (#1197).

    Only the LIVE continuous-batching cache needs this best-effort, config-level
    decision: it is chosen ONCE up front (before any token) and cannot fall back
    to bf16 mid-stream, so an incompatible first write would crash the request.
    The retained prefix cache is NOT gated here — it self-coerces per layer
    against the real stored dims at quantize time (see ``_quantize_cache``), which
    also handles MLA models whose cached dims differ from any config head dim.

    ``mx.quantize`` needs the head dim divisible by a group size in {32,64,128}.
    Returns ``(group_size, live_disabled)``:

    * probe failed (either dim unknown) OR no supported size divides both dims
      (e.g. head_dim=80) -> ``live_disabled=True``; the live cache stays bf16.
      (``group_size`` is returned unchanged and unused.)
    * otherwise -> the coerced size valid for both dims, live cache enabled.

    Note: because the probe reads generic attention dims, an MLA model
    (DeepSeek-V3) may be conservatively reported incompatible here and keep a
    bf16 LIVE cache even though its real cached dims quantize fine — that is a
    missed optimization, not a regression (the retained cache still quantizes).
    """
    if k_head_dim is None or v_head_dim is None:
        return requested_group_size, True
    gs = supported_group_size(k_head_dim, requested_group_size)
    if gs is not None and v_head_dim != k_head_dim:
        gs = supported_group_size(v_head_dim, gs)
    if gs is None:
        return requested_group_size, True
    return gs, False


def normalize_caches_for_quantization(caches: list, group_size: int, bits: int) -> list:
    """Convert plain ``KVCache`` layers into :class:`_QuantizableKVCache`.

    A prefix-cache HIT hands mlx-lm a restored ``KVCache`` list; without this it
    would merge into a plain ``BatchKVCache`` while a MISS builds a
    ``QuantizedBatchKVCache``, and mixing the two in ``extend`` crashes. Running
    restored caches through here makes both paths converge on the quantized
    batch type. Existing content (keys/values/offset) is preserved.

    Only exact top-level ``KVCache`` layers are converted, mirroring
    :func:`install_quantized_batch_cache` exactly so the MISS and HIT paths agree:
    ``RotatingKVCache`` / hybrid / ``CacheList`` layers pass through untouched
    (see that function for why ``CacheList`` models stay bf16).
    """
    out = []
    for c in caches:
        if type(c) is KVCache:
            q = _QuantizableKVCache(group_size, bits)
            q.keys, q.values, q.offset = c.keys, c.values, c.offset
            out.append(q)
        else:
            out.append(c)
    return out
