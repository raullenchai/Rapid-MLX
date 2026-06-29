# SPDX-License-Identifier: Apache-2.0
"""Surgical indexer gate for REAP-pruned DeepseekV32 models.

Background
----------
Upstream ``mlx_lm.models.deepseek_v32.DeepseekV32Attention.__init__`` builds
``self.indexer = Indexer(config)`` on **every** layer unconditionally. The base
GLM-5.2 architecture (and its REAP-pruned variant
``mlx-community/pipenetwork-GLM-5.2-REAP50-MLX-4bit``) publishes a per-layer
``indexer_types: List[str]`` field where each entry is either ``"full"`` (this
layer owns Indexer weights in the safetensors) or ``"shared"`` (this layer has
**no** Indexer weights and is expected at inference time to reuse the
``topk_indices`` produced by the **most recent preceding "full" layer**'s
indexer). The GLM-5.2 base config pattern is e.g.
``["full", "full", "full", "shared", "shared", "shared", "full", "shared", ...]``
— full anchors at positions 0,1,2,6,10,14,...; shared layers in between.

On such configs every ``"shared"`` layer's Indexer parameters are missing from
the safetensors and ``mlx_lm.load`` aborts with
``Missing N parameters: ...indexer...`` (on the full 78-layer model: 11 keys ×
57 shared layers = 627 missing tensors).

What this module patches
------------------------
1. ``DeepseekV32DecoderLayer.__init__`` — after the upstream init runs, if
   ``config.indexer_types`` is present and the current layer is ``"shared"``,
   drop ``self_attn.indexer`` (``= None``) so it is removed from the model's
   parameter tree (see :class:`mlx.nn.Module.__setattr__`: assigning a
   non-array/dict/list/tuple deregisters the child from the parameter dict).
   ``layer_idx`` is threaded onto ``self_attn``.

2. ``Indexer.__call__`` — wrap to stash the return tensor on
   ``self._last_topk_indices`` so the next shared layer in the same forward
   pass can pick it up.

3. ``DeepseekV32Model.__call__`` — wrap so it threads the most-recent
   ``"full"`` layer's ``topk_indices`` into each subsequent ``"shared"``
   layer's attention via an instance attribute
   ``self_attn._shared_topk_indices`` (cleared after each shared layer
   reads it). On non-REAP configs (``indexer_types is None``) the wrapped
   call delegates immediately to the upstream implementation.

4. ``DeepseekV32Attention.__call__`` — wrap so layers where ``self.indexer is
   None`` take a shared-layer code path that:

   * Reads ``self._shared_topk_indices`` (set by the model-level wrapper).
     If non-None, applies the same sparse-mask block as the upstream full
     layer (so the layer attends to the same top-K KV slots the most-recent
     full layer selected — the REAP reuse contract).
   * Falls back to dense MLA attention only when the threaded topk is
     ``None`` (e.g. when the indexer would itself have returned ``None``
     because ``k.shape[2] <= self.index_topk``). This matches the natural
     upstream fallback for short sequences.
   * Skips the trailing
     ``cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))``
     line because ``cache[1]`` is never populated on a shared layer
     (``cache[1].keys`` is ``None`` and the ``mx.depends`` call would crash).

5. ``glm_moe_dsa.ModelArgs.from_dict`` — extend ``BaseModelArgs.from_dict``
   filtering so ``indexer_types`` is preserved on the dataclass instance.
   Upstream's filter (``inspect.signature(cls).parameters``) drops keys not
   declared on the dataclass; without this patch ``indexer_types`` is
   silently dropped and the gates above never fire.

What it does NOT change
-----------------------
* Configs **without** ``indexer_types`` (legitimate non-REAP DSv32 / GLM-4.6)
  take the upstream path verbatim. Every patched function delegates straight
  to the original on the no-types branch.
* No vendored copy of ``deepseek_v32.py``. Total surface area: this module
  plus one import + one call in ``vllm_mlx.model_runner``.

Concurrency note
----------------
The shared-topk thread runs through per-attention-instance attributes
(``self_attn._shared_topk_indices``). Concurrent forward passes on the same
model instance would race on these attributes. rapid-mlx's engine serializes
forward passes per model (one event loop, one model_runner per process), so
this is fine in production. The model.__call__ wrapper clears each shared
layer's attribute IMMEDIATELY before the layer runs so stale values from a
prior forward cannot leak into a fresh one.

When upstream mlx_lm lands first-class ``indexer_types`` support, delete this
module and the import in ``vllm_mlx.model_runner`` — nothing else needs to
change.
"""

from __future__ import annotations

import inspect
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_INSTALLED = False
_ALLOWED_MODES = ("full", "shared")

# Counters (test-observable) used by the regression suite to prove the
# gated code paths actually fire. Production code ignores them.
_SHARED_LAYER_FORWARD_COUNT = 0
_SHARED_LAYER_REUSE_COUNT = 0  # shared layers that consumed a non-None prior topk
_SHARED_LAYER_DENSE_FALLBACK_COUNT = 0  # shared layers that hit the dense fallback
_WARNED_DENSE_FALLBACK = False

# Saved references to the upstream callables; used to undo the patch in tests
# and (rarely) at runtime via ``uninstall_deepseek_v32_indexer_gate``. ``None``
# until the gate is installed.
_orig_attn_call: Any | None = None
_orig_decoder_init: Any | None = None
_orig_indexer_call: Any | None = None
_orig_model_call: Any | None = None
_orig_from_dict: Any | None = None


def _resolve_mode(types, layer_idx: int) -> str:
    """Return ``"full"`` or ``"shared"`` for ``layer_idx``.

    Backward-compatible: when ``types`` is missing/empty/over-the-end we fall
    back to ``"full"`` (= upstream behavior).
    """
    if types is None or not isinstance(types, (list, tuple)):
        return "full"
    if layer_idx is None or layer_idx < 0 or layer_idx >= len(types):
        return "full"
    mode = types[layer_idx]
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            f"indexer_types[{layer_idx}]={mode!r}; expected one of {_ALLOWED_MODES}"
        )
    return mode


def _validate_anchor(types, num_hidden_layers: int | None = None) -> None:
    """Fail fast on structurally invalid ``indexer_types`` configs.

    REAP-pruned configs always (a) start with a ``"full"`` layer and
    (b) have at least one ``"full"`` somewhere — every ``"shared"`` layer
    by definition reuses a previous full layer's indexer output, so
    a ``"shared"`` at index 0 has nothing to reuse and the config is
    structurally invalid. Also (c) the ``indexer_types`` length must match
    ``num_hidden_layers`` when both are known — a mismatched config would
    silently treat the over-the-end layers as ``"full"`` and then crash
    later with "missing weights" instead of a clear validation error
    (codex finding #2 on PR #967 round 2).

    Malformed non-``None`` values (wrong type, empty, invalid mode) are
    rejected up-front rather than silently treated as all-``"full"``
    (codex NIT on PR #967 round 4).
    """
    if types is None:
        return
    # Reject malformed non-None values up-front — previously this branch
    # silently returned and downstream code would treat the config as all-
    # "full", masking the misconfiguration until safetensors loading
    # eventually crashed with a less clear "missing weights" error.
    if not isinstance(types, (list, tuple)):
        raise ValueError(
            f"indexer_types must be a list or tuple when present, got "
            f"{type(types).__name__}"
        )
    if len(types) == 0:
        raise ValueError(
            "indexer_types is present but empty; omit the field entirely "
            "for non-REAP configs"
        )
    for i, m in enumerate(types):
        if m not in _ALLOWED_MODES:
            raise ValueError(
                f"indexer_types[{i}]={m!r}; expected one of {_ALLOWED_MODES}"
            )
    # (a) the very first entry must be "full" — there is no prior layer
    # for an index-0 "shared" to reuse from.
    if types[0] != "full":
        raise ValueError(
            f"indexer_types[0]={types[0]!r}; the first layer must be "
            "'full' (a 'shared' layer at index 0 has no previous full "
            "layer to reuse the indexer output from)."
        )
    # (b) at least one "full" anchor must exist. Redundant given (a),
    # but kept defensively so refactors of (a) don't silently weaken
    # the validator.
    if all(m == "shared" for m in types):
        raise ValueError(
            "indexer_types has no 'full' anchor — at least one layer must be "
            "'full' (a REAP-pruned config without any full layer is invalid; "
            "shared layers reuse the previous full layer's indexer output)."
        )
    # (c) length must equal num_hidden_layers when both are known.
    if num_hidden_layers is not None and len(types) != num_hidden_layers:
        raise ValueError(
            f"indexer_types has length {len(types)} but num_hidden_layers="
            f"{num_hidden_layers}; the lists must be the same length so each "
            "layer has an explicit 'full'/'shared' designation."
        )


def _shared_layer_attn_call(self, x, mask, cache):
    """Replica of upstream ``DeepseekV32Attention.__call__`` for layers
    where ``self.indexer is None``.

    Per the REAP reuse contract: if the model-level wrapper threaded a
    non-None ``topk_indices`` from the prior ``"full"`` layer onto
    ``self._shared_topk_indices``, apply the same sparse-mask block as
    the upstream full layer (so this shared layer attends to the same
    top-K KV slots). Otherwise (no prior topk — typically because the
    KV cache is shorter than ``self.index_topk`` so the prior indexer
    itself returned ``None``), fall through to dense MLA attention over
    the full KV window — the same natural fallback upstream takes.

    Always skips the ``cache[0].keys = mx.depends(cache[0].keys,
    (cache[1].keys, cache[1].values))`` line because ``cache[1]`` is
    never populated on a shared layer.

    Kept inline (instead of monkey-patching the indexer with a stub) so
    the failure traceback says ``_shared_layer_attn_call`` if anything
    goes wrong — easier to grep than a stack ending in upstream code.

    Bumps test-observable counters:

    * ``_SHARED_LAYER_FORWARD_COUNT`` — total shared-layer forwards.
    * ``_SHARED_LAYER_REUSE_COUNT`` — shared forwards that successfully
      consumed a non-None prior topk (proves the REAP reuse path fired).
    * ``_SHARED_LAYER_DENSE_FALLBACK_COUNT`` — shared forwards that hit
      the dense fallback (no prior topk available).
    """
    global \
        _SHARED_LAYER_FORWARD_COUNT, \
        _SHARED_LAYER_REUSE_COUNT, \
        _SHARED_LAYER_DENSE_FALLBACK_COUNT, \
        _WARNED_DENSE_FALLBACK

    _SHARED_LAYER_FORWARD_COUNT += 1
    # Consume + clear the threaded topk so a stale value from a prior
    # forward cannot leak into a fresh one.
    topk_indices = getattr(self, "_shared_topk_indices", None)
    self._shared_topk_indices = None

    if topk_indices is None:
        _SHARED_LAYER_DENSE_FALLBACK_COUNT += 1
        if not _WARNED_DENSE_FALLBACK:
            _WARNED_DENSE_FALLBACK = True
            logger.warning(
                "[deepseek_v32_indexer_gate] no prior-full-layer topk "
                "available on shared layer (layer_idx=%s); falling back to "
                "dense MLA attention. This is the same behavior the "
                "upstream Indexer takes when k.shape[2] <= index_topk (short "
                "KV cache), so it is typically benign.",
                getattr(self, "layer_idx", "?"),
            )
    else:
        _SHARED_LAYER_REUSE_COUNT += 1

    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, _ = x.shape
    qr = self.q_a_layernorm(self.q_a_proj(x))
    q = self.q_b_proj(qr)
    q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
    compressed_kv = self.kv_a_proj_with_mqa(x)
    compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
    k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
    kv_latent = self.kv_a_layernorm(compressed_kv)

    offset = cache[0].offset if cache is not None else 0
    q_pe = self.rope(q_pe, offset)
    k_pe = self.rope(k_pe, offset)

    kv_latent = mx.expand_dims(kv_latent, axis=1)
    if cache is not None:
        kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
    else:
        cache = [None] * 2

    # REAP reuse: apply the prior full layer's top-K KV selection to this
    # shared layer, exactly as the upstream full path does after running
    # its own indexer. The sparse-mask block is a verbatim copy of
    # ``DeepseekV32Attention.__call__`` in mlx_lm 0.31.3.
    if topk_indices is not None:
        if L == 1:
            idx = topk_indices[:, :, 0, :, None]
            kv_latent = mx.take_along_axis(
                kv_latent,
                mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                axis=2,
            )
            k_pe = mx.take_along_axis(
                k_pe,
                mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                axis=2,
            )
            if mask is not None:
                mask = mx.take_along_axis(mask, topk_indices, axis=-1)
        else:
            shape = list(topk_indices.shape)
            shape[-1] = kv_latent.shape[2]
            sparse_mask = mx.zeros(shape, dtype=mx.bool_)
            sparse_mask = mx.put_along_axis(
                sparse_mask, topk_indices, mx.array(True), axis=-1
            )
            if mask is not None:
                sparse_mask = sparse_mask & mask
            mask = sparse_mask

    # NB: NO ``cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys,
    # cache[1].values))`` — cache[1] is never populated on a shared layer
    # so ``cache[1].keys`` is ``None`` and ``mx.depends`` would crash.
    # Skipping is safe: that line is an upstream graph-pruning hint, not a
    # correctness requirement.

    pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
    if mask is not None:
        pe_scores = mx.where(
            mask,
            pe_scores,
            mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
        )

    if L == 1:
        q_nope = self.embed_q(q_nope)
        k = v = kv_latent
    else:
        k = self.embed_q(kv_latent, transpose=False)
        v = self.unembed_out(kv_latent)

    output = scaled_dot_product_attention(
        q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
    )
    if L == 1:
        output = self.unembed_out(output)

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


def install_deepseek_v32_indexer_gate() -> None:
    """Install the indexer gate. Idempotent and thread-safe."""
    global \
        _INSTALLED, \
        _orig_attn_call, \
        _orig_decoder_init, \
        _orig_indexer_call, \
        _orig_model_call, \
        _orig_from_dict

    with _LOCK:
        if _INSTALLED:
            return

        # Import lazily so this module is cheap to import even when mlx_lm
        # is unavailable (e.g. tests that patch import paths).
        try:
            from mlx_lm.models import deepseek_v32 as ds
            from mlx_lm.models import glm_moe_dsa as glm
        except ImportError:  # pragma: no cover - mlx_lm always present in our env
            logger.debug(
                "[deepseek_v32_indexer_gate] mlx_lm not importable; skipping install"
            )
            return

        # Stash the upstream originals on the ds module itself the FIRST
        # time we install. On a later install after a module reload, we
        # retrieve them from there instead of re-capturing the currently-
        # patched callables (which would self-reference and either
        # infinite-loop in the wrappers or leak originals through a
        # subsequent uninstall — codex finding #2 on PR #967 round 3).
        if not getattr(ds, "_RAPID_MLX_INDEXER_GATE_INSTALLED", False):
            ds._RAPID_MLX_ORIG_ATTN_CALL = ds.DeepseekV32Attention.__call__
            ds._RAPID_MLX_ORIG_DECODER_INIT = ds.DeepseekV32DecoderLayer.__init__
            ds._RAPID_MLX_ORIG_INDEXER_CALL = ds.Indexer.__call__
            ds._RAPID_MLX_ORIG_MODEL_CALL = ds.DeepseekV32Model.__call__
            ds._RAPID_MLX_ORIG_FROM_DICT = glm.ModelArgs.from_dict

        # Always copy from the stash into this module's globals so a later
        # ``uninstall_deepseek_v32_indexer_gate()`` from a freshly-loaded
        # module instance still restores the true upstream callables.
        _orig_attn_call = ds._RAPID_MLX_ORIG_ATTN_CALL
        _orig_decoder_init = ds._RAPID_MLX_ORIG_DECODER_INIT
        _orig_indexer_call = ds._RAPID_MLX_ORIG_INDEXER_CALL
        _orig_model_call = ds._RAPID_MLX_ORIG_MODEL_CALL
        _orig_from_dict = ds._RAPID_MLX_ORIG_FROM_DICT

        # Re-entry guard: if a previous module instance already installed
        # the gate on the upstream classes, mark this instance as
        # installed and return WITHOUT re-wrapping (otherwise the new
        # wrappers would call the old wrappers as "originals" → infinite
        # delegation). The captured originals above are the un-patched
        # callables, so uninstall still works.
        if getattr(ds, "_RAPID_MLX_INDEXER_GATE_INSTALLED", False):
            _INSTALLED = True
            return

        def _patched_decoder_init(self, config, layer_idx):
            _orig_decoder_init(self, config, layer_idx)
            types = getattr(config, "indexer_types", None)
            if types is None:
                return
            # Fail fast on structurally-invalid configs (no anchor, shared at
            # index 0, length mismatch).
            if layer_idx == 0:
                _validate_anchor(types, num_hidden_layers=config.num_hidden_layers)
            self.self_attn.layer_idx = layer_idx
            # Initialize the topk reuse slot; the model-level wrapper writes
            # the prior full layer's topk here before invoking the shared
            # layer's ``__call__``.
            self.self_attn._shared_topk_indices = None
            mode = _resolve_mode(types, layer_idx)
            if mode == "shared":
                # Assigning ``None`` to an nn.Module child removes it from
                # the parameter dict (see ``Module.__setattr__``), which is
                # exactly what we want so ``load_weights(strict=True)`` no
                # longer demands the absent ``indexer.*`` keys.
                self.self_attn.indexer = None

        def _patched_indexer_call(self, x, qr, mask, cache=None):
            # Cache the return tensor on the instance so the model-level
            # wrapper can pick it up after the full-layer forward and thread
            # it into the subsequent shared layers.
            result = _orig_indexer_call(self, x, qr, mask, cache)
            self._last_topk_indices = result
            return result

        def _patched_attn_call(self, x, mask=None, cache=None):
            if getattr(self, "indexer", None) is not None:
                return _orig_attn_call(self, x, mask, cache)
            return _shared_layer_attn_call(self, x, mask, cache)

        def _patched_model_call(self, x, cache=None):
            # Lazy import so the patch module doesn't pull mlx at import.
            import mlx.core as mx
            from mlx_lm.models.base import create_attention_mask

            # ``DeepseekV32Model`` (the inner module) doesn't store
            # ``config`` as an attribute. Pick the config off the first
            # decoder layer's attention — every layer's
            # ``self_attn.config`` is set by upstream
            # ``DeepseekV32Attention.__init__``. Bounds-check
            # ``start_idx`` so a pipeline/sharded layout with an empty
            # local slice (or out-of-range ``start_idx``) safely
            # delegates to upstream (codex finding #3 on PR #967 round 5).
            types = None
            if (
                self.layers
                and 0 <= self.start_idx < len(self.layers)
                and self.layers[self.start_idx] is not None
            ):
                cfg = getattr(self.layers[self.start_idx].self_attn, "config", None)
                if cfg is not None:
                    types = getattr(cfg, "indexer_types", None)

            if types is None:
                # Non-REAP path (or out-of-range slice) — delegate to
                # upstream verbatim.
                return _orig_model_call(self, x, cache)

            # Replica of upstream ``DeepseekV32Model.__call__`` with topk
            # threading inserted into the layer loop.
            h = self.embed_tokens(x)
            pipeline_rank = self.pipeline_rank
            pipeline_size = self.pipeline_size

            if cache is None:
                cache = [None] * self.num_layers
            mask = create_attention_mask(
                h, cache[0][0] if cache[0] else None, return_array=True
            )

            if pipeline_rank < pipeline_size - 1:
                h = mx.distributed.recv_like(h, (pipeline_rank + 1))

            # ``last_topk_indices`` is set fresh by each ``"full"``
            # layer's forward in THIS call and consumed by the
            # immediately-following ``"shared"`` layers. We deliberately
            # do NOT seed it from a prior forward's persisted
            # ``_last_topk_indices`` — those indices reference KV-slot
            # positions computed at a different query/cache length and
            # applying them to a fresh decode step would be a stale
            # selection (codex finding #1 on PR #967 round 5; supersedes
            # the round-3 seed attempt). When this rank's iteration
            # window starts with a ``"shared"`` layer (e.g. a pipeline-
            # parallel slice without a local full-layer anchor), those
            # leading shared layers correctly hit the dense fallback —
            # the same path the upstream Indexer takes when
            # ``k.shape[2] <= index_topk``. Cross-rank communication of
            # the current full layer's topk would be the
            # architecturally-correct remedy if needed in the future;
            # that is out of scope for the surgical D1 fix here.
            last_topk_indices = None

            for i in range(self.num_layers):
                layer = self.layers[self.start_idx + i]
                layer_idx = self.start_idx + i
                mode = _resolve_mode(types, layer_idx)
                if mode == "shared":
                    # Thread the most-recent full layer's topk into this
                    # shared layer's attention BEFORE it runs.
                    layer.self_attn._shared_topk_indices = last_topk_indices
                h = layer(h, mask, cache[i])
                if mode == "full":
                    # Pick up the topk this layer's indexer just produced
                    # (may be None if k.shape[2] <= index_topk).
                    indexer = getattr(layer.self_attn, "indexer", None)
                    if indexer is not None:
                        last_topk_indices = getattr(indexer, "_last_topk_indices", None)

            if pipeline_rank != 0:
                h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
                if cache[-1] is not None:
                    cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

            if pipeline_size > 1:
                h = mx.distributed.all_gather(h)[: h.shape[0]]

            return self.norm(h)

        @classmethod
        def _patched_from_dict(cls, params):
            sig = set(inspect.signature(cls).parameters)
            instance = cls(**{k: v for k, v in params.items() if k in sig})
            # Preserve the REAP extension field. Plain attribute assignment
            # is fine on a non-frozen dataclass; downstream code reads via
            # ``getattr(config, "indexer_types", None)``.
            #
            # Scope the validation strictly to the GLM-MoE-DSA model
            # type. Although ``glm.ModelArgs.from_dict`` is itself class-
            # specific (inherited subclasses of ``BaseModelArgs`` are not
            # affected by this assignment), gating on ``model_type``
            # provides a defense-in-depth signal that the validation
            # only applies to REAP-pruned DeepseekV32/GLM-MoE-DSA
            # configs and not to any future extension that happens to
            # reuse this dataclass for a different architecture (codex
            # finding #2 on PR #967 round 5).
            if "indexer_types" in params and params.get("model_type") == "glm_moe_dsa":
                # Validate at config-parse time so a malformed REAP config
                # fails clearly here rather than later in weight loading
                # (codex NIT, PR #967 round 4). ``_validate_anchor`` enforces
                # type, allowed values, length, anchor-at-0 and at-least-one-
                # anchor — see its docstring.
                _validate_anchor(
                    params["indexer_types"],
                    num_hidden_layers=params.get("num_hidden_layers"),
                )
            if "indexer_types" in params:
                instance.indexer_types = params["indexer_types"]
            return instance

        ds.DeepseekV32Attention.__call__ = _patched_attn_call
        ds.DeepseekV32DecoderLayer.__init__ = _patched_decoder_init
        ds.Indexer.__call__ = _patched_indexer_call
        ds.DeepseekV32Model.__call__ = _patched_model_call
        glm.ModelArgs.from_dict = _patched_from_dict
        ds._RAPID_MLX_INDEXER_GATE_INSTALLED = True
        _INSTALLED = True
        logger.debug("[deepseek_v32_indexer_gate] installed")


def uninstall_deepseek_v32_indexer_gate() -> None:
    """Undo the gate. Test-only; production code does not call this."""
    global \
        _INSTALLED, \
        _orig_attn_call, \
        _orig_decoder_init, \
        _orig_indexer_call, \
        _orig_model_call, \
        _orig_from_dict

    with _LOCK:
        if not _INSTALLED:
            return
        try:
            from mlx_lm.models import deepseek_v32 as ds
            from mlx_lm.models import glm_moe_dsa as glm
        except ImportError:  # pragma: no cover
            return
        if _orig_attn_call is not None:
            ds.DeepseekV32Attention.__call__ = _orig_attn_call
        if _orig_decoder_init is not None:
            ds.DeepseekV32DecoderLayer.__init__ = _orig_decoder_init
        if _orig_indexer_call is not None:
            ds.Indexer.__call__ = _orig_indexer_call
        if _orig_model_call is not None:
            ds.DeepseekV32Model.__call__ = _orig_model_call
        if _orig_from_dict is not None:
            glm.ModelArgs.from_dict = _orig_from_dict
        ds._RAPID_MLX_INDEXER_GATE_INSTALLED = False
        _INSTALLED = False
        _orig_attn_call = None
        _orig_decoder_init = None
        _orig_indexer_call = None
        _orig_model_call = None
        _orig_from_dict = None


def is_installed() -> bool:
    return _INSTALLED
