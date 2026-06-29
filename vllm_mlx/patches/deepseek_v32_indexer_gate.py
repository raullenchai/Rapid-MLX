# SPDX-License-Identifier: Apache-2.0
"""Surgical indexer gate for REAP-pruned DeepseekV32 models.

Background
----------
Upstream ``mlx_lm.models.deepseek_v32.DeepseekV32Attention.__init__`` builds
``self.indexer = Indexer(config)`` on **every** layer unconditionally. REAP-
pruned variants (e.g. ``mlx-community/pipenetwork-GLM-5.2-REAP50-MLX-4bit``)
publish a per-layer ``indexer_types: List[str]`` field where each entry is
either ``"full"`` (this layer owns Indexer weights in the safetensors) or
``"shared"`` (this layer has **no** Indexer weights and reuses the previous
full layer's indexer output at inference time). On such configs every
``"shared"`` layer's Indexer parameters are missing from the safetensors and
``mlx_lm.load`` aborts with ``Missing N parameters: ...indexer...``.

What this module patches
------------------------
1. ``DeepseekV32DecoderLayer.__init__`` — after the upstream init runs, if
   ``config.indexer_types`` is present and the current layer is ``"shared"``,
   drop ``self_attn.indexer`` (``= None``) so it is removed from the model's
   parameter tree (see :class:`mlx.nn.Module.__setattr__`: assigning a
   non-array/dict/list/tuple deregisters the child from the parameter dict).
   The ``layer_idx`` is also threaded onto ``self_attn`` for diagnostics.

2. ``DeepseekV32Attention.__call__`` — wrap so layers where ``self.indexer is
   None`` take a shared-layer code path that (a) skips the Indexer call and
   the sparse-mask block, and (b) skips the trailing
   ``cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))``
   line. On a shared layer ``cache[1]`` is never populated so ``cache[1].keys``
   is ``None`` and the ``mx.depends`` call would crash. The shared path falls
   through to dense MLA attention over the full KV window.

3. ``glm_moe_dsa.ModelArgs.from_dict`` — extend ``BaseModelArgs.from_dict``
   filtering so ``indexer_types`` is preserved on the dataclass instance.
   Upstream's filter (``inspect.signature(cls).parameters``) drops keys not
   declared on the dataclass; without this patch ``indexer_types`` is
   silently dropped and the gate above never fires.

What it does NOT change
-----------------------
* Configs **without** ``indexer_types`` (legitimate non-REAP DSv32 / GLM-4.6)
  take the upstream path verbatim. The patched ``__init__`` returns early,
  the patched ``__call__`` immediately delegates to the original, and
  ``from_dict`` matches upstream's behavior when ``indexer_types`` is absent.
* No vendored copy of ``deepseek_v32.py``. Total surface area: this module
  plus one import + one call in ``vllm_mlx.model_runner``.

When upstream mlx_lm lands ``indexer_types`` support, delete this module and
the import in ``vllm_mlx.model_runner`` — nothing else needs to change.
"""

from __future__ import annotations

import inspect
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_INSTALLED = False
_ALLOWED_MODES = ("full", "shared")

# Saved references to the upstream callables; used to undo the patch in tests
# and (rarely) at runtime via ``uninstall_deepseek_v32_indexer_gate``. ``None``
# until the gate is installed.
_orig_attn_call: Optional[Any] = None
_orig_decoder_init: Optional[Any] = None
_orig_from_dict: Optional[Any] = None


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
            f"indexer_types[{layer_idx}]={mode!r}; "
            f"expected one of {_ALLOWED_MODES}"
        )
    return mode


def _validate_anchor(types) -> None:
    """Fail fast on configs with no ``"full"`` anchor.

    A real REAP-pruned model always has at least one ``"full"`` layer at the
    front; an all-``"shared"`` config would have no Indexer weights at all
    and could never be inferenced.
    """
    if types is None:
        return
    if not isinstance(types, (list, tuple)) or len(types) == 0:
        return
    if all(m == "shared" for m in types):
        raise ValueError(
            "indexer_types has no 'full' anchor — at least one layer must be "
            "'full' (a REAP-pruned config without any full layer is invalid; "
            "shared layers reuse the previous full layer's indexer output)."
        )


def _shared_layer_attn_call(self, x, mask, cache):  # pragma: no cover - exercised by smoke test
    """Replica of upstream ``DeepseekV32Attention.__call__`` minus the indexer.

    The shared path simply runs dense MLA attention over the full KV window
    — the same fallback the upstream code naturally takes when
    ``topk_indices is None`` (e.g. on the very first decode step), but here
    we also skip the ``mx.depends`` on ``cache[1]`` since the indexer cache
    is never populated on shared layers and ``cache[1].keys`` is ``None``.

    Kept inline (instead of monkey-patching the indexer with a stub) so the
    failure traceback says ``_shared_layer_attn_call`` if anything goes
    wrong — easier to grep than a stack ending in upstream code.
    """
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

    # No indexer => no sparse mask, no ``cache[0].keys = mx.depends(...)`` on
    # the (unpopulated) ``cache[1]``. Dense MLA attention over the full KV
    # window is the natural fallback the upstream code already takes when
    # ``topk_indices is None``.
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
    global _INSTALLED, _orig_attn_call, _orig_decoder_init, _orig_from_dict

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

        # Detect double-install across module reloads (e.g. pytest restart).
        if getattr(ds, "_RAPID_MLX_INDEXER_GATE_INSTALLED", False):
            _INSTALLED = True
            return

        _orig_attn_call = ds.DeepseekV32Attention.__call__
        _orig_decoder_init = ds.DeepseekV32DecoderLayer.__init__
        _orig_from_dict = glm.ModelArgs.from_dict

        def _patched_decoder_init(self, config, layer_idx):
            _orig_decoder_init(self, config, layer_idx)
            types = getattr(config, "indexer_types", None)
            if types is None:
                return
            # Fail fast on the all-shared edge case (no valid REAP config has
            # this shape; bail before load_weights spits a confusing error).
            if layer_idx == 0:
                _validate_anchor(types)
            self.self_attn.layer_idx = layer_idx
            mode = _resolve_mode(types, layer_idx)
            if mode == "shared":
                # Assigning ``None`` to an nn.Module child removes it from
                # the parameter dict (see ``Module.__setattr__``), which is
                # exactly what we want so ``load_weights(strict=True)`` no
                # longer demands the absent ``indexer.*`` keys.
                self.self_attn.indexer = None

        def _patched_attn_call(self, x, mask=None, cache=None):
            if getattr(self, "indexer", None) is not None:
                return _orig_attn_call(self, x, mask, cache)
            return _shared_layer_attn_call(self, x, mask, cache)

        @classmethod
        def _patched_from_dict(cls, params):
            sig = set(inspect.signature(cls).parameters)
            instance = cls(**{k: v for k, v in params.items() if k in sig})
            # Preserve the REAP extension field. Plain attribute assignment
            # is fine on a non-frozen dataclass; downstream code reads via
            # ``getattr(config, "indexer_types", None)``.
            if "indexer_types" in params:
                instance.indexer_types = params["indexer_types"]
            return instance

        ds.DeepseekV32Attention.__call__ = _patched_attn_call
        ds.DeepseekV32DecoderLayer.__init__ = _patched_decoder_init
        glm.ModelArgs.from_dict = _patched_from_dict
        ds._RAPID_MLX_INDEXER_GATE_INSTALLED = True
        _INSTALLED = True
        logger.debug("[deepseek_v32_indexer_gate] installed")


def uninstall_deepseek_v32_indexer_gate() -> None:
    """Undo the gate. Test-only; production code does not call this."""
    global _INSTALLED, _orig_attn_call, _orig_decoder_init, _orig_from_dict

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
        if _orig_from_dict is not None:
            glm.ModelArgs.from_dict = _orig_from_dict
        ds._RAPID_MLX_INDEXER_GATE_INSTALLED = False
        _INSTALLED = False
        _orig_attn_call = None
        _orig_decoder_init = None
        _orig_from_dict = None


def is_installed() -> bool:
    return _INSTALLED
