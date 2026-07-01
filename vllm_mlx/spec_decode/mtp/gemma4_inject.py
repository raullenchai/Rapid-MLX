# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP injection for Gemma 4 base models.

This module is the Gemma 4 counterpart to
:mod:`vllm_mlx.spec_decode.mtp.qwen3_5_inject`. It wires the same
four contract surfaces (``.mtp``, ``.mtp_forward``,
``.make_mtp_cache``, and the ``__call__(return_hidden, n_confirmed)``
kwargs) onto a loaded Gemma 4 model so
:func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step` can
drive it.

Sidecar architecture note
-------------------------

The community-shipped Gemma 4 MTP sidecar
(``Mia-AiLab/Gemmable-4-12B-MTP-GGUF``, ~98 k HF downloads at
release-time) is a **``gemma4-assistant`` architecture**, not the
Qwen3.5-style layered MTP head this inject was originally scoped for.
Inspected via ``gguf.GGUFReader``:

* ``general.architecture = 'gemma4-assistant'``.
* Its own transformer stack — ``block_count=4``,
  ``embedding_length=1024`` (distinct from Gemma 4 12B's base
  ``hidden_size=3840``).
* Cross-hidden-dim projection layers — ``nextn.pre_projection`` maps
  ``7680 -> 1024`` (concat of base ``hidden`` + base ``embed`` at
  ``3840*2``, projected down to assistant space) and
  ``nextn.post_projection`` maps ``1024 -> 3840`` (assistant space
  projected back to base ``hidden`` for the shared ``lm_head``).
* No K/V weights on the assistant's own attention (``attn_k`` /
  ``attn_v`` / ``attn_k_norm`` / ``attn_v_norm`` absent from every
  block). ``gemma4-assistant.attention.shared_kv_layers = 4`` = all
  layers borrow K/V from the corresponding base layer.
* Runs UP TO 4 draft tokens per verify step (metadata
  ``nextn_predict_layers = 4``).

This is a **fundamentally different draft-model shape** from Qwen3.5's
MTP head (which is a single decoder layer at the base's hidden_size,
sitting atop the base's last hidden state and reusing base's
``embed_tokens`` + ``lm_head`` directly). Wiring the ``gemma4-assistant``
sidecar into ``mtp_generate_step`` requires:

1. A dedicated ``AssistantModel`` class with its own hidden dim.
2. A cross-model K/V bridge (the base's per-layer K/V cache tap must
   feed the assistant's ``shared_kv`` arg).
3. Extending the generator to project through the pre/post
   projection layers at ``mtp_forward`` boundaries.

Landing all three cleanly is a follow-up PR (tracked in the PR body
of the PR that lands this file). This module's role in the *current*
PR is to:

* Own the routing surface for ``model_type in {gemma4, gemma4_unified}``
  so :func:`vllm_mlx.spec_decode.mtp.dispatch.dispatch_mtp_inject`
  has a place to send the call.
* Pass the **wiring probe** (all four contract surfaces attach)
  under ``allow_random_init=True`` — the CI unit tests exercise
  every model_type against the shared wiring contract.
* **REFUSE any real ``gemma4-assistant`` sidecar** (fail-closed on
  ``mtp_sidecar=<real_file>``) so a production rapid-mlx serve on
  Gemma 4 cannot silently ship a broken/random-init MTP head. The
  refusal path emits a specific "architectural mismatch, redesign
  needed" log message that the operator can grep for.

Once the follow-up PR lands the ``AssistantModel`` class, the
sidecar-loading branch of :func:`inject_mtp_support` below flips from
refusal to actual load — no external contract change.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Tensor-key markers that identify a ``gemma4-assistant`` sidecar
# (produced by ``scripts/convert_gemma4_mtp_gguf.py`` from the
# Mia-AiLab GGUF). Presence of ANY of these keys means we're looking
# at the draft-model architecture, which this inject cannot yet drive.
_ASSISTANT_SIDECAR_MARKERS: frozenset[str] = frozenset(
    {
        "mtp.pre_projection.weight",
        "mtp.post_projection.weight",
        "assistant.pre_projection.weight",
        "assistant.post_projection.weight",
        # Raw GGUF-style names (if operator forgot to run the converter).
        "nextn.pre_projection.weight",
        "nextn.post_projection.weight",
    }
)


def _resolve_inner_text_model(model: Any) -> Any:
    """Return the ``Gemma4TextModel``-like instance the patch targets.

    Gemma 4's ``mlx_lm.models.gemma4.Model`` wraps a
    ``gemma4_text.Model`` under ``language_model``. That inner
    ``gemma4_text.Model`` in turn wraps the ``Gemma4TextModel`` (the
    actual backbone with ``embed_tokens`` / ``layers`` / ``norm``)
    under ``.model``. Rapid-MLX's own ``Gemma4TextWrapper`` (for the
    ``gemma4_unified`` model_type not yet in mlx-lm) exposes the same
    ``.language_model`` / ``.language_model.model`` shape.

    Three shapes are accepted:

    * Outer VLM-style ``Model`` (``gemma4`` model_type) with
      ``.language_model``.
    * Rapid-MLX's ``Gemma4TextWrapper`` (``gemma4_unified`` model_type)
      — same ``.language_model`` surface.
    * The inner ``gemma4_text.Model`` itself (test path — matches the
      Qwen3.5 helper contract: tests bypass the heavy VLM wrapper by
      constructing the inner model directly).
    """
    # Case 1: VLM wrapper — text model lives under ``.language_model``.
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "args") and hasattr(lm, "model"):
        return lm

    # Case 2: Already the inner ``gemma4_text.Model`` (or a test shell).
    if hasattr(model, "model") and hasattr(model, "args"):
        return model

    return None


def _detect_base_quantization(inner: Any) -> dict | None:
    """Detect the base model's ``(bits, group_size)`` — same helper as Qwen3.5.

    Walks the inner model looking for a ``QuantizedLinear``. Gemma 4's
    quantized weights land on the same ``self_attn.q_proj`` +
    ``embed_tokens`` surfaces the Qwen3.5 helper checks, so the same
    walk works here.
    """
    try:
        from mlx.nn import QuantizedEmbedding, QuantizedLinear
    except ImportError:  # pragma: no cover — mlx.nn always available
        return None

    backbone = getattr(inner, "model", None)
    if backbone is None:
        return None

    for layer in getattr(backbone, "layers", []):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
            qp = layer.self_attn.q_proj
            if isinstance(qp, QuantizedLinear):
                return {
                    "bits": int(qp.bits),
                    "group_size": int(qp.group_size),
                }

    embed = getattr(backbone, "embed_tokens", None)
    if isinstance(embed, QuantizedEmbedding):
        return {
            "bits": int(embed.bits),
            "group_size": int(embed.group_size),
        }

    return None


def _resolve_sidecar_file(mtp_sidecar: str | Path) -> Path | None:
    """Resolve a sidecar reference to a concrete safetensors file.

    Accepts a file path (``*.safetensors``), a directory containing
    ``model.safetensors`` / ``model-mtp.safetensors``, or an HF Hub
    repo id (``mlx-community/gemma-4-12b-mtp-4bit``). Mirrors the
    resolver in :mod:`.qwen3_5_inject` so operators see the same
    accepted shapes across families.
    """
    if mtp_sidecar is None:
        return None

    path = Path(mtp_sidecar)
    if path.is_file():
        return path
    if path.is_dir():
        return _find_mtp_weights_file(path)

    # Treat as HF repo id.
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=str(mtp_sidecar))
        return _find_mtp_weights_file(Path(local))
    except Exception as exc:  # pragma: no cover — network failure path
        logger.warning(
            "[mtp.inject.gemma4] could not resolve sidecar %r: %s",
            mtp_sidecar,
            exc,
        )
        return None


def _find_mtp_weights_file(sidecar_dir: Path) -> Path | None:
    candidates = (
        sidecar_dir / "model-mtp.safetensors",
        sidecar_dir / "model.safetensors",
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def _looks_like_assistant_sidecar(weights: dict[str, Any]) -> bool:
    """Fingerprint the ``gemma4-assistant`` architecture by tensor names.

    The Mia-AiLab GGUF (and its MLX conversion produced by
    ``scripts/convert_gemma4_mtp_gguf.py``) ships pre/post projection
    layers that the current Qwen3.5-style MTP head lacks. Any of
    those keys → this is an assistant-model sidecar we can't load
    yet.
    """
    keys = set(weights.keys())
    return bool(keys & _ASSISTANT_SIDECAR_MARKERS)


def _build_scaffold_mtp_module(args: Any, num_layers: int):
    """Build the wiring-probe MTP head for Gemma 4.

    This is a **placeholder** module built to satisfy the four MTP
    contract surfaces during CI wiring probes. It follows the
    Qwen3.5-style layout (single-layer decoder + pre_fc norms + fc
    concat) so the same load/quantize/attach machinery works. It
    does NOT reproduce the ``gemma4-assistant`` architecture — the
    module is only meant to prove that ``inject_mtp_support`` can
    attach surfaces to a Gemma 4 base model.

    Args:
        args: The inner ``gemma4_text`` model's ``args`` (a
            ``ModelArgs`` dataclass with ``hidden_size``,
            ``rms_norm_eps``, ``intermediate_size``,
            ``num_attention_heads``, ``head_dim``,
            ``num_key_value_heads``, and RoPE fields).
        num_layers: How many MTP layers to construct. Currently
            always ``1`` — matches the Qwen3.5 vendor PR contract.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.gemma4_text import MLP

    if num_layers < 1:
        raise ValueError(
            f"_build_scaffold_mtp_module requires num_layers >= 1; got {num_layers}"
        )

    class _ScaffoldDecoderLayer(nn.Module):
        """One transformer layer — Gemma 4 shapes, unshared attention.

        This is a bare-bones self-attn + MLP block sized for the
        scaffold. We intentionally do NOT reuse
        ``gemma4_text.DecoderLayer`` because that class carries
        Gemma 4's kv-shared-layers / per-layer-input-gate / MoE
        machinery which the scaffold has no reason to exercise. The
        Q/K/V/O projections match ``num_attention_heads * head_dim``
        the same way ``gemma4_text.Attention`` does.
        """

        def __init__(self, layer_args):
            super().__init__()
            hidden = layer_args.hidden_size
            n_heads = layer_args.num_attention_heads
            head_dim = getattr(layer_args, "head_dim", hidden // n_heads)
            n_kv_heads = layer_args.num_key_value_heads
            self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
            self.input_layernorm = nn.RMSNorm(hidden, eps=layer_args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                hidden, eps=layer_args.rms_norm_eps
            )
            self.mlp = MLP(layer_args, layer_idx=0)
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.head_dim = head_dim

        def __call__(self, x, mask=None, cache=None):
            # Wiring-probe forward — proves the module can be called
            # without shape errors. Production draft quality is not a
            # goal here; the real Gemma 4 MTP forward path lands in
            # the follow-up "AssistantModel" PR.
            B, L, _ = x.shape
            h = self.input_layernorm(x)
            q = self.q_proj(h).reshape(B, L, self.n_heads, self.head_dim)
            k = self.k_proj(h).reshape(B, L, self.n_kv_heads, self.head_dim)
            v = self.v_proj(h).reshape(B, L, self.n_kv_heads, self.head_dim)
            # Repeat KV to match Q heads (GQA / MQA).
            if self.n_kv_heads != self.n_heads:
                repeats = self.n_heads // self.n_kv_heads
                k = mx.repeat(k, repeats, axis=2)
                v = mx.repeat(v, repeats, axis=2)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            scale = 1.0 / (self.head_dim**0.5)
            attn = (q @ k.transpose(0, 1, 3, 2)) * scale
            if mask is not None:
                attn = attn + mask
            attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(x.dtype)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
            h = x + self.post_attention_layernorm(self.o_proj(out))
            return h + self.mlp(h)

    class _ScaffoldMTPModule(nn.Module):
        """Wiring-only MTP head — matches Qwen3.5 head's PUBLIC surface.

        The parameter names (``pre_fc_norm_hidden`` /
        ``pre_fc_norm_embedding`` / ``fc`` / ``layers`` / ``norm``)
        are chosen to line up with ``_MTPModule`` in ``head.py`` so
        a hand-written scaffold sidecar (used in the wiring tests)
        can round-trip through ``mx.save_safetensors`` /
        ``mtp.load_weights``.
        """

        def __init__(self, mod_args, n_layers):
            super().__init__()
            hidden = mod_args.hidden_size
            eps = mod_args.rms_norm_eps
            self.pre_fc_norm_hidden = nn.RMSNorm(hidden, eps=eps)
            self.pre_fc_norm_embedding = nn.RMSNorm(hidden, eps=eps)
            self.fc = nn.Linear(hidden * 2, hidden, bias=False)
            self.layers = [_ScaffoldDecoderLayer(mod_args) for _ in range(n_layers)]
            self.norm = nn.RMSNorm(hidden, eps=eps)

        def __call__(
            self,
            hidden_states: mx.array,
            next_token_ids: mx.array,
            embed_tokens: nn.Embedding,
            cache: Any | None = None,
        ) -> mx.array:
            embeds = embed_tokens(next_token_ids)
            e = self.pre_fc_norm_embedding(embeds)
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))

            if cache is None:
                cache = [None] * len(self.layers)

            mask = create_attention_mask(fused, cache[0])
            for layer, c in zip(self.layers, cache):
                fused = layer(fused, mask, c)

            return self.norm(fused)

    return _ScaffoldMTPModule(args, num_layers)


def _num_mtp_layers_from_args(inner: Any, outer: Any) -> int:
    """Read ``mtp_num_hidden_layers`` from the dataclass or outer wrapper.

    Copy of the resolution logic from ``qwen3_5_inject`` — the field
    may live on the inner ``args`` (test path), the outer wrapper's
    ``text_config`` dict (real runtime path), or nowhere at all
    (checkpoint doesn't ship MTP).
    """
    args = inner.args
    n = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
    if n >= 1:
        return n

    outer_args = getattr(outer, "args", None)
    text_config = getattr(outer_args, "text_config", None) or {}
    if isinstance(text_config, dict):
        n = int(text_config.get("mtp_num_hidden_layers", 0) or 0)
    if n >= 1:
        try:
            object.__setattr__(args, "mtp_num_hidden_layers", n)
        except (TypeError, AttributeError):  # pragma: no cover
            pass
    return n


def inject_mtp_support(
    model: Any,
    mtp_sidecar: str | Path | None = None,
    *,
    allow_random_init: bool = False,
    _accept_scaffold_sidecar_for_tests: bool = False,
) -> bool:
    """Inject MTP support into a loaded Gemma 4 (or ``gemma4_unified``) model.

    Args:
        model: A model loaded via ``mlx_lm.load()`` (either the outer
            VLM wrapper or Rapid-MLX's ``Gemma4TextWrapper``, or the
            inner ``gemma4_text.Model`` directly for tests).
        mtp_sidecar: Optional reference to the MTP sidecar. Accepts
            a repo id (``mlx-community/gemma-4-12b-mtp-4bit``), a
            directory path, or a direct ``.safetensors`` path.
        allow_random_init: Test-only opt-in — permits ``mtp_sidecar
            =None`` and ships a random-init scaffold MTP head. Match
            the ``qwen3_5_inject`` default of ``False`` — production
            callers MUST pass a sidecar.
        _accept_scaffold_sidecar_for_tests: **PRIVATE / TEST-ONLY**.
            When ``True``, the sidecar path is allowed to load
            weights that match the scaffold module's parameter tree.
            Codex round-2 flagged that the scaffold layer does not
            carry a working ``KVCache`` update path, so any accepted
            production sidecar would break multi-token spec decode.
            The default of ``False`` REFUSES every sidecar for the
            duration of PR-3 — the follow-up ``AssistantModel`` PR
            replaces this gate with the real consumer. Tests set
            this to True to exercise the sidecar resolver + coverage
            check without pretending the scaffold can drive
            production traffic.

    Returns:
        ``True`` if the four contract surfaces attached. ``False``
        under any of:

        * Not a Gemma 4-shaped model (no ``.language_model`` or
          ``.model`` / ``.args``).
        * Config has no ``mtp_num_hidden_layers >= 1``.
        * ``mtp_sidecar=None`` with ``allow_random_init=False`` (the
          fail-closed default codex round-5 installed on the Qwen3.5
          side).
        * Sidecar file is missing / unresolvable / not a valid
          safetensors.
        * **The sidecar fingerprints as a ``gemma4-assistant``
          architecture** (see :func:`_looks_like_assistant_sidecar`
          / the module docstring). That case is REFUSED with a
          specific "architecture-not-yet-supported" log — a
          follow-up PR lands the ``AssistantModel`` code path.
        * A non-assistant sidecar was passed AND
          ``_accept_scaffold_sidecar_for_tests`` is ``False``. The
          scaffold module cannot drive production multi-token spec
          decode (no KVCache path in its forward); refusing here
          prevents the codex round-2 broken-cache regression.

    Never raises — every failure returns ``False`` and leaves the
    model unmodified so the caller can fall back to non-spec-decode.
    """
    import mlx.core as mx
    import mlx.nn as nn

    inner = _resolve_inner_text_model(model)
    if inner is None:
        logger.warning(
            "[mtp.inject.gemma4] model %s has neither .language_model nor "
            "(.model + .args); skipping MTP injection.",
            type(model).__name__,
        )
        return False

    num_mtp_layers = _num_mtp_layers_from_args(inner, model)
    if num_mtp_layers < 1:
        logger.info(
            "[mtp.inject.gemma4] config has no mtp_num_hidden_layers; skipping MTP "
            "injection. Stock mlx-community/gemma-4-12b-* checkpoints ship "
            "without this field — the operator must layer it in via a config "
            "override or use a converted MTP-enabled checkpoint."
        )
        return False

    # --- Codex round-7 blocking fix: EARLY refusal gates ------------------
    # Move the fail-closed refusals BEFORE the (expensive) MTP module
    # construction + quantization. A production Gemma 4 boot that hits
    # any of the refuse paths below previously allocated a full
    # scaffold head (num_mtp_layers × hidden_size × ~4× MLP fan-out
    # weights, then quantized copy) just to throw it away on return
    # False. Cheap parameter-only checks first; expensive allocation
    # after.
    if mtp_sidecar is None and not allow_random_init:
        logger.warning(
            "[mtp.inject.gemma4] inject_mtp_support called without "
            "mtp_sidecar and allow_random_init=False; refusing to ship "
            "a random-init MTP head. Pass a converted sidecar via "
            "scripts/convert_gemma4_mtp_gguf.py, or set "
            "allow_random_init=True for unit-test wiring probes."
        )
        return False

    # Pre-flight the sidecar (resolve + architecture-fingerprint +
    # scaffold-refusal) BEFORE constructing the scaffold module. If any
    # of these fail we return False without having allocated the
    # scaffold head. We stash the parsed weights in a local so the
    # loading branch below can skip the redundant load.
    resolved_weights_file: Path | None = None
    resolved_mtp_weights: dict[str, Any] | None = None
    if mtp_sidecar is not None:
        resolved_weights_file = _resolve_sidecar_file(mtp_sidecar)
        if resolved_weights_file is None:
            logger.warning(
                "[mtp.inject.gemma4] sidecar %r could not be resolved to a "
                "safetensors file; skipping MTP injection.",
                mtp_sidecar,
            )
            return False

        try:
            _raw_preflight = mx.load(str(resolved_weights_file))
        except Exception as exc:
            logger.warning(
                "[mtp.inject.gemma4] sidecar %s failed to load via mx.load: %s. "
                "Refusing to inject; caller can fall back to non-spec-decode.",
                resolved_weights_file.name,
                exc,
            )
            return False
        _mtp_weights_preflight = {
            (k.removeprefix("mtp.") if k.startswith("mtp.") else k): v
            for k, v in _raw_preflight.items()
        }

        if _looks_like_assistant_sidecar(
            _raw_preflight
        ) or _looks_like_assistant_sidecar(_mtp_weights_preflight):
            logger.warning(
                "[mtp.inject.gemma4] sidecar %s fingerprints as a "
                "'gemma4-assistant' architecture (Mia-AiLab-style draft "
                "model with pre/post projection layers). This inject "
                "currently only carries the wiring-probe scaffold — the "
                "AssistantModel code path (cross-hidden-dim projections + "
                "base-model K/V bridge) lands in a follow-up PR. Refusing "
                "to inject rather than ship a broken/random-init draft. "
                "Track: `gemma4-assistant MTP` follow-up in the PR body.",
                resolved_weights_file.name,
            )
            return False

        if not _accept_scaffold_sidecar_for_tests:
            logger.warning(
                "[mtp.inject.gemma4] Refusing to load sidecar %s in "
                "production. The Gemma 4 inject in this PR is scaffold-"
                "only — its decoder layer does not update KVCache, so "
                "any accepted sidecar would break multi-token spec "
                "decode. Track: `gemma4-assistant MTP` follow-up in "
                "the PR body. For CI wiring probes pass "
                "allow_random_init=True (no sidecar).",
                resolved_weights_file.name,
            )
            return False

        resolved_mtp_weights = _mtp_weights_preflight

    # --- Step 1: Build the scaffold MTP module -----------------------------
    args = inner.args
    mtp = _build_scaffold_mtp_module(args, num_mtp_layers)
    logger.info(
        "[mtp.inject.gemma4] Built SCAFFOLD MTP module (%d layer(s), "
        "hidden_size=%d). NOTE: this is the wiring-probe head — the true "
        "gemma4-assistant sidecar architecture (pre/post projections, "
        "independent hidden dim) is a follow-up PR.",
        num_mtp_layers,
        getattr(args, "hidden_size", -1),
    )

    # --- Step 2: Quantize to match base ------------------------------------
    quant_info = _detect_base_quantization(inner)
    if quant_info is not None:
        nn.quantize(
            mtp,
            group_size=quant_info["group_size"],
            bits=quant_info["bits"],
        )
        logger.info(
            "[mtp.inject.gemma4] Quantized MTP: %d-bit, group_size=%d",
            quant_info["bits"],
            quant_info["group_size"],
        )

    # --- Step 3: Load sidecar weights --------------------------------------
    # All refusal gates ran in the pre-flight block above (before the
    # scaffold allocation) — codex round-7 blocking fix. Here we only
    # do the coverage-check + weight load using the already-parsed
    # ``resolved_mtp_weights``, or fall through to the random-init
    # branch when no sidecar was passed.
    if resolved_mtp_weights is not None:
        assert resolved_weights_file is not None  # narrow for the type checker
        from mlx.utils import tree_flatten

        expected_keys = {k for k, _ in tree_flatten(mtp.parameters())}
        loaded_keys = set(resolved_mtp_weights.keys())
        missing = expected_keys - loaded_keys
        if missing:
            logger.warning(
                "[mtp.inject.gemma4] sidecar %s is missing %d required MTP "
                "tensor(s); refusing to ship a partially-random-init head. "
                "Missing keys (first 8): %s",
                resolved_weights_file.name,
                len(missing),
                sorted(missing)[:8],
            )
            return False

        mtp.load_weights(list(resolved_mtp_weights.items()), strict=False)
        mx.eval(mtp.parameters())
        extra = loaded_keys - expected_keys
        logger.info(
            "[mtp.inject.gemma4] Loaded %d/%d expected MTP weight tensors from %s%s",
            len(expected_keys),
            len(expected_keys),
            resolved_weights_file.name,
            f" (+{len(extra)} extra sidecar key(s) ignored)" if extra else "",
        )
    else:
        # ``mtp_sidecar is None and allow_random_init=True`` — the
        # ``allow_random_init=False`` case was already refused in the
        # early-refusal block above.
        mx.eval(mtp.parameters())
        logger.warning(
            "[mtp.inject.gemma4] inject_mtp_support called with "
            "allow_random_init=True — SCAFFOLD MTP head retains RANDOM init "
            "weights (accept rate ~0%%, and the head shape doesn't match "
            "the gemma4-assistant sidecar anyway). Test-only path."
        )

    # --- Step 4: Global ArraysCache patch (parity with qwen3_5) -----------
    # Gemma 4 doesn't ship GatedDeltaNet layers, so the linear-attention
    # rollback machinery is a no-op here — but the ``rollback_state``
    # slot is what the generator's ``_rollback_draft`` reads from, and
    # installing it universally keeps that path uniform across
    # architectures.
    from .cache_patch import patch_arrays_cache_rollback_state

    patch_arrays_cache_rollback_state()

    # --- Step 5: Attach + monkey-patch the inner class --------------------
    inner.mtp = mtp
    original_class = type(inner)

    class _Gemma4WithMTP(original_class):  # type: ignore[valid-type, misc]
        """``gemma4_text.Model`` + MTP surfaces (SCAFFOLD).

        See module docstring for the difference between this scaffold
        wiring and the eventual ``gemma4-assistant`` code path.
        """

        def __call__(  # type: ignore[override]
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            per_layer_inputs=None,
            return_hidden: bool = False,
            n_confirmed: int = 0,
        ):
            # Codex round-3: neither ``return_hidden`` nor ``n_confirmed``
            # can be honestly implemented in the scaffold — we'd have to
            # tap the pre-norm hidden state (requires forking the
            # backbone's __call__) and thread n_confirmed through the
            # base model's linear-attention rollback (Gemma 4 has none;
            # for Qwen3.5 this is the GatedDeltaNet rollback path). Both
            # semantics land in the follow-up ``AssistantModel`` PR. If a
            # caller reaches here with either flag set, RAISE loudly
            # instead of silently returning fake data — that's the codex
            # round-3 fix (was: returned all-zero hidden; hidden was
            # then fed to mtp_forward and produced garbage drafts).
            if return_hidden or n_confirmed:
                raise NotImplementedError(
                    "[mtp.inject.gemma4] scaffold __call__ received "
                    f"return_hidden={return_hidden} / n_confirmed={n_confirmed}, "
                    "which the scaffold cannot honestly implement. This is "
                    "wiring-only surface parity — the semantically-correct "
                    "path lands in the follow-up AssistantModel PR. Only the "
                    "bare forward (return_hidden=False, n_confirmed=0) is "
                    "supported here."
                )
            return original_class.__call__(
                self,
                inputs,
                cache=cache,
                input_embeddings=input_embeddings,
                per_layer_inputs=per_layer_inputs,
            )

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache,
        ):
            """Scaffold placeholder — raises to prevent silent misuse.

            Codex round-3 blocking fix: the scaffold cannot produce
            production-correct drafts (no KVCache update in the
            decoder layer, no valid hidden-state tap in ``__call__``).
            Instead of returning wrong output, raise so any caller
            that reaches here gets a loud, actionable error. The real
            forward lands in the follow-up AssistantModel PR.
            """
            raise NotImplementedError(
                "[mtp.inject.gemma4] scaffold mtp_forward is not "
                "implemented — the scaffold module attaches surfaces so "
                "the wiring probe can validate the injection contract, "
                "but the production-quality forward (using pre/post "
                "projections + base-model K/V bridge) lands in the "
                "follow-up AssistantModel PR. If you see this error at "
                "runtime, `rapid-mlx serve --spec-decode mtp` should NOT "
                "be enabled for Gemma 4 aliases yet."
            )

        def make_mtp_cache(self):
            """One ``KVCache`` per scaffold MTP layer (all full attention).

            Returned cache slot is real (matches the KVCache contract
            the generator expects). Since ``mtp_forward`` raises
            immediately, the cache is never actually populated — the
            slot exists purely to satisfy the wiring probe's
            ``callable(make_mtp_cache)`` check.
            """
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.mtp.layers]

    inner.__class__ = _Gemma4WithMTP
    # Codex round-4 blocking fix: stamp the scaffold marker on the
    # inner model so ``validate_mtp_support`` can honestly report the
    # model as NOT production-ready. Without this, a caller through
    # the dispatcher sees ``inject=True`` + ``validate=True`` and
    # trusts the model — but every actual invocation raises.
    inner._mtp_is_scaffold = True

    # Codex round-6 blocking fix: if the caller passed the OUTER
    # Gemma 4 wrapper (mlx_lm.load()'s standard return shape —
    # ``gemma4.Model`` wrapping ``.language_model``, or Rapid-MLX's
    # ``Gemma4TextWrapper`` for ``gemma4_unified``), the four
    # contract surfaces (``.mtp``, ``.mtp_forward``,
    # ``.make_mtp_cache``, and the ``__call__(return_hidden, n_confirmed)``
    # extended signature) currently exist only on ``inner``. That
    # means a caller who does ``model.mtp_forward(...)`` on the
    # object they passed in gets ``AttributeError``, even though
    # ``inject_mtp_support`` returned ``True``. Delegate the three
    # attribute surfaces from the outer wrapper down to inner so
    # the advertised contract holds. ``__call__`` is deliberately
    # NOT delegated — the outer VLM wrapper's ``__call__`` signature
    # differs (it accepts pixels, tokens, etc.), and the vendored
    # ``mtp_generate_step`` in ``generator.py`` only ever invokes
    # ``make_mtp_cache`` / ``mtp_forward`` directly on a base
    # text-model shape.
    if model is not inner:
        import types as _types

        model.mtp = inner.mtp

        def _delegate_mtp_forward(_self, hidden_states, next_token_ids, mtp_cache):
            return inner.mtp_forward(hidden_states, next_token_ids, mtp_cache)

        def _delegate_make_mtp_cache(_self):
            return inner.make_mtp_cache()

        model.mtp_forward = _types.MethodType(_delegate_mtp_forward, model)
        model.make_mtp_cache = _types.MethodType(_delegate_make_mtp_cache, model)
        # Mirror the scaffold marker on the outer so any validator
        # that resolves the outer directly (rather than via
        # ``_resolve_inner_text_model``) still sees a scaffold model.
        model._mtp_is_scaffold = True

    logger.info(
        "[mtp.inject.gemma4] Patched %s with MTP surfaces "
        "(return_hidden, n_confirmed, mtp_forward, make_mtp_cache). "
        "SCAFFOLD wiring only — validate_mtp_support will return False "
        "so production callers do NOT trust this model as MTP-capable.",
        original_class.__name__,
    )
    return True


def validate_mtp_support(model: Any) -> bool:
    """Verify that :func:`inject_mtp_support` succeeded on ``model``.

    Same shape as :func:`vllm_mlx.spec_decode.mtp.qwen3_5_inject.validate_mtp_support`
    — checks ``.mtp``, ``.mtp_forward``, ``.make_mtp_cache``, and the
    ``__call__`` signature.

    **Scaffold guard (codex round-4)**: if the model carries the
    ``_mtp_is_scaffold`` marker set by :func:`inject_mtp_support`, this
    validator returns ``False``. The scaffold's methods raise
    ``NotImplementedError`` when invoked, so from any production
    caller's perspective the model is NOT MTP-capable — even though
    the four surfaces attach. This matches the dispatcher contract:
    ``dispatch_mtp_validate`` seeing ``False`` here tells the bench
    (or CLI) that Gemma 4 spec-decode is not yet ready.
    """
    import inspect

    inner = _resolve_inner_text_model(model)
    if inner is None:
        return False

    if getattr(inner, "mtp", None) is None:
        logger.warning("[mtp.validate.gemma4] model.mtp is missing.")
        return False
    if not callable(getattr(inner, "mtp_forward", None)):
        logger.warning("[mtp.validate.gemma4] model.mtp_forward is missing.")
        return False
    if not callable(getattr(inner, "make_mtp_cache", None)):
        logger.warning("[mtp.validate.gemma4] model.make_mtp_cache is missing.")
        return False
    sig = inspect.signature(type(inner).__call__)
    if "return_hidden" not in sig.parameters:
        logger.warning(
            "[mtp.validate.gemma4] model.__call__ does not accept return_hidden."
        )
        return False
    if "n_confirmed" not in sig.parameters:
        logger.warning(
            "[mtp.validate.gemma4] model.__call__ does not accept n_confirmed."
        )
        return False
    if getattr(inner, "_mtp_is_scaffold", False):
        logger.warning(
            "[mtp.validate.gemma4] model carries the scaffold marker — "
            "the four surfaces attach but every invocation of "
            "mtp_forward / __call__(return_hidden=True) raises "
            "NotImplementedError. Reporting False so production callers "
            "do not enable spec-decode. The real AssistantModel-based "
            "inject lands in the follow-up PR."
        )
        return False
    return True
