# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP injection for Gemma 4 via Google's official assistant models.

Google published a family of MTP drafters for Gemma 4 on 2026-05-05 under
Apache 2.0, one per base size:

* ``google/gemma-4-E2B-it-assistant``
* ``google/gemma-4-E4B-it-assistant``
* ``google/gemma-4-12B-it-assistant``      ← paired with ``mlx-community/gemma-4-12B-it-4bit``
* ``google/gemma-4-26B-A4B-it-assistant``
* ``google/gemma-4-31B-it-assistant``

Each assistant is a standalone 4-layer transformer that reuses the target
model's K/V cache. The safetensors ship with weights at:

  ``model.embed_tokens.weight``   (vocab, hidden)          — own token embed
  ``model.layers.{i}.self_attn.q_proj.weight``             — Q only (no K/V!)
  ``model.layers.{i}.self_attn.q_norm.weight``
  ``model.layers.{i}.self_attn.o_proj.weight``
  ``model.layers.{i}.mlp.{gate,up,down}_proj.weight``
  ``model.layers.{i}.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight``
  ``model.layers.{i}.layer_scalar``                        — per-layer residual scale
  ``model.norm.weight``
  ``pre_projection.weight``       (hidden, 2 × backbone_hidden)
  ``post_projection.weight``      (backbone_hidden, hidden)

Key architectural properties (verified against Google's official
``config.json``, e.g. the 12B assistant at
<https://huggingface.co/google/gemma-4-12B-it-assistant/blob/main/config.json>):

* ``num_hidden_layers = 4``
* ``hidden_size = 1024`` (the drafter's OWN hidden dim)
* ``backbone_hidden_size`` = the target's hidden dim (3840 for 12B target,
  5376 for 31B target). Read from the assistant's top-level config, not
  from ``text_config``.
* ``layer_types = [sliding, sliding, sliding, full_attention]`` — matches
  the last 4 layers of the base's sliding/full pattern.
* ``num_kv_shared_layers = 4`` — every drafter layer borrows K/V from a
  corresponding target layer. Concretely, drafter layer ``i`` reads target
  layer ``(target_num_layers - 4) + i``, so its ``layer_type`` matches.
* ``tie_word_embeddings = True`` — the drafter's own ``embed_tokens.weight``
  serves as its tied lm_head.
* ``num_centroids = 2048`` and ``centroid_intermediate_top_k = 32`` appear
  in the top-level config but NO centroid tensors ship in the safetensors,
  so those are NOT applied in this MVP consumer. See PR body for the
  post-MVP tuning note.

Forward pass (MVP interpretation)
---------------------------------

The drafter chains the target's activations into its own 4-layer block:

    fused_7680 = concat(target_last_hidden_3840, target_embed(next_tok)_3840)
    h_1024     = pre_projection(fused_7680)
    for layer_i, target_kv_i in zip(drafter.layers, target_kv_tail):
        h_1024 = layer_i(h_1024, shared_kv=target_kv_i.state, offset=target_kv_i.offset)
    h_1024   = drafter.norm(h_1024)
    logits   = drafter.embed_tokens.as_linear(h_1024)   # tied

The post_projection (1024 → backbone_hidden) is retained on the module
so weights round-trip, and can be consumed by a future extension that
chains the drafter's hidden back into the target space; MVP does not use
it on the emit path.

The target's forward is untouched. The drafter's Q is computed with its
own weights; K and V come DIRECTLY from the target's KVCache/RotatingKVCache
state (already RoPE-applied by the target). The drafter's ``mtp_cache``
returned by :meth:`make_mtp_cache` is a set of empty ``KVCache`` slots
that satisfy the generator's ``.trim(1)`` / ``quantize_cache_fn`` contract
but are never written into — the sharing is one-way and reads only.

This module deliberately does NOT touch ``qwen3_5_inject.py`` or
``cache_patch.py``. The Qwen3.5 lane is untouched by this PR.

Honest MVP caveats (locked to the PR body's ``post-MVP TODO`` list)
------------------------------------------------------------------

* The N > 1 (cache-commit) branch of ``_step_mtp`` in the generator is
  handled by processing only the LAST position; the drafter's own hidden
  chaining across draft-token iterations is not implemented in this MVP.
* The centroid embedder (``num_centroids`` / ``centroid_intermediate_top_k``)
  is not applied — the assistant's own tied embed is used as the lm_head.
* No final-logit softcapping on the drafter (base Gemma 4 applies it, but
  the assistant config sets ``final_logit_softcapping = null``). At
  ``temp=0`` this preserves argmax equality; at ``temp>0`` acceptance
  ratios may run lower but the lossless contract still holds via
  residual sampling in the generator.

These caveats are called out in the PR body and are the first things
the operator smoke test will validate against Google's published
reference implementation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Assistant model_type strings Google ships in the checkpoint's
# top-level config. Both variants land here (2B/4B/12B use
# ``gemma4_assistant``; 26B / 31B unified variants use
# ``gemma4_unified_assistant``).
_ASSISTANT_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "gemma4_assistant",
        "gemma4_unified_assistant",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_inner_text_model(model: Any) -> Any:
    """Return the ``gemma4_text.Model``-like inner text model.

    Three shapes accepted (mirrors qwen3_5_inject's helper):

    * The outer VLM-style ``Model`` (``gemma4`` / ``gemma4_unified``)
      whose ``.language_model`` field is the inner text model.
    * The inner ``gemma4_text.Model`` itself (also carries
      ``.language_model = None`` in some builds — check ``.args`` +
      ``.model``).
    * A fake shell with just ``.args`` + ``.model`` (test path).
    """
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "args") and hasattr(lm, "model"):
        return lm
    if hasattr(model, "model") and hasattr(model, "args"):
        return model
    return None


def _resolve_sidecar_dir(mtp_sidecar: str | Path) -> Path | None:
    """Resolve a sidecar reference to a directory holding config + safetensors.

    Accepts:
      * A directory path — must contain ``config.json`` + ``model.safetensors``.
      * An HF repo id like ``google/gemma-4-12B-it-assistant`` — download via
        ``snapshot_download`` and return the local snapshot dir.

    Returns ``None`` if the reference cannot be resolved. Never raises.
    """
    if mtp_sidecar is None:
        return None

    path = Path(mtp_sidecar)
    if path.is_dir():
        return path
    if path.is_file():
        # Historical: some operators pass a direct safetensors path.
        # Google's checkpoints ship as a repo with config.json — we need
        # both files. Treat "file" as the safetensors and use the parent
        # as the dir; fail if config.json is absent.
        parent = path.parent
        if (parent / "config.json").exists():
            return parent
        logger.warning(
            "[mtp.inject.gemma4] sidecar file %s has no sibling config.json; "
            "cannot resolve as an assistant checkpoint.",
            path,
        )
        return None

    # Codex round-9 nit fix: only attempt snapshot_download for
    # strings that LOOK LIKE HF repo ids (``owner/name`` shape). A
    # typo'd local path like ``/tmp/missing-assistant`` should fail
    # as a local-path error immediately instead of spending a network
    # round-trip against a nonexistent HF repo. Detection: string
    # must contain exactly one non-leading '/', no path separators
    # that indicate an absolute or nested filesystem path.
    ref = str(mtp_sidecar)
    is_hf_shape = (
        "/" in ref
        and not ref.startswith("/")
        and not ref.startswith("./")
        and not ref.startswith("../")
        and ref.count("/") == 1
        and not ref.endswith("/")
    )
    if not is_hf_shape:
        logger.warning(
            "[mtp.inject.gemma4] sidecar %r is neither an existing local path "
            "nor an HF-repo-id-shape (``owner/name``). Refusing to attempt "
            "snapshot_download.",
            ref,
        )
        return None

    # Treat as HF repo id.
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=str(mtp_sidecar))
        return Path(local)
    except Exception as exc:  # pragma: no cover — network path
        logger.warning(
            "[mtp.inject.gemma4] could not resolve sidecar repo %r: %s",
            mtp_sidecar,
            exc,
        )
        return None


def _load_assistant_config(sidecar_dir: Path) -> dict | None:
    """Read + validate ``config.json`` from an assistant checkpoint dir.

    Returns the parsed dict on success; ``None`` on any parse / schema
    problem so the caller falls back to no-op.
    """
    cfg_path = sidecar_dir / "config.json"
    if not cfg_path.exists():
        logger.warning(
            "[mtp.inject.gemma4] assistant dir %s has no config.json.",
            sidecar_dir,
        )
        return None
    try:
        return json.loads(cfg_path.read_text())
    except Exception as exc:  # pragma: no cover — malformed JSON
        logger.warning(
            "[mtp.inject.gemma4] could not parse config.json in %s: %s",
            sidecar_dir,
            exc,
        )
        return None


def _find_safetensors(sidecar_dir: Path) -> Path | None:
    """Return the sole ``.safetensors`` file in the assistant dir.

    Google's official assistant release ships a single ``model.safetensors``
    (806 MB for the 12B assistant). If the directory contains multiple
    ``*.safetensors`` (accidental sharding, mixed downloads), refuse
    to guess — sharded / indexed loading is out of scope for the MVP
    consumer and picking the "first" file could silently load the
    wrong tensors.
    """
    # Codex round-10 nit fix: enumerate ALL .safetensors first and
    # refuse when there is more than one — including when one of them
    # happens to be named ``model.safetensors``. Previously the
    # well-known-name shortcut returned early and ignored any
    # ``model-00001-of-00003.safetensors`` shards that would have
    # combined into a different weight tree.
    matches = sorted(sidecar_dir.glob("*.safetensors"))
    if len(matches) > 1:
        logger.warning(
            "[mtp.inject.gemma4] %s contains %d .safetensors files: %s. "
            "Refusing to guess — sharded / multi-file assistant loading "
            "is not supported.",
            sidecar_dir,
            len(matches),
            [p.name for p in matches[:5]],
        )
        return None
    if len(matches) == 1:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# AssistantModel — matches Google's checkpoint layout
# ---------------------------------------------------------------------------


def _build_assistant_model_args(
    assistant_cfg: dict,
    target_backbone_hidden: int,
) -> Any:
    """Assemble a ``gemma4_text.ModelArgs`` for the assistant's inner block.

    Google's ``config.json`` puts model architecture fields under
    ``text_config``. We surface the ones we need onto a
    ``gemma4_text.ModelArgs`` so we can reuse mlx-lm's ``DecoderLayer`` —
    which already implements Gemma-sandwich norms, layer_scalar,
    per-layer-input gating (unused here), and the ``shared_kv`` path we
    need for cross-model K/V reuse.
    """
    from mlx_lm.models.gemma4_text import ModelArgs

    tc = assistant_cfg.get("text_config", {}) or {}

    # Layer types come pre-listed on the assistant config. Length must
    # match num_hidden_layers.
    n_layers = int(tc.get("num_hidden_layers", 4))
    layer_types = list(tc.get("layer_types") or [])
    if len(layer_types) != n_layers:
        # Fail closed on the schema mismatch — a bad list would land in
        # ModelArgs and later crash inside DecoderLayer's Attention with
        # an opaque IndexError. Better to refuse at inject time.
        logger.warning(
            "[mtp.inject.gemma4] layer_types has %d entries but num_hidden_layers=%d; "
            "refusing to build assistant args (schema mismatch).",
            len(layer_types),
            n_layers,
        )
        return None

    args = ModelArgs(
        model_type=str(tc.get("model_type", "gemma4_unified_text")),
        hidden_size=int(tc.get("hidden_size", 1024)),
        num_hidden_layers=n_layers,
        intermediate_size=int(tc.get("intermediate_size", 8192)),
        num_attention_heads=int(tc.get("num_attention_heads", 16)),
        head_dim=int(tc.get("head_dim", 256)),
        global_head_dim=int(tc.get("global_head_dim", 512)),
        rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
        vocab_size=int(tc.get("vocab_size", 262144)),
        vocab_size_per_layer_input=int(tc.get("vocab_size_per_layer_input", 0)),
        num_key_value_heads=int(tc.get("num_key_value_heads", 8)),
        num_global_key_value_heads=tc.get("num_global_key_value_heads"),
        # num_kv_shared_layers=N causes DecoderLayer's Attention to skip
        # k_proj / v_proj / k_norm / v_norm — matches Google's checkpoint
        # (Q-only weights, K/V taps from target).
        num_kv_shared_layers=int(tc.get("num_kv_shared_layers", n_layers)),
        pad_token_id=int(tc.get("pad_token_id", 0)),
        hidden_size_per_layer_input=int(tc.get("hidden_size_per_layer_input", 0)),
        rope_parameters=tc.get("rope_parameters"),
        sliding_window=int(tc.get("sliding_window", 1024)),
        # ``sliding_window_pattern`` unused because we set layer_types
        # explicitly; still required by ModelArgs.__post_init__ default
        # fallback → pick sane default.
        sliding_window_pattern=int(tc.get("sliding_window_pattern", 6)),
        max_position_embeddings=int(tc.get("max_position_embeddings", 262144)),
        attention_k_eq_v=bool(tc.get("attention_k_eq_v", False)),
        final_logit_softcapping=tc.get("final_logit_softcapping"),
        use_double_wide_mlp=bool(tc.get("use_double_wide_mlp", False)),
        enable_moe_block=bool(tc.get("enable_moe_block", False)),
        tie_word_embeddings=bool(tc.get("tie_word_embeddings", True)),
        layer_types=layer_types if layer_types else None,
    )
    # Backbone-hidden lives on the OUTER config, not text_config.
    # Preferring the config value, but if absent, fall back to the
    # target model's hidden_size (passed in).
    bb = int(
        assistant_cfg.get("backbone_hidden_size")
        or tc.get("backbone_hidden_size")
        or target_backbone_hidden
    )
    if bb != target_backbone_hidden:
        logger.warning(
            "[mtp.inject.gemma4] assistant backbone_hidden_size=%d does not match "
            "target hidden_size=%d; drafter cross-projection shapes will not fit. "
            "Refusing to inject.",
            bb,
            target_backbone_hidden,
        )
        return None
    # Stash on the dataclass for downstream use (frozen=False by default).
    try:
        object.__setattr__(args, "backbone_hidden_size", bb)
    except (TypeError, AttributeError):  # pragma: no cover
        pass
    return args


def _build_assistant_model(args: Any, backbone_hidden_size: int):
    """Instantiate the AssistantModel matching Google's safetensors layout.

    Weight key contract (after :meth:`load_weights`):

      ``model.embed_tokens.weight``
      ``model.layers.{i}.*``            — via mlx_lm gemma4_text.DecoderLayer
      ``model.norm.weight``
      ``pre_projection.weight``
      ``post_projection.weight``

    All under a top-level ``nn.Module`` so ``mx.load(safetensors)`` +
    ``model.load_weights(...)`` round-trips cleanly.
    """
    import mlx.nn as nn
    from mlx_lm.models.gemma4_text import DecoderLayer

    class _AssistantBackbone(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
            self.layers = [
                DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    class _AssistantModel(nn.Module):
        """Google gemma-4-*-it-assistant — 4-layer drafter with cross-KV."""

        def __init__(self, args, backbone_hidden_size):
            super().__init__()
            self.args = args
            self.backbone_hidden_size = backbone_hidden_size
            self.model = _AssistantBackbone(args)
            # pre_projection: [2 × backbone_hidden, hidden] — concat of
            # target's last hidden + target's embed(next_tok), both in
            # target's backbone_hidden space.
            self.pre_projection = nn.Linear(
                2 * backbone_hidden_size,
                args.hidden_size,
                bias=False,
            )
            # post_projection: [hidden, backbone_hidden] — retained so
            # weights round-trip; not on the MVP emit path (see module
            # docstring for the chaining follow-up).
            self.post_projection = nn.Linear(
                args.hidden_size,
                backbone_hidden_size,
                bias=False,
            )

    return _AssistantModel(args, backbone_hidden_size)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def inject_mtp_support(
    model: Any,
    mtp_sidecar: str | Path | None = None,
    *,
    allow_random_init: bool = False,
) -> bool:
    """Attach the Google Gemma 4 assistant drafter to ``model``.

    Args:
        model: A Gemma 4 target loaded via ``mlx_lm.load()``. Either the
            outer VLM wrapper (``model_type='gemma4'`` /
            ``'gemma4_unified'``) or the inner ``gemma4_text.Model``
            (test path).
        mtp_sidecar: Reference to Google's assistant checkpoint. Accepts:

              * A local directory path (``~/…/gemma-4-12B-it-assistant``)
                containing ``config.json`` + ``model.safetensors``.
              * An HF Hub repo id (``google/gemma-4-12B-it-assistant``) —
                downloaded via ``snapshot_download`` to the HF cache.

        allow_random_init: Test-only opt-in. When ``True``, permit
            ``mtp_sidecar=None`` and ship an assistant module with random
            init weights. Same fail-closed default as ``qwen3_5_inject``.

    Returns:
        ``True`` when the four contract surfaces attach and — under the
        real-sidecar path — weights load without missing tensors.
        ``False`` on any refusal (never raises).
    """
    import mlx.core as mx

    inner = _resolve_inner_text_model(model)
    if inner is None:
        logger.warning(
            "[mtp.inject.gemma4] model %s has no .language_model or (.model + .args); "
            "skipping.",
            type(model).__name__,
        )
        return False

    # Refuse the ``no sidecar + no allow_random_init`` production
    # default — matches the qwen3_5 fail-closed shape.
    if mtp_sidecar is None and not allow_random_init:
        logger.warning(
            "[mtp.inject.gemma4] inject_mtp_support called without mtp_sidecar and "
            "allow_random_init=False; refusing to ship a random-init drafter. "
            "Pass mtp_sidecar='google/gemma-4-12B-it-assistant' (or equivalent) "
            "for production use."
        )
        return False

    target_hidden = int(getattr(inner.args, "hidden_size", 0) or 0)
    if target_hidden <= 0:
        logger.warning(
            "[mtp.inject.gemma4] target hidden_size=%d unresolved; skipping.",
            target_hidden,
        )
        return False

    # ── Resolve sidecar (skipped under allow_random_init) ────────────
    assistant_cfg: dict | None = None
    weights_file: Path | None = None
    if mtp_sidecar is not None:
        sidecar_dir = _resolve_sidecar_dir(mtp_sidecar)
        if sidecar_dir is None:
            return False

        assistant_cfg = _load_assistant_config(sidecar_dir)
        if assistant_cfg is None:
            return False

        # Architecture guard: config must announce one of the known
        # assistant model_types so we don't accidentally load a random
        # Gemma 4 CHECKPOINT (base target, wrong shape).
        cfg_type = str(assistant_cfg.get("model_type", ""))
        if cfg_type not in _ASSISTANT_MODEL_TYPES:
            logger.warning(
                "[mtp.inject.gemma4] sidecar %s has model_type=%r; expected one of %s. "
                "Refusing to inject.",
                sidecar_dir,
                cfg_type,
                sorted(_ASSISTANT_MODEL_TYPES),
            )
            return False

        weights_file = _find_safetensors(sidecar_dir)
        if weights_file is None:
            logger.warning(
                "[mtp.inject.gemma4] no safetensors found under %s.", sidecar_dir
            )
            return False

    # ── Build args + assistant ───────────────────────────────────────
    if assistant_cfg is not None:
        args = _build_assistant_model_args(assistant_cfg, target_hidden)
        if args is None:
            return False
        backbone_hidden = int(getattr(args, "backbone_hidden_size", target_hidden))
        # Codex round-7 blocking fix: vocab_size mismatch would let a
        # sidecar with the correct backbone_hidden_size but a
        # different tokenizer vocab inject cleanly and later return
        # logits over the WRONG vocabulary. Compare against the
        # target's inner vocab_size and refuse before loading weights.
        # ``tie_word_embeddings`` on the assistant means its
        # ``embed_tokens`` doubles as the tied ``lm_head`` — the
        # emitted logits are (B, N, assistant.vocab_size), so a
        # mismatch here is fatal for both drafter attention (embed
        # lookup on next_token_ids from the target's tokenizer) and
        # drafter output (logits reindexed into the wrong tokens).
        target_vocab = int(getattr(inner.args, "vocab_size", 0) or 0)
        assistant_vocab = int(getattr(args, "vocab_size", 0) or 0)
        if target_vocab > 0 and assistant_vocab > 0 and target_vocab != assistant_vocab:
            logger.warning(
                "[mtp.inject.gemma4] vocab_size mismatch: target=%d assistant=%d. "
                "The assistant tokenizer differs from the target's — refusing to "
                "inject; drafter logits would be indexed into the wrong vocab.",
                target_vocab,
                assistant_vocab,
            )
            return False

        # Codex round-10 blocking fix: verify that the target's TAIL
        # layer_types match the assistant's layer_types before wiring
        # up cross-model K/V. The drafter maps its layer i to target
        # layer ``(target_num_layers - n_assistant_layers) + i`` and
        # reads that layer's K/V state, applying the SAME sliding /
        # full attention treatment defined in the drafter's own
        # ``layer_types``. If the target variant has a different tail
        # pattern (e.g. all-full, or [sliding, full, sliding, full]),
        # the drafter would use the wrong RoPE / mask semantics
        # against a cache that was populated under a different
        # attention type. Refuse on mismatch.
        assistant_layer_types = list(getattr(args, "layer_types", []) or [])
        target_layer_types = list(getattr(inner.args, "layer_types", []) or [])
        n_assistant = len(assistant_layer_types)
        if (
            assistant_layer_types
            and target_layer_types
            and len(target_layer_types) >= n_assistant
        ):
            target_tail = target_layer_types[-n_assistant:]
            if target_tail != assistant_layer_types:
                logger.warning(
                    "[mtp.inject.gemma4] target tail layer_types %r do not match "
                    "assistant layer_types %r; refusing to inject cross-KV under "
                    "mismatched attention semantics.",
                    target_tail,
                    assistant_layer_types,
                )
                return False
        elif not target_layer_types and assistant_layer_types:
            logger.warning(
                "[mtp.inject.gemma4] target has no layer_types published on "
                "``inner.args``; cannot verify tail layer_types match the "
                "assistant's %r. Refusing to inject rather than silently ship "
                "a semantics-mismatched drafter.",
                assistant_layer_types,
            )
            return False
    else:
        # allow_random_init path — synthesize a minimal config sized to
        # the target so the drafter surfaces still attach on tests.
        from mlx_lm.models.gemma4_text import ModelArgs

        # Codex round-9 blocking fix: derive vocab_size from the
        # target's ``inner.args.vocab_size`` instead of hard-coding
        # 128. Under random-init the drafter's embed_tokens can never
        # be correct (it's random by design), but the SHAPE must at
        # least match the target's tokenizer so that any test which
        # calls ``embed_tokens(next_token_ids)`` doesn't OOB-lookup on
        # a token id from the target's real vocab. Refuse if the
        # target's vocab_size cannot be resolved.
        random_vocab = int(getattr(inner.args, "vocab_size", 0) or 0)
        if random_vocab <= 0:
            logger.warning(
                "[mtp.inject.gemma4] allow_random_init=True but target has no "
                "resolvable vocab_size; refusing to build random drafter.",
            )
            return False

        args = ModelArgs(
            model_type="gemma4_unified_text",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            head_dim=16,
            global_head_dim=32,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            rms_norm_eps=1e-6,
            vocab_size=random_vocab,
            vocab_size_per_layer_input=0,
            num_kv_shared_layers=2,
            hidden_size_per_layer_input=0,
            sliding_window=64,
            sliding_window_pattern=2,
            max_position_embeddings=128,
            final_logit_softcapping=None,
            enable_moe_block=False,
            use_double_wide_mlp=False,
            tie_word_embeddings=True,
            layer_types=["sliding_attention", "full_attention"],
            attention_k_eq_v=False,
        )
        backbone_hidden = target_hidden

    try:
        assistant = _build_assistant_model(args, backbone_hidden)
    except Exception as exc:
        logger.warning(
            "[mtp.inject.gemma4] failed to instantiate AssistantModel: %s", exc
        )
        return False

    # ── Load weights ─────────────────────────────────────────────────
    if weights_file is not None:
        try:
            raw = mx.load(str(weights_file))
        except Exception as exc:
            logger.warning(
                "[mtp.inject.gemma4] mx.load(%s) failed: %s. Refusing to inject.",
                weights_file,
                exc,
            )
            return False
        from mlx.utils import tree_flatten

        expected_pairs = list(tree_flatten(assistant.parameters()))
        expected_shapes = {k: tuple(v.shape) for k, v in expected_pairs}
        expected_keys = set(expected_shapes)
        loaded_keys = set(raw.keys())
        missing = expected_keys - loaded_keys
        if missing:
            logger.warning(
                "[mtp.inject.gemma4] assistant weights missing %d tensor(s) — "
                "first 8: %s. Refusing to inject with a partially-random head.",
                len(missing),
                sorted(missing)[:8],
            )
            return False
        # Codex round-6 blocking fix (defense-in-depth): MLX's
        # ``load_weights`` accepts mismatched shapes silently (lazy
        # array assignment doesn't validate), so a sidecar with all
        # the right KEYS but a wrong-shape tensor would sneak through
        # and later crash inside ``mtp_forward``. Explicitly compare
        # shapes up front and refuse before any monkey-patching.
        shape_mismatches: list[tuple[str, tuple, tuple]] = []
        for k in expected_keys:
            got = tuple(raw[k].shape)
            want = expected_shapes[k]
            if got != want:
                shape_mismatches.append((k, got, want))
        if shape_mismatches:
            logger.warning(
                "[mtp.inject.gemma4] %d tensor(s) in %s have shapes "
                "incompatible with the AssistantModel. First 4: %s. "
                "Refusing to inject.",
                len(shape_mismatches),
                weights_file,
                [(k, g, w) for (k, g, w) in shape_mismatches[:4]],
            )
            return False
        # ``strict=False`` tolerates EXTRA keys that our AssistantModel
        # doesn't consume (metadata / centroid tables that Google
        # publishes but this MVP consumer doesn't load). We already
        # asserted no REQUIRED keys are missing, so ``strict=False``
        # cannot mask a partial-load bug. Extras are elevated from
        # INFO to WARNING per codex round-4 so the operator sees them.
        #
        # Codex round-6 blocking fix: wrap the load + eval in a
        # try/except — a sidecar carrying all the expected KEYS but
        # tensors with INCOMPATIBLE SHAPES (e.g. a checkpoint from a
        # different assistant size that happens to share the layout
        # names) would raise here otherwise, breaking the
        # documented "fail closed on any refusal" contract.
        try:
            assistant.load_weights(list(raw.items()), strict=False)
            mx.eval(assistant.parameters())
        except Exception as exc:
            logger.warning(
                "[mtp.inject.gemma4] load_weights / eval failed on %s: %s. "
                "Refusing to inject with an inconsistent drafter.",
                weights_file,
                exc,
            )
            return False
        extra = loaded_keys - expected_keys
        if extra:
            logger.warning(
                "[mtp.inject.gemma4] sidecar %s carries %d tensor(s) the MVP "
                "consumer does not load (first 8: %s). Loading continues; "
                "review the follow-up TODOs (centroid embedder / draft chain) "
                "before treating these as unused.",
                weights_file.name,
                len(extra),
                sorted(extra)[:8],
            )
        logger.info(
            "[mtp.inject.gemma4] Loaded %d/%d assistant tensors from %s",
            len(expected_keys),
            len(expected_keys),
            weights_file.name,
        )
    else:
        mx.eval(assistant.parameters())
        logger.warning(
            "[mtp.inject.gemma4] allow_random_init=True — assistant drafter has "
            "RANDOM init weights (accept rate ~0%%). Test-only wiring probe."
        )

    # ── Install rollback_state (no-op for Gemma 4 — no GatedDeltaNet — but
    #    keeps generator._rollback_draft's ``hasattr(c, 'rollback_state')``
    #    walk uniform across families).
    from .cache_patch import patch_arrays_cache_rollback_state

    patch_arrays_cache_rollback_state()

    # ── Prepare the patched class + delegation ───────────────────────
    # Class swap and attribute delegation are staged locally, then
    # committed transactionally at the very end: on any failure the
    # target model is left bit-for-bit unmodified (codex round-5
    # blocking fix — a mid-inject exception would previously leave
    # ``inner.__class__`` swapped + ``inner.mtp`` set even though
    # ``inject_mtp_support`` returned False).
    original_class = type(inner)
    _target_num_layers = len(getattr(inner.model, "layers", []) or [])
    _n_assistant_layers = len(assistant.model.layers)

    class _Gemma4WithMTP(original_class):  # type: ignore[valid-type, misc]
        """Target model + MTP surfaces for the Google assistant drafter.

        Codex round-9 blocking-fix documentation: the shared-K/V path
        used by ``mtp_forward`` reads the target's cache list — one
        list per request. Consequently this drafter is a strict
        single-request path today; a scheduler that folds multiple
        requests into a single (B, T, H) tensor must NOT dispatch
        through the MTP fast-path. The class attribute
        :attr:`mtp_max_batch_size` (=1) is set here so a future
        scheduler can read it as a static gate. ``mtp_forward`` itself
        also raises ``ValueError`` on B>1 as a defense-in-depth — the
        vendored ``mtp_generate_step`` is inherently single-request so
        the raise never fires under normal operation.
        """

        # Static gate for schedulers — see class docstring.
        mtp_max_batch_size = 1

        def __call__(  # type: ignore[override]
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            per_layer_inputs=None,
            *args,
            return_hidden: bool = False,
            n_confirmed: int = 0,
            **kwargs,
        ):
            # Forward any positional / keyword args mlx-lm might add in
            # a future gemma4_text.Model.__call__ revision — this
            # patched shape adds ``return_hidden`` + ``n_confirmed``
            # only, and refuses to mask new base-model kwargs.
            # Stash target cache for mtp_forward — read-only reference,
            # single-request scheduler flow (no concurrent MTP).
            # Always overwrite: a subsequent target forward for a
            # different request must NOT let mtp_forward read K/V
            # from a stale prior-request cache. When ``cache is
            # None`` (e.g. a stray forward outside the MTP loop), we
            # store None too; mtp_forward validates and raises a
            # clear error rather than reusing stale state.
            self._mtp_target_cache = cache
            # Run the target's backbone directly to get pre-lm-head
            # hidden — inner.model returns ``norm(h)`` which is exactly
            # what we feed to ``pre_projection`` on the drafter side.
            hidden = self.model(
                inputs,
                *args,
                cache=cache,
                input_embeddings=input_embeddings,
                per_layer_inputs=per_layer_inputs,
                **kwargs,
            )
            # Logits path — matches mlx_lm gemma4_text.Model.__call__:
            # tied embed_tokens.as_linear + optional softcap.
            if self.args.tie_word_embeddings:
                out = self.model.embed_tokens.as_linear(hidden)
            else:
                out = self.lm_head(hidden)
            if self.args.final_logit_softcapping is not None:
                from mlx_lm.models.gemma4_text import logit_softcap

                out = logit_softcap(self.args.final_logit_softcapping, out)
            # ``n_confirmed`` is a no-op for Gemma 4 (no linear-attn
            # rollback state to snapshot). The generator's KVCache
            # trim(1) path handles rollback for full/sliding attention.
            _ = n_confirmed
            if return_hidden:
                return out, hidden
            return out

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache,
        ):
            """Run the Google assistant drafter over target's tail K/V.

            **Single-query contract**: this MVP consumer processes
            ONE query position — the last of ``hidden_states``. The
            generator's ``_step_mtp`` reads only
            ``mtp_logits[:, -1, :]`` so the returned shape
            ``(B, 1, vocab_size)`` slices identically to the caller's
            existing pull. When the generator passes ``N > 1`` (its
            cache-commit path concatenates ``align_h`` with the
            current hidden), only the last row is used; earlier rows
            would carry the WRONG RoPE offset under the shared-K/V
            path (every drafter Q gets ``offset=cache.offset``, and
            per-row RoPE offset per query position is not modeled
            in this MVP). Rejecting N>1 explicitly is deferred until
            the draft-chain follow-up ships a correct per-row offset
            path; today the generator only ever consumes position -1
            so the semantics stay right.
            """
            import mlx.core as _mx

            target_cache = getattr(self, "_mtp_target_cache", None)
            if target_cache is None:
                raise RuntimeError(
                    "[mtp.inject.gemma4] mtp_forward invoked before a target "
                    "backbone forward — target KV cache is not populated. This "
                    "should not happen with the vendored mtp_generate_step."
                )
            # Slice target's LAST N cache slots (matches assistant's
            # layer_types order: 3 sliding + 1 full = target's last 4).
            n_take = _n_assistant_layers
            if len(target_cache) < n_take:
                raise RuntimeError(
                    f"[mtp.inject.gemma4] target has {len(target_cache)} cache slots "
                    f"but the assistant requires {n_take}."
                )
            shared_kv_slots = target_cache[-n_take:]

            # Normalize unbatched shapes — the vendored
            # ``mtp_generate_step`` uses batched (B=1) tensors, but
            # defensive handling covers callers that pass raw
            # ``(T, H)`` / ``(T,)`` arrays.
            if hidden_states.ndim == 2:
                hidden_states = hidden_states[None]  # (1, T, H)
            if next_token_ids.ndim == 1:
                next_token_ids = next_token_ids[None]  # (1, T)

            # Codex round-7 blocking fix: MVP shared-K/V drafter only
            # supports batch=1 today. The ``_mtp_target_cache`` we
            # slice is ONE cache list (belonging to a single request's
            # target forward); if a caller passed B>1 we would fan out
            # queries against that single cache and either
            # shape-broadcast K/V into the wrong seats or silently mix
            # per-request state. Reject B>1 explicitly. Per-batch
            # shared-K/V support is a follow-up (see PR body's
            # post-MVP TODOs).
            if hidden_states.shape[0] != 1:
                raise ValueError(
                    "[mtp.inject.gemma4] mtp_forward only supports batch=1 "
                    f"today; got hidden_states.shape[0]={hidden_states.shape[0]}. "
                    "Per-batch shared-K/V is a post-MVP follow-up."
                )
            if next_token_ids.shape[0] != 1:
                raise ValueError(
                    "[mtp.inject.gemma4] mtp_forward only supports batch=1 "
                    f"today; got next_token_ids.shape[0]={next_token_ids.shape[0]}."
                )

            # ── Codex round-13 blocking-fix documentation ────────────
            # The generator invokes mtp_forward on TWO paths:
            #
            #   1. ``_step_mtp`` — line 365 in generator.py — hands us
            #      hidden_last of shape (B, 1, H) OR (B, 2, H) when
            #      ``align_h`` is prepended. It then extracts
            #      ``mtp_logits[:, -1, :]`` so ONLY the last position's
            #      logit is used.
            #
            #   2. ``_prefill`` — line 398 in generator.py — hands us
            #      (B, n, H) during warmup. The return value is
            #      DISCARDED — the call exists to warm the drafter's
            #      internal cache. For Gemma 4 cross-KV, the drafter
            #      has NO internal cache to warm (it reads target's),
            #      so this call is a pure no-op computation.
            #
            # In both cases the observable behavior is determined only
            # by the last-position logit (or nothing at all). Any
            # earlier position under the shared-K/V path would carry
            # the wrong RoPE offset (each row needs its own scalar
            # offset, but shared_kv is applied uniformly). Rather
            # than pay N-1 extra forward passes to compute logits the
            # generator never reads, we deliberately compute ONLY the
            # last-position logit and return shape ``(B, 1, vocab)``.
            #
            # A caller that ever grows to consume position [-2] or
            # earlier will IndexError against the (B, 1, ...) shape —
            # a loud red that surfaces the change immediately, better
            # than silent stale-hidden semantics.
            # ────────────────────────────────────────────────────────
            # Take the LAST position from hidden_states / next_token_ids.
            # The generator only USES the last-position logit anyway,
            # and running a single query means shared_kv offset = target
            # cache offset is unambiguously the correct RoPE position
            # (any earlier position would need a smaller offset that
            # this MVP does not model).
            hidden_last = hidden_states[:, -1:, :]
            next_last = next_token_ids[:, -1:]

            # Embed next_token_ids via TARGET's embedding table (drafter's
            # own embed_tokens is 1024-dim; pre_projection expects
            # backbone_hidden × 2 = 3840 × 2 for the 12B pair).
            target_embed = self.model.embed_tokens
            next_embed = target_embed(next_last)  # (B, 1, backbone_hidden)

            fused = _mx.concatenate(
                [hidden_last, next_embed], axis=-1
            )  # (B, 1, 2 * backbone_hidden)
            h = self.mtp.pre_projection(fused)  # (B, 1, hidden_size)

            for layer, tgt_cache in zip(self.mtp.model.layers, shared_kv_slots):
                # target cache may be empty in edge cases — guard.
                #
                # Codex round-8 blocking fix: ``getattr(cache, 'state',
                # default)`` DOES swallow AttributeError raised inside
                # a property fget (verified: empty ``KVCache.state``
                # raises AttributeError via ``self.keys.shape[2]`` when
                # keys is None, and getattr returns the default).
                # BUT: to keep the intent obvious to future readers and
                # match codex's suggestion of an explicit guard, wrap
                # the property access in try/except AttributeError and
                # raise the loud runtime error directly. Belt-and-
                # suspenders — the getattr fallback also stays fine.
                try:
                    state = tgt_cache.state
                except AttributeError:
                    state = None
                if state is None or (
                    isinstance(state, tuple) and state[0] is None
                ):  # pragma: no cover — defensive
                    raise RuntimeError(
                        "[mtp.inject.gemma4] target cache slot has empty state; "
                        "cannot compute drafter attention without any K/V."
                    )
                if isinstance(state, tuple):
                    keys, values = state[0], state[1]
                else:
                    keys = state
                    values = state
                # ``offset`` is passed through to ``self.rope(queries,
                # offset=...)``. Upstream Gemma 4 attention uses
                # ``mx.array(cache.offset)`` in the non-shared path
                # (see ``mlx_lm/models/gemma4_text.py::Attention.__call__``);
                # keeping the mx.array wrap here matches that pattern
                # rather than relying on the RoPE class silently
                # accepting a bare int (which it does today, but the
                # explicit array is safer against future changes).
                offset = _mx.array(int(tgt_cache.offset))
                # mask=None is correct for a SINGLE drafter query
                # attending to a target cache that only holds
                # COMMITTED positions:
                # * Fresh draft path: cache holds [0..main_pos];
                #   drafter Q is at main_pos+1; attends to
                #   [0..main_pos]. No future.
                # * Post-verify path: cache holds
                #   [0..committed, draft]; drafter Q is at
                #   draft+1; attends to [0..draft]. Still all
                #   committed. No future.
                # * Sliding layers: target uses ``RotatingKVCache``
                #   which naturally truncates to the window, so
                #   the K/V slice is already window-bounded.
                # Adding a causal / sliding mask here would be a
                # no-op — the cache slice IS the causal / sliding
                # window.
                h, _shared, _off = layer(
                    h,
                    mask=None,
                    cache=None,
                    per_layer_input=None,
                    shared_kv=(keys, values),
                    offset=offset,
                )

            h = self.mtp.model.norm(h)
            # Tied output: assistant's own embed_tokens serves as lm_head.
            logits = self.mtp.model.embed_tokens.as_linear(h)
            # ``mtp_cache`` argument is unused: cross-KV means the
            # drafter reads from target's cache, never writes its own.
            # Retained on the signature to keep generator compatibility.
            _ = mtp_cache
            return logits

        def make_mtp_cache(self):
            """Empty KVCache slots — safe no-ops for the generator.

            **Length note:** returns one slot per ASSISTANT layer (4
            for Gemma 4), NOT one per target layer. This matches the
            Qwen3.5 contract (``[KVCache() for _ in self.mtp.layers]``
            in ``qwen3_5_inject``) and the generator's own split:
            ``model_cache = prompt_cache[:n_main]; mtp_cache =
            prompt_cache[n_main:] or model.make_mtp_cache()``, where
            ``n_main == len(model.layers)`` is target-layer count.
            The generator's ``quantize_cache_fn`` and ``_prefill``
            walks operate on both lists independently, so a size
            mismatch between them is expected and correct.
            ``_rollback_draft`` walks ``model_cache`` only.

            The drafter reads K/V from TARGET's cache; its own MTP
            cache is never written into by ``mtp_forward`` above. But
            the generator still walks ``mtp_cache`` on three paths:

            1. ``quantize_cache_fn(mtp_cache)`` — the underlying
               ``maybe_quantize_kv_cache`` short-circuits when
               ``kv_bits is None`` (the default). When ``kv_bits`` is
               set, the walk calls ``c.to_quantized(...)`` on each
               slot; empty ``KVCache.to_quantized()`` returns an
               empty ``QuantizedKVCache`` without touching
               ``self.keys / self.values``.
            2. ``_prefill``'s ``mx.eval([c.state for c in ... if
               hasattr(c, 'state')])`` — on an empty ``KVCache``,
               ``state`` raises ``AttributeError`` (``self.keys`` is
               None → ``.shape[2]``), which ``hasattr`` swallows to
               return False. The empty slot is skipped.
            3. ``_rollback_draft`` walks ``model_cache`` NOT
               ``mtp_cache``, so ``.trim(1)`` never fires here.

            Returning empty ``KVCache`` instances is therefore safe
            with the current generator. This analysis is locked by a
            dedicated test that exercises ``trim``, ``to_quantized``,
            and ``hasattr(_, 'state')`` on the returned list.
            """
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.mtp.model.layers]

    # ── Transactional commit: swap class + delegate surfaces ─────────
    # Prepare all outer-wrapper delegation methods BEFORE mutating any
    # inner state. If anything goes wrong (attribute setattr blocked,
    # method construction fails), abort with the target unmodified.
    import types as _types

    try:
        _delegate_forward = None
        _delegate_cache = None
        if model is not inner:

            def _delegate_forward(_self, hidden_states, next_token_ids, mtp_cache):
                return inner.mtp_forward(hidden_states, next_token_ids, mtp_cache)

            def _delegate_cache(_self):
                return inner.make_mtp_cache()

        # Commit: class swap first so the surfaces are visible via the
        # inner's own method resolution, then set attributes on inner
        # + (optionally) delegate onto the outer.
        inner.__class__ = _Gemma4WithMTP
        inner.mtp = assistant
        if model is not inner:
            model.mtp = inner.mtp
            model.mtp_forward = _types.MethodType(_delegate_forward, model)
            model.make_mtp_cache = _types.MethodType(_delegate_cache, model)
            # Codex round-11 blocking fix: mirror the static batch-size
            # gate onto the OUTER wrapper. Schedulers that inspect the
            # caller-visible ``model`` object (not the buried inner
            # text model) need to read this attribute at dispatch time
            # to skip B>1 requests off the MTP fast-path. Inner's
            # class-level ``mtp_max_batch_size`` was invisible to
            # them; setting it as an instance attribute on the outer
            # wrapper closes that gap.
            model.mtp_max_batch_size = _Gemma4WithMTP.mtp_max_batch_size
    except Exception as exc:
        # Roll back any partial state.
        #
        # Codex round-8 blocking fix: previous rollback used
        # ``hasattr(inner, 'mtp')`` which invokes descriptors on the
        # possibly-half-patched object (the class swap already ran, so
        # attribute lookup goes through ``_Gemma4WithMTP.__mro__``).
        # Use direct ``__dict__.pop`` on the instance to bypass
        # descriptors entirely, and unconditional guarded ``delattr``
        # instead of ``hasattr + delattr``.
        try:
            if type(inner) is _Gemma4WithMTP:
                inner.__class__ = original_class
            # Direct __dict__ removal — bypasses any potential
            # descriptor / __getattr__ interception on inner.
            try:
                inner.__dict__.pop("mtp", None)
            except Exception:  # pragma: no cover
                pass
            if model is not inner:
                for attr in (
                    "mtp",
                    "mtp_forward",
                    "make_mtp_cache",
                    "mtp_max_batch_size",
                ):
                    # Direct __dict__ removal first (fast path), then
                    # a guarded delattr as a fallback for objects
                    # that route attribute writes through __setattr__.
                    try:
                        model.__dict__.pop(attr, None)
                    except Exception:  # pragma: no cover
                        pass
                    try:
                        delattr(model, attr)
                    except AttributeError:  # attribute already gone
                        pass
                    except Exception:  # pragma: no cover — best effort
                        pass
        except Exception:  # pragma: no cover — best-effort rollback
            pass
        logger.warning(
            "[mtp.inject.gemma4] failed during inject commit; rolled back. Error: %s",
            exc,
        )
        return False

    logger.info(
        "[mtp.inject.gemma4] Injected Google assistant drafter "
        "(layers=%d, hidden=%d, backbone_hidden=%d) onto %s.",
        _n_assistant_layers,
        assistant.args.hidden_size,
        backbone_hidden,
        original_class.__name__,
    )
    _ = _target_num_layers  # currently unused, reserved for the follow-up
    return True


def validate_mtp_support(model: Any) -> bool:
    """Verify the four contract surfaces attached to ``model``.

    Same shape as :func:`qwen3_5_inject.validate_mtp_support`. Returns
    ``True`` when the injection landed and the model is ready to be
    driven by :func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step`.
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

    # Codex round-6 blocking fix: when ``model`` is an outer wrapper
    # (e.g. ``Gemma4Unified`` with ``model.language_model`` as the
    # actual text model), we resolved down to ``inner`` above. But
    # the generator will call ``model.mtp_forward(...)`` /
    # ``model.make_mtp_cache()`` on the object the CALLER passed —
    # which is the outer wrapper. Inject wires those delegations onto
    # the outer, but a partially-rolled-back inject could leave inner
    # correctly patched while the outer stays bare. Explicitly
    # re-check the delegated surfaces on the outer to catch that.
    if model is not inner:
        # Codex round-11 blocking fix: mtp_max_batch_size is a
        # scheduler-visible static gate. The commit path sets it on
        # the outer; a partially-rolled-back inject that leaves it
        # missing means schedulers can't safely gate. Include it in
        # the required-delegated-surface list.
        for attr in (
            "mtp",
            "mtp_forward",
            "make_mtp_cache",
            "mtp_max_batch_size",
        ):
            if not hasattr(model, attr):
                logger.warning(
                    "[mtp.validate.gemma4] outer wrapper missing delegated "
                    "surface: %s. Inner-only patch would cause the generator "
                    "to AttributeError on the outer.",
                    attr,
                )
                return False
        # Codex round-12 blocking fix: hasattr can't distinguish an
        # attribute that was cleared to ``None`` from one that was
        # never set. A partial rollback that assigned ``None`` (e.g.
        # via a well-meaning ``setattr(model, 'mtp', None)`` in some
        # future teardown path) would validate green via hasattr but
        # deliver no usable drafter to the generator. Match the inner
        # check by requiring the actual attribute value.
        if getattr(model, "mtp", None) is None:
            logger.warning(
                "[mtp.validate.gemma4] outer wrapper's ``mtp`` attribute is "
                "None; drafter is not attached."
            )
            return False
        # Codex round-13 nit fix: verify ``mtp_max_batch_size`` value
        # (not just presence). A corrupted / hand-patched outer with
        # ``mtp_max_batch_size = 2`` (or any other truthy value) would
        # pass ``hasattr`` but let a scheduler dispatch B=2 requests
        # into ``mtp_forward``'s B=1-only path. Require the exact
        # value the inject commit path set.
        outer_max_batch = getattr(model, "mtp_max_batch_size", None)
        if outer_max_batch != 1:
            logger.warning(
                "[mtp.validate.gemma4] outer wrapper's mtp_max_batch_size=%r "
                "does not equal 1 (the value the inject path sets). This "
                "would let schedulers dispatch B>1 requests into the "
                "batch=1-only mtp_forward path.",
                outer_max_batch,
            )
            return False
        if not callable(getattr(model, "mtp_forward", None)):
            logger.warning(
                "[mtp.validate.gemma4] outer wrapper's mtp_forward is not callable."
            )
            return False
        if not callable(getattr(model, "make_mtp_cache", None)):
            logger.warning(
                "[mtp.validate.gemma4] outer wrapper's make_mtp_cache is not callable."
            )
            return False
    return True
