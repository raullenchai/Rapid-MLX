# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 text-only model loaders for the LLM path.

mlx-lm 0.31+ added native ``gemma4`` (used by the 26B / 31B aliases), but
``gemma4_unified`` (the model_type the four ``gemma-4-12b-*`` aliases ship
under) is still not in mlx-lm. This module loads the language model
portion from mlx-vlm (or the vendored copy) and wraps it to be compatible
with mlx-lm's generate_step() interface, enabling:
- Prompt cache (KV reuse across requests)
- DeltaNet state snapshots (if applicable)
- All LLM-path optimizations

Three outer model_types are served, routed to two loaders via an
EXPLICIT exact-match allow-list so routing is unambiguous (see #509 —
the old ``"gemma4" in model_type`` substring test would silently
misroute a hypothetical ``gemma4_videogen`` or the inner sub-config's
own ``gemma4_text`` label):

- ``gemma4``           → :func:`is_gemma4_nonunified_model` /
                         :func:`load_gemma4_text`
- ``gemma4_assistant`` → :func:`is_gemma4_nonunified_model` /
                         :func:`load_gemma4_text`
                         (the ``gemma-4-*-assistant`` aliases; its nested
                         ``text_config`` is a ``gemma4_text`` shape, so it
                         rides the same non-unified loader the old
                         substring match sent it down — kept for
                         backward compat)
- ``gemma4_unified``   → :func:`is_gemma4_unified_model` /
                         :func:`load_gemma4_unified_text`

:func:`is_gemma4_family_model` is the OR of all three for call sites that
just need "is this a Gemma 4 text-servable arch?"; :func:`gemma4_family_kind`
classifies with a single config read for dispatch sites.
:func:`is_gemma4_model` is a family-wide back-compat alias for
:func:`is_gemma4_family_model` (NOT the narrow non-unified router — new
code should use the precise predicates above). Both loaders prefer the
matching upstream ``mlx_vlm`` subpackage when installed and fall back to
the vendored copy under ``vllm_mlx/models/gemma4_vendored/`` so a fresh
``pip install rapid-mlx`` (no ``[vision]`` extra) still boots.

The wrapper is thin: it just ensures model(input_ids, cache=cache) returns
a raw logits tensor instead of LanguageModelOutput.

TODO: Remove once mlx-lm adds native ``gemma4_unified`` support (12B variants).
"""

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


_FP_DTYPES_NOT_QUANTIZED = (
    mx.bfloat16,
    mx.float16,
    mx.float32,
)


def _bare_fp_weight_paths(sanitized: dict) -> set[str]:
    """Return paths whose ``.weight`` is stored as bare fp (no scales/biases).

    These are layers the checkpoint deliberately left unquantized (e.g.
    the ``per_layer_model_projection`` "altup" projection on gemma-4-e2b/
    e4b). The blanket ``nn.quantize`` path would otherwise convert them
    to QuantizedLinear and then ``mx.quantized_matmul`` would raise at
    inference time on the bf16 weight that disk supplied.

    We return the bare-path (no ``.weight`` suffix) so it can be matched
    against ``nn.quantize``'s dotted module paths via suffix-equality.
    """
    weight_paths = {k[: -len(".weight")] for k in sanitized if k.endswith(".weight")}
    skip: set[str] = set()
    for base in weight_paths:
        wkey = f"{base}.weight"
        scales = f"{base}.scales"
        biases = f"{base}.biases"
        w = sanitized.get(wkey)
        if w is None:
            continue
        # A truly quantized weight will have dtype uint32 AND a matching
        # ``.scales`` companion. A bare fp16/bf16 ``.weight`` with no
        # scales companion means "checkpoint kept this layer in full
        # precision on purpose".
        if w.dtype in _FP_DTYPES_NOT_QUANTIZED and (
            scales not in sanitized and biases not in sanitized
        ):
            skip.add(base)
    return skip


def _path_matches_any_suffix(path: str, suffixes: set[str]) -> bool:
    """Is ``path`` a suffix-match for any of ``suffixes``?

    ``nn.quantize`` visits each ``to_quantized``-able module with a
    dotted path relative to the root (e.g.
    ``language_model.model.per_layer_model_projection``). The
    sanitized-weights keys come from disk and may carry slightly
    different prefixes (``model.``, ``language_model.model.``)
    depending on which wrapper layer they live under. So we match
    by suffix on the bare module path (everything before ``.weight``)
    rather than equality.
    """
    if not suffixes:
        return False
    for suffix in suffixes:
        # Direct suffix match handles "language_model.model.X" vs
        # nn.quantize path "language_model.model.X" naturally; the
        # second form ("model.X") falls out because Python's str.endswith
        # also accepts the bare tail.
        if path == suffix or path.endswith("." + suffix.split(".")[-1]):
            # Verify the FULL final segment matches (don't let
            # "self_attn.k_proj" suffix-match against
            # "self_attn.k_proj_extra" — both end with "k_proj" but
            # the latter isn't the same module).
            if path.split(".")[-1] == suffix.split(".")[-1]:
                # Match more of the path to avoid colliding with sibling
                # modules in different layers. Compare the last N tokens
                # of each.
                p_tokens = path.split(".")
                s_tokens = suffix.split(".")
                n = min(len(p_tokens), len(s_tokens))
                if p_tokens[-n:] == s_tokens[-n:]:
                    return True
    return False


def _read_model_type(model_path: str | Path) -> str | None:
    """Read the top-level ``model_type`` from a model's ``config.json``.

    Returns ``None`` when the config is unreachable or unparseable so
    callers can treat "can't tell" as "not this family".

    The previous ``is_gemma4_model`` implementation called
    ``snapshot_download(repo_id)`` to populate a local cache before
    reading ``config.json``. That works, but ``snapshot_download``
    validates/fetches the ENTIRE model tree (all safetensors shards,
    tokenizer files, generation config), which for an 8-bit 35B model is
    ~35 GB of Xet-protocol revalidation on every cold ``rapid-mlx serve``
    start. We fetch ``config.json`` directly via ``hf_hub_download``: a
    ~5 KB file, validated against the existing HF cache. This was the
    root cause of stress_e2e_bench server-boot timeouts on large models
    in PR #600 validation.
    """
    p = Path(model_path)
    config_path = p / "config.json" if p.is_dir() else None
    if config_path is None or not config_path.exists():
        try:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(repo_id=str(model_path), filename="config.json")
            )
        except Exception:
            return None
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
        return config.get("model_type", "")
    except Exception:
        return None


# Model_types the Gemma 4 text loader path claims. This is a deliberate
# exact-match allow-list, NOT a ``"gemma4" in model_type`` substring test
# (see #509). The substring check also matched a hypothetical future
# ``gemma4_videogen`` (or ``gemma4_text`` — the inner text sub-config's
# own model_type) and would silently misroute it. Each member is
# classified by :func:`gemma4_family_kind` and routed to the matching
# ``load_gemma4_*`` loader; adding a new supported arch is a one-line
# edit here plus a loader branch, which is the point — routing is
# explicit and unknown arches surface loudly instead of silently riding
# the text path.
#
# - ``gemma4``           : non-unified text arch (26B/31B/e2b/e4b,
#                          ``Gemma4ForConditionalGeneration``).
# - ``gemma4_unified``   : unified arch (the four ``gemma-4-12b-*``
#                          aliases, ``Gemma4UnifiedForConditionalGeneration``).
# - ``gemma4_assistant`` : the ``gemma-4-*-assistant`` aliases
#                          (``Gemma4AssistantForCausalLM``). Its nested
#                          ``text_config`` is a ``gemma4_text`` shape, so
#                          it loads through the SAME non-unified path the
#                          old substring match sent it down. Kept in the
#                          allow-list to preserve that pre-#509 behavior
#                          (dropping it would regress those aliases to the
#                          unsupported native-load path).
_GEMMA4_NONUNIFIED_MODEL_TYPES = ("gemma4", "gemma4_assistant")
_GEMMA4_UNIFIED_MODEL_TYPES = ("gemma4_unified",)
_GEMMA4_FAMILY_MODEL_TYPES = (
    _GEMMA4_NONUNIFIED_MODEL_TYPES + _GEMMA4_UNIFIED_MODEL_TYPES
)


def is_gemma4_family_model(model_path: str | Path) -> bool:
    """Check if the model belongs to the Gemma 4 *family* (text loader path).

    True for any model_type the Gemma 4 text loaders serve — see
    :data:`_GEMMA4_FAMILY_MODEL_TYPES`. Use this as the routing gate;
    then split with :func:`is_gemma4_unified_model` to pick the loader.
    """
    # Read the model_type once (may hit hf_hub_download for a remote repo,
    # cached ~5 KB) rather than paying multiple lookups via the exact
    # detectors.
    return _read_model_type(model_path) in _GEMMA4_FAMILY_MODEL_TYPES


def is_gemma4_nonunified_model(model_path: str | Path) -> bool:
    """Check if the model is a NON-unified Gemma 4 text arch.

    Matches ``gemma4`` and ``gemma4_assistant`` — the arches served by
    :func:`load_gemma4_text`. This is the explicitly-named narrowed
    predicate (the complement of :func:`is_gemma4_unified_model` within
    the family). Callers that want "any Gemma 4 text-servable arch"
    should use :func:`is_gemma4_family_model` (which the family-wide
    back-compat alias :func:`is_gemma4_model` delegates to); callers that
    specifically want the unified arch should use
    :func:`is_gemma4_unified_model`.
    """
    return _read_model_type(model_path) in _GEMMA4_NONUNIFIED_MODEL_TYPES


def is_gemma4_model(model_path: str | Path) -> bool:
    """Back-compat alias: is this ANY Gemma 4 text-servable arch?

    Pre-#509 this did ``"gemma4" in model_type``, returning True for the
    base ``gemma4`` arch AND (by substring) ``gemma4_unified`` /
    ``gemma4_assistant``. To keep that documented family-wide meaning for
    any existing caller of this name, it now delegates to
    :func:`is_gemma4_family_model` (the exact-match allow-list) rather
    than the old substring test. The only behavioral difference vs the
    substring version is that a NON-family arch that merely contains the
    text ``"gemma4"`` (a hypothetical ``gemma4_videogen``) is no longer
    falsely claimed — which is the whole point of #509.

    New code should call the precise predicate for its intent:
    :func:`is_gemma4_unified_model`, :func:`is_gemma4_nonunified_model`,
    or :func:`gemma4_family_kind` (single-read classification).
    """
    return is_gemma4_family_model(model_path)


def is_gemma4_unified_model(model_path: str | Path) -> bool:
    """Check if the model is the unified Gemma 4 arch (``gemma4_unified``).

    Exact match on ``model_type == "gemma4_unified"`` — the arch the four
    ``gemma-4-12b-*`` aliases ship under
    (``Gemma4UnifiedForConditionalGeneration``). Routed to
    :func:`load_gemma4_unified_text`, which pins to the matching
    ``mlx_vlm.models.gemma4_unified`` subpackage when mlx-vlm is
    installed and falls back to the vendored copy otherwise. See #509.
    """
    return _read_model_type(model_path) in _GEMMA4_UNIFIED_MODEL_TYPES


def gemma4_family_kind(model_path: str | Path) -> str | None:
    """Classify a Gemma 4 family model with a SINGLE config read.

    Returns ``"unified"`` for ``gemma4_unified``, ``"nonunified"`` for
    ``gemma4`` / ``gemma4_assistant``, or ``None`` if the model isn't in
    the Gemma 4 text-loader family. Prefer this over calling
    :func:`is_gemma4_family_model` then :func:`is_gemma4_unified_model` at
    a dispatch site — it reads ``config.json`` once (one Hub lookup for a
    remote repo) and the retained classification can't be flipped by a
    transient second lookup failure.
    """
    mt = _read_model_type(model_path)
    if mt in _GEMMA4_UNIFIED_MODEL_TYPES:
        return "unified"
    if mt in _GEMMA4_NONUNIFIED_MODEL_TYPES:
        return "nonunified"
    return None


class Gemma4TextWrapper(nn.Module):
    """Wraps mlx-vlm's Gemma4 LanguageModel for mlx-lm compatibility.

    mlx-lm's generate_step() expects model(input_ids, cache=cache) -> logits.
    mlx-vlm's LanguageModel returns LanguageModelOutput(logits=...).
    This wrapper extracts .logits so the interface matches.
    """

    def __init__(self, language_model, routed_model_type: str = "gemma4"):
        super().__init__()
        self.language_model = language_model
        # Expose config for mlx-lm compatibility
        self.config = language_model.config
        self.model = language_model.model
        # Report the arch the caller ACTUALLY routed for (``gemma4`` /
        # ``gemma4_unified`` / ``gemma4_assistant``). The wrapped
        # ``LanguageModel`` is the shared text stack and always reports the
        # generic inner label ``"gemma4_text"`` regardless of the outer
        # arch, so we normalize that to the routed model_type. If the
        # wrapped model ever reports a more specific type, honor it.
        inner = getattr(language_model, "model_type", None)
        self.model_type = (
            inner if inner and inner != "gemma4_text" else routed_model_type
        )

    def __call__(self, input_ids, cache=None, **kwargs):
        out = self.language_model(input_ids, cache=cache, **kwargs)
        # LanguageModelOutput -> raw logits tensor
        return out.logits if hasattr(out, "logits") else out

    def sanitize(self, weights):
        """Strip language_model. prefix from VLM-format weights."""
        sanitized = {}
        for k, v in weights.items():
            new_key = k
            # Strip top-level "model." wrapper
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            # Strip "language_model." to get bare model weights,
            # then re-add "language_model." for our wrapper structure
            if new_key.startswith("language_model."):
                pass  # keep as-is — our wrapper has .language_model attribute
            elif not any(
                new_key.startswith(p)
                for p in ["vision_tower", "audio_tower", "embed_vision", "embed_audio"]
            ):
                new_key = "language_model." + new_key
            else:
                continue  # skip vision/audio weights
            # Skip rotary embeddings (computed dynamically)
            if "rotary_emb" in new_key:
                continue
            # Skip clipping params (vision-only)
            if any(
                s in new_key
                for s in ["input_max", "input_min", "output_max", "output_min"]
            ):
                continue
            sanitized[new_key] = v
        return sanitized

    def make_cache(self):
        """Delegate to LanguageModel for proper sliding window + full attention cache."""
        return self.language_model.make_cache()

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def head_dim(self):
        return self.language_model.head_dim

    @property
    def n_kv_heads(self):
        return self.language_model.n_kv_heads


# Shared preamble for both loaders. As of 0.10.1 we vendor the Gemma 4
# text classes (~50 KB, ~1200 lines) directly under
# `vllm_mlx/models/gemma4_vendored/` so a fresh `pip install rapid-mlx`
# boots Gemma 4 out of the box — no `[vision]` extra required.
# Previously we imported from `mlx_vlm.models.gemma4.*`, but that
# required either promoting mlx-vlm to a core dep (~+483 MB transitive
# bloat: opencv-python, pyarrow, pandas, scipy, mlx-audio — none of
# which text-only Gemma 4 inference touches) or making users know to
# `pip install --no-deps 'mlx-vlm>=0.6.1'` themselves. See
# `vllm_mlx/models/gemma4_vendored/__init__.py` for the sync policy.
#
# We still prefer upstream mlx-vlm when it's already importable
# (e.g. `[vision]` users): mlx-vlm may ship a bug fix or Gemma 4.1
# update before we sync the vendored copy. The vendored fallback keeps
# the fresh-install path working with zero extras.


def _resolve_gemma4_text_classes():
    """Return ``(TextConfig, LanguageModel)`` for the NON-unified ``gemma4``
    arch — upstream ``mlx_vlm.models.gemma4`` when importable, else the
    vendored copy (dataclass-identical, no ``[vision]`` extra needed)."""
    try:
        from mlx_vlm.models.gemma4.config import TextConfig
        from mlx_vlm.models.gemma4.language import LanguageModel

        return TextConfig, LanguageModel
    except ImportError:
        from vllm_mlx.models.gemma4_vendored import (
            config as _v_cfg,
        )
        from vllm_mlx.models.gemma4_vendored import (
            language as _v_lang,
        )

        return _v_cfg.TextConfig, _v_lang.LanguageModel


def _resolve_gemma4_unified_text_classes():
    """Return ``(TextConfig, LanguageModel)`` for the ``gemma4_unified`` arch.

    Prefer upstream ``mlx_vlm.models.gemma4_unified`` when mlx-vlm is
    installed, so we pin to the subpackage that actually matches the
    ``Gemma4UnifiedForConditionalGeneration`` checkpoints and surface any
    future upstream drift instead of silently reusing the non-unified
    classes. Notes on the upstream layout (mlx-vlm 0.6.3):

    - ``gemma4_unified.config.TextConfig`` subclasses
      ``gemma4.config.TextConfig`` with unified-specific defaults; the
      concrete field values still come from the checkpoint's
      ``text_config`` via ``from_dict``, so it stays dataclass-compatible
      with the vendored copy.
    - ``gemma4_unified.LanguageModel`` is literally re-exported from
      ``gemma4.language`` (``from ..gemma4.language import LanguageModel``
      in ``gemma4_unified.py``) — the unified arch only adds vision/audio
      embedders around the SAME text stack. So the text forward path is
      identical to the non-unified loader by construction, which is why
      serving ``gemma-4-12b`` through the ``gemma4`` classes worked
      empirically (see #509). Routing here just makes the intent explicit.

    Falls back to the vendored ``gemma4`` copy when mlx-vlm is absent
    (fresh install). The vendored copy has no separate ``unified``
    variant, but since upstream's ``LanguageModel`` IS the ``gemma4`` one
    and the ``TextConfig`` is dataclass-identical for text purposes, the
    vendored classes serve ``gemma4_unified`` correctly — same behavior
    as today, just reached through an explicit branch. A real
    ``ImportError`` is only possible if BOTH upstream-unified AND the
    vendored copy are unavailable, which cannot happen because the
    vendored copy ships inside the wheel.
    """
    try:
        from mlx_vlm.models.gemma4_unified import LanguageModel
        from mlx_vlm.models.gemma4_unified.config import TextConfig

        return TextConfig, LanguageModel
    except ImportError:
        # No upstream unified subpackage (fresh install, or an older
        # mlx-vlm without gemma4_unified). Fall back to the vendored
        # text classes — same as the non-unified path.
        from vllm_mlx.models.gemma4_vendored import (
            config as _v_cfg,
        )
        from vllm_mlx.models.gemma4_vendored import (
            language as _v_lang,
        )

        return _v_cfg.TextConfig, _v_lang.LanguageModel


def load_gemma4_unified_text(model_path: str | Path, tokenizer_config: dict = None):
    """Load a ``gemma4_unified`` Gemma 4 checkpoint as a text-only model.

    Explicit loader for the ``gemma-4-12b-*`` aliases
    (``Gemma4UnifiedForConditionalGeneration``). Pins to the matching
    ``mlx_vlm.models.gemma4_unified`` subpackage when mlx-vlm is
    installed, else falls back to the vendored copy. See #509 and
    :func:`_resolve_gemma4_unified_text_classes` for the fallback
    rationale. Returns ``(model, tokenizer)`` compatible with mlx-lm's
    ``generate_step()``.
    """
    return _load_gemma4_text_impl(
        model_path,
        tokenizer_config,
        resolve_classes=_resolve_gemma4_unified_text_classes,
        default_model_type="gemma4_unified",
    )


def load_gemma4_text(model_path: str | Path, tokenizer_config: dict = None):
    """Load a NON-unified ``gemma4`` checkpoint as a text-only model.

    For the 26B / 31B / e2b / e4b aliases
    (``Gemma4ForConditionalGeneration``). Returns ``(model, tokenizer)``
    compatible with mlx-lm's ``generate_step()``. For ``gemma4_unified``
    (12B) use :func:`load_gemma4_unified_text` instead.
    """
    return _load_gemma4_text_impl(
        model_path,
        tokenizer_config,
        resolve_classes=_resolve_gemma4_text_classes,
        default_model_type="gemma4",
    )


def _text_config_default_num_kv_shared(tc) -> int:
    """The ``num_kv_shared_layers`` dataclass DEFAULT for ``tc``'s class.

    Used only when the config key was ABSENT and the dataclass field itself
    resolved to ``None`` (an explicit ``null`` is rejected upstream in
    :func:`_check_kv_share_config`, so it never reaches here). We fall back to
    the same default an absent key already receives via ``from_dict`` rather
    than forcing 0, which would silently change a shared-KV checkpoint's
    architecture. Robust across the upstream mlx-vlm and vendored
    ``TextConfig`` dataclasses; returns 0 if the default cannot be introspected
    or is itself ``None`` (fail-safe: inactive rather than a wrong active
    split).
    """
    import dataclasses

    try:
        for f in dataclasses.fields(tc):
            if f.name == "num_kv_shared_layers":
                default = f.default
                if isinstance(default, int) and not isinstance(default, bool):
                    return default
    except (TypeError, ValueError):
        pass
    return 0


def _check_kv_share_config(text_config: dict, tc, model_id: str) -> None:
    """Guard Gemma 4 cross-layer KV-sharing at load time.

    Gemma 3n / Gemma 4's last ``num_kv_shared_layers`` decoder layers are
    "borrowers": they compute no K/V and reuse the last same-type producer
    layer's K/V (split at ``num_hidden_layers - num_kv_shared_layers``). This
    is the mechanism behind the smaller resident KV cache (measured ~2.3x
    footprint reduction on gemma-4-e2b-4bit; the prefill/TTFT wall-time win is
    negligible on the small quantized sizes because the eliminated
    K/V-projection compute is <3% of prefill) — see
    ``models/gemma4_vendored/language.py`` (``make_cache`` returns a
    producer-only cache list; borrowers get no cache object).

    Severity is decided on ``tc.num_kv_shared_layers`` — the value the model
    is actually built from (``LanguageModel(tc)`` uses it to split producers
    from borrowers). The raw ``text_config`` dict is consulted only to enrich
    the message (distinguish "checkpoint omitted the key, dataclass default
    applied" from "checkpoint explicitly set 0"), because ``from_dict`` fills
    an absent key with the dataclass default and thereby masks its absence.

    Failure modes this guards against:

    * ``tc.num_kv_shared_layers == 0`` → ``first_kv_shared`` equals
      ``num_hidden_layers`` → NO layer borrows → every layer allocates its
      own KV, so the resident cache is not reduced by cross-layer reuse. We
      log at INFO (not WARNING, and never hard-fail): the dense large Gemma 4
      sizes — 12B / 26B-A4B / 31B — legitimately ship
      ``num_kv_shared_layers=0`` and never share, so warning on every such
      load would be a false-positive alert. INFO still makes the "no
      cross-layer reuse" fact observable so a user can never silently believe
      sharing is on when it is not.
    * ``tc.num_kv_shared_layers`` invalid — not a non-negative integer, or
      ``>= num_hidden_layers``, or a borrower attention type with no
      producer of that type below the split — → malformed config that cannot
      share correctly (no producers, out-of-range split, or a borrower with
      nothing to borrow). We RAISE — a broken checkpoint, not a silent
      degrade.
    * ``0 < tc.num_kv_shared_layers < num_hidden_layers`` (and every borrower
      type has a producer) → sharing active; log the producer/borrower split
      at debug.

    Placed on the shared build path (``_load_gemma4_text_impl``) so it fires
    for every Gemma 4 size and both loaders (``gemma4`` / ``gemma4_unified``),
    regardless of whether ``resolve_classes`` returns the upstream mlx-vlm or
    the vendored text classes (both expose ``num_hidden_layers`` /
    ``num_kv_shared_layers`` on the dataclass ``TextConfig``).
    """
    # ``num_hidden_layers`` must be a positive, non-boolean int for any of the
    # split math to make sense. A malformed value (0, negative, bool, string,
    # None) is a broken config — raise the clear diagnostic here rather than
    # letting the model build fail cryptically or silently produce an empty
    # stack. (``bool`` is an ``int`` subclass in Python, so exclude it first.)
    num_hidden = getattr(tc, "num_hidden_layers", None)
    if (
        isinstance(num_hidden, bool)
        or not isinstance(num_hidden, int)
        or num_hidden <= 0
    ):
        raise ValueError(
            f"Gemma 4 config INVALID for {model_id}: num_hidden_layers="
            f"{num_hidden!r} must be a positive integer."
        )

    key_present = "num_kv_shared_layers" in text_config
    raw_dict_val = text_config.get("num_kv_shared_layers", "__absent__")
    raw_shared = getattr(tc, "num_kv_shared_layers", 0)

    # Reject an EXPLICIT JSON null: a config that writes ``num_kv_shared_layers:
    # null`` is malformed (it can't declare a share count and leave it unset).
    # We must not guess a size-specific value — forcing the class default would
    # give E4B 20 borrowers instead of 18, or a dense model 20 instead of 0,
    # silently changing its architecture. An ABSENT key is different: it keeps
    # whatever the dataclass default already filled in via ``from_dict`` (the
    # legitimate "checkpoint didn't override the default" case).
    if key_present and raw_dict_val is None:
        raise ValueError(
            f"Gemma 4 KV-sharing config INVALID for {model_id}: "
            "num_kv_shared_layers is explicitly null. A checkpoint must declare "
            "a concrete share count (an int) or omit the key entirely; a null "
            "value is malformed and cannot be resolved to a size-specific split."
        )

    # Normalize the value the model will be built from.
    #   * A plain non-negative int → used as-is.
    #   * ``None`` reaching here means the key was ABSENT (not explicit null,
    #     handled above) and the dataclass field itself is ``None`` — fall back
    #     to the config dataclass DEFAULT for the field, the same value an
    #     absent key already gets via ``from_dict``.
    #   * Anything else (string, list, float, bool) → malformed config → raise.
    if raw_shared is None:
        num_shared = _text_config_default_num_kv_shared(tc)
        key_present = False  # surface "not explicitly specified" in the log
    elif isinstance(raw_shared, bool) or not isinstance(raw_shared, int):
        raise ValueError(
            f"Gemma 4 KV-sharing config INVALID for {model_id}: "
            f"num_kv_shared_layers={raw_shared!r} is not a non-negative "
            "integer. This is a malformed checkpoint config."
        )
    else:
        num_shared = raw_shared

    # Write the normalized value back onto ``tc`` so the subsequent
    # ``LanguageModel(tc)`` build sees a plain int (its producer-split math
    # ``num_hidden - num_kv_shared_layers`` would otherwise raise a cryptic
    # ``TypeError`` on a ``None`` that reached it via an explicit config null).
    # If the writeback can't take (e.g. a frozen config), fail loudly here
    # rather than let the model crash later on the unchanged value.
    if raw_shared is not num_shared:
        try:
            tc.num_kv_shared_layers = num_shared
        except Exception as exc:
            raise ValueError(
                f"Gemma 4 KV-sharing config for {model_id}: could not normalize "
                f"num_kv_shared_layers={raw_shared!r} → {num_shared} on the "
                f"config object ({exc}); the model would then fail its "
                "producer-split computation."
            ) from exc

    if num_shared < 0 or num_shared >= num_hidden:
        raise ValueError(
            f"Gemma 4 KV-sharing config INVALID for {model_id}: "
            f"num_kv_shared_layers={num_shared} must satisfy "
            f"0 <= num_kv_shared_layers < num_hidden_layers={num_hidden}. "
            "This is a malformed checkpoint config (borrowers would exceed "
            "producers, or the split would be out of range)."
        )

    if num_shared == 0:
        logger.info(
            "[gemma4] KV-sharing INACTIVE for %s: num_kv_shared_layers=0; "
            "no cross-layer KV reuse (every layer allocates its own KV). "
            "Expected for the dense sizes (12B / 26B-A4B / 31B); only the "
            "E-series (E2B / E4B) shares.",
            model_id,
        )
        return

    # 0 < num_shared < num_hidden → sharing candidate. Active sharing REQUIRES
    # a usable ``layer_types`` (one entry per layer): the producer→borrower map
    # is built by matching each borrower's attention type to the last same-type
    # producer. Without it we cannot establish that map, so declaring sharing
    # ACTIVE would be unfounded — raise instead of logging a claim we did not
    # validate. (The dataclass ``__post_init__`` always derives ``layer_types``;
    # a missing/short one is a genuinely malformed config.)
    layer_types = getattr(tc, "layer_types", None)
    num_producers = num_hidden - num_shared
    if not isinstance(layer_types, (list, tuple)) or len(layer_types) != num_hidden:
        raise ValueError(
            f"Gemma 4 KV-sharing config INVALID for {model_id}: sharing is on "
            f"(num_kv_shared_layers={num_shared}) but layer_types is missing or "
            f"the wrong length (need {num_hidden} entries, got "
            f"{len(layer_types) if isinstance(layer_types, (list, tuple)) else layer_types!r}). "
            "Cannot establish the producer→borrower map."
        )
    # Entries must be strings (attention-type labels) — guard against
    # unhashable / non-string entries producing an incidental TypeError from
    # the set() below instead of this clear malformed-config diagnostic.
    if not all(isinstance(t, str) for t in layer_types):
        raise ValueError(
            f"Gemma 4 KV-sharing config INVALID for {model_id}: layer_types "
            "must be a list of attention-type strings; found a non-string entry."
        )
    # Every borrower attention type must have a producer of that type below the
    # split; otherwise a borrower would have nothing to borrow.
    producer_types = set(layer_types[:num_producers])
    borrower_types = set(layer_types[num_producers:])
    orphan_types = borrower_types - producer_types
    if orphan_types:
        raise ValueError(
            f"Gemma 4 KV-sharing config INVALID for {model_id}: borrower "
            f"attention type(s) {sorted(orphan_types)} have no producer "
            f"layer of that type in the first {num_producers} layer(s). "
            "This layer_types layout cannot share K/V correctly."
        )

    default_note = (
        ""
        if key_present
        else " (checkpoint omitted the key; dataclass default applied)"
    )
    logger.debug(
        "[gemma4] KV-sharing ACTIVE for %s: %d producer layer(s) + "
        "%d borrower layer(s) (first_kv_shared_layer_idx=%d)%s",
        model_id,
        num_producers,
        num_shared,
        num_producers,
        default_note,
    )


def _load_gemma4_text_impl(
    model_path: str | Path,
    tokenizer_config: dict = None,
    *,
    resolve_classes,
    default_model_type: str,
):
    """Shared build for both Gemma 4 text loaders.

    ``resolve_classes`` returns ``(TextConfig, LanguageModel)`` for the
    target arch (unified vs non-unified); everything downstream —
    weight sanitize, checkpoint-driven quantization, tokenizer load — is
    identical between the two because the unified arch reuses the same
    text ``LanguageModel``.
    """
    from mlx_lm.utils import load_tokenizer

    p = Path(model_path)
    if not p.is_dir():
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(str(model_path)))

    config = json.loads((p / "config.json").read_text())
    text_config = config.get("text_config", config)

    # The outer arch the checkpoint actually declares (``gemma4`` /
    # ``gemma4_unified`` / ``gemma4_assistant``). Prefer it over the
    # ``default_model_type`` hint so the wrapper reports the real arch
    # (e.g. ``gemma4_assistant``) instead of a coarser default.
    routed_model_type = config.get("model_type") or default_model_type

    TextConfig, LanguageModel = resolve_classes()

    tc = TextConfig.from_dict(text_config)

    # Guard cross-layer KV-sharing before building the model: log (INFO) if the
    # checkpoint won't share (num_kv_shared_layers=0 — normal for the dense
    # sizes), raise on a malformed split. See _check_kv_share_config.
    _check_kv_share_config(text_config, tc, str(model_path))

    language_model = LanguageModel(tc)

    # Wrap for mlx-lm compatibility
    model = Gemma4TextWrapper(language_model, routed_model_type=routed_model_type)

    # Load weights once up front (mmap-backed, cheap) — we'll feed these
    # back into ``model.load_weights`` after quantization. Sanitize per
    # shard so peak memory stays proportional to the text-only model,
    # not the full multimodal checkpoint (#123).
    weight_files = sorted(
        f for f in p.glob("*.safetensors") if not f.name.startswith("._")
    )
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files in {p}")
    sanitized = {}
    for wf in weight_files:
        shard = mx.load(str(wf))
        sanitized.update(model.sanitize(shard))
        del shard

    # Identify layers the checkpoint stored as bare fp16/bf16 instead of
    # quantized (uint32 ``.weight`` + ``.scales`` + ``.biases``).
    # mlx-community's QAT pipeline intentionally leaves a few layers in
    # full precision — on the gemma-4-e2b/e4b "altup" variants this
    # includes ``per_layer_model_projection``. Our prior code applied
    # ``nn.quantize`` blanket, converted that Linear to QuantizedLinear,
    # then loaded a bare bfloat16 ``.weight`` into it. At inference time
    # ``mx.quantized_matmul`` raised:
    #
    #   ValueError: [quantized_matmul] The weight matrix should be
    #   uint32 but received bfloat16
    #
    # …and the server emitted 0 tokens with finish_reason=length. Self-
    # tuning the predicate from disk dtype keeps the post-quantize model
    # bit-exactly consistent with what mlx-community shipped: layers
    # they quantized stay quantized, layers they left fp16 stay fp16.
    # On variants where every ``.weight`` IS quantized (12b/26b/31b
    # don't even have per_layer_model_projection), this set is empty
    # and behavior is identical to the previous code.
    skip_quant_paths = _bare_fp_weight_paths(sanitized)

    # Apply quantization config if present (converts Linear → QuantizedLinear)
    quant_config = config.get("quantization", config.get("quantization_config"))
    if quant_config:
        default_bits = quant_config.get("bits", 4)
        default_gs = quant_config.get("group_size", 64)

        # Build per-layer override map from config (mixed quantization)
        # Keys like "language_model.model.layers.0.mlp.gate_proj" → {bits:8, group_size:64}
        overrides = {}
        for k, v in quant_config.items():
            if isinstance(v, dict) and "bits" in v:
                overrides[k] = {
                    kk: vv
                    for kk, vv in v.items()
                    if kk in ("bits", "group_size", "mode")
                }

        if skip_quant_paths:
            logger.info(
                "[gemma4] %d layer(s) kept as fp16 per checkpoint "
                "(e.g. %s) — won't be quantized",
                len(skip_quant_paths),
                next(iter(skip_quant_paths)),
            )

        if overrides:
            logger.info(
                "[gemma4] Mixed quantization: %d-bit default, %d overrides (8-bit MLP)",
                default_bits,
                len(overrides),
            )

            def _class_predicate(path, module):
                if not hasattr(module, "to_quantized"):
                    return False
                if _path_matches_any_suffix(path, skip_quant_paths):
                    return False
                # Check per-layer overrides
                # Override keys use "language_model.model.layers..." but nn.quantize
                # sees "model.layers..." (relative to wrapper). Match by suffix.
                for override_path, override_cfg in overrides.items():
                    # Strip common prefixes for matching
                    suffix = override_path.split("language_model.model.")[-1]
                    if path.endswith(suffix):
                        return override_cfg
                return {"bits": default_bits, "group_size": default_gs}

            nn.quantize(model, class_predicate=_class_predicate)
        else:
            logger.info(
                "[gemma4] Applying %d-bit quantization (group_size=%d)",
                default_bits,
                default_gs,
            )

            def _class_predicate(path, module):
                if not hasattr(module, "to_quantized"):
                    return False
                if _path_matches_any_suffix(path, skip_quant_paths):
                    return False
                return True

            nn.quantize(
                model,
                class_predicate=_class_predicate,
                group_size=default_gs,
                bits=default_bits,
            )

    model.load_weights(list(sanitized.items()), strict=False)

    # Verify weights loaded
    test_param = model.language_model.model.embed_tokens
    if hasattr(test_param, "scales") and mx.all(test_param.scales == 0).item():
        logger.warning(
            "[gemma4] Embedding scales are zero — quantized model may have issues"
        )

    # Load tokenizer
    tokenizer_config = tokenizer_config or {}
    eos_token_ids = config.get("eos_token_id", text_config.get("eos_token_id"))
    tokenizer = load_tokenizer(p, tokenizer_config, eos_token_ids=eos_token_ids)

    logger.info(
        "[gemma4] Loaded %s text-only model via LLM path (%d layers)",
        routed_model_type,
        len(model.layers),
    )
    return model, tokenizer
