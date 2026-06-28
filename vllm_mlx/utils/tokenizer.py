# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities with fallback support for non-standard tokenizers.

Some models (e.g., Nemotron) use non-standard tokenizer configurations
that transformers doesn't recognize. This module provides fallback loading
directly from tokenizer.json.
"""

import json
import logging
from pathlib import Path

from .chat_templates import DEFAULT_CHATML_TEMPLATE, NEMOTRON_CHAT_TEMPLATE

logger = logging.getLogger(__name__)

# Models that require tokenizer fallback
FALLBACK_MODELS = [
    "nemotron",
    "NVIDIA-Nemotron",
]


def _needs_tokenizer_fallback(model_name: str) -> bool:
    """Check if model needs tokenizer fallback."""
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in FALLBACK_MODELS)


# Attribute name used to stash the union of ``generation_config.json``
# EOS ids on raw HF tokenizers (mlx-vlm processors). Read by
# ``Scheduler._get_stop_tokens`` and ``MLLMScheduler._get_stop_tokens``
# as a fourth-source union, alongside the legacy
# ``eos_token_id`` / ``eos_token_ids`` / ``_eos_token_ids`` surfaces.
# Public so consumers outside this module (DFlash drafter, future
# code paths) can read it without importing private symbols.
RAPID_EXTRA_EOS_ATTR = "_rapid_extra_eos_token_ids"


# Characters that mark a *broken* GPT-2 byte-level BPE decode path.
# When ``tokenizer.decode([id])`` leaks any of these, the underlying
# fast-tokenizer ``decoder`` is mis-configured (typically a Llama
# SentencePiece decoder paired with a Qwen3 / GPT-2 byte-level BPE
# vocab — see ``repair_byte_level_decoder`` docstring for the
# full diagnosis). The repair probe samples a known byte-level pretty
# token and asserts the decode is clean before declaring the tokenizer
# healthy.
_BYTE_LEVEL_MOJIBAKE_MARKERS: tuple[str, ...] = (
    "Ġ",  # 'Ġ' — GPT-2 byte-level encoding of space
    "Ċ",  # 'Ċ' — GPT-2 byte-level encoding of newline
    "ĉ",  # 'ĉ' — GPT-2 byte-level encoding of tab
)

# SentencePiece metaspace marker — ``▁`` (U+2581). Hybrid SP/byte-level
# tokenizers (Gemma 4, future GG-style models) encode word boundaries
# with this character and rely on a ``Replace("▁", " ")`` decoder step
# to surface them as ASCII spaces. Issue #950 (Gemma 4): swapping the
# whole decoder for a bare GPT-2 ``ByteLevel`` drops that ``Replace``
# step and corrupts EVERY space in model output. Gate 2 in
# ``repair_byte_level_decoder`` detects this configuration and bails
# before any mutation; gate 3 catches future hybrids that slip past
# gate 2 by post-swap-decoding a spaced sample and reverting if any
# ``▁`` leaks. Both gates use this marker.
_METASPACE_MARKER = "▁"  # ▁


def _decoder_has_metaspace_replace(decoder) -> bool:
    """Return True if ``decoder`` contains a ``Replace("▁", " ")`` step.

    SentencePiece-metaspace tokenizers (Gemma family, Llama base, ...)
    encode word boundaries as ``▁`` (U+2581) in the vocab and rely on a
    ``Replace("▁", " ")`` decoder step to surface them as ASCII spaces.
    A ``Replace`` step may be the top-level decoder OR nested inside a
    ``Sequence`` (e.g. Gemma 4 ships
    ``Sequence([Replace("▁"," "), ByteFallback(), Fuse()])``).

    Gate 2 of issue #950 (Gemma 4): when this returns True, the caller
    must NOT swap the decoder out for a bare ``ByteLevel`` — that would
    drop the ``Replace`` step and corrupt every space in model output.
    Such tokenizers are HYBRIDS (SP metaspace + legit GPT-2-pretty byte
    tokens) and the rare cosmetic byte-token issue PR #793 was solving
    is not worth universal space corruption.

    Inspects the Rust decoder via ``__getstate__`` which returns a JSON
    bytes blob describing the decoder tree. We walk it for any
    ``{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}``
    node (or ``Regex`` variant of the pattern). Returns False on any
    introspection failure — a fail-open default that does not block
    legitimate repairs on tokenizers whose state can't be parsed.
    """
    try:
        state_raw = decoder.__getstate__()
    except Exception:
        return False
    try:
        state = json.loads(state_raw)
    except Exception:
        return False

    def _walk(node) -> bool:
        if not isinstance(node, dict):
            return False
        ntype = node.get("type")
        if ntype == "Replace":
            pattern = node.get("pattern") or {}
            content = node.get("content", "")
            # The ``pattern`` slot is a discriminated union — either
            # ``{"String": "<lit>"}`` or ``{"Regex": "<re>"}``. We
            # accept either if it matches the metaspace marker.
            pattern_str = pattern.get("String") or pattern.get("Regex") or ""
            if pattern_str == _METASPACE_MARKER and content == " ":
                return True
        if ntype == "Sequence":
            for child in node.get("decoders", []) or []:
                if _walk(child):
                    return True
        return False

    return _walk(state)


def repair_byte_level_decoder(tokenizer) -> bool:
    """Repair a mis-configured byte-level BPE decoder in place.

    Bug D-DETOK-BPE (rapid-mlx 0.7/0.8 series): every DeepSeek-R1
    distill on Qwen3 / Llama bases (``mlx-community/DeepSeek-R1-0528-
    Qwen3-8B-4bit``, ``DeepSeek-R1-Distill-Qwen-32B-4bit``, etc.) ships
    a ``tokenizer_config.json`` declaring ``tokenizer_class:
    LlamaTokenizerFast``. The Rust fast tokenizer is loaded with the
    correct GPT-2 ``ByteLevel`` *encoder* (so encode is fine), but
    transformers' ``LlamaTokenizerFast`` then *overrides* the decoder
    chain with the SentencePiece convention::

        Sequence([Replace("▁", " "), ByteFallback(), Fuse(),
                  Strip(" ", start=1, stop=0)])

    even though the vocab uses GPT-2 byte-level pretty tokens (``Ġ``,
    ``Ċ``, ``âĢľ``, ``Â°``…). Result: ``tokenizer.decode([6771])``
    returns ``"ĠLet"`` instead of ``" Let"``, so every byte-level pretty
    token leaks **verbatim** into ``reasoning_content``, ``content``,
    streaming ``delta.*`` fields, and ``/v1/completions[0].text``. This
    happens at the tokenizer layer, *not* per-parser, which is why all
    user-facing surfaces (chat stream, chat non-stream, raw completions)
    are affected on every affected alias.

    The repair: detect the mismatch by probing a token whose pretty form
    starts with ``Ġ`` or ``Ċ`` (we use the first vocab id whose
    ``convert_ids_to_tokens`` output begins with such a marker), and if
    ``decode([id])`` still contains the marker, swap the live
    ``backend_tokenizer.decoder`` for a plain GPT-2 ``ByteLevel`` decoder.
    The vocab itself is correct — only the decoder side needs swapping.

    **Two safety gates for HYBRID tokenizers (issue #950, Gemma 4):**

    * **Gate 2** — short-circuit when the existing decoder already
      contains a ``Replace("▁", " ")`` step. Such tokenizers are
      SentencePiece-metaspace hybrids: vocab has both ``▁``-prefixed
      space tokens (the dominant case) AND a few legit GPT-2-pretty
      byte tokens (e.g. Gemma 4's id-240630 ``ĉ`` for tab). The byte
      probe trips on those rare tokens, but swapping the decoder out
      drops the ``Replace`` step and corrupts every space — a
      universal cosmetic regression in exchange for a rare cosmetic
      fix. Bail before any mutation.

    * **Gate 3** — even after the swap clears the probe, decode a
      spaced sample (``encode("a b c")`` → ``decode(...)``) and assert
      no ``▁`` (U+2581) leaks. If it does, restore the original
      decoder and return False. This catches any future hybrid that
      slips past gate 2.

    Idempotent: a second call on a healthy tokenizer is a no-op.

    Returns ``True`` if a repair was applied, ``False`` otherwise.

    Note: also unwraps ``mlx_lm.tokenizer_utils.TokenizerWrapper`` —
    ``decode`` is forwarded to ``_tokenizer`` via ``__getattr__`` so
    patching the inner backend is sufficient; both the wrapper's own
    ``decode`` callers and the raw HF ``decode`` callers see the fix.
    """
    if tokenizer is None:
        return False

    # Three tokenizer shapes flow through Rapid-MLX:
    # 1. ``mlx_lm.tokenizer_utils.TokenizerWrapper`` — wraps an HF
    #    tokenizer; ``decode`` is forwarded via ``__getattr__``.
    # 2. ``transformers.PreTrainedTokenizerFast`` (and subclasses,
    #    including ``LlamaTokenizerFast``) — the canonical HF fast
    #    shape; the Rust backend lives on ``backend_tokenizer``.
    # 3. Slow / pure-Python HF tokenizers — no Rust backend, byte-level
    #    handling is built in to the slow decoder, so no repair needed.
    #
    # The mlx-lm wrapper *also* has a ``_tokenizer`` attribute, but on
    # an HF fast tokenizer ``_tokenizer`` is the raw Rust ``Tokenizer``
    # object (no ``backend_tokenizer``). We probe both candidates and
    # pick the one that exposes ``backend_tokenizer``.
    candidates = [tokenizer]
    if hasattr(tokenizer, "_tokenizer"):
        candidates.append(tokenizer._tokenizer)
    inner = next(
        (c for c in candidates if hasattr(c, "backend_tokenizer")),
        None,
    )
    if inner is None:
        # Slow / pure-Python tokenizer — no Rust decoder to swap. The
        # slow decoder paths handle byte-level natively, so this branch
        # is healthy by construction.
        return False
    backend = inner.backend_tokenizer

    # Gate 2 (issue #950): HYBRID tokenizers (SentencePiece metaspace +
    # legit GPT-2-pretty byte tokens) — e.g. Gemma 4 — keep their
    # existing decoder. Their vocab has both ``▁``-prefixed tokens AND
    # a few legit byte tokens like ``ĉ`` (tab); the byte probe trips
    # on the legit byte tokens, but swapping the decoder out drops the
    # ``Replace("▁", " ")`` step and corrupts every space.
    #
    # PR #793's target — DeepSeek/Qwen with a mis-paired Llama SP
    # decoder over a pure-GPT-2-byte-level vocab — ALSO carries a
    # ``Replace("▁", " ")`` step in its (broken) decoder, so the decoder-
    # shape check alone would over-fire. The disambiguator: pure-GPT-2-
    # byte-level vocabs (DeepSeek distills, Qwen3) ENCODE spaces as
    # ``Ġ`` and have NO ``▁`` in their vocab — so the (mis-applied)
    # ``Replace`` step is a no-op and the swap is safe. Hybrid Gemma-4-
    # style vocabs ENCODE spaces as ``▁`` — encoding "a b c" yields
    # tokens containing ``▁``. We tell the two apart by encoding a
    # known-spaced sample: if the resulting tokens contain ``▁``, the
    # ``Replace`` step is LOAD-BEARING and we must not swap.
    if _decoder_has_metaspace_replace(backend.decoder):
        try:
            spaced_ids = inner.encode("a b c", add_special_tokens=False)
            spaced_tokens = inner.convert_ids_to_tokens(spaced_ids)
        except Exception:
            spaced_tokens = []
        if any(
            isinstance(t, str) and _METASPACE_MARKER in t for t in spaced_tokens
        ):
            # Hybrid tokenizer: vocab uses ``▁`` for spaces AND the
            # decoder has the matching ``Replace`` step. Bail without
            # mutation — the cosmetic byte-token quirk PR #793 was
            # chasing is not worth corrupting every space.
            logger.debug(
                "repair_byte_level_decoder: skipping %s — decoder has "
                "load-bearing Replace('%s', ' ') step (hybrid "
                "SentencePiece-metaspace tokenizer)",
                type(inner).__name__,
                _METASPACE_MARKER,
            )
            return False

    # Find a probe id whose pretty token starts with a byte-level marker.
    # We scan the *entire* vocab (codex r2 NIT) — a 4 KB id prefix cap
    # silently skips valid byte-level vocabs whose byte tokens all live
    # past id 4096 (e.g. tokenizers that pack specials + reserved ids
    # ahead of the BPE merges). The scan walks the dict from
    # ``get_vocab()`` (token→id), which is O(vocab) one-shot and short-
    # circuits on the first match — no per-id ``convert_ids_to_tokens``
    # round-trip, so even a 200k-entry vocab probes in <5 ms.
    probe_id: int | None = None
    probe_pretty: str | None = None
    try:
        vocab = inner.get_vocab()
    except Exception:
        return False
    # ``get_vocab`` returns ``{pretty: id}``. Sort by id so the probe
    # is deterministic across HF tokenizer versions (some return dicts
    # in insertion order, others in hash order).
    for pretty, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        if not isinstance(pretty, str):
            continue
        if any(pretty.startswith(m) for m in _BYTE_LEVEL_MOJIBAKE_MARKERS):
            probe_id = tid
            probe_pretty = pretty
            break

    if probe_id is None:
        # Not a byte-level vocab — nothing to repair.
        return False

    try:
        decoded = inner.decode([probe_id], skip_special_tokens=False)
    except Exception:
        return False

    if not any(m in decoded for m in _BYTE_LEVEL_MOJIBAKE_MARKERS):
        # Decoder is already correct.
        return False

    # Decoder is broken: swap in a plain ByteLevel decoder. Save the
    # original so we can restore it on verification failure (codex r1
    # BLOCKING: a "revert" comment that doesn't actually revert leaves
    # an unverified mutation in place).
    original_decoder = backend.decoder
    try:
        from tokenizers import decoders as _decoders

        backend.decoder = _decoders.ByteLevel()
    except Exception as exc:  # noqa: BLE001 — defensive only
        logger.warning(
            "repair_byte_level_decoder: failed to swap decoder on %s: %s",
            type(inner).__name__,
            exc,
        )
        return False

    # Verify the swap actually fixed the decode. If the model genuinely
    # uses a non-ByteLevel pretty token (unlikely on real models), put
    # the original decoder back so we don't silently corrupt output.
    try:
        verify = inner.decode([probe_id], skip_special_tokens=False)
    except Exception:
        verify = decoded
    if any(m in verify for m in _BYTE_LEVEL_MOJIBAKE_MARKERS):
        # Restore the original decoder — if ByteLevel can't clear the
        # mojibake either, the vocab is in a shape we don't understand
        # and changing the decoder is a net behaviour change. Honour
        # the "non-destructive on unknown vocab" contract by undoing
        # the swap before returning False.
        try:
            backend.decoder = original_decoder
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "repair_byte_level_decoder: could not restore original "
                "decoder on %s after failed verification: %s",
                type(inner).__name__,
                exc,
            )
        logger.warning(
            "repair_byte_level_decoder: swap did not clear mojibake on %s "
            "(probe id=%d pretty=%r decoded=%r); restored original decoder",
            type(inner).__name__,
            probe_id,
            probe_pretty,
            verify,
        )
        return False

    # Gate 3 (issue #950): even after the probe clears, decode a spaced
    # sample and ensure no ``▁`` (U+2581) leaks. A hybrid tokenizer
    # whose decoder didn't trip gate 2 — for instance one whose
    # ``Replace`` step has a different shape we don't recognise, or one
    # where the metaspace marker appears via a different decoder
    # primitive — would here surface ``▁`` in the round-trip and we
    # must revert so we never ship corrupted spaces to users.
    try:
        spaced_ids = inner.encode("a b c", add_special_tokens=False)
        spaced_decoded = inner.decode(spaced_ids, skip_special_tokens=False)
    except Exception:
        spaced_decoded = ""
    if _METASPACE_MARKER in spaced_decoded:
        try:
            backend.decoder = original_decoder
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "repair_byte_level_decoder: could not restore original "
                "decoder on %s after spaced-sample verification failed: %s",
                type(inner).__name__,
                exc,
            )
        logger.warning(
            "repair_byte_level_decoder: post-swap spaced-sample decode "
            "leaked metaspace marker on %s (encode('a b c') -> %r); "
            "restored original decoder",
            type(inner).__name__,
            spaced_decoded,
        )
        return False

    logger.info(
        "repair_byte_level_decoder: swapped %s.backend_tokenizer.decoder to "
        "ByteLevel (probe id=%d pretty=%r -> decoded=%r)",
        type(inner).__name__,
        probe_id,
        probe_pretty,
        verify,
    )
    return True


def augment_eos_token_ids_from_generation_config(
    tokenizer, model_path_or_name: str
) -> None:
    """Union ``generation_config.json``'s ``eos_token_id`` list into
    the tokenizer's stop-token surface so the chat-template
    terminator halts generation.

    Why this is necessary:

    The HuggingFace convention is that ``tokenizer_config.json``
    declares a single primary ``eos_token`` (and therefore a single
    ``tokenizer.eos_token_id``), while ``generation_config.json``
    declares the *full* set of stop tokens — including the
    chat-template terminator that's distinct from the model-level
    ``<eos>``. Concretely:

    * Gemma 3 / 3n: ``tokenizer.eos_token_id == 1`` (``<eos>``);
      ``generation_config.json`` declares ``[1, 106]`` where 106 is
      ``<end_of_turn>``.
    * Qwen3 / Qwen2.5: ``tokenizer.eos_token_id == 151645``
      (``<|im_end|>``); ``generation_config.json`` declares
      ``[151645, 151643]`` where 151643 is ``<|endoftext|>``.
    * Llama 3: ``tokenizer.eos_token_id == 128001``
      (``<|end_of_text|>``); ``generation_config.json`` declares
      ``[128001, 128009]`` where 128009 is ``<|eot_id|>``.

    Without this augmentation every downstream consumer that halts
    on ``eos_token_id`` (our schedulers, mlx-lm's ``BatchGenerator``,
    DFlash drafter, streaming detokenizer) misses the chat-template
    terminator and the model emits it as a literal token until
    ``max_tokens`` is hit. User-visible symptom on Gemma 3n:
    ``hello -> "Okay.<end_of_turn><end_of_turn>..."``.

    Two tokenizer shapes flow through Rapid-MLX:

    1. **mlx-lm ``TokenizerWrapper``** — has a curated
       ``_eos_token_ids: set[int]`` plus an ``add_eos_token`` method
       that grows it. mlx-lm's own ``BatchGenerator`` reads this
       set, so mutating it here also fixes upstream batching.

    2. **Raw HF tokenizer** (mlx-vlm processors return these
       directly — ``Gemma3Processor.tokenizer`` is a
       ``GemmaTokenizer``, not a wrapper). HF defines both
       ``eos_token_id`` and ``eos_token_ids`` as property
       descriptors backed by setters that reject non-string values,
       so we can't assign a list to either. Instead we stash the
       union on a Rapid-MLX-owned attribute name
       (``RAPID_EXTRA_EOS_ATTR``) that doesn't collide with any HF
       descriptor; both schedulers' source-4 union branch reads it.

    The fix is one mutation point per model load rather than an
    N-way patch across every consumer.
    """
    from .generation_config import load_generation_config_eos_ids

    extras = load_generation_config_eos_ids(model_path_or_name)
    if not extras:
        return

    # Shape 1: mlx-lm TokenizerWrapper. The ``_eos_token_ids`` set
    # is the curated stop set mlx-lm's BatchGenerator consults; we
    # add to it directly rather than going through ``add_eos_token``
    # (which exists but is also defined on raw HF tokenizers with
    # totally different semantics — see Shape 2 below).
    wrapper_set = getattr(tokenizer, "_eos_token_ids", None)
    if isinstance(wrapper_set, set):
        before = set(wrapper_set)
        wrapper_set.update(extras)
        added = sorted(set(wrapper_set) - before)
        if added:
            logger.info(
                "augment_eos: added %s to TokenizerWrapper stop set for %s",
                added,
                model_path_or_name,
            )
        return

    # Shape 2: raw HF tokenizer (e.g. ``GemmaTokenizer`` returned by
    # mlx-vlm processors). HF defines ``eos_token_id`` and
    # ``eos_token_ids`` as property descriptors backed by setters
    # that reject non-string values — so we can't just assign a
    # list. Instead stash on a Rapid-MLX-owned attribute name that
    # doesn't collide with any HF descriptor, and have the
    # schedulers' source-4 union branch read it. This avoids
    # monkey-patching HF internals and keeps ``tokenizer.eos_token``
    # (used by other HF code paths) untouched.
    try:
        existing = getattr(tokenizer, RAPID_EXTRA_EOS_ATTR, None) or ()
        merged_set = set(int(x) for x in existing) | set(extras)
        merged = tuple(sorted(merged_set))
        setattr(tokenizer, RAPID_EXTRA_EOS_ATTR, merged)
        logger.info(
            "augment_eos: set %s=%s on %s for %s",
            RAPID_EXTRA_EOS_ATTR,
            list(merged),
            type(tokenizer).__name__,
            model_path_or_name,
        )
    except Exception as exc:  # noqa: BLE001 — defensive only
        logger.debug(
            "augment_eos: could not stash extras on %s (%s)",
            type(tokenizer).__name__,
            exc,
        )


def _apply_chat_template_sidecar(model_path: Path, tokenizer) -> bool:
    """Populate ``tokenizer.chat_template`` from a sidecar file if missing.

    Newer HuggingFace repos ship the chat template as a standalone file
    next to ``tokenizer_config.json`` instead of embedding it. Two
    conventions exist:

      - ``chat_template.jinja`` (raw jinja, the modern transformers
        ≥4.43 default — DeepSeek V4, some Qwen builds)
      - ``chat_template.json`` (single-key ``{"chat_template": "..."}``
        wrapper — used by mlx-community Mistral Small 3.1 and newer
        repos that follow the older HF Tokenizers sidecar convention)

    Both ``AutoTokenizer.from_pretrained`` and ``mlx_lm.load``'s
    ``TokenizerWrapper`` fail to auto-merge ``chat_template.json`` on
    transformers ≤5.6 — ``tokenizer.chat_template`` comes back ``None``
    and every ``/v1/chat/completions`` request 400s with
    "Cannot use chat template functions". Surfaced on 2026-05-22
    fresh-PyPI v0.6.65 onboarding sweep against
    ``mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit``.

    Returns True if a sidecar template was applied, False otherwise.
    """
    if getattr(tokenizer, "chat_template", None):
        return False

    jinja_path = model_path / "chat_template.jinja"
    if jinja_path.exists():
        # utf-8-sig strips a UTF-8 BOM if the file was saved with one —
        tokenizer.chat_template = jinja_path.read_text(encoding="utf-8-sig")
        logger.info("Chat template loaded from chat_template.jinja sidecar")
        return True

    json_path = model_path / "chat_template.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"Found chat_template.json at {json_path} but failed to parse: {e}"
            )
            return False
        template = data.get("chat_template")
        if isinstance(template, str) and template:
            tokenizer.chat_template = template
            logger.info("Chat template loaded from chat_template.json sidecar")
            return True
        logger.warning(
            f"chat_template.json at {json_path} has no 'chat_template' string key; "
            f"got keys={list(data.keys())}"
        )
    return False


def _resolve_model_path(model_name: str) -> Path | None:
    """Resolve a HuggingFace ``model_name`` to a local snapshot directory.

    Returns ``None`` (instead of raising) when the model can't be located
    locally — callers use this for best-effort sidecar lookup and should
    skip the sidecar branch silently if the path can't be resolved
    (offline / non-existent model / weird hub state).
    """
    local = Path(model_name)
    if local.is_dir():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_name))
    except Exception as e:
        logger.debug(f"_resolve_model_path({model_name}) failed: {e}")
        return None


def _register_vendored_archs() -> None:
    """Make vendored model architectures visible to mlx-lm's importlib lookup.

    mlx-lm resolves model_type → module via `importlib.import_module(
    f"mlx_lm.models.{model_type}")`. Pre-registering our vendored modules in
    sys.modules under that path lets it find them transparently. Idempotent.
    """
    import sys

    if "mlx_lm.models.deepseek_v4" not in sys.modules:
        try:
            from ..models import deepseek_v4 as _ds_v4

            # setdefault is atomic under the GIL; harmless if a concurrent
            # caller raced ahead (we'd cache the same module either way).
            sys.modules.setdefault("mlx_lm.models.deepseek_v4", _ds_v4)
        except Exception as e:
            logger.debug(f"deepseek_v4 vendored module unavailable: {e}")


# model_types served by vllm_mlx.models.* shims. transformers' AutoConfig /
# PreTrainedConfig won't recognize these, and mlx-lm's load() internally
# uses AutoTokenizer (which routes through AutoConfig). We must skip that
# path entirely for these models and use the lower-level load_model() +
# direct tokenizer.json load instead.
_VENDORED_MODEL_TYPES = {"deepseek_v4"}


def _is_vendored_arch_model(model_name: str) -> bool:
    """Return True if model's config.json declares a model_type we vendor."""
    try:
        local = Path(model_name)
        if local.is_dir():
            config_path = local / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(repo_id=model_name, filename="config.json")
            )
        if not config_path.exists():
            return False
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("model_type") in _VENDORED_MODEL_TYPES
    except Exception as e:
        logger.debug(f"_is_vendored_arch_model({model_name}) failed: {e}")
        return False


def load_model_with_fallback(model_name: str, tokenizer_config: dict = None):
    """
    Load model and tokenizer with fallback for non-standard tokenizers.

    Args:
        model_name: HuggingFace model name or local path
        tokenizer_config: Optional tokenizer configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    from mlx_lm import load

    _register_vendored_archs()
    tokenizer_config = tokenizer_config or {}

    # Check if model needs fallback (e.g., Nemotron)
    if _needs_tokenizer_fallback(model_name):
        logger.info(
            f"Model {model_name} requires tokenizer fallback, loading directly..."
        )
        return _load_with_tokenizer_fallback(model_name)

    # Vendored architectures (e.g. deepseek_v4) — transformers' AutoConfig
    # doesn't know about them, so mlx-lm's high-level load() blows up
    # before we get a chance to handle the error. Route directly to the
    # lower-level load_model() + raw tokenizer.json fallback.
    if _is_vendored_arch_model(model_name):
        logger.info(
            f"Model {model_name} uses a vendored architecture, "
            "skipping AutoConfig path and loading directly..."
        )
        return _load_with_tokenizer_fallback(model_name)

    # Gemma 4: mlx-lm 0.31+ supports it natively. Only use our wrapper
    # for older mlx-lm versions that lack gemma4 model support.
    from ..models.gemma4_text import is_gemma4_model

    if is_gemma4_model(model_name):
        try:
            # Try native mlx-lm load first (0.31+)
            model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
            logger.info("Gemma 4 loaded natively via mlx-lm")
            if not getattr(tokenizer, "chat_template", None):
                mp = _resolve_model_path(model_name)
                if mp is not None:
                    _apply_chat_template_sidecar(mp, tokenizer)
            augment_eos_token_ids_from_generation_config(tokenizer, model_name)
            repair_byte_level_decoder(tokenizer)
            return model, tokenizer
        except Exception as e:
            # Fall back to our wrapper for older mlx-lm versions
            # that lack native gemma4 architecture support
            from ..models.gemma4_text import load_gemma4_text

            logger.info(
                f"Gemma 4 native load failed ({e}), "
                "falling back to text-only wrapper (legacy mlx-lm)"
            )
            return load_gemma4_text(model_name, tokenizer_config)

    try:
        model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
        # mlx_lm.load() succeeds but sanitize() may have silently
        # stripped mtp.* weights.  Check if the config declares MTP
        # layers and the model came back without a .mtp attribute;
        # if so, re-inject from the safetensors on disk.
        _try_inject_mtp_post_load(model, model_name)
        # Sidecar chat-template recovery: AutoTokenizer doesn't merge
        # ``chat_template.json`` on transformers ≤5.6, leaving
        # ``tokenizer.chat_template`` None for newer mlx-community repos
        # like Mistral Small 3.1. /v1/chat/completions then 400s. Try
        # to load the sidecar before returning so chat endpoints work.
        if not getattr(tokenizer, "chat_template", None):
            mp = _resolve_model_path(model_name)
            if mp is not None:
                _apply_chat_template_sidecar(mp, tokenizer)
        augment_eos_token_ids_from_generation_config(tokenizer, model_name)
        repair_byte_level_decoder(tokenizer)
        return model, tokenizer
    except ValueError as e:
        # Fallback for models with non-standard tokenizers, OR newer model_types
        # transformers' AutoConfig hasn't learned about yet (e.g. deepseek_v4
        # before transformers PR #45643 lands). The vendored arch can still load
        # the weights — we just need to bypass AutoTokenizer.
        if (
            "TokenizersBackend" in str(e)
            or "Tokenizer class" in str(e)
            or "does not recognize this architecture" in str(e)
        ):
            logger.warning(f"Standard tokenizer loading failed, using fallback: {e}")
            return _load_with_tokenizer_fallback(model_name)
        # Fallback for models with extra/missing weights (e.g., vision tower, MTP layers).
        # Retry with strict=False to discard extra weights.
        elif "parameters not in model" in str(e) or (
            "Missing" in str(e) and "parameters" in str(e)
        ):
            logger.warning(
                f"Model has extra/missing parameters (likely VLM / MTP weights), "
                f"retrying with strict=False: {e}"
            )
            return _load_strict_false(model_name, tokenizer_config)
        else:
            raise


def _load_strict_false(model_name: str, tokenizer_config: dict = None):
    """Load model with strict=False to discard extra weights (e.g., vision tower, MTP)."""
    from mlx_lm.utils import load_model, load_tokenizer

    local_path = Path(model_name)
    if local_path.is_dir():
        model_path = local_path
    else:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_name))

    model, config = load_model(model_path, strict=False)
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config or {},
        eos_token_ids=config.get("eos_token_id", None),
    )
    # Inject MTP support if model has MTP config + weights
    _try_inject_mtp(model, model_path, config)
    _apply_chat_template_sidecar(model_path, tokenizer)
    augment_eos_token_ids_from_generation_config(tokenizer, str(model_path))
    repair_byte_level_decoder(tokenizer)
    return model, tokenizer


def _read_num_mtp_layers(config: dict) -> int:
    """Read num_nextn_predict_layers from config, checking text_config too.

    Multimodal checkpoints (VLM + MTP) store this under text_config,
    while text-only checkpoints put it at the top level.  Fixes #121.
    """
    n = config.get("num_nextn_predict_layers", 0)
    if n == 0:
        n = config.get("text_config", {}).get("num_nextn_predict_layers", 0)
    return n


def _try_inject_mtp(model, model_path, config):
    """Inject MTP support if model has MTP config + weights."""
    num = _read_num_mtp_layers(config)
    if num > 0:
        from ..patches.qwen3_next_mtp import inject_mtp_support

        # inject_mtp_support reads config["num_nextn_predict_layers"]
        # directly.  For VLM checkpoints where the field lives under
        # text_config, surface it to the top level so the injector
        # doesn't skip with "num_nextn_predict_layers=0".
        if config.get("num_nextn_predict_layers", 0) == 0:
            config = {**config, "num_nextn_predict_layers": num}
        inject_mtp_support(model, model_path, config)


def _try_inject_mtp_post_load(model, model_name):
    """Check if MTP weights exist but were stripped by sanitize(), and inject."""
    import json

    from mlx_lm.utils import _download

    model_path = _download(model_name)
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    num_mtp = _read_num_mtp_layers(config)
    if num_mtp > 0 and getattr(model, "mtp", None) is None:
        mtp_file = Path(model_path) / "model-mtp.safetensors"
        if mtp_file.exists():
            logger.info(
                f"[MTP] Found MTP config (layers={num_mtp}) and weights, injecting..."
            )
            _try_inject_mtp(model, model_path, config)
        else:
            logger.info(
                f"[MTP] Config has num_nextn_predict_layers={num_mtp} "
                "but model-mtp.safetensors not found, skipping MTP."
            )


def _load_non_strict(model_name: str, tokenizer_config: dict = None):
    """Load model with strict=False to skip extra weights (e.g., vision tower)."""
    from mlx_lm.utils import load_model, load_tokenizer

    local_path = Path(model_name)
    if local_path.is_dir():
        model_path = local_path
    else:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_name))

    model, _ = load_model(model_path, strict=False)
    tokenizer = load_tokenizer(model_path, tokenizer_config or {})
    _apply_chat_template_sidecar(model_path, tokenizer)
    augment_eos_token_ids_from_generation_config(tokenizer, str(model_path))
    repair_byte_level_decoder(tokenizer)
    return model, tokenizer


def _load_with_tokenizer_fallback(model_name: str):
    """Load model with fallback tokenizer for non-standard models like Nemotron."""
    from mlx_lm.utils import load_model

    logger.info("Loading with tokenizer fallback...")

    # Get model path - use local path if it exists, otherwise download from Hub
    local_path = Path(model_name)
    if local_path.is_dir():
        model_path = local_path
    else:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_name))

    # Load model
    model, _ = load_model(model_path)

    # Try to load tokenizer from tokenizer.json directly
    tokenizer_json = model_path / "tokenizer.json"
    if tokenizer_json.exists():
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        logger.info("Loading tokenizer from tokenizer.json")
        base_tokenizer = Tokenizer.from_file(str(tokenizer_json))

        # Read tokenizer_config.json for special tokens and chat template
        tokenizer_config_path = model_path / "tokenizer_config.json"
        bos_token = "<s>"
        eos_token = "</s>"
        unk_token = "<unk>"
        chat_template = None

        if tokenizer_config_path.exists():
            with open(tokenizer_config_path) as f:
                config = json.load(f)
                bos_token = config.get("bos_token", bos_token)
                eos_token = config.get("eos_token", eos_token)
                unk_token = config.get("unk_token", unk_token)
                chat_template = config.get("chat_template")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token="<pad>",
        )

        # Set chat template if available. Sidecar fallback (.jinja then
        # .json) is delegated to ``_apply_chat_template_sidecar`` so the
        # primary load path and this fallback stay in sync (Mistral
        # Small 3.1 ships .json sidecar; DeepSeek V4 ships .jinja).
        if chat_template:
            tokenizer.chat_template = chat_template
            logger.info("Chat template loaded from tokenizer_config.json")
        elif _apply_chat_template_sidecar(model_path, tokenizer):
            pass  # helper logs the sidecar source
        elif _needs_tokenizer_fallback(model_name):
            # Use official Nemotron chat template with thinking support
            tokenizer.chat_template = NEMOTRON_CHAT_TEMPLATE
            logger.info("Using official Nemotron chat template with thinking support")
        else:
            # Default simple ChatML format for other models
            tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE
            logger.info("Using default ChatML chat template")

        repair_byte_level_decoder(tokenizer)
        logger.info("Tokenizer loaded via fallback successfully")
        return model, tokenizer
    else:
        raise ValueError(f"No tokenizer.json found in {model_path}")
