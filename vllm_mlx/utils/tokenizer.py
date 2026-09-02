# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities with fallback support for non-standard tokenizers.

Some models (e.g., Nemotron) use non-standard tokenizer configurations
that transformers doesn't recognize. This module provides fallback loading
directly from tokenizer.json.
"""

import json
import logging
import os
from pathlib import Path

from .chat_templates import DEFAULT_CHATML_TEMPLATE, NEMOTRON_CHAT_TEMPLATE
from .model_file_guard import validate_local_model_file

logger = logging.getLogger(__name__)

_FALSE_ENV_VALUES = frozenset({"0", "false", "no", "off"})


def apply_remote_code_policy(
    tokenizer_config: dict | None,
) -> tuple[dict | None, bool]:
    """Apply the process-wide remote-code opt-out without changing defaults.

    ``None`` remains ``None`` when the environment variable is unset, which
    preserves the loader's historical defaults for non-serve call sites.  An
    explicit false value is authoritative across every caller of the shared
    loader, including bench and disk-stream paths.
    """
    configured = tokenizer_config is not None
    config = dict(tokenizer_config or {})
    requested = bool(config.get("trust_remote_code", True))
    raw = os.environ.get("RAPID_MLX_TRUST_REMOTE_CODE")
    if raw is not None and raw.strip().lower() in _FALSE_ENV_VALUES:
        config["trust_remote_code"] = False
        return config, False
    return (config if configured else None), requested


# Install the per-layer Indexer gate for REAP-pruned DeepseekV32 configs
# (e.g. mlx-community/pipenetwork-GLM-5.2-REAP50-MLX-4bit). The hook is
# placed here because this is the real `rapid-mlx serve` boot path:
#   cli -> server -> engine.batched._start_llm -> utils.tokenizer.load_model_with_fallback
#   -> mlx_lm.load -> mlx_lm.utils.load_model
# Install is idempotent (_LOCK + _INSTALLED early-return) and a no-op on
# configs that don't publish ``indexer_types``.
from ..patches.deepseek_v32_indexer_gate import (
    install_deepseek_v32_indexer_gate as _install_dsv32_indexer_gate,
)

# Same wiring rationale as the indexer gate above: install the Qwen3.5/3.6
# norm-shift correction on the canonical production model-load path.
# Idempotent + a no-op on checkpoints whose norm gains are already
# zero-centered.
from ..patches.qwen3_5_norm_shift import (
    install_qwen3_5_norm_shift_fix as _install_qwen3_5_norm_shift_fix,
)

# Both installers run below the import block so no module-level import
# follows an executable statement (E402).
_install_dsv32_indexer_gate()
_install_qwen3_5_norm_shift_fix()

# Models that require tokenizer fallback
FALLBACK_MODELS = [
    "nemotron",
    "NVIDIA-Nemotron",
]


def _needs_tokenizer_fallback(model_name: str) -> bool:
    """Check if model needs tokenizer fallback."""
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in FALLBACK_MODELS)


def _special_token_text(value, default: str | None) -> str | None:
    """Normalize tokenizer_config special tokens across HF representations."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return default


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
        if any(isinstance(t, str) and _METASPACE_MARKER in t for t in spaced_tokens):
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


# codex round 3 [NIT #2]: model_types served by vllm_mlx.models.* shims.
# transformers' AutoConfig / PreTrainedConfig won't recognize these, and
# mlx-lm's load() internally uses AutoTokenizer (which routes through
# AutoConfig). We must skip that path entirely for these models and use
# the lower-level load_model() + direct tokenizer.json load instead.
#
# Initialized with the archs whose vendor modules ship inside vllm_mlx and
# never need dynamic registration (``deepseek_v4`` is pure Python inside
# ``vllm_mlx.models`` — if that import fails, the wheel is broken and the
# earlier ``from ..models import deepseek_v4`` at every serve entry point
# has already crashed). Conditionally registered archs like ``hy_v3`` and
# ``gpt_oss_puzzle`` are added by ``_register_vendored_archs()`` only after
# their vendored module successfully installs in ``sys.modules``, so a failure
# there does not leave the arch advertised as vendored while the shim path
# silently short-circuits (which would push users into an opaque later
# model-load error instead of surfacing the actual registration failure).
_VENDORED_MODEL_TYPES: set[str] = {"deepseek_v4"}


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

    if "mlx_lm.models.cohere2_moe" not in sys.modules:
        try:
            from ..models import cohere2_moe as _cohere2_moe

            sys.modules.setdefault("mlx_lm.models.cohere2_moe", _cohere2_moe)
        except Exception as e:
            logger.warning(
                "cohere2_moe vendored module failed to register; North-Mini-Code "
                "will not load until resolved: %s",
                e,
            )
        else:
            _VENDORED_MODEL_TYPES.add("cohere2_moe")

    if "mlx_lm.models.hy_v3" not in sys.modules:
        # If mlx-lm ever ships native ``hy_v3`` support (upstream PR #1211
        # merges into 0.32+), defer to their copy so we don't shadow real
        # upstream bug fixes with a stale vendor. ``find_spec`` returns
        # ``None`` when the sub-module doesn't exist, which is the current
        # state on mlx-lm 0.31.3 → we fall through to our vendored install.
        import importlib.util as _importlib_util

        _native_spec = None
        try:
            _native_spec = _importlib_util.find_spec("mlx_lm.models.hy_v3")
        except (ImportError, ValueError):
            _native_spec = None

        if _native_spec is None:
            try:
                from ..models import hy_v3 as _hy_v3

                # Tencent Hunyuan 3 (295B/21B active MoE) — vendored from
                # ml-explore/mlx-lm PR #1211 (open, unreviewed since 2026-04-27).
                # Auto-defers to native support once mlx-lm 0.32+ merges the
                # upstream PR; delete this vendor block after that.
                sys.modules.setdefault("mlx_lm.models.hy_v3", _hy_v3)
            except Exception as e:
                # codex round 3 [NIT #2]: log at WARNING (not DEBUG) so the
                # actionable root cause surfaces the moment the vendor fails
                # to register — otherwise the user sees only a confusing
                # later model-load failure with no pointer at what actually
                # went wrong. Membership in ``_VENDORED_MODEL_TYPES`` is
                # only granted below on the success branch, so a failure
                # here leaves downstream code on the AutoTokenizer path
                # rather than the (broken) shim path.
                logger.warning(
                    "hy_v3 vendored module failed to register — "
                    "mlx-community/Hy3-preview-4bit will not load until "
                    "resolved: %s",
                    e,
                )
            else:
                # Success: promote to the vendored-arch set so the
                # tokenizer fallback path (``_is_vendored_arch_model``)
                # routes HY3 loads through the vendor shim instead of
                # ``AutoTokenizer`` (which doesn't recognize the arch in
                # transformers ≤ 5.12).
                _VENDORED_MODEL_TYPES.add("hy_v3")
        else:
            # Native ``mlx_lm.models.hy_v3`` is available — use it and
            # graduate ``hy_v3`` to the vendored set so the tokenizer
            # fallback still bypasses AutoTokenizer (transformers ≤ 5.12
            # still doesn't know the arch, native mlx-lm module or not).
            _VENDORED_MODEL_TYPES.add("hy_v3")

    if "mlx_lm.models.muse_glimmer" not in sys.modules:
        # Meta Muse Glimmer 30B — vendored text backbone (see
        # ``vllm_mlx/models/muse_glimmer.py`` for the why + sync policy).
        # Defer to native mlx-lm support the moment upstream ships it,
        # same probe as ``hy_v3`` above.
        import importlib.util as _importlib_util

        _muse_native_spec = None
        try:
            _muse_native_spec = _importlib_util.find_spec("mlx_lm.models.muse_glimmer")
        except (ImportError, ValueError):
            _muse_native_spec = None

        if _muse_native_spec is None:
            try:
                from ..models import muse_glimmer as _muse

                sys.modules.setdefault("mlx_lm.models.muse_glimmer", _muse)
            except Exception as e:
                logger.warning(
                    "muse_glimmer vendored module failed to register — "
                    "mlx-community/Muse-Glimmer-30B-* will not load until "
                    "resolved: %s",
                    e,
                )
            else:
                _VENDORED_MODEL_TYPES.add("muse_glimmer")
        else:
            _VENDORED_MODEL_TYPES.add("muse_glimmer")

    if "mlx_lm.models.bailing_hybrid" not in sys.modules:
        # inclusionAI Ling 3.0 family (tiny/flash) + Ling 2.6 — vendored
        # KDA+MLA hybrid backbone (see ``vllm_mlx/models/bailing_hybrid.py``
        # for the why + sync policy). Defers to native mlx-lm support the
        # moment upstream ships it (mlx-lm PR #1227 lineage), same probe
        # as ``hy_v3`` above.
        import importlib.util as _importlib_util

        _bailing_native_spec = None
        try:
            _bailing_native_spec = _importlib_util.find_spec(
                "mlx_lm.models.bailing_hybrid"
            )
        except (ImportError, ValueError):
            _bailing_native_spec = None

        if _bailing_native_spec is None:
            try:
                from ..models import bailing_hybrid as _bailing

                sys.modules.setdefault("mlx_lm.models.bailing_hybrid", _bailing)
            except Exception as e:
                logger.warning(
                    "bailing_hybrid vendored module failed to register — "
                    "inclusionAI/Ling-3.0-* will not load until resolved: %s",
                    e,
                )
            else:
                _VENDORED_MODEL_TYPES.add("bailing_hybrid")
        else:
            _VENDORED_MODEL_TYPES.add("bailing_hybrid")

    if "mlx_lm.models.gpt_oss_puzzle" not in sys.modules:
        # Puzzle is a heterogeneous GPT-OSS architecture from mlx-lm #1488.
        # Prefer a future native implementation rather than shadowing upstream
        # fixes with this vendor module indefinitely — same probe as ``hy_v3``
        # above.
        import importlib.util as _importlib_util

        _puzzle_native_spec = None
        try:
            _puzzle_native_spec = _importlib_util.find_spec(
                "mlx_lm.models.gpt_oss_puzzle"
            )
        except (ImportError, ValueError):
            _puzzle_native_spec = None

        if _puzzle_native_spec is None:
            try:
                from ..models import gpt_oss_puzzle as _gpt_oss_puzzle

                sys.modules.setdefault("mlx_lm.models.gpt_oss_puzzle", _gpt_oss_puzzle)
            except Exception as e:
                logger.warning(
                    "gpt_oss_puzzle vendored module failed to register — "
                    "NVIDIA Puzzle checkpoints will not load until resolved: %s",
                    e,
                )
            else:
                _VENDORED_MODEL_TYPES.add("gpt_oss_puzzle")
        else:
            _VENDORED_MODEL_TYPES.add("gpt_oss_puzzle")

    if "mlx_lm.models.nemotron_labs_diffusion" not in sys.modules:
        # NVIDIA Nemotron-Labs-Diffusion (3B/8B/14B) — AR (autoregressive)
        # mode = a Ministral3-style decoder + untied diffusion_head. Vendored
        # because mlx-lm (0.31.3, 2026-08-21) ships no support for the arch.
        # Same native-probe + defer-to-upstream policy as ``hy_v3`` above:
        # if mlx-lm ever lands native support we use theirs, not ours.
        import importlib.util as _importlib_util

        _nld_native_spec = None
        try:
            _nld_native_spec = _importlib_util.find_spec(
                "mlx_lm.models.nemotron_labs_diffusion"
            )
        except (ImportError, ValueError):
            _nld_native_spec = None

        if _nld_native_spec is None:
            try:
                from ..models import nemotron_labs_diffusion as _nld

                sys.modules.setdefault("mlx_lm.models.nemotron_labs_diffusion", _nld)
            except Exception as e:
                logger.warning(
                    "nemotron_labs_diffusion vendored module failed to register — "
                    "Nemotron-Labs-Diffusion checkpoints will not load until "
                    "resolved: %s",
                    e,
                )
            else:
                # Promote to the vendored set only on success so the
                # tokenizer fallback path routes the arch through the vendor
                # shim instead of auto-config heuristics.
                _VENDORED_MODEL_TYPES.add("nemotron_labs_diffusion")
        else:
            _VENDORED_MODEL_TYPES.add("nemotron_labs_diffusion")

    if "mlx_lm.models.qwen4_exp" not in sys.modules:
        # Qwen3.8-Flash-Next's qwen4_exp text decoder. Prefer a future native
        # mlx-lm implementation as soon as one ships; until then the vendored
        # module owns the typed QSA/PLE/HC architecture contract.
        import importlib.util as _importlib_util

        _qwen4_native_spec = None
        try:
            _qwen4_native_spec = _importlib_util.find_spec("mlx_lm.models.qwen4_exp")
        except (ImportError, ValueError):
            _qwen4_native_spec = None

        if _qwen4_native_spec is None:
            try:
                from ..models import qwen4_exp as _qwen4_exp

                sys.modules.setdefault("mlx_lm.models.qwen4_exp", _qwen4_exp)
            except Exception as e:
                logger.warning(
                    "qwen4_exp vendored module failed to register — "
                    "Qwen4-Exp checkpoints will not load until resolved: %s",
                    e,
                )
            else:
                _VENDORED_MODEL_TYPES.add("qwen4_exp")
        else:
            _VENDORED_MODEL_TYPES.add("qwen4_exp")
    # mlx-lm <=0.31.x has ``nemotron_h`` but assumes uniform MoE widths.
    # Puzzle's ``block_configs`` needs the small compatibility vendor above.
    # Once #1536 (or equivalent native support) is installed, leave native
    # Nemotron-H untouched and let its own remapping handle Puzzle configs.
    _native_nemotron_h = sys.modules.get("mlx_lm.models.nemotron_h")
    if _native_nemotron_h is None:
        try:
            import importlib

            _native_nemotron_h = importlib.import_module("mlx_lm.models.nemotron_h")
        except ImportError:
            _native_nemotron_h = None
    _native_args = getattr(_native_nemotron_h, "ModelArgs", None)
    _native_fields = getattr(_native_args, "__dataclass_fields__", {})
    if "block_configs" not in _native_fields:
        try:
            from ..models import nemotron_h as _puzzle_nemotron_h

            # The community 6-bit conversion identifies as ``nemotron_h``;
            # NVIDIA's original config identifies as ``nemotron_h_puzzle``.
            # Register both names without changing mlx-lm's general loader.
            sys.modules["mlx_lm.models.nemotron_h"] = _puzzle_nemotron_h
            sys.modules.setdefault(
                "mlx_lm.models.nemotron_h_puzzle", _puzzle_nemotron_h
            )
        except Exception as e:
            logger.warning("Nemotron Puzzle vendor failed to register: %s", e)
        else:
            _VENDORED_MODEL_TYPES.update({"nemotron_h", "nemotron_h_puzzle"})


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


def _post_load_ubc_evict(model_name: str) -> None:
    """Defect 4: evict the UBC mirror of safetensors shards on Darwin.

    Called from the public ``load_model_with_fallback`` after the
    underlying loader returns successfully. macOS keeps mmap'd
    safetensors pages in the Unified Buffer Cache after ``munmap`` +
    ``close``, which doubles the effective memory pressure of a large
    MoE load (mmap mirror + materialised UMA tensors) and trips Jetsam
    on GLM-5.2 / DeepSeek-V3.2 boots. Evicting the mirror via
    ``msync(MS_INVALIDATE)`` releases those pages back to the free pool.

    No-op on non-Darwin platforms (see :mod:`vllm_mlx.runtime.ubc_evict`).
    The platform gate is at the **TOP** of the function — codex round 1
    BLOCKING — so Linux/Windows callers don't pay for path resolution
    (which on a cache-miss could trigger a Hub ``snapshot_download``).

    Wrapped in a broad try/except: a failure here MUST NEVER block a
    model from loading. The eviction is opportunistic memory cleanup,
    not a correctness gate.
    """
    # Platform gate FIRST — every line below this point is Darwin-only
    # bookkeeping. Codex round 1 caught that without this early return,
    # Linux/Windows callers paid for ``_resolve_model_path``, which can
    # invoke ``huggingface_hub.snapshot_download`` on a cache miss —
    # a potentially expensive side effect inside the load-path
    # ``finally`` clause for a feature that has no effect on those
    # platforms.
    import sys as _sys

    if _sys.platform != "darwin":
        return

    try:
        from ..runtime.ubc_evict import ubc_evict_paths

        model_path = _resolve_model_path(model_name)
        if model_path is None:
            return
        # Enumerate every safetensors shard at the snapshot root.
        # Top-level ``glob`` (not ``rglob``) — every mlx-community
        # checkpoint we serve keeps safetensors at the snapshot root
        # alongside ``config.json`` / ``tokenizer.json``. Nested
        # safetensors would be a vendor-specific repo layout we have
        # not seen, and walking subdirectories on a HF snapshot risks
        # picking up unrelated payloads (e.g. variant siblings the HF
        # client materialised but the loader didn't bind).
        shards = sorted(str(p) for p in model_path.glob("*.safetensors"))
        if not shards:
            return
        ubc_evict_paths(shards)
    except Exception as e:  # pragma: no cover — defensive belt + suspenders
        logger.debug(f"Defect 4 post-load UBC evict skipped (non-fatal): {e}")


def _local_snapshot_if_cached(model_name: str) -> str:
    """Resolve a verified-complete cached repo id to its on-disk snapshot dir,
    so the loader is handed a local path and never makes a network round-trip.

    ``mlx_lm``'s ``get_model_path`` calls ``snapshot_download`` with no
    ``local_files_only``, so even a fully cached flat repo does a metadata
    round-trip (``refs/main`` HEAD + per-file ETag) on every start. On a
    poisoned-DNS network that round-trip hangs in SYN_SENT rather than failing
    fast, and the UI sits at "Starting" until the outer deadline.

    Resolving the cached snapshot here with ``local_files_only=True`` and
    handing the loader that local directory is the explicit, concurrency-safe
    fix: no process-global ``HF_HUB_OFFLINE`` toggling (which is racy across
    overlapping loads and does not reconfigure an already-created HTTP session),
    just a path the loader treats as a local checkpoint that never round-trips.

    Gated on ``is_repo_cached`` — the same completeness signal the pre-download
    gate trusts to skip fetching. On anything else (cold cache, not a repo id, a
    local path, or a resolve failure) return ``model_name`` unchanged so the
    normal online pull still runs.
    """
    try:
        from .._download_gate import is_repo_cached
        from ..model_metadata import (
            resolve_offline_cached_snapshot,
            resolve_unreferenced_cached_snapshot,
        )

        if not is_repo_cached(model_name):
            snapshot = resolve_unreferenced_cached_snapshot(model_name)
            if snapshot is None:
                snapshot = resolve_offline_cached_snapshot(model_name)
            return str(snapshot) if snapshot is not None else model_name
    except Exception:
        return model_name
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        # The cache probe said complete but the local resolve failed — fall
        # back to the normal path rather than blocking a load that may still
        # succeed online.
        return model_name


def _resolve_subfolder_checkpoint(model_name: str) -> str:
    """Turn ``org/repo`` into ``…/snapshots/<sha>/<subfolder>`` when the
    alias for that repo pins one; otherwise return ``model_name`` as-is.

    Only the declared subfolder is fetched — ``allow_patterns`` keeps a
    ``rapid-mlx serve lfm2.5-2.6b-4bit`` from pulling the repo's other
    seven quantizations (~20 GB for LFM2.5-2.6B-MLX).

    A local path is passed through untouched — it is already a checkpoint
    directory, and re-deriving a subfolder from a reverse alias lookup on a
    local path would be wrong. So is a repo with no declared subfolder.

    Once a subfolder IS declared this must not fall back to the bare repo
    id. Handing an unresolved remote id to ``mlx_lm.load`` sends it to its
    own *unfiltered* ``snapshot_download`` — the caller would wait out the
    full ~20 GB repo and then still fail, because the repo root is not a
    checkpoint for this publisher. Raise instead: an immediate, named error
    beats a very expensive one. The offline path is preserved explicitly by
    retrying against the local cache before giving up.
    """
    import os

    if os.path.exists(model_name):
        return model_name
    # Registry errors PROPAGATE. Swallowing them and returning the bare
    # repo id is precisely the outcome this helper exists to prevent: for
    # a subfolder repo, mlx-lm would then run its own unfiltered
    # ``snapshot_download`` and pull every quantization before failing.
    # A malformed aliases.json is a hard error everywhere else too
    # (``resolve_model`` loads the same registry at CLI startup), so this
    # is consistent, not a new failure mode.
    from huggingface_hub import snapshot_download

    from .._download_gate import (
        _escape_variant_glob_literal,
        _snapshot_is_complete,
        pulled_variant,
    )
    from ..model_aliases import resolve_model, resolve_subfolder

    repo_id = resolve_model(model_name)

    # #2340 precedence is intentional:
    #
    # 1. An explicit alias names a specific catalog checkpoint and must win.
    # 2. A bare repo id uses the latest successful ``pull --bits/--format``
    #    marker, when present.
    # 3. Otherwise retain the historical reverse-catalog default/root lookup.
    #
    # This lets ``pull --bits 8 <repo>`` followed by ``serve <repo>`` recover
    # 8bit without letting stale repo-level cache metadata change what
    # ``serve lfm2.5-2.6b-4bit`` explicitly means.
    catalog_subfolder = resolve_subfolder(model_name)
    explicit_alias_subfolder = catalog_subfolder if model_name != repo_id else None
    subfolder = explicit_alias_subfolder or pulled_variant(repo_id) or catalog_subfolder
    if not subfolder:
        return model_name
    # ``resolve_subfolder`` answers for BOTH spellings — the alias the user
    # typed and the repo id the CLI resolves it to. The Hub only knows the
    # latter. ``server.load_model`` is also a public entry point that
    # programmatic callers reach with a bare alias, skipping the CLI's
    # pre-resolution, so normalize here instead of assuming someone
    # upstream already did.
    patterns = [f"{_escape_variant_glob_literal(subfolder)}/*"]

    # Offline-first: a warm, COMPLETE cache resolves with zero network. The
    # online call used to run first and, on a poisoned-DNS network, hangs in
    # SYN_SENT indefinitely instead of raising — so the cached fallback it
    # reached only inside ``except`` never ran, and an already-downloaded
    # subfolder (e.g. the default starter ``lfm2.5-1b-4bit``) sat at
    # "Starting" until the outer deadline. Only a verified-complete on-disk
    # subfolder short-circuits: a half-pulled one still falls through to the
    # Hub so an interrupted download is finished rather than loaded as-is.
    cached_local = None
    try:
        cached_local = snapshot_download(
            repo_id, allow_patterns=patterns, local_files_only=True
        )
    except Exception:
        cached_local = None

    if cached_local is not None and _snapshot_is_complete(
        os.path.join(cached_local, subfolder)
    ):
        local = cached_local
    else:
        try:
            local = snapshot_download(repo_id, allow_patterns=patterns)
        except Exception as online_exc:
            # The Hub call failed. The cause could be anything — no network, a
            # gated repo, a bad token, a full disk — and we deliberately do NOT
            # try to tell them apart: the recovery is the same for all of them,
            # namely "is it already on disk?", and a wrong guess about the
            # cause is worse than not guessing. Reuse whatever the cache holds
            # (even if incomplete) so the completeness check below produces the
            # precise "present but incomplete" diagnosis instead of a raw
            # network error.
            if cached_local is not None:
                local = cached_local
                logger.warning(
                    "Could not reach %s (%s) — falling back to the cached %r subfolder.",
                    repo_id,
                    online_exc,
                    subfolder,
                )
            else:
                raise RuntimeError(
                    f"Could not fetch the {subfolder!r} subfolder of {repo_id}, "
                    f"and it is not in the local cache. This publisher ships one "
                    f"checkpoint per quantization folder, so the repo root cannot "
                    f"be loaded instead. Original error: {online_exc}"
                ) from online_exc

    resolved = os.path.join(local, subfolder)
    if not os.path.isdir(resolved):
        raise RuntimeError(
            f"{repo_id} resolves to subfolder {subfolder!r} but {resolved} "
            "does not exist after download — the publisher has probably "
            "reorganized the repo, or the variant was pulled to a different "
            f"cache. Re-run `rapid-mlx pull --format {subfolder} {repo_id}` "
            "(or update the alias) rather than loading the repo root, which "
            "is not a checkpoint."
        )
    # A directory is not a checkpoint. An interrupted or disk-full pull
    # leaves the folder present with its shards missing, and a publisher who
    # reorganizes the repo can leave a config.json with no weights beside
    # it. Both reach here on the SUCCESS path too, so the check belongs
    # after both branches, not only after the offline fallback. Reuse the
    # download gate's implementation — it already mirrors mlx-lm's own
    # ``model*.safetensors`` glob and shard-index validation (imported above).
    if not _snapshot_is_complete(resolved):
        raise RuntimeError(
            f"The {subfolder!r} subfolder of {repo_id} is present but "
            "incomplete — its weight shards are missing. A previous download "
            "did not finish, or the publisher reorganized the repo."
        )
    logger.info("Loading %s from its %r subfolder", repo_id, subfolder)
    return resolved


def load_model_with_fallback(
    model_name: str,
    tokenizer_config: dict = None,
    *,
    enable_dspark: bool = False,
    chat_template_id: str | None = None,
    lazy: bool = False,
    return_config: bool = False,
    return_source: bool = False,
):
    """
    Load model and tokenizer with fallback for non-standard tokenizers.

    Args:
        model_name: HuggingFace model name or local path
        tokenizer_config: Optional tokenizer configuration
        lazy: ``--disk-stream`` path only. When True, skip every
            *branch-selection* fallback / tokenizer-quirk path below
            (Gemma 4 native/legacy routing, vendored-arch tokenizer
            fallback, strict=False retry — all orthogonal to laziness,
            per PRD-rapid-mlx-integration.md's CLI-wiring decision) and
            load straight through ``mlx_lm.load(model_name, lazy=True)``
            so MoE expert weights are never materialized. The
            architecture-independent *post-load* fixups
            (``_neutralize_unbundled_template_types`` before the load,
            ``_try_inject_mtp_post_load`` /
            ``_apply_chat_template_sidecar`` /
            ``augment_eos_token_ids_from_generation_config`` /
            ``repair_byte_level_decoder`` after it) still run — see the
            ``if lazy:`` block below for why each is safe on a lazily
            loaded model. ``vllm_mlx.disk_stream_patch.install`` is
            called by the caller afterward, before the model reaches
            serving.
        return_config: When True (only meaningful with ``lazy=True``),
            also return the model's config dict as a third tuple element
            (needed to read ``model_type`` for
            ``disk_stream_patch.install``) — passed straight through to
            ``mlx_lm.load(..., return_config=True)``.
        return_source: Append the concrete checkpoint source selected for this
            load. Engine startup uses this to pin persisted-cache identity
            without requiring the returned model object to accept attributes.

    Returns:
        ``(model, tokenizer)`` by default; ``(model, tokenizer, config)``
        with ``return_config=True``; ``(model, tokenizer, source)`` with
        ``return_source=True``; or ``(model, tokenizer, config, source)``
        when both flags are true.
    """
    # Resolve model-owned prompt serialization before ``model_name`` becomes a
    # cache snapshot path.  The selected template is installed once after the
    # tokenizer loads; request rendering does no model/template inference.
    from ..model_aliases import resolve_profile

    requested_profile = resolve_profile(model_name)
    requested_chat_template_id = (
        chat_template_id
        if chat_template_id is not None
        else (
            requested_profile.chat_template_id
            if requested_profile is not None
            else None
        )
    )
    raw_explicit_chat_template = (
        tokenizer_config.get("chat_template")
        if isinstance(tokenizer_config, dict)
        else None
    )
    explicit_chat_template = (
        raw_explicit_chat_template
        if isinstance(raw_explicit_chat_template, (str, dict, list))
        else None
    )

    def _resolve_loaded_template(result) -> None:
        from .chat_template_registry import resolve_chat_template

        resolve_chat_template(
            result[1],
            requested_chat_template_id,
            explicit_template=explicit_chat_template,
        )

    # Publishers who ship one repo per model with a folder per quant
    # (``LiquidAI/LFM2.5-2.6B-MLX`` → ``4bit/``, ``8bit/``, ``bf16/`` …)
    # need the repo id turned into a concrete directory before mlx-lm
    # sees it — ``mlx_lm.load`` has no subfolder parameter. This is the
    # ONLY place that happens: everything upstream (download gate, R2
    # mirror catalog, ``model_sizes``, telemetry) keeps working with the
    # bare repo id. No-op for the ~99% of aliases whose repo root is the
    # checkpoint.
    model_name = _resolve_subfolder_checkpoint(model_name)

    # Hand the loader a local snapshot path for a verified-complete cached repo,
    # so mlx_lm's own ``snapshot_download`` never fires the online metadata
    # round-trip that hangs a start on a poisoned-DNS network. No-op for a cold
    # cache or an already-local path (see the helper).
    model_name = _local_snapshot_if_cached(model_name)

    # Pin remote repositories to the concrete snapshot that THIS load will
    # consume. Persisted KV identity must never be derived later from mutable
    # refs/main state, which can advance while the server is running.
    if not Path(model_name).is_dir():
        resolved_snapshot = _resolve_model_path(model_name)
        if resolved_snapshot is not None:
            model_name = str(resolved_snapshot)

    # ``mlx_lm.load`` may import config.json::model_file.  Validate that
    # caller-supplied local path once at this shared boundary before any native
    # or fallback loader runs.  Remote repository ids are intentionally a no-op
    # here; see validate_local_model_file for the containment boundary.
    validate_local_model_file(model_name)

    tokenizer_config, trust_remote_code = apply_remote_code_policy(tokenizer_config)

    # Security hardening: when remote-code execution is enabled (the default,
    # for maximal community-model compatibility) and this model actually needs
    # it (its config declares ``auto_map``), say so BEFORE any repo code is
    # downloaded and run. This turns silent code execution into an informed
    # choice; opt out process-wide with RAPID_MLX_TRUST_REMOTE_CODE=0 (see
    # BatchedEngine). A probe failure is silent — never breaks loading.
    if _model_requires_remote_code(model_name):
        if trust_remote_code:
            logger.warning(
                "Security: model %r declares auto_map (custom Python code). "
                "Loading may DOWNLOAD AND EXECUTE that repo's code locally. "
                "Only continue if you trust this model's source. Disable with "
                "RAPID_MLX_TRUST_REMOTE_CODE=0 if you do not need it.",
                model_name,
            )
        else:
            logger.warning(
                "Model %r declares auto_map custom code, but remote code is "
                "disabled; loading will continue with trust_remote_code=False.",
                model_name,
            )

    if lazy:
        from mlx_lm import load as _mlx_lm_load

        # #1420 guard: must run BEFORE mlx_lm.load, same as the non-lazy
        # path (_load_model_with_fallback_impl calls it at :1068-1074,
        # ahead of any load()). It mutates tokenizer_config, which is
        # consumed by mlx_lm.load's own internal tokenizer loading
        # during the call below — running it after load() would be too
        # late to pre-empt the crashing importlib.import_module() this
        # guard exists to stop (#1420, unbundled chat_template_type /
        # tool_parser_type). Reading tokenizer_config.json off disk does
        # not touch model weights, so it's exactly as valid ahead of a
        # lazy load as an eager one.
        tokenizer_config = _neutralize_unbundled_template_types(
            model_name, tokenizer_config or {}
        )
        result = _mlx_lm_load(
            model_name,
            tokenizer_config=tokenizer_config,
            lazy=True,
            return_config=return_config,
        )
        model, tokenizer = result[0], result[1]

        # The four fixups below are pure post-load tokenizer/generation-
        # config/model-attribute adjustments: each one reads the already-
        # returned `model`/`tokenizer` objects and `model_name`/on-disk
        # sidecar files, and mutates `tokenizer` (or `model.mtp`) in
        # place. None of them materializes or inspects lazily-loaded
        # weight *tensors* — they don't care whether the checkpoint was
        # loaded lazily or eagerly — so skipping them for `lazy=True` was
        # a bug in the original short-circuit, not a deliberate scope
        # cut (see review-notes.md's blocking finding). Concretely:
        # `augment_eos_token_ids_from_generation_config`'s own docstring
        # names Qwen3/Qwen2.5 (151645/151643) as the exact scenario it
        # exists to fix, and `qwen2_moe` is one of the registered
        # --disk-stream architectures (vllm_mlx/registry.py) — without
        # this call a qwen2_moe checkpoint loaded with --disk-stream
        # would silently fail to stop at its chat-template terminator
        # and run to max_tokens instead, the same checkpoint stopping
        # correctly without the flag.
        _try_inject_mtp_post_load(model, model_name)
        if not getattr(tokenizer, "chat_template", None):
            mp = _resolve_model_path(model_name)
            if mp is not None:
                _apply_chat_template_sidecar(mp, tokenizer)
        augment_eos_token_ids_from_generation_config(tokenizer, model_name)
        repair_byte_level_decoder(tokenizer)
        _resolve_loaded_template(result)

        # Still legitimately skipped for `lazy=True`: the Gemma-4 native/
        # legacy routing (`gemma4_family_kind` gate + its own duplicate
        # augment/chat-template/repair calls), the vendored-arch dispatch
        # (`_is_vendored_arch_model`), the Nemotron tokenizer fallback
        # (`_needs_tokenizer_fallback`), and the `except ValueError`
        # strict=False retry. All four are branch-*selection* logic that
        # decides which loader to call (`mlx_lm.load` vs. the vendor/
        # fallback loaders) or how to recover from a `ValueError` those
        # alternate loaders raise — genuinely orthogonal to whether the
        # chosen loader materializes weights eagerly or lazily. They are
        # also unreachable in practice for a --disk-stream load: neither
        # Gemma 4, nor any vendored architecture, nor Nemotron is a
        # registered --disk-stream architecture (`lfm2_moe`, `qwen2_moe`,
        # and `qwen3_next` are, per vllm_mlx/registry.py), so none of these
        # branches would ever fire for a checkpoint this code path is
        # actually used for.
        _post_load_ubc_evict(model_name)
        return (*result, str(model_name)) if return_source else result
    if enable_dspark:
        result = _load_model_with_fallback_impl(
            model_name, tokenizer_config, enable_dspark=True
        )
    else:
        # Preserve the historical two-argument call shape for downstream
        # wrappers and tests that instrument this internal dispatch boundary.
        result = _load_model_with_fallback_impl(model_name, tokenizer_config)
    _resolve_loaded_template(result)
    # Defect 4: evict UBC mirror of safetensors shards on Darwin so
    # the (mmap mirror + materialised weights) burst does not double
    # the load-window memory footprint. Runs ONLY after a successful
    # load — pr_validate codex BLOCKING #1: a failed load might be
    # operating on an uncached HF repo, and resolving the path inside
    # the failure path could trigger ``snapshot_download`` side effects
    # for a model whose tensors never materialised. We only have a
    # mirror worth evicting after the inner loader succeeded.
    # No-op on non-Darwin.
    _post_load_ubc_evict(model_name)
    return (*result, str(model_name)) if return_source else result


# mlx-lm's tokenizer loader (``mlx_lm.tokenizer_utils.load``) imports a
# chat-template / tool-parser module BY NAME whenever the model's
# ``tokenizer_config.json`` declares one, with no guard:
#     if chat_template_type := cfg.get("chat_template_type", False):
#         importlib.import_module(f"mlx_lm.chat_templates.{chat_template_type}")
#     if tool_parser_type := cfg.get("tool_parser_type", ...):
#         importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")
# When the *bundled* mlx-lm doesn't ship that module the import raises
# ``ModuleNotFoundError`` and the server dies at startup with NO fallback to
# the model's on-disk ``chat_template.jinja``. #1420: mlx-lm 0.31.x dropped
# ``mlx_lm/chat_templates/gemma4.py`` while ``mlx-community/gemma-4-26b-a4b-it-4bit``
# still declares ``"chat_template_type": "gemma4"`` → every such Gemma 4
# checkpoint 500s on ``rapid-mlx serve`` (regression vs 0.6.71, which
# shipped the module). Weights load BEFORE the tokenizer in ``mlx_lm.load``,
# so a catch-and-retry would re-load multi-GB weights — we neutralize the
# offending field up front instead.
_UNBUNDLED_TEMPLATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("chat_template_type", "mlx_lm.chat_templates"),
    ("tool_parser_type", "mlx_lm.tool_parsers"),
)


def _read_tokenizer_config_json(model_name: str) -> dict | None:
    """Best-effort read of a model's ``tokenizer_config.json``.

    Local dir → read directly. HF repo id → fetch ONLY that one small file
    (never the weights) via the hub cache. Returns ``None`` on any failure
    so the caller falls through to the unmodified load path — this is a
    pre-load probe and must never itself break loading.
    """
    try:
        local = Path(model_name)
        if local.is_dir():
            cfg_path = local / "tokenizer_config.json"
            if not cfg_path.is_file():
                return None
        else:
            from huggingface_hub import hf_hub_download

            try:
                # Cache-only first: a model that's already downloaded (the
                # common case, and always so for the desktop app which pulls
                # before it serves) resolves with NO network — we don't add a
                # hub round-trip to every serve start, and stay functional
                # offline.
                cfg_path = Path(
                    hf_hub_download(
                        model_name, "tokenizer_config.json", local_files_only=True
                    )
                )
            except Exception:
                # Not cached yet (a fresh ``serve <repo-id>``): fetch just
                # this one small file so a first load of a Gemma 4 checkpoint
                # is guarded too. The weights download on the very next step.
                cfg_path = Path(hf_hub_download(model_name, "tokenizer_config.json"))
        with open(cfg_path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 — a probe must not break loading
        logger.debug("tokenizer_config.json probe failed for %s: %s", model_name, e)
        return None


def _read_model_config_json(model_name: str) -> dict | None:
    """Best-effort read of a model's ``config.json``.

    Mirrors :func:`_read_tokenizer_config_json` (local dir first, cache-only
    hub fetch with a network fallback for a fresh ``serve <repo-id>``). Any
    failure returns ``None`` so callers fall through to the unmodified load
    path — this is a pre-load probe and must never itself break loading.
    """
    try:
        local = Path(model_name)
        if local.is_dir():
            cfg_path = local / "config.json"
            if not cfg_path.is_file():
                return None
        else:
            from huggingface_hub import hf_hub_download

            try:
                cfg_path = Path(
                    hf_hub_download(model_name, "config.json", local_files_only=True)
                )
            except Exception:
                cfg_path = Path(hf_hub_download(model_name, "config.json"))
        with open(cfg_path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 — a probe must not break loading
        logger.debug("config.json probe failed for %s: %s", model_name, e)
        return None


def _model_requires_remote_code(model_name: str) -> bool:
    """Return True if the model's ``config.json`` / ``tokenizer_config.json``
    declares ``auto_map`` custom code.

    ``auto_map`` is HF transformers' opt-in for executing model/tokenizer
    Python shipped inside a repo (the remote-code execution gate). When a
    model declares it, loading with ``trust_remote_code=True`` will download
    and run that code locally — so surfacing it lets operators make an
    informed choice before the code executes. Any probe failure returns
    ``False`` (no warning) so loading is never broken by the probe.
    """
    try:
        tok_cfg = _read_tokenizer_config_json(model_name)
        if tok_cfg and tok_cfg.get("auto_map"):
            return True
        model_cfg = _read_model_config_json(model_name)
        if model_cfg and model_cfg.get("auto_map"):
            return True
    except Exception as e:  # noqa: BLE001 — a probe must never break loading
        logger.debug("remote-code probe failed for %s: %s", model_name, e)
    return False


def _neutralize_unbundled_template_types(
    model_name: str, tokenizer_config: dict
) -> dict:
    """Guard against #1420: strip any ``chat_template_type`` /
    ``tool_parser_type`` whose ``mlx_lm`` module the bundled mlx-lm doesn't
    ship, so ``mlx_lm.load`` skips the crashing ``import_module`` and the
    model boots from its ``chat_template.jinja`` sidecar instead.

    Returns the original dict unchanged when there is nothing to neutralize
    (the common case — no field declared, or its module IS bundled), so the
    happy path pays only one small config read.
    """
    cfg = _read_tokenizer_config_json(model_name)
    if not cfg:
        return tokenizer_config

    import importlib.util as _iu

    patched: dict | None = None
    for field, pkg in _UNBUNDLED_TEMPLATE_FIELDS:
        # The caller's tokenizer_config override wins over the on-disk value
        # in mlx-lm (kwargs → AutoTokenizer init_kwargs → the ``:= get()``
        # branch), so if the caller has already specified this field — any
        # value, truthy or falsy — it owns it: don't probe the on-disk value
        # and don't clobber a deliberately-requested bundled template
        # (codex r1 MAJOR: a truthy override was being overwritten to None).
        if field in tokenizer_config:
            continue
        type_name = cfg.get(field)
        if not type_name or not isinstance(type_name, str):
            continue
        module = f"{pkg}.{type_name}"
        try:
            spec = _iu.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            if patched is None:
                patched = dict(tokenizer_config)
            patched[field] = None
            logger.warning(
                "%s declares %s=%r but bundled mlx-lm ships no %s — "
                "neutralizing it so the model loads from its chat_template.jinja "
                "sidecar rather than crashing at startup (#1420).",
                model_name,
                field,
                type_name,
                module,
            )
    return patched if patched is not None else tokenizer_config


def _load_model_with_fallback_impl(
    model_name: str,
    tokenizer_config: dict = None,
    *,
    enable_dspark: bool = False,
):
    """Inner load implementation — kept separate so the public wrapper can
    install a try/finally for the Defect 4 UBC eviction without rewriting
    every return branch in the loader."""
    from mlx_lm import load

    _register_vendored_archs()
    tokenizer_config = tokenizer_config or {}
    # #1420: neutralize any declared chat-template / tool-parser type whose
    # mlx-lm module isn't bundled, BEFORE any load() — covers the native
    # Gemma 4 path, its legacy-wrapper fallback, and the general path, all of
    # which feed this dict to ``mlx_lm.load`` / ``load_tokenizer``.
    tokenizer_config = _neutralize_unbundled_template_types(
        model_name, tokenizer_config
    )

    # Check if model needs fallback (e.g., Nemotron)
    if _needs_tokenizer_fallback(model_name):
        logger.info(
            f"Model {model_name} requires tokenizer fallback, loading directly..."
        )
        return _load_with_tokenizer_fallback(model_name, enable_dspark=enable_dspark)

    # Vendored architectures (e.g. deepseek_v4) — transformers' AutoConfig
    # doesn't know about them, so mlx-lm's high-level load() blows up
    # before we get a chance to handle the error. Route directly to the
    # lower-level load_model() + raw tokenizer.json fallback.
    if _is_vendored_arch_model(model_name):
        logger.info(
            f"Model {model_name} uses a vendored architecture, "
            "skipping AutoConfig path and loading directly..."
        )
        return _load_with_tokenizer_fallback(model_name, enable_dspark=enable_dspark)

    # Gemma 4: mlx-lm 0.31+ supports it natively. Only use our wrapper
    # for older mlx-lm versions that lack gemma4 model support. Several
    # model_types ride this path — the non-unified ``gemma4`` (26B/31B/
    # e2b/e4b) and ``gemma4_assistant`` (assistant aliases), plus
    # ``gemma4_unified`` (12B). Read the arch ONCE and retain the
    # classification through the native-load fallback so a remote repo's
    # config isn't fetched twice and a transient second lookup can't
    # flip the loader choice (see #509).
    from ..models.gemma4_text import (
        gemma4_family_kind,
    )

    gemma4_kind = gemma4_family_kind(model_name)  # "unified" | "nonunified" | None
    if gemma4_kind is not None:
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
            # that lack native gemma4 architecture support. Route
            # ``gemma4_unified`` to the explicit unified loader so it
            # pins to ``mlx_vlm.models.gemma4_unified`` (with vendored
            # fallback); everything else uses the non-unified loader.
            if gemma4_kind == "unified":
                from ..models.gemma4_text import load_gemma4_unified_text

                logger.info(
                    f"Gemma 4 unified native load failed ({e}), "
                    "falling back to unified text-only wrapper (legacy mlx-lm)"
                )
                return load_gemma4_unified_text(model_name, tokenizer_config)

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
            return _load_with_tokenizer_fallback(
                model_name, enable_dspark=enable_dspark
            )
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


def _load_with_tokenizer_fallback(model_name: str, *, enable_dspark: bool = False):
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

    # The published 0731 MXFP checkpoint's quantization paths match its
    # standalone model (``layers.*``).  Our mlx-lm-compatible vendored model
    # nests the transformer under ``model`` and renames shared-expert
    # projections, so mlx-lm would otherwise apply the global MXFP4 default to
    # MXFP8 attention tensors and reject their packed shapes.
    model_config = _deepseek_v4_quantization_override(
        model_path, enable_dspark=enable_dspark
    )

    # DeepSeek-style fp8 block checkpoints (Ling 3.0 fp8): mlx has no fp8
    # dtype, so ``mx.load`` cannot open the shards at all. Repack the
    # original e4m3 bytes + ue8m0 block scales into mlx's mxfp8 layout at
    # load time — bit-lossless, no offline conversion needed (see
    # ``vllm_mlx/fp8_repack.py``).
    from ..fp8_repack import is_fp8_block_checkpoint, load_fp8_model_online

    if is_fp8_block_checkpoint(model_path):
        logger.info("fp8 block checkpoint detected — online mxfp8 repack")
        model = load_fp8_model_online(model_path)
    else:
        # Load model
        model, _ = load_model(model_path, model_config=model_config)

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
        pad_token = "<pad>"
        chat_template = None

        if tokenizer_config_path.exists():
            with open(tokenizer_config_path) as f:
                config = json.load(f)
                bos_token = _special_token_text(config.get("bos_token"), bos_token)
                eos_token = _special_token_text(config.get("eos_token"), eos_token)
                unk_token = _special_token_text(config.get("unk_token"), unk_token)
                pad_token = _special_token_text(config.get("pad_token"), pad_token)
                chat_template = config.get("chat_template")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
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
        # Union in generation_config EOS ids — this was the ONLY load
        # path missing the call (the AutoTokenizer paths all have it).
        # Muse Glimmer surfaced the gap: its tokenizer_config eos is
        # <|end_of_text|> (200001) but turns end with <|eot|> (200008,
        # declared only in generation_config.json) — without the union
        # the model generates past every turn end until max_tokens,
        # repeating its answer (real-weights mini smoke, 2026-08-10).
        augment_eos_token_ids_from_generation_config(tokenizer, str(model_path))
        logger.info("Tokenizer loaded via fallback successfully")
    else:
        raise ValueError(f"No tokenizer.json found in {model_path}")
    return model, tokenizer


def _deepseek_v4_quantization_override(
    model_path: Path, *, enable_dspark: bool = False
) -> dict | None:
    """Translate standalone DeepSeek-V4 quantization paths for mlx-lm.

    Returns a ``model_config`` overlay only for ``model_type=deepseek_v4``.
    Older mlx-community V4 checkpoints already use the vendored module paths;
    translating is idempotent for those keys.
    """
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if config.get("model_type") != "deepseek_v4":
        return None
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        return None

    scalar_keys = {"group_size", "bits", "mode"}
    translated = {k: v for k, v in quantization.items() if k in scalar_keys}
    projection_names = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    for path, value in quantization.items():
        if path in scalar_keys:
            continue
        new_path = path
        if new_path.startswith("layers."):
            new_path = "model." + new_path
        elif new_path == "embed":
            new_path = "model.embed_tokens"
        elif new_path == "head":
            new_path = "lm_head"
        for old, new in projection_names.items():
            new_path = new_path.replace(
                f".ffn.shared_experts.{old}", f".ffn.shared_experts.{new}"
            )
        translated[new_path] = value
    overlay = {"quantization": translated}
    try:
        from ..spec_decode.dspark import detect_dspark_metadata

        dspark = detect_dspark_metadata(model_path) if enable_dspark else None
    except Exception:  # pragma: no cover - optional checkpoint metadata
        dspark = None
    if dspark is not None:
        overlay.update(
            {
                "dspark_num_layers": dspark.num_layers,
                "dspark_block_size": dspark.block_size,
                "dspark_noise_token_id": dspark.noise_token_id,
                "dspark_target_layer_ids": list(dspark.target_layer_ids),
                "dspark_markov_rank": dspark.markov_rank,
            }
        )
    return overlay
