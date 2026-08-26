# SPDX-License-Identifier: Apache-2.0
"""PFlash-style long-prompt token-statistical compression for prefill (#287).

PFlash trades a small amount of recall on the middle of very long prompts
for a large cold-prefill TTFT win. Scoring is deterministic and uses only
``collections.Counter`` so it runs without a Metal device and adds no
new dependency.

Original design + reference fork by @michaelasper on the
``pflash-qwen36-ttft`` branch of github.com/michaelasper/Rapid-MLX
(commits d7a2797 + b6089ce). See issue #287 for the discussion.

This adaptation differs from the fork in three places:

* It is disabled by default (``--pflash off``).
* The compressor's output bypasses the prefix cache entirely on the
  scheduler side — see ``scheduler.add_request`` — so a later
  uncompressed request that shares a sink-token prefix with a compressed
  request cannot inherit position-shifted KV. The fork only suppressed
  the ``prefix_boundary`` boundary save; that left four other cache
  store sites poisoning the trie.
* Multimodal models are rejected up front instead of silently no-op.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

logger = logging.getLogger(__name__)

PFlashMode = Literal["off", "auto", "always"]


@dataclass(frozen=True)
class PFlashConfig:
    """Configuration for PFlash prompt compression.

    Defaults match the validated profile from PR #649 needle + TTFT
    runs: threshold 32 768 tokens, keep ratio 0.20 (~5× prefill
    reduction), minimum 2 048 kept tokens so very-long prompts still
    retain a usable amount of body context, large 2 048-token tail
    because the user's actual query tends to live there. The fork's
    default was 0.10 but our bench evidence (TTFT 3.87x-8.5x, needle
    recall 5/5) is all at 0.20 — the verified-tier auto-ON default
    must match the validated number, so we use 0.20 here.
    """

    mode: PFlashMode = "off"
    threshold: int = 32_768
    keep_ratio: float = 0.20
    min_keep_tokens: int = 2_048
    sink_tokens: int = 256
    tail_tokens: int = 2_048
    block_size: int = 128
    query_window: int = 512
    stride_blocks: int = 8
    skip_when_tools: bool = True

    def validate(self) -> PFlashConfig:
        if self.mode not in ("off", "auto", "always"):
            raise ValueError("--pflash must be one of: off, auto, always")
        if self.threshold < 0:
            raise ValueError("--pflash-threshold must be >= 0")
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("--pflash-keep-ratio must be > 0.0 and <= 1.0")
        if self.min_keep_tokens < 0:
            raise ValueError("--pflash-min-keep-tokens must be >= 0")
        if self.sink_tokens < 0:
            raise ValueError("--pflash-sink-tokens must be >= 0")
        if self.tail_tokens < 0:
            raise ValueError("--pflash-tail-tokens must be >= 0")
        if self.block_size <= 0:
            raise ValueError("--pflash-block-size must be > 0")
        if self.query_window <= 0:
            raise ValueError("--pflash-query-window must be > 0")
        if self.stride_blocks < 0:
            raise ValueError("--pflash-stride-blocks must be >= 0")
        return self


@dataclass(frozen=True)
class PFlashResult:
    tokens: list[int]
    compressed: bool
    reason: str
    original_tokens: int
    kept_tokens: int
    # Number of *middle* tokens (between the leading sink and trailing tail) that
    # survived compression. A zero-middle budget now refuses compression rather
    # than deleting the body. Defaults to 0 so existing keyword construction
    # (e.g. ``_unchanged``) is unaffected.
    middle_tokens_kept: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.kept_tokens / self.original_tokens

    @property
    def endpoints_only(self) -> bool:
        """True when a compressed result kept zero middle tokens.

        The compressor now refuses this lossy state with
        ``reason="insufficient_middle_budget"`` and returns the prompt unchanged,
        so production results should always report False. The property remains
        for metadata compatibility and as a defensive invariant signal.
        """
        return self.compressed and self.middle_tokens_kept == 0


def config_from_args(args: Any) -> PFlashConfig:
    """Build and validate a PFlashConfig from argparse-style attributes.

    ``args.pflash`` may be ``None`` when the CLI hasn't been run through
    :func:`resolve_pflash_mode_default` yet (e.g. unit tests that build a
    ``SimpleNamespace`` directly, or callers that opt out of the per-alias
    default resolution). Treat ``None`` as the conservative ``"off"`` so
    a forgotten resolver call never silently enables compression — the
    intent of the tier-based default is *opt-in for verified aliases*,
    not *opt-in by accident anywhere else*.
    """
    mode = args.pflash if args.pflash is not None else "off"
    # ``pflash_keep_ratio`` mirrors ``pflash`` (mode): the CLI sentinel default
    # is ``None`` and :func:`resolve_pflash_keep_ratio_default` materializes it
    # (explicit flag > per-alias override > engine default 0.20) before we get
    # here. Fall back to 0.20 if that resolver never ran (unit tests that build
    # a bare SimpleNamespace, or the ``--enable-dflash`` path that skips PFlash
    # resolution) so a forgotten resolve never fails validation with ``None``.
    keep_ratio = args.pflash_keep_ratio
    if keep_ratio is None:
        keep_ratio = 0.20
    return PFlashConfig(
        mode=mode,
        threshold=args.pflash_threshold,
        keep_ratio=keep_ratio,
        min_keep_tokens=args.pflash_min_keep_tokens,
        sink_tokens=args.pflash_sink_tokens,
        tail_tokens=args.pflash_tail_tokens,
        block_size=args.pflash_block_size,
        query_window=args.pflash_query_window,
        stride_blocks=args.pflash_stride_blocks,
        skip_when_tools=not getattr(args, "pflash_include_tools", False),
    ).validate()


# Sentinel for "the caller has not pre-resolved the alias profile — detect it
# yourself". Distinct from ``None``, which is the legitimate "detected, but this
# path is not an alias" result. Lets ``resolve_pflash_config`` detect the
# profile ONCE and hand it to both resolvers instead of each re-detecting on the
# no-flag startup path (codex #1458 NIT: same metadata was detected twice).
_DETECT = object()


def _detect_or(model_name: str, pre: Any) -> Any:
    """Return ``pre`` if the caller already resolved the profile, else detect it.

    Detection import errors (broken/partial install) degrade to ``None`` — the
    same conservative "no alias decision" both resolvers already fall back to.
    """
    if pre is not _DETECT:
        return pre
    try:
        from .model_auto_config import detect_model_config
    except ImportError:
        return None
    return detect_model_config(model_name)


def resolve_pflash_mode_default(
    args: Any,
    *,
    model_name: str,
    is_multimodal: bool = False,
    _detected_config: Any = _DETECT,
) -> str:
    """Resolve ``args.pflash`` when the user passed nothing on the CLI.

    Per-alias tier-based default (#287 alias-profile integration):

    * If ``args.pflash`` is already set (user passed ``--pflash off|auto|always``)
      it wins — return it unchanged. This preserves the explicit-override
      contract documented on the CLI flag and the env var.
    * Otherwise, look up the model's profile via ``detect_model_config``
      and switch on ``pflash_tier``:

      - ``"verified"`` → ``"always"``  (Qwen3.5 / Qwen3.6 family, bench
        evidence in PR #649: 3.87x-8.5x TTFT speedup at keep_ratio=0.20
        with 100% needle recall across tested cells) — UNLESS the model is
        multimodal, see below.
      - anything else → ``"off"`` (today's behaviour preserved for every
        alias we haven't measured).

    ``is_multimodal`` — the SAME ``is_mllm`` verdict the caller passes to
    :func:`validate_model_support` — suppresses the verified-tier
    ``"always"`` promotion. PFlash cannot serve the MLLM/VLM lane
    (``validate_model_support`` rejects it), so a verified alias that is
    ALSO multimodal (a vision-config Qwen3.6-27B checkpoint is both) must
    NOT auto-enable PFlash — otherwise the naive ``rapid-mlx serve
    <flagship>`` command dies on ``--pflash is not supported for
    multimodal models``, a flag the user never set (#352 dogfood P1-②).
    An explicit ``--pflash always`` still wins via the early return above
    and, for the MLLM lane, errors loudly in ``validate_model_support`` —
    the user asked for it, so they get the actionable message. Defaults to
    ``False`` so callers that don't route multimodally (and the unit tests)
    keep the pure tier-based behaviour.

    The result is the string to assign back to ``args.pflash`` before
    calling :func:`config_from_args`. Splitting resolution from
    construction keeps unit tests trivial: build a ``SimpleNamespace``
    with ``pflash=None`` and assert against the returned mode.
    """
    if args.pflash is not None:
        return args.pflash
    # Resolve the alias profile (or reuse one ``resolve_pflash_config`` already
    # detected). ``_detect_or`` does the late import so importing ``pflash``
    # stays cheap for callers that never resolve a default, and degrades a
    # broken install to ``None`` = PFlash off (a malformed ``aliases.json``
    # still raises ValueError from ``_coerce`` — the user must see that).
    cfg = _detect_or(model_name, _detected_config)
    if cfg is not None and cfg.pflash_tier == "verified":
        # A multimodal (MLLM/VLM) model can NOT run PFlash — the MLLM lane
        # is rejected outright by ``validate_model_support``. Auto-enabling
        # the verified-tier default for such an alias makes the naive
        # ``rapid-mlx serve <flagship>`` command die on a ``--pflash`` flag
        # the user never set (a vision-config Qwen3.6-27B checkpoint is
        # pflash_tier=verified AND multimodal — #352 dogfood P1-②). Leave
        # PFlash off in that case. Note: this is intentionally scoped to
        # MULTIMODAL, not hybrid — a hybrid MoE like Qwen3.5-35B-A3B still
        # has full-attention layers with standard KV to compress and is a
        # verified PFlash target.
        if is_multimodal:
            # Do NOT advise ``--pflash always`` here: PFlash genuinely cannot
            # serve the MLLM/VLM lane, so forcing it on would only be rejected
            # at startup by ``validate_model_support``. Keep the user's mental
            # model correct — state that PFlash is unavailable for this model,
            # and that an explicit override would error (codex #2 nit on #1178).
            logger.info(
                "PFlash default: alias %r is multimodal — leaving PFlash off. "
                "PFlash cannot serve the MLLM/VLM lane, so it is unavailable "
                "for this model (an explicit --pflash always/auto would be "
                "rejected at startup).",
                model_name,
            )
            return "off"
        # Surface the alias-driven flip at INFO so a developer running
        # ``rapid-mlx bench qwen3.5-4b-4bit`` immediately sees that
        # PFlash is on by default — the verified-tier policy is
        # uniform across ``serve``/``bench`` by design, but the bench
        # workflow specifically expects to see what mode is being
        # measured (codex r4 BLOCKING called out this surprise).
        logger.info(
            "PFlash default: alias %r is pflash_tier=verified — "
            "engine defaults to --pflash always. Pass --pflash off to "
            "compare against the no-compression baseline.",
            model_name,
        )
        return "always"
    return "off"


def resolve_pflash_keep_ratio_default(
    args: Any, *, model_name: str, _detected_config: Any = _DETECT
) -> float:
    """Resolve ``args.pflash_keep_ratio`` when the user passed nothing on the CLI.

    Precedence (mirrors :func:`resolve_pflash_mode_default`):

    * If ``args.pflash_keep_ratio`` is already a number (user passed
      ``--pflash-keep-ratio``), it wins — return it unchanged.
    * Else if the model's alias profile pins a ``pflash_keep_ratio``, use it.
      This is how an alias verified at a NON-default ratio (e.g. a ternary
      arch whose mid-prompt recall only survives at 0.50) gets its safe
      ratio applied whenever PFlash auto-enables — without it, a bare
      ``pflash_tier=verified`` would run the lossy 0.20 default.
    * Else the engine default 0.20.

    Returns the float to assign back to ``args.pflash_keep_ratio`` before
    :func:`config_from_args`. Kept separate from construction so unit tests
    can build a ``SimpleNamespace`` with ``pflash_keep_ratio=None`` and assert
    the resolved value directly.
    """
    if args.pflash_keep_ratio is not None:
        return args.pflash_keep_ratio
    cfg = _detect_or(model_name, _detected_config)
    if cfg is not None and cfg.pflash_keep_ratio is not None:
        return cfg.pflash_keep_ratio
    return 0.20


def resolve_pflash_config(
    args: Any,
    *,
    model_name: str,
    is_multimodal: bool = False,
    _detected_config: Any = _DETECT,
) -> PFlashConfig:
    """Resolve BOTH per-alias PFlash defaults (mode + keep_ratio) and build the
    validated :class:`PFlashConfig`. This is the single wiring shared by the
    ``serve`` and ``bench`` commands so the two never drift — and so the
    end-to-end alias→config path is testable in one call (a test that asserts
    both effective values here fails if either resolver is unwired, which a
    test calling the resolvers directly would not catch).

    Mutates ``args.pflash`` and ``args.pflash_keep_ratio`` in place (both were
    the CLI None-sentinels until now) so any later reader sees the resolved
    values, then returns the built config. ``validate_model_support`` is left
    to the caller because its ``is_mllm`` verdict and error handling differ
    between the two commands. ``_detected_config`` lets the serve entrypoint
    share its single resolved checkpoint profile with other startup defaults;
    the private sentinel preserves lazy detection for existing callers.
    """
    # Detect the alias profile ONCE and share it with both resolvers (each
    # would otherwise re-detect on the no-flag path). Skip detection entirely
    # when the user pinned both flags — neither resolver would consult it.
    if _detected_config is not _DETECT:
        detected = _detected_config
    elif args.pflash is None or args.pflash_keep_ratio is None:
        detected = _detect_or(model_name, _DETECT)
    else:
        detected = _DETECT
    args.pflash = resolve_pflash_mode_default(
        args,
        model_name=model_name,
        is_multimodal=is_multimodal,
        _detected_config=detected,
    )
    args.pflash_keep_ratio = resolve_pflash_keep_ratio_default(
        args, model_name=model_name, _detected_config=detected
    )
    return config_from_args(args)


def validate_model_support(
    config: PFlashConfig,
    *,
    model_name: str,
    is_mllm: bool = False,
) -> None:
    """Reject combinations PFlash cannot serve so they fail loudly at startup
    instead of silently no-op'ing inside the scheduler hot path."""
    if config.mode != "off" and is_mllm:
        raise ValueError(
            f"--pflash is not supported for multimodal models ({model_name}); "
            "disable --pflash for MLLM/VLM serving."
        )


@dataclass(frozen=True)
class _BlockScore:
    start: int
    end: int
    score: float


def compress_tokens(
    tokens: list[int],
    config: PFlashConfig,
    *,
    has_tools: bool = False,
    requires_prompt_integrity: bool = False,
) -> PFlashResult:
    """Compress a token list according to PFlash settings.

    Always preserves the leading sink and trailing tail; fills the
    remaining budget with middle blocks ranked by tail-query overlap and
    token rarity. Output preserves original order. Repeated filler tends
    to drop; uncommon tokens that reappear near the query are kept.
    """

    n_tokens = len(tokens)
    # Structural eligibility checks first — these are properties of the
    # request itself (schema-protected? tool prompt? empty?) and don't
    # depend on the engine's current PFlash mode. Reporting the
    # structural reason in skip telemetry is more actionable than
    # "off": a downstream telemetry consumer can tell whether the
    # request would have been a compression candidate IF the mode
    # were on. Operational checks (mode, threshold) come after so
    # ``--pflash off`` still short-circuits before the budget math.
    if requires_prompt_integrity:
        return _unchanged(tokens, "protected_prompt")
    if has_tools and config.skip_when_tools:
        return _unchanged(tokens, "tools")
    if n_tokens == 0:
        return _unchanged(tokens, "empty")
    if config.mode == "off":
        return _unchanged(tokens, "off")
    if config.mode == "auto" and n_tokens < config.threshold:
        return _unchanged(tokens, "threshold")

    block_size = max(1, config.block_size)
    keep_budget = _keep_budget(n_tokens, config)
    if keep_budget >= n_tokens:
        return _unchanged(tokens, "budget")

    sink_end = min(max(0, config.sink_tokens), n_tokens)
    tail_start = max(sink_end, n_tokens - max(0, config.tail_tokens))

    keep_positions = set(range(sink_end))
    keep_positions.update(range(tail_start, n_tokens))

    # Tokens available in the middle span (everything the sink/tail don't cover)
    # and the budget left for them after the endpoints are reserved. When
    # ``remaining_budget <= 0`` the sink+tail already meet or exceed the keep
    # budget, so not a single middle block can be selected — the whole body is
    # dropped. Track how many middle tokens actually survive so the result can
    # flag the degenerate "endpoints-only" regime.
    middle_span = tail_start - sink_end
    remaining_budget = keep_budget - len(keep_positions)
    if middle_span > 0 and remaining_budget <= 0:
        # The endpoints already consume the entire keep budget. Compressing in
        # this regime used to return a normal success after silently deleting
        # every token in the body — exactly where ordinary 2.3k-11.5k agent
        # conversations land under the verified Qwen3.5/3.6 defaults. There is
        # no middle selection to score, so preserve the prompt instead. These
        # prompts are also too short for the long-prefill win PFlash targets.
        return _unchanged(tokens, "insufficient_middle_budget")

    middle_selected = 0
    if remaining_budget > 0:
        scored_blocks = _score_middle_blocks(
            tokens=tokens,
            start=sink_end,
            stop=tail_start,
            block_size=block_size,
            query_window=max(1, config.query_window),
            stride_blocks=max(0, config.stride_blocks),
        )

        for block in scored_blocks:
            block_len = block.end - block.start
            slots = remaining_budget - middle_selected
            if slots <= 0:
                break
            take = min(block_len, slots)
            keep_positions.update(range(block.start, block.start + take))
            middle_selected += take
            if middle_selected >= remaining_budget:
                break

    kept = [tokens[i] for i in sorted(keep_positions)]
    if len(kept) >= n_tokens:
        return _unchanged(tokens, "budget")
    return _changed(tokens, kept, "compressed", middle_tokens_kept=middle_selected)


def compress_request_tokens(
    tokens: list[int],
    config: PFlashConfig,
    *,
    has_tools: bool = False,
    requires_prompt_integrity: bool = False,
) -> tuple[list[int], dict[str, int | bool | str | float]]:
    """Compress request tokens and return compact metadata for logging/state."""
    result = compress_tokens(
        tokens,
        config,
        has_tools=has_tools,
        requires_prompt_integrity=requires_prompt_integrity,
    )
    return result.tokens, {
        "compressed": result.compressed,
        "reason": result.reason,
        "original_tokens": result.original_tokens,
        "kept_tokens": result.kept_tokens,
        "dropped_tokens": result.original_tokens - result.kept_tokens,
        "compression_ratio": result.compression_ratio,
        "middle_tokens_kept": result.middle_tokens_kept,
        "endpoints_only": result.endpoints_only,
    }


def _keep_budget(n_tokens: int, config: PFlashConfig) -> int:
    ratio_budget = ceil(n_tokens * _clamp(config.keep_ratio, 0.0, 1.0))
    return max(1, min(n_tokens, max(config.min_keep_tokens, ratio_budget)))


def _score_middle_blocks(
    *,
    tokens: list[int],
    start: int,
    stop: int,
    block_size: int,
    query_window: int,
    stride_blocks: int,
) -> list[_BlockScore]:
    if start >= stop:
        return []

    # counts is global token frequency across the whole prompt — used as
    # the rarity denominator and the overlap weight. query_counts is the
    # tail query window. The blend tilts the score toward blocks whose
    # tokens reappear near the query (overlap) while still keeping rare
    # tokens (rarity) so a needle that only shows up once doesn't get
    # buried under chatty filler.
    counts = Counter(tokens)
    query = tokens[max(0, len(tokens) - query_window) :]
    query_counts = Counter(query)
    span = max(1, stop - start)

    blocks: list[_BlockScore] = []
    for block_index, block_start in enumerate(range(start, stop, block_size)):
        block_end = min(block_start + block_size, stop)
        block = tokens[block_start:block_end]

        overlap = sum(query_counts.get(token, 0) / counts[token] for token in block)
        rarity = sum(1.0 / counts[token] for token in block) / len(block)
        recency = (block_end - start) / span
        stride_bonus = (
            0.25 if stride_blocks and block_index % stride_blocks == 0 else 0.0
        )

        score = (4.0 * overlap) + rarity + (0.05 * recency) + stride_bonus
        blocks.append(_BlockScore(block_start, block_end, score))

    # Deterministic: identical scores fall back to start position so the
    # same input always yields the same output across runs.
    return sorted(blocks, key=lambda item: (-item.score, item.start))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _unchanged(tokens: list[int], reason: str) -> PFlashResult:
    return PFlashResult(
        tokens=tokens,
        compressed=False,
        reason=reason,
        original_tokens=len(tokens),
        kept_tokens=len(tokens),
    )


def _changed(
    tokens: list[int],
    kept: list[int],
    reason: str,
    *,
    middle_tokens_kept: int = 0,
) -> PFlashResult:
    return PFlashResult(
        tokens=kept,
        compressed=True,
        reason=reason,
        original_tokens=len(tokens),
        kept_tokens=len(kept),
        middle_tokens_kept=middle_tokens_kept,
    )
