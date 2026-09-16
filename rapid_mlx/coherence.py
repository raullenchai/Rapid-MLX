# SPDX-License-Identifier: Apache-2.0
"""Output-coherence primitives — the reusable core of the release coherence gate.

Motivation (#1247). Qwen3.6-35B-A3B and Qwen3.5-35B-A3B-8bit shipped producing
**pure garbage from the first token** (a doubled RMSNorm scale from an mlx-lm
``+1.0`` norm-shift misfire; fixed in #1234). The garbage passed *every*
automated gate — 278 delta tests, lint, install/import smoke, perf thresholds —
because **no gate ever generates a token and checks whether the output is
coherent**. This module supplies two pieces with deliberately different roles:

  * :data:`GOLDEN` + :func:`evaluate_case` — the **BLOCKING** layer. Fixed
    prompts with a strict, normalized, *checkable* answer (``capital of Japan``
    → ``Tokyo``, ``17 × 23`` → ``391``). A coherent-but-wrong regression fails
    (not an exact match), and garbage fails too (also not a match), so the
    blocking layer needs no heuristic help and cannot false-green on
    plausible-looking token soup.
  * :func:`looks_like_garbage` — an **ADVISORY / diagnostic** detector for the
    obvious collapse classes we have actually shipped (``!!!!!!`` prefix-cache
    poison, doubled-norm single-token soup, exact loops). It is *not* a reliable
    classifier — a frequency heuristic cannot separate diverse token soup
    (``"Ocean qzxv blorp fnarg glip."``) from prose without trading false
    negatives for false positives — so the runner surfaces it as a warning and
    it **never blocks a release**. Also intended for reuse by the telemetry
    garbage-rate alert (#1250), where an aggregate advisory signal is useful.

Everything here is pure (no network, no MLX, no server) so it can be unit-tested
in ordinary CI on a GitHub-hosted runner. The serve-path runner that feeds real
generations through these predicates lives in ``evals/coherence_gate.py`` and
runs on the Apple-Silicon release gauntlet (``scripts/release_check_m3.sh``).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

__all__ = [
    "GoldenCase",
    "GOLDEN",
    "looks_like_garbage",
    "is_degenerate_completion",
    "evaluate_case",
]

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Literal reasoning-channel markers that must never survive into the visible
# assistant message (the OutputRouter strips them; a routing regression leaks
# them). Kept lowercase — callers compare against a lowercased copy.
_THINK_MARKERS = ("<think>", "</think>", "<reasoning>", "</reasoning>")

# Closed <think>…</think> and <reasoning>…</reasoning> blocks plus a
# trailing unclosed opener. Reasoning-distill models (e.g. DeepSeek-R1-Distill)
# may leave chain-of-thought in the visible message, so the gate strips it
# before scoring the concluded answer. The block patterns match the same
# _THINK_MARKERS tag bytes.
_THINK_BLOCK_RE = re.compile(
    r"<think>[\s\S]*?</think>\s*"
    r"|<reasoning>[\s\S]*?</reasoning>\s*"
    r"|<think>[\s\S]*"
    r"|<reasoning>[\s\S]*",
    re.IGNORECASE,
)

# A reasoning model may put its terse final result in a terminal LaTeX box
# after a visible explanation.  Only accept a box at the very end; searching
# anywhere in the prose would let a model false-green merely by mentioning the
# expected token before reaching a different conclusion.
_LATEX_TEXT_WRAPPER_RE = re.compile(
    r"\\(?:text|mathrm|operatorname)\{([^{}]+)\}", re.IGNORECASE
)


def strip_thinking(text: str) -> str:
    """Remove reasoning-channel markers/blocks, returning only visible text."""
    if not text:
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


def _terminal_boxed_content(text: str) -> str | None:
    """Return the balanced content of a terminal ``\\boxed{...}``, if any."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                suffix = text[index + 1 :].strip()
                if suffix not in {"", r"\]"}:
                    return None
                content = text[content_start:index].strip()
                wrapper = _LATEX_TEXT_WRAPPER_RE.fullmatch(content)
                return wrapper.group(1) if wrapper else content
    return None


def _max_char_run(s: str) -> int:
    """Length of the longest run of a single non-space character."""
    best = run = 0
    prev = ""
    for ch in s:
        if ch == prev and not ch.isspace():
            run += 1
        else:
            run = 1
            prev = ch
        if run > best:
            best = run
    return best


def looks_like_garbage(text: str, *, min_words: int = 10) -> tuple[bool, str]:
    """Return ``(is_garbage, reason)`` for a completion string.

    Conservative: fires only on unambiguous degeneracy so real prose passes.
    Detects the classes we have actually shipped or seen:

    * empty / whitespace-only output;
    * punctuation/symbol-only output (``!!!!!``, CJK fill) at any length;
    * a single character dominating a non-trivial output (``aaaaa``);
    * a very long single-character run embedded in otherwise-mixed text;
    * a single word repeated (``ocean ocean ocean ocean``);
    * a tiny vocabulary over a long output (looping token soup);
    * one bigram dominating a long output (``the the the …``).

    Short legitimate answers (``"7"``, ``"42"``, ``"391"``, ``"Tokyo"``) are
    never flagged: the character-repetition heuristics only apply above a small
    length floor, and a valid answer always carries at least one word
    character. ``min_words`` guards the vocabulary/bigram checks so a short
    answer is never judged a loop.
    """
    s = (text or "").strip()
    if not s:
        return True, "empty"

    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return True, "whitespace-only"

    words = _WORD_RE.findall(s.lower())

    # (a) no word characters at all -> pure punctuation/symbol collapse
    # ("!!!!!", "。。。。", "?????"). Fires at any length; a legitimate answer to
    # any prompt carries at least one alphanumeric word character.
    if not words:
        return True, "no word characters (punctuation/symbol-only)"

    # (b) a single word repeated -> "ocean ocean ocean ocean". Unambiguous at
    # any length >= 4 (no legitimate answer is one word repeated 4+ times).
    if len(words) >= 4 and len(set(words)) == 1:
        return True, f"single word {words[0]!r} repeated {len(words)}x"

    # (c) character-repetition heuristics. Guarded by a length floor so short
    # legitimate answers ("7", "42", "OK") are never flagged: a doubled-norm /
    # cache-poison collapse that carries word characters ("aaaaa…") is long.
    short_numeric_answer = s.isdecimal() and len(non_space) < 20
    if len(non_space) >= 5 and not short_numeric_answer:
        top_char, top_n = Counter(non_space).most_common(1)[0]
        if top_n >= 5 and top_n / len(non_space) > 0.5:
            return True, (
                f"char {top_char!r} is {top_n}/{len(non_space)} of non-space output"
            )
        if _max_char_run(s) >= 20:
            return True, "single-character run >= 20"

    if len(words) >= min_words:
        # (c) tiny vocabulary over a long output -> looping token soup
        uniq_ratio = len(set(words)) / len(words)
        if uniq_ratio < 0.20:
            return True, (
                f"distinct-word ratio {uniq_ratio:.2f} < 0.20 over {len(words)} words"
            )

        # (d) one bigram dominates -> "the the the the …"
        bigrams = list(zip(words, words[1:]))
        if bigrams:
            _, bn = Counter(bigrams).most_common(1)[0]
            if bn / len(bigrams) > 0.30:
                return True, f"top bigram is {bn}/{len(bigrams)} of the output"

    return False, "ok"


def is_degenerate_completion(text: str | None) -> bool:
    """Boolean form of :func:`looks_like_garbage` for the #1250 telemetry
    canary: is a **non-empty** completion degenerate (repetition / single-token
    collapse)?

    Empty / whitespace-only input returns ``False`` — absence of output is a
    *separate* signal (the zero completion-token bucket), not degeneracy — so
    this stays a clean "non-empty content looks like garbage" indicator for the
    #1234 class (normal-length but garbage output). Pure and text-only: callers
    run it locally and emit only the returned bool, never the text.
    """
    s = text or ""
    if not s.strip():
        return False
    return looks_like_garbage(s)[0]


@dataclass(frozen=True)
class GoldenCase:
    """A fixed prompt plus a *deterministic* checkable predicate. Every golden
    case is BLOCKING and has a known-right answer — there are no heuristic
    (open-ended / prose) cases in the blocking set, because a structural check
    on free-form text still false-greens on diverse token soup that happens to
    include the required words (``"Ocean qzxv blorp. Water wug traz."``). Only
    exact-answer checks belong here.

    ``kind`` selects the predicate applied by :func:`evaluate_case`:

    * ``exact``         — normalized completion exactly matches one of ``expect``
    * ``no_think_leak`` — exact match AND no raw reasoning tag
    """

    id: str
    prompt: str
    kind: str
    expect: tuple[str, ...] = field(default_factory=tuple)
    max_tokens: int = 64


# Deterministic anchors a garbage / doubled-norm model cannot produce, and that
# a coherent-but-wrong regression also fails. Kept small + fast: on the starter
# alias (qwen3.5-4b-4bit) the whole set generates in well under a minute.
GOLDEN: tuple[GoldenCase, ...] = (
    GoldenCase(
        "capital-japan",
        "What is the capital of Japan? Answer in one word.",
        "exact",
        ("Tokyo",),
        max_tokens=32,
    ),
    GoldenCase(
        "arithmetic",
        "What is 17 multiplied by 23? Reply with just the number.",
        "exact",
        ("391",),
        max_tokens=32,
    ),
    GoldenCase(
        "sky-color",
        "What color is a clear daytime sky? Answer in one word.",
        "exact",
        ("blue",),
        max_tokens=32,
    ),
    GoldenCase(
        "days-in-week",
        "How many days are in a week? Reply with just the number.",
        "exact",
        ("7", "seven"),
        max_tokens=32,
    ),
    GoldenCase(
        # Instruction-following + coherence: a garbage / doubled-norm model
        # cannot echo a distinctive token. "banana" is 6 chars and vanishingly
        # unlikely to appear by chance, so this is far more discriminating than
        # a case-insensitive single-letter check.
        "echo-word",
        "Repeat exactly this word back to me, nothing else: banana",
        "exact",
        ("banana",),
        max_tokens=32,
    ),
    GoldenCase(
        "no-think-leak",
        "What is the capital of France? Answer in one word.",
        "no_think_leak",
        ("Paris",),
        max_tokens=64,
    ),
)


_ANSWER_WRAPPER = " \t\r\n`*_~\"'“”‘’.,!?;:。！？；："


def _normalize_exact_answer(text: str) -> str:
    """Normalize harmless presentation around a requested one-token answer."""
    return " ".join(text.casefold().strip(_ANSWER_WRAPPER).split())


def _matches_exact(text: str, expected: tuple[str, ...]) -> bool:
    answer = _normalize_exact_answer(text)
    return any(answer == _normalize_exact_answer(item) for item in expected)


def evaluate_concluded(case: GoldenCase, text: str) -> tuple[bool, str]:
    """Score the *concluded* answer of a reasoning-distill completion.

    Reasoning-distill models (DeepSeek-R1-Distill) emit chain-of-thought in the
    visible channel even when the server was told not to think; the reasoning
    prose is stripped and the remaining text is checked as an exact match, so
    the gate measures the concluded answer rather than format compliance.
    """
    if not isinstance(text, str):
        return False, f"invalid response content type {type(text).__name__}"
    concluded = strip_thinking(text)
    if not concluded:
        return False, "no concluded answer after stripping reasoning channel"
    if _matches_exact(concluded, case.expect):
        return True, f"concluded answer exactly matches {case.expect!r}"
    boxed = _terminal_boxed_content(concluded)
    if boxed is not None and _matches_exact(boxed, case.expect):
        return True, f"terminal boxed conclusion exactly matches {case.expect!r}"
    return False, f"concluded answer not an exact match for {case.expect!r}"


def evaluate_case(case: GoldenCase, text: str) -> tuple[bool, str]:
    """Apply ``case``'s deterministic golden predicate to a completion ``text``.

    Returns ``(passed, reason)``. This is the BLOCKING layer: a strict,
    normalized golden-answer check with NO heuristic garbage screen. Garbage or
    a fluent-but-wrong answer fails naturally — it is simply not an exact match
    for the expected token — so the blocking layer cannot false-green on
    plausible-looking token soup. Garbage *detection*
    (:func:`looks_like_garbage`) is a separate ADVISORY signal the runner prints
    but never blocks on, because a frequency heuristic cannot reliably separate
    diverse token soup (``"Ocean qzxv blorp fnarg glip."``) from prose.
    """
    if not isinstance(text, str):
        return False, f"invalid response content type {type(text).__name__}"

    if case.kind == "exact":
        if _matches_exact(text, case.expect):
            return True, f"exactly matches {case.expect!r}"
        return False, f"not an exact match for {case.expect!r}"

    if case.kind == "no_think_leak":
        low = text.lower()
        leaked = [m for m in _THINK_MARKERS if m in low]
        if leaked:
            return False, f"leaked reasoning marker(s) {leaked!r} into visible output"
        if _matches_exact(text, case.expect):
            return True, f"exactly matches {case.expect!r}, no think-leak"
        return False, f"not an exact match for {case.expect!r}"

    return False, f"unknown case kind {case.kind!r}"
