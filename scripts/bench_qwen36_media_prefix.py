#!/usr/bin/env python3
"""Qualify the serialized-MLLM media prefix cache against the cold lane.

Paired A/B on one machine, one model revision, one process:

* baseline phase — ``SchedulerConfig(mllm_media_prefix_cache="off")``;
* candidate phase — ``"auto"`` (the shipped default).

Both phases run the shipped singleton fast path (``mllm_singleton_fastpath``
default) so the A/B isolates the media boundary cache alone. Conversations
come from the tracked manifest
``evals/prompts/qwen36_media_prefix_conversations.json`` (20 multi-turn
media conversations, 3 user turns each, one image per conversation). Each
measured pass replays every conversation turn by turn, appending the
assistant replies as it goes, so turn 2+ renders a strict extension of the
turn-1 media prefix — the resume path's precondition.

The primary deterministic gates are:

* **within-phase determinism** — every pass inside a phase produces the
  identical SHA-256 per turn (both phases; requires ``--pairs >= 2``);
* **semantic correctness** — every turn passes its manifest checker in both
  phases, guarding against two equally wrong outputs passing;
* **engagement** — the candidate phase must actually store boundary
  snapshots and serve warm resumes (``stores > 0`` and ``hits > 0``) while
  the baseline does neither, else the A/B compares off against off;
* **resume coverage** — the turns that are supposed to resume (turn 2+;
  turn 1 is the documented store turn) must actually serve a media
  resume in at least 75% of their measured samples (exit 4 otherwise),
  with every miss reported. The per-sample signal is the engine's media
  hit-counter delta around that one request — ``cached_tokens`` alone
  conflates the text exact-cache path with the media resume path;
* **no resume-turn regression** — every measured sample that actually
  served a media resume must not exceed the baseline median TTFT for its
  (conversation, turn) by more than a 15% stall margin (exit 3
  otherwise). Samples that store (or miss) pay the documented bounded
  snapshot cost and are excluded.

Scope: this harness gates determinism, semantics, engagement, and
warm-turn latency on the qualified model. The remaining design-note gates
(bounded-memory soak, cancellation-recovery probes, sampled fixed-seed A/B
through the real request API) are covered by separate evidence — see the
design note's qualification section.

Cross-phase per-turn SHA equality is *reported* (``exact_by_conversation``)
but not gated: the warm turn's suffix forward runs its input-projection
GEMMs at the suffix batch size while the cold lane runs them at the full
sequence length, and Metal's GEMM tiling regimes make per-row results
coincide for some (suffix, full) length pairs and differ by one bf16 ULP
for others. That single-ULP difference compounds through the hybrid
recurrence into ~1-2 logits of drift with identical argmax structure —
equally valid phrasing at temperature 0, the same exposure any warm-prefix
resume has on this lane. The media boundary is aligned to the recurrent
scan's tile grid (``_MEDIA_BOUNDARY_ALIGN_TOKENS``) so the fixable half of
this (GatedDeltaNet re-tiling) is bit-exact; the GEMM-regime half is
inherent to splitting a prefill.

Usage (Studio qualification, 256 GB, offline model resolution; local
manifest images require the media-root lockdown env):

    RAPID_MLX_MEDIA_ROOT=<repo-root> python -m scripts.bench_qwen36_media_prefix \\
        --model <snapshot-path> --pairs 2 --output /tmp/media-prefix.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "evals/prompts/qwen36_media_prefix_conversations.json"

try:
    from scripts.bench_metadata import write_bench_json
except ImportError:  # direct-script execution fallback
    from bench_metadata import write_bench_json


def _media_root_from_manifest(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    """Resolve the manifest's media root, anchored to this repository.

    ``images_root`` comes from the caller-controlled manifest, so it is not
    itself trustworthy: the containment anchor is the repository the harness
    ships in. A manifest may relocate the media root within the repo (shared
    image directories) but never outside it — ``images_root: "/"`` with
    arbitrary readable files is rejected here rather than defeating the
    per-image path checks in ``_conversation_messages``.
    """
    repo_root = (
        manifest_path.resolve().parent / manifest.get("images_root", ".")
    ).resolve()
    if repo_root != ROOT and ROOT not in repo_root.parents:
        raise ValueError(
            f"manifest images_root must resolve inside the repository: {repo_root}"
        )
    return repo_root


def _conversation_messages(
    conversation: dict[str, Any],
    repo_root: Path,
    turn_index: int,
    replies: list[str],
) -> list[dict[str, Any]]:
    """Messages for ``turn_index`` (0-based) of a conversation.

    Prior user turns are replayed verbatim and each recorded assistant reply
    is appended, so the rendered prompt strictly extends the previous turn's
    prompt — the media resume path's strict token-prefix precondition. The
    image rides only on the first user turn and reaches later turns through
    that history message: the engine collects images from the whole message
    list, and re-sending it per turn would add a second placeholder run
    into the resumed suffix (whose forward intentionally carries
    ``pixel_values=None``).
    """
    # A caller-supplied manifest must not aim the engine at arbitrary local
    # files: image references are relative to the manifest's media root and
    # their resolved location must stay inside it (absolute paths and
    # ``..`` escapes are rejected, and symlinks are resolved before the
    # containment check so they cannot point out either).
    media_root = repo_root.resolve()
    image_paths = []
    for image in conversation["images"]:
        supplied = Path(image)
        if supplied.is_absolute():
            raise ValueError(f"manifest image must be a relative path: {image}")
        resolved = (repo_root / supplied).resolve()
        if resolved != media_root and media_root not in resolved.parents:
            raise ValueError(f"manifest image escapes the media root: {image}")
        if not resolved.is_file():
            raise FileNotFoundError(f"manifest image missing: {resolved}")
        image_paths.append(resolved)
    messages: list[dict[str, Any]] = []
    for index in range(turn_index + 1):
        content: list[dict[str, Any]] = [
            {"type": "text", "text": conversation["turns"][index]["prompt"]}
        ]
        if index == 0:
            # Every declared image rides on the first user turn — the
            # multi-image comparison cases must actually run multi-image.
            for image_path in image_paths:
                content.append(
                    {"type": "image_url", "image_url": {"url": str(image_path)}}
                )
        messages.append({"role": "user", "content": content})
        if index < turn_index:
            messages.append({"role": "assistant", "content": replies[index]})
    return messages


_PUNCT = '.,;:!?’”"()[]{}<>`—–-“„«»'

# Characters that fuse words in prose ("skip"/"next", "dark-themed",
# "qwen3.6-27b" on the text side): normalized to spaces before tokenizing so
# compound forms split into the same tokens the term side does.
_INNER_BREAKS = str.maketrans({"/": " ", "-": " ", "–": " ", "—": " "})

# Tokens that invert a preceding claim: a required term directly preceded by
# one of these does not satisfy the requirement ("not ready" must not pass a
# "ready" requirement). Applied to *required* matching only — ``forbidden``
# terms stay negation-blind (claiming "not ready" on a screen whose chip says
# otherwise is still wrong content).
# Contrast conjunctions close the preceding clause: a negator behind one
# targets the contrast, not the term after it ("not idle, but ready").
_CONTRASTS = frozenset({"but", "yet", "however", "though", "although", "while"})

_NEGATORS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "cannot",
        "without",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "don't",
        "doesn't",
        "didn't",
        "can't",
        "won't",
    }
)


def _tokens(text: str) -> list[str]:
    """Whitespace tokens, casefolded, edge punctuation stripped, empties out.

    Curly apostrophes normalize to ASCII first: contractions written with
    typographic quotes ("isn’t") must hit the same negator forms as their
    ASCII spellings, or an explicitly false answer slips past the negation
    guard.

    Slash/hyphen runs are broken into separate tokens first so compound
    forms ("skip"/"next", "dark-themed") tokenize the same on the term and
    text sides.
    """
    tokens = []
    for raw in (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .casefold()
        .translate(_INNER_BREAKS)
        .split()
    ):
        token = raw.strip(_PUNCT)
        if token:
            tokens.append(token)
    return tokens


def _term_matches(text_tokens: list[str], term: str, *, guard_negation: bool) -> bool:
    """Token/phrase-boundary term match.

    Substring matching admits wrong answers ("bright side" for "right",
    "not ready" for "ready"); matching on normalized token sequences fixes
    both — the phrase must appear as a contiguous token run, and with
    ``guard_negation`` a negator in the three tokens before the run does
    not count ("not currently ready", "is not the ready state").
    """
    phrase = _tokens(term)
    if not phrase:
        return False
    width = len(phrase)
    for start in range(len(text_tokens) - width + 1):
        if text_tokens[start : start + width] != phrase:
            continue
        if guard_negation:
            # Scan the three tokens before the phrase right-to-left: a
            # negator attached to the phrase rejects the match, but a
            # contrast conjunction ends the clause the negation lives in —
            # "not idle, but ready" asserts ``ready``.
            window = text_tokens[max(0, start - 3) : start]
            negated = False
            for token in reversed(window):
                if token in _CONTRASTS:
                    break
                if token in _NEGATORS:
                    negated = True
                    break
            if negated:
                continue
            # Negation after the phrase inverts it too: "ready is not the
            # status" asserts the chip is anything but ready. Only a
            # copula-linked pattern counts — "ready, not idle" contrasts
            # the term against another and stays a positive claim about
            # ``ready``.
            after = text_tokens[start + width : start + width + 3]
            if (
                after
                and after[0] in {"is", "are", "was", "were"}
                and any(token in _NEGATORS for token in after[1:])
            ):
                continue
        return True
    return False


def _structural_pass(checker: dict[str, Any], text: str) -> bool:
    """Explicit per-turn formatting constraints from the prompt wording.

    "Two short lines", "one short line", "only its exact label",
    "at least 200 words" — keyword presence alone lets a terse blob with the
    right substrings pass, so the manifest's structural constraints are
    validated here: word-count bounds and line-count bounds.
    """
    min_words = int(checker.get("min_words", 0) or 0)
    max_words = int(checker.get("max_words", 0) or 0)
    min_lines = int(checker.get("min_lines", 0) or 0)
    max_lines = int(checker.get("max_lines", 0) or 0)
    words = len(text.split())
    # Lines are non-blank splitlines: a trailing newline cannot fake a second
    # line ("Ready\n" is one line), and blank padding lines neither satisfy a
    # minimum nor violate a maximum — the prompts ask for content lines.
    lines = sum(1 for line in text.splitlines() if line.strip())
    if min_words and words < min_words:
        return False
    if max_words and words > max_words:
        return False
    if min_lines and lines < min_lines:
        return False
    return not (max_lines and lines > max_lines)


def _required_any_groups_pass(checker: dict[str, Any], text_tokens: set[str]) -> bool:
    """``required_any_groups``: AND over groups of OR-alternatives.

    The summary turns use one group per prior answer, so a response that
    drops an entire answer fails even though its anchors also occur in the
    other answer. Each group's alternatives span the subject's phrasings
    observed across hosts (detail answers vary at temperature 0 across
    machines; a single phrasing is not groundable). Shared by both the
    ``terms`` and ``any`` kinds — a ``terms`` checker that pins per-control
    groups (accessibility descriptions) relies on it too.
    """
    for group in checker.get("required_any_groups", []):
        if not any(
            all(
                _term_matches(text_tokens, term, guard_negation=True)
                for term in alternative
            )
            for alternative in group
        ):
            return False
    return True


def _checker_pass(
    checker: dict[str, Any], text: str, earlier_texts: tuple[str, ...] = ()
) -> bool:
    text_tokens = _tokens(text)
    kind = checker.get("type", "any")
    if kind == "terms":
        if not all(
            _term_matches(text_tokens, term, guard_negation=True)
            for term in checker.get("required", [])
        ):
            return False
        # ``required_any``: alternatives — at least one term list fully
        # satisfied (e.g. a shortcut rendered as "⌘N" or "Cmd+N").
        required_any = checker.get("required_any", [])
        if required_any and not any(
            all(
                _term_matches(text_tokens, term, guard_negation=True)
                for term in alternative
            )
            for alternative in required_any
        ):
            return False
        # ``line_expect`` pins ordered per-line content for the multi-image
        # turns ("report the first bar, then the second, one line each"):
        # line 1 must carry the first bar's terms and line 2 the second's,
        # so swapping the two image answers fails — with flat ``required``
        # both orderings pass although image ordering is the behavior under
        # test. Paired with ``min_lines``/``max_lines`` it pins the
        # one-line-per-bar shape too.
        line_expect = checker.get("line_expect", [])
        if line_expect:
            # The same non-blank-line sequence _structural_pass counts:
            # blank padding must not shift which line carries which terms.
            lines = [line for line in text.splitlines() if line.strip()]
            for line_index, alternative in enumerate(line_expect):
                terms = alternative if isinstance(alternative, list) else [alternative]
                line_tokens = (
                    _tokens(lines[line_index]) if line_index < len(lines) else []
                )
                if not all(
                    _term_matches(line_tokens, term, guard_negation=True)
                    for term in terms
                ):
                    return False
        if not _structural_pass(checker, text):
            return False
        if not _required_any_groups_pass(checker, text_tokens):
            return False
        return not any(
            _term_matches(text_tokens, term, guard_negation=False)
            for term in checker.get("forbidden", [])
        )
    if kind == "json_shape":
        # "Output JSON only": the stripped response must parse as one JSON
        # object — searching for the first ``{`` would let prose wrapped
        # around an all-null payload pass ("JSON only" with entirely wrong
        # field values). A fenced response passes only when the fence is
        # well-formed — a supported label on the opening fence, a matching
        # terminal fence, and nothing but the JSON object between them:
        # the qualified model deterministically formats its JSON-only
        # answers as a single fenced code block, so the fence is that
        # model's markup for "JSON only", not prose — prose outside the
        # fence still fails the whole-payload parse, and an unclosed or
        # mislabeled fence is malformed output, not markup.
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline == -1:
                return False
            label = stripped[3:first_newline].strip()
            if label not in ("", "json"):
                return False
            body = stripped[first_newline + 1 :]
            closing = body.rfind("\n```")
            if closing == -1 or body[closing + 4 :].strip():
                return False
            stripped = body[:closing]
        try:
            payload = json.loads(stripped)
        except (ValueError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        if not all(key in payload for key in checker.get("keys", [])):
            return False
        # ``list_keys`` must be non-empty lists; ``item_keys`` requires
        # these keys on every item — key presence alone lets any value
        # qualify. ``list_len`` pins exact lengths and ``list_expect``
        # validates each item's designated values in order (a bars list
        # missing its second bar, or with null/wrong statuses, fails).
        # ``field_terms`` pins the value of a designated field (a bare
        # ``{"model": null, ...}`` no longer satisfies ``required``).
        item_keys = checker.get("item_keys", {})
        list_len = checker.get("list_len", {})
        list_expect = checker.get("list_expect", {})
        for key in checker.get("list_keys", []):
            value = payload.get(key)
            if not isinstance(value, list) or not value:
                return False
            if key in list_len and len(value) != int(list_len[key]):
                return False
            # ``item_keys`` applies to object lists (``{"model", "status"}``
            # bars); plain string lists (buttons) only need to be non-empty.
            if key in item_keys:
                for item in value:
                    if not isinstance(item, dict) or not all(
                        item_key in item for item_key in item_keys[key]
                    ):
                        return False
            if key in list_expect:
                expected_items = list_expect[key]
                if len(value) != len(expected_items):
                    return False
                for item, expected in zip(value, expected_items):
                    if isinstance(expected, dict):
                        if not isinstance(item, dict):
                            return False
                        for field, terms in expected.items():
                            field_tokens = _tokens(str(item.get(field, "")))
                            if not isinstance(terms, list):
                                terms = [terms]
                            if not any(
                                _term_matches(field_tokens, term, guard_negation=True)
                                for term in terms
                            ):
                                return False
                    else:
                        alternatives = (
                            expected if isinstance(expected, list) else [expected]
                        )
                        item_tokens = _tokens(str(item))
                        if not any(
                            _term_matches(item_tokens, term, guard_negation=True)
                            for term in alternatives
                        ):
                            return False
        for key, terms in checker.get("field_terms", {}).items():
            if key not in payload:
                return False
            # ``ensure_ascii=False``: the default JSON escaping rewrites
            # non-ASCII characters as backslash-unicode escapes before
            # tokenizing, so a field term written as the on-screen text
            # (e.g. a keyboard-shortcut glyph) could never match its own
            # designated field.
            field_tokens = _tokens(
                json.dumps(payload.get(key), default=str, ensure_ascii=False)
            )
            if not isinstance(terms, list):
                terms = [terms]
            if not all(
                _term_matches(field_tokens, term, guard_negation=True) for term in terms
            ):
                return False
        if not _structural_pass(checker, text):
            return False
        return not any(
            _term_matches(text_tokens, term, guard_negation=False)
            for term in checker.get("forbidden", [])
        ) and all(
            _term_matches(text_tokens, term, guard_negation=True)
            for term in checker.get("required", [])
        )
    if kind == "any":
        # Open-ended follow-up turns are still grounded three ways:
        # ``min_words`` rejects degenerate outputs (empty, single looping
        # token) — word count alone is gameable ("foo foo foo foo foo"), so
        # the vocabulary must spread too: at least half the floor (minimum
        # 2) distinct words. ``required_any`` carries the semantics: at
        # least one alternative term list must be fully present, anchored
        # to content a correct answer must reference (the conversation's
        # own screen elements or its prior answers). ``min_lines``/
        # ``max_lines``/``max_words`` pin the prompt's formatting asks.
        min_words = int(checker.get("min_words", 0) or 0)
        if min_words > 0:
            if len(text.split()) < min_words:
                return False
            distinct = {word.casefold().strip('.,;:!?’”"()') for word in text.split()}
            distinct.discard("")
            if len(distinct) < max(2, min_words // 2):
                return False
        if not _structural_pass(checker, text):
            return False
        # ``required_any`` anchors semantics: at least ``required_any_min``
        # (default 1) alternatives must be fully present.
        required_any = checker.get("required_any", [])
        if required_any:
            required_min = int(checker.get("required_any_min", 1) or 1)
            satisfied = [
                alternative
                for alternative in required_any
                if all(
                    _term_matches(text_tokens, term, guard_negation=True)
                    for term in alternative
                )
            ]
            if len(satisfied) < required_min:
                return False
            # ``novel``: the prompt asked for a detail "not mentioned yet",
            # so an alternative whose every term already appears in the
            # earlier turns of this pass cannot count — an answer that only
            # repeats an already-mentioned anchor fails even though the
            # anchor itself is required vocabulary.
            if checker.get("novel") and earlier_texts:
                earlier_tokens = _tokens("\n".join(earlier_texts))
                if not any(
                    not all(
                        _term_matches(earlier_tokens, term, guard_negation=True)
                        for term in alternative
                    )
                    for alternative in satisfied
                ):
                    return False
        # ``required_any_groups``: AND over groups of OR-alternatives — the
        # summary turns use one group per prior answer, so a response that
        # drops an entire answer fails even though its anchors also occur
        # in the other answer. Each group's alternatives span the subject's
        # phrasings observed across hosts (detail answers vary at
        # temperature 0 across machines; a single phrasing is not
        # groundable).
        if not _required_any_groups_pass(checker, text_tokens):
            return False
        return True
    raise ValueError(f"unknown checker type: {kind}")


def _median(samples: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(sample[key]) for sample in samples)


def _media_hits(engine: Any) -> int:
    """Media-resume hit counter, 0 when the engine reports no media stats."""
    media = engine.get_stats().get("media_prefix_cache") or {}
    return int(media.get("hits", 0) or 0)


WARM_REGRESSION_MARGIN = 1.15
RESUME_COVERAGE_FLOOR = 0.75


def _resume_gate_samples(
    auto_passes: list[list[dict[str, Any]]],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Per-sample resume accounting for one conversation.

    Turn 1 is the documented store turn (turn-0 prompts fall below the
    boundary min-tokens floor, so turn 1 snapshots instead of resuming);
    turns 2+ are the expected resume slots. Returns
    ``(expected_samples, resumed_samples, misses)`` where each miss names
    the measured pass and turn whose media counter did not move. Gating
    per measured sample — not per-slot median — so a slot where only one
    pass resumed cannot hide behind the other pass's cold miss. The
    signal is the per-sample media-hit delta, not ``cached_tokens``: that
    field is stamped by the text exact-cache path too, and a text warm
    hit would otherwise pass the media coverage gate.
    """
    expected = 0
    resumed = 0
    misses: list[dict[str, Any]] = []
    for pass_index, passes in enumerate(auto_passes):
        for sample in passes:
            if sample["turn"] < 2:
                continue
            expected += 1
            if sample["media_hit"]:
                resumed += 1
            else:
                misses.append({"pass": pass_index, "turn": sample["turn"]})
    return expected, resumed, misses


def _resume_regressions(
    auto_passes: list[list[dict[str, Any]]],
    baseline_median_by_turn: dict[int, float],
    margin: float = WARM_REGRESSION_MARGIN,
) -> list[dict[str, Any]]:
    """Per-sample warm-regression check for one conversation.

    Every measured sample that actually served a media resume (per-sample
    media-hit delta; ``cached_tokens`` alone would also count text
    exact-cache warm hits, which are not this feature's latency to
    defend) is compared against the baseline median TTFT for its turn;
    medians never mix cold and resumed executions. Turns 0-1 are excluded
    (turn 1 is the documented store turn — its bounded snapshot cost is in
    the design note — and a turn-1 "resume" can only come from an earlier
    pass's same-prompt entry, not this pass's lifecycle).
    """
    flagged: list[dict[str, Any]] = []
    for pass_index, passes in enumerate(auto_passes):
        for sample in passes:
            turn = sample["turn"]
            if turn < 2 or not sample["media_hit"]:
                continue
            baseline = baseline_median_by_turn.get(turn)
            if baseline is None or baseline <= 0:
                continue
            ttft = float(sample["ttft_s"])
            if ttft / baseline > margin:
                flagged.append(
                    {
                        "pass": pass_index,
                        "turn": turn,
                        "ttft_s": ttft,
                        "baseline_median_ttft_s": baseline,
                        "cached_tokens": int(sample["cached_tokens"]),
                    }
                )
    return flagged


async def _run_turn(
    engine: Any,
    conversation: dict[str, Any],
    repo_root: Path,
    turn_index: int,
    replies: list[str],
) -> dict[str, Any]:
    turn = conversation["turns"][turn_index]
    sampling = turn.get("sampling", {})
    started = time.perf_counter()
    first_token_at: float | None = None
    final = None
    # Per-sample media attribution: ``cached_tokens`` conflates the text
    # exact-cache path with the media resume path (both stamp the same
    # field), so the gates cannot tell a media resume from a text warm hit
    # off it. Snapshot the media hit counter around this one request — the
    # serialized lane runs exactly one request per turn, so the delta is
    # this sample's media-resume boolean.
    media_hits_before = _media_hits(engine)
    async for output in engine.stream_chat(
        messages=_conversation_messages(conversation, repo_root, turn_index, replies),
        max_tokens=int(turn["max_tokens"]),
        temperature=float(sampling.get("temperature", 0.0)),
        top_p=float(sampling.get("top_p", 1.0)),
        enable_thinking=False,
    ):
        if first_token_at is None and (output.new_text or output.completion_tokens > 0):
            first_token_at = time.perf_counter()
        final = output
    ended = time.perf_counter()
    if final is None:
        raise RuntimeError(
            f"conversation {conversation['id']} turn {turn_index} produced no output"
        )
    text = final.raw_text or final.text or ""
    return {
        "ttft_s": first_token_at - started if first_token_at else 0.0,
        "elapsed_s": ended - started,
        "media_hit": _media_hits(engine) - media_hits_before > 0,
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "checker_pass": _checker_pass(
            turn.get("checker", {}), text, earlier_texts=tuple(replies)
        ),
        "prompt_tokens": int(final.prompt_tokens),
        "completion_tokens": int(final.completion_tokens),
        "cached_tokens": int(final.cached_tokens),
    }


async def _replay_conversation(
    engine: Any, conversation: dict[str, Any], repo_root: Path
) -> list[dict[str, Any]]:
    """One measured pass: every turn in order, replies feeding the next turn."""
    replies: list[str] = []
    samples = []
    for turn_index in range(len(conversation["turns"])):
        sample = await _run_turn(engine, conversation, repo_root, turn_index, replies)
        sample["turn"] = turn_index
        replies.append(sample["text"])
        samples.append(sample)
    return samples


async def _run_phase(
    engine: Any,
    conversations: list[dict[str, Any]],
    repo_root: Path,
    pairs: int,
) -> dict[str, Any]:
    """Warmup pass, then ``pairs`` measured passes over every conversation."""
    for conversation in conversations:
        await _replay_conversation(engine, conversation, repo_root)

    # The warmup pass must not leak into the measured passes. Warmup stores
    # boundary snapshots; left in place, measured pass 1 would resume them
    # and the latency gates would never observe the documented store-turn
    # cost (and a warmup-only store would keep the cache alive without any
    # measured pass proving it can store). Drop every cached entry and
    # reset the counters before EACH measured pass so every pass runs the
    # full store→resume lifecycle self-contained — store on the store turn,
    # resume on the turns after — and no pass starts from a prior pass's
    # leftovers.
    by_conversation: dict[str, list[list[dict[str, Any]]]] = {
        conversation["id"]: [] for conversation in conversations
    }
    media = {key: 0 for key in ("hits", "misses", "stores", "budget_evictions")}
    for _ in range(pairs):
        engine.clear_prefix_cache(reset_stats=True)
        for conversation in conversations:
            by_conversation[conversation["id"]].append(
                await _replay_conversation(engine, conversation, repo_root)
            )
        # Counters were reset at the pass start, so the post-pass values are
        # that pass's totals; sum across passes. Gauges (entries/bytes/
        # budget) report the end-of-phase state as-is.
        pass_stats = engine.get_stats().get("media_prefix_cache") or {}
        for key in media:
            media[key] += int(pass_stats.get(key, 0) or 0)
        if "entries" in pass_stats:
            for key in ("entries", "bytes", "budget_bytes"):
                if key in pass_stats:
                    media[key] = pass_stats[key]
    return {
        "per_conversation": by_conversation,
        "media_prefix_cache": media,
        "summary": {
            conversation_id: {
                "median_ttft_s": [
                    _median(passes, "ttft_s") for passes in zip(*turn_streams)
                ]
                if turn_streams
                else [],
                "median_elapsed_s": [
                    _median(passes, "elapsed_s") for passes in zip(*turn_streams)
                ]
                if turn_streams
                else [],
                "median_cached_tokens": [
                    _median(passes, "cached_tokens") for passes in zip(*turn_streams)
                ]
                if turn_streams
                else [],
                "checker_passes": sum(
                    bool(sample["checker_pass"])
                    for passes in turn_streams
                    for sample in passes
                ),
            }
            for conversation_id, turn_streams in by_conversation.items()
        },
    }


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model snapshot path")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--conversations",
        action="append",
        default=[],
        help="Conversation id filter (repeatable); default runs the whole manifest",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=2,
        help="Measured passes per phase; >= 2 so the determinism gate is meaningful",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    # One measured pass cannot demonstrate within-phase determinism: the
    # gate would pass vacuously on a single sample. Fail at the door.
    if args.pairs < 2:
        parser.error("--pairs must be >= 2 (within-phase determinism gate)")

    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.scheduler import SchedulerConfig

    manifest = json.loads(args.manifest.read_text())
    repo_root = _media_root_from_manifest(args.manifest, manifest)
    conversations = [
        conversation
        for conversation in manifest["conversations"]
        if not args.conversations or conversation["id"] in args.conversations
    ]
    if not conversations:
        raise SystemExit("no manifest conversations selected")

    result: dict[str, Any] = {
        "model": str(Path(args.model).expanduser().resolve()),
        "manifest": str(args.manifest),
        "pairs": args.pairs,
        "phases": {},
    }

    async def engine_for(flag: str) -> BatchedEngine:
        engine = BatchedEngine(
            str(Path(args.model).expanduser().resolve()),
            force_mllm=True,
            scheduler_config=SchedulerConfig(mllm_media_prefix_cache=flag),
        )
        try:
            await engine.start()
        except BaseException:
            # A partially-initialized engine may already hold worker/model
            # resources; the phase-level ``finally`` below is not installed
            # yet, so stop() what was built before propagating.
            with contextlib.suppress(Exception):
                await engine.stop()
            raise
        return engine

    try:
        # Baseline runs first: the candidate phase then executes on a
        # thermally warmer machine, so any ordering bias works AGAINST the
        # candidate — the headline saving understates and the 15%
        # resume-regression gate is measured against a cooler-machine
        # baseline (stricter). Compilation, allocator, and filesystem
        # warmth are per-engine and already absorbed by each phase's own
        # warmup pass, so the measured comparison is warm-vs-warm; only
        # residual thermal drift remains, and it points the safe way.
        # Alternating phase order would need one engine per phase per pair
        # (load/unload dominates the run) for a bias the fixed order
        # already points the safe way.
        # Baseline: cold lane, no boundary snapshots.
        baseline = await engine_for("off")
        try:
            result["phases"]["off"] = await _run_phase(
                baseline, conversations, repo_root, args.pairs
            )
        finally:
            await baseline.stop()

        # Candidate: shipped default media prefix cache.
        candidate = await engine_for("auto")
        try:
            result["phases"]["auto"] = await _run_phase(
                candidate, conversations, repo_root, args.pairs
            )
        finally:
            await candidate.stop()

        # Per-turn hashes across phases and passes. A turn is "exact" only
        # when every pass in both phases produced the identical hash
        # (reported; see module docstring for why this is not gated).
        # Turn completeness: every conversation must produce exactly its
        # declared turn count in every pass of both phases. Deriving the
        # count from the observed minimum would silently omit a missing or
        # zero-turn conversation while determinism and checker gates stay
        # green over the surviving turns.
        for conversation, phase_streams in (
            (
                c,
                (
                    result["phases"]["off"]["per_conversation"].get(c["id"]),
                    result["phases"]["auto"]["per_conversation"].get(c["id"]),
                ),
            )
            for c in conversations
        ):
            declared = len(conversation["turns"])
            for phase_name, streams in zip(("off", "auto"), phase_streams):
                if streams is None:
                    raise SystemExit(
                        f"{conversation['id']}: phase {phase_name} produced no samples"
                    )
                for pass_index, passes in enumerate(streams):
                    if len(passes) != declared:
                        raise SystemExit(
                            f"{conversation['id']}: phase {phase_name} pass "
                            f"{pass_index} produced {len(passes)} of {declared} "
                            "declared turns"
                        )
        exact_by_turn: dict[str, list[bool]] = {}
        within_phase_deterministic: dict[str, bool] = {}
        checker_pass_both: dict[str, list[bool]] = {}
        for conversation_id, turn_streams in result["phases"]["off"][
            "per_conversation"
        ].items():
            auto_streams = result["phases"]["auto"]["per_conversation"][conversation_id]
            # Completeness was validated above; the count is the manifest's.
            turn_count = len(
                next(c["turns"] for c in conversations if c["id"] == conversation_id)
            )
            flags = []
            checker_flags = []
            for turn_index in range(turn_count):
                phase_hashes: dict[str, set[str]] = {}
                for phase_name, streams in (
                    ("off", turn_streams),
                    ("auto", auto_streams),
                ):
                    hashes = {
                        sample["sha256"]
                        for passes in streams
                        for sample in passes
                        if sample["turn"] == turn_index
                    }
                    phase_hashes[phase_name] = hashes
                    # Within-phase determinism: every pass of this phase
                    # produced the identical hash for this turn.
                    if len(hashes) != 1:
                        within_phase_deterministic[conversation_id] = False
                flags.append(
                    bool(phase_hashes["off"])
                    and bool(phase_hashes["auto"])
                    and phase_hashes["off"] == phase_hashes["auto"]
                )
                # Eager lists: a generator closed over turn_index would
                # evaluate every aggregate with the loop's final value.
                checker_flags.append(
                    [
                        all(
                            sample["checker_pass"]
                            for passes in streams
                            for sample in passes
                            if sample["turn"] == turn_index
                        )
                        for streams in (turn_streams, auto_streams)
                    ]
                )
            exact_by_turn[conversation_id] = flags
            # ``checker_flags`` is turn-major ([off, auto] per turn);
            # ``zip(*...)`` transposes to phase-major columns, so each
            # ``all(pair)`` is one phase's all-turns verdict. The gate only
            # needs "checkers pass in both phases on every turn", which this
            # preserves — a failed turn fails its phase column.
            checker_pass_both[conversation_id] = [
                all(pair) for pair in zip(*checker_flags)
            ]
            within_phase_deterministic.setdefault(conversation_id, True)
        result["exact_by_conversation"] = {
            conversation_id: all(flags)
            for conversation_id, flags in exact_by_turn.items()
        }
        result["exact_conversations"] = sum(result["exact_by_conversation"].values())
        result["total_conversations"] = len(result["exact_by_conversation"])
        result["exact_turns"] = sum(sum(flags) for flags in exact_by_turn.values())
        result["total_turns"] = sum(len(flags) for flags in exact_by_turn.values())

        # Hard gate: every pass inside a phase is identical.
        result["within_phase_deterministic"] = all(
            within_phase_deterministic.values()
        ) and bool(within_phase_deterministic)

        # Hard gate: semantic checkers pass in both phases on every turn.
        result["checkers_pass"] = all(
            all(flags) for flags in checker_pass_both.values()
        ) and bool(checker_pass_both)

        # Hard gate: the candidate phase must store snapshots and serve warm
        # resumes; the baseline must do neither. Without this gate a silently
        # disabled feature would qualify as an off-vs-off vacuous pass.
        def media_stats(phase: str) -> dict[str, Any]:
            return result["phases"][phase].get("media_prefix_cache") or {}

        off_media = media_stats("off")
        auto_media = media_stats("auto")
        result["media_engaged"] = (
            auto_media.get("stores", 0) > 0
            and auto_media.get("hits", 0) > 0
            and off_media.get("stores", 0) == 0
            and off_media.get("hits", 0) == 0
        )

        # Turn-level deltas (auto vs off), warm turn 2+ being where the
        # resume win lands.
        result["summary_change_pct"] = {}
        for conversation_id, summary in result["phases"]["auto"]["summary"].items():
            off_summary = result["phases"]["off"]["summary"][conversation_id]
            per_turn = []
            for turn_index, turn_ttft in enumerate(summary["median_ttft_s"]):
                baseline_ttft = off_summary["median_ttft_s"][turn_index]
                baseline_elapsed = off_summary["median_elapsed_s"][turn_index]
                per_turn.append(
                    {
                        "ttft": 100.0 * (turn_ttft / baseline_ttft - 1.0)
                        if baseline_ttft
                        else 0.0,
                        "elapsed": 100.0
                        * (
                            summary["median_elapsed_s"][turn_index] / baseline_elapsed
                            - 1.0
                        )
                        if baseline_elapsed
                        else 0.0,
                    }
                )
            result["summary_change_pct"][conversation_id] = per_turn

        # Hard gate: a resume sample must not be slower than the cold
        # baseline by more than the stall margin. Gated per measured
        # sample — not per-slot median — so a slot where one pass resumed
        # slowly and another resumed fast cannot average its way under the
        # margin, and a slot whose cached_tokens median is positive only
        # because a minority of passes resumed cannot drag cold samples
        # into the comparison. Each resumed sample (``cached_tokens > 0``)
        # is compared against the baseline (off-phase) median TTFT for its
        # (conversation, turn). The storing turn's bounded snapshot cost is
        # documented in the design note and deliberately excluded here.
        regressions = {}
        for conversation_id, auto_passes in result["phases"]["auto"][
            "per_conversation"
        ].items():
            off_passes = result["phases"]["off"]["per_conversation"][conversation_id]
            baseline_median_by_turn = {
                turn_index: _median(list(passes), "ttft_s")
                for turn_index, passes in enumerate(zip(*off_passes))
            }
            flagged = _resume_regressions(
                auto_passes, baseline_median_by_turn, WARM_REGRESSION_MARGIN
            )
            if flagged:
                regressions[conversation_id] = flagged
        result["warm_turn_regressions"] = regressions

        # Hard gate: the turns that are SUPPOSED to resume must resume.
        # The engagement gate demands only one aggregate hit, so a feature
        # that resumed a single lucky turn while every other follow-up
        # missed silently would still qualify. Turn 1 is the documented
        # store turn (turn-0 prompts fall below the boundary min-tokens
        # floor, so turn 1 is where the snapshot is taken — it pays the
        # bounded snapshot cost and does not resume); the expected resume
        # slots are the follow-up turns from turn 2 on. Gated per measured
        # sample — a slot whose cached_tokens median is positive because
        # only one of two passes resumed is still a miss for the cold
        # pass. Every miss is reported; below the declared minimum hit
        # rate the run fails.
        expected_samples = 0
        resumed_samples = 0
        resume_misses: dict[str, list[dict[str, Any]]] = {}
        for conversation_id, auto_passes in result["phases"]["auto"][
            "per_conversation"
        ].items():
            slot_expected, slot_resumed, misses = _resume_gate_samples(auto_passes)
            expected_samples += slot_expected
            resumed_samples += slot_resumed
            if misses:
                resume_misses[conversation_id] = misses
        result["resume_misses"] = resume_misses
        result["resume_coverage"] = (
            resumed_samples / expected_samples if expected_samples else 0.0
        )
        # Declared minimum hit rate: 75% of expected resume samples. The
        # qualified run measured 85% (17/20 turn-2 samples; the misses are
        # strict-prefix template mismatches reported as clean misses).
        result["resume_coverage_ok"] = (
            expected_samples > 0 and result["resume_coverage"] >= RESUME_COVERAGE_FLOOR
        )
    finally:
        pass

    if args.output:
        write_bench_json(args.output, result, Path(__file__))
    payload = {
        key: result[key]
        for key in (
            "exact_by_conversation",
            "exact_conversations",
            "total_conversations",
            "exact_turns",
            "total_turns",
            "within_phase_deterministic",
            "checkers_pass",
            "media_engaged",
            "resume_coverage",
            "resume_misses",
            "warm_turn_regressions",
            "summary_change_pct",
            "phases",
        )
        if key in result
    }
    if args.summary_only:
        payload.pop("phases", None)
    try:
        from scripts.bench_metadata import format_bench_json
    except ImportError:
        from bench_metadata import format_bench_json

    print(format_bench_json(payload, Path(__file__)))
    if not result.get("within_phase_deterministic", False) or not result.get(
        "checkers_pass", False
    ):
        raise SystemExit(1)
    if result.get("warm_turn_regressions"):
        raise SystemExit(3)
    if not result.get("resume_coverage_ok", False):
        raise SystemExit(4)
    if not result.get("media_engaged", False):
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(_main())
