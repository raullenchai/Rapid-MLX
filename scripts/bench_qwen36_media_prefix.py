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
  turn 1 is the documented store turn) must actually resume in at least
  75% of their measured samples (exit 4 otherwise), with every miss
  reported;
* **no resume-turn regression** — every measured sample that actually
  served a media resume (auto ``cached_tokens > 0``) must not exceed the
  baseline median TTFT for its (conversation, turn) by more than a 15%
  stall margin (exit 3 otherwise). Samples that store (or miss) pay the
  documented bounded snapshot cost and are excluded.

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
    image_paths = [(repo_root / image).resolve() for image in conversation["images"]]
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"manifest image missing: {image_path}")
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

    Slash/hyphen runs are broken into separate tokens first so compound
    forms ("skip"/"next", "dark-themed") tokenize the same on the term and
    text sides.
    """
    tokens = []
    for raw in text.casefold().translate(_INNER_BREAKS).split():
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
            window = text_tokens[max(0, start - 3) : start]
            if any(token in _NEGATORS for token in window):
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
    lines = text.count("\n") + 1
    if min_words and words < min_words:
        return False
    if max_words and words > max_words:
        return False
    if min_lines and lines < min_lines:
        return False
    return not (max_lines and lines > max_lines)


def _checker_pass(checker: dict[str, Any], text: str) -> bool:
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
        if not _structural_pass(checker, text):
            return False
        return not any(
            _term_matches(text_tokens, term, guard_negation=False)
            for term in checker.get("forbidden", [])
        )
    if kind == "json_shape":
        # "Output JSON only": the stripped response itself must parse as one
        # JSON object — searching for the first ``{`` would let prose
        # wrapped around an all-null payload pass ("JSON only" with
        # entirely wrong field values).
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
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
            field_tokens = _tokens(json.dumps(payload.get(key), default=str))
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
        required_any = checker.get("required_any", [])
        if required_any and not any(
            all(
                _term_matches(text_tokens, term, guard_negation=True)
                for term in alternative
            )
            for alternative in required_any
        ):
            return False
        return True
    raise ValueError(f"unknown checker type: {kind}")


def _median(samples: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(sample[key]) for sample in samples)


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
    the measured pass and turn whose ``cached_tokens`` was 0. Gating per
    measured sample — not per-slot median — so a slot where only one pass
    resumed cannot hide behind the other pass's cold miss.
    """
    expected = 0
    resumed = 0
    misses: list[dict[str, Any]] = []
    for pass_index, passes in enumerate(auto_passes):
        for sample in passes:
            if sample["turn"] < 2:
                continue
            expected += 1
            if int(sample["cached_tokens"]) > 0:
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

    Every measured sample that actually resumed (``cached_tokens > 0``) is
    compared against the baseline median TTFT for its turn; medians never
    mix cold and resumed executions. Turn 1 (the store turn) is excluded —
    its bounded snapshot cost is documented in the design note.
    """
    flagged: list[dict[str, Any]] = []
    for pass_index, passes in enumerate(auto_passes):
        for sample in passes:
            turn = sample["turn"]
            if turn < 1 or int(sample["cached_tokens"]) <= 0:
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
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "checker_pass": _checker_pass(turn.get("checker", {}), text),
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

    # Engagement counters must come from the MEASURED passes only: warmup
    # would otherwise contribute stores/hits and let the engagement (and
    # warm-regression) gates pass green while every measured pass missed.
    warmup_stats = engine.get_stats().get("media_prefix_cache") or {}
    by_conversation: dict[str, list[list[dict[str, Any]]]] = {
        conversation["id"]: [] for conversation in conversations
    }
    for _ in range(pairs):
        for conversation in conversations:
            by_conversation[conversation["id"]].append(
                await _replay_conversation(engine, conversation, repo_root)
            )
    measured_stats = engine.get_stats().get("media_prefix_cache") or {}
    media = {
        # Counters delta over the measured passes; gauges (entries/bytes/
        # budget) report the end-of-phase state as-is.
        **{
            key: int(measured_stats.get(key, 0) or 0)
            - int(warmup_stats.get(key, 0) or 0)
            for key in ("hits", "misses", "stores", "budget_evictions")
        },
        **{
            key: measured_stats[key]
            for key in ("entries", "bytes", "budget_bytes")
            if key in measured_stats
        },
    }
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
    repo_root = (
        args.manifest.resolve().parent / manifest.get("images_root", ".")
    ).resolve()
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
        exact_by_turn: dict[str, list[bool]] = {}
        within_phase_deterministic: dict[str, bool] = {}
        checker_pass_both: dict[str, list[bool]] = {}
        for conversation_id, turn_streams in result["phases"]["off"][
            "per_conversation"
        ].items():
            auto_streams = result["phases"]["auto"]["per_conversation"][conversation_id]
            # Conversation's own turn count — manifests need not be uniform,
            # and an all()-over-empty would vacuously pass the gates.
            turn_count = (
                min(len(passes) for passes in turn_streams + auto_streams if passes)
                if (turn_streams or auto_streams)
                else 0
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
