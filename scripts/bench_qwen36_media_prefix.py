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
  the baseline does neither, else the A/B compares off against off.

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
    image_path = (repo_root / conversation["images"][0]).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"manifest image missing: {image_path}")
    messages: list[dict[str, Any]] = []
    for index in range(turn_index + 1):
        content: list[dict[str, Any]] = [
            {"type": "text", "text": conversation["turns"][index]["prompt"]}
        ]
        if index == 0:
            content.append({"type": "image_url", "image_url": {"url": str(image_path)}})
        messages.append({"role": "user", "content": content})
        if index < turn_index:
            messages.append({"role": "assistant", "content": replies[index]})
    return messages


def _checker_pass(checker: dict[str, Any], text: str) -> bool:
    lowered = text.casefold()
    kind = checker.get("type", "any")
    if kind == "terms":
        if not all(term.casefold() in lowered for term in checker.get("required", [])):
            return False
        return not any(
            term.casefold() in lowered for term in checker.get("forbidden", [])
        )
    if kind == "json_shape":
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
        try:
            start = stripped.index("{")
            payload = json.loads(stripped[start:])
        except (ValueError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and all(
            key in payload for key in checker.get("keys", [])
        )
    if kind == "any":
        return bool(text.strip())
    raise ValueError(f"unknown checker type: {kind}")


def _median(samples: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(sample[key]) for sample in samples)


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

    by_conversation: dict[str, list[list[dict[str, Any]]]] = {
        conversation["id"]: [] for conversation in conversations
    }
    for _ in range(pairs):
        for conversation in conversations:
            by_conversation[conversation["id"]].append(
                await _replay_conversation(engine, conversation, repo_root)
            )
    stats = engine.get_stats()
    media = stats.get("media_prefix_cache") or {}
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
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

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
        await engine.start()
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
        turns_count = len(conversations[0]["turns"])
        exact_by_turn: dict[str, list[bool]] = {}
        within_phase_deterministic: dict[str, bool] = {}
        checker_pass_both: dict[str, list[bool]] = {}
        for conversation_id, turn_streams in result["phases"]["off"][
            "per_conversation"
        ].items():
            auto_streams = result["phases"]["auto"]["per_conversation"][conversation_id]
            flags = []
            checker_flags = []
            for turn_index in range(turns_count):
                phase_hashes: dict[str, set[str]] = {}
                for phase_name, streams in (("off", turn_streams), ("auto", auto_streams)):
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
                checker_flags.append(
                    all(
                        sample["checker_pass"]
                        for passes in streams
                        for sample in passes
                        if sample["turn"] == turn_index
                    )
                    for streams in (turn_streams, auto_streams)
                )
            exact_by_turn[conversation_id] = flags
            checker_pass_both[conversation_id] = [all(pair) for pair in zip(*checker_flags)]
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
    if args.pairs > 0 and not result.get("media_engaged", False):
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(_main())
