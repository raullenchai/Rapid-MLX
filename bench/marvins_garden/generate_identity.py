# SPDX-License-Identifier: Apache-2.0
"""Identity-mix samples for Marvin's Garden v1.1 (anti-forgetting anchors).

The v1 adapter collapses generation mode on out-of-distribution prompts
(see iq_probe results): letter-only supervision gives the model no anchor
for ordinary prose answers. This generator emits ~200 deterministic,
factually correct instruction→prose samples mixed into SFT so the adapter
keeps its general generation mode while learning the decision task.

All answers are curated facts or deterministic transformations — no
placeholder text. Same chat wire format as to_chat_sft.py output.

Usage::

    python bench/marvins_garden/generate_identity.py [--n 192] [--seed 20260919]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("idempotent (distributed systems)", "An operation is idempotent when running it more than once has the same effect as running it once — retries become safe."),
    ("CAP theorem", "A distributed store can guarantee at most two of consistency, availability, and partition tolerance; under a partition you must pick consistency or availability."),
    ("cache eviction (LRU)", "Least-recently-used eviction discards the entry whose last access is oldest, approximating recency of future reuse."),
    ("exponential backoff", "A retry policy that doubles the wait after each failed attempt, spreading load and avoiding thundering herds."),
    ("database index", "An auxiliary structure that maps key values to row locations so lookups avoid scanning the whole table."),
    ("rate limiting (token bucket)", "A limiter that refills tokens at a fixed rate and spends one per request, allowing short bursts while capping the average rate."),
    ("connection pooling", "Reusing a set of open connections across requests so the per-request handshake cost is avoided."),
    ("atomic commit (two-phase)", "A coordinator asks all participants to prepare, then commits only if every participant acknowledged the prepare phase."),
    ("memoization", "Caching a function's results keyed by its inputs so repeated calls with the same inputs skip recomputation."),
    ("graceful degradation", "Designing a system so that when a component fails, it sheds optional features and keeps serving the core ones."),
)

CAPITALS: dict[str, str] = {
    "Japan": "Tokyo", "Kenya": "Nairobi", "Canada": "Ottawa", "Brazil": "Brasília",
    "Norway": "Oslo", "Vietnam": "Hanoi", "Morocco": "Rabat", "Chile": "Santiago",
    "Portugal": "Lisbon", "Indonesia": "Jakarta", "Egypt": "Cairo", "Turkey": "Ankara",
    "Australia": "Canberra", "Switzerland": "Bern", "Peru": "Lima", "Ghana": "Accra",
}

POLITE_REWRITES: tuple[tuple[str, str], ...] = (
    ("Send me the report now.", "Could you please send me the report when you have a moment?"),
    ("This is wrong, fix it.", "I think there may be an issue here — could you take another look and help fix it?"),
    ("Why is this broken again?", "Could you help me understand what might be causing this to break again?"),
    ("Call me back ASAP.", "Would you mind giving me a call back at your earliest convenience?"),
    ("You missed the deadline.", "It looks like the deadline slipped — is there anything I can do to help get things back on track?"),
)

TRANSFORMS: tuple[tuple[str, str, str], ...] = (
    ("Formal", "we're gonna ship the update on Friday", "We plan to ship the update on Friday."),
    ("Concise", "Due to the fact that the server was down, we were not able to finish the deployment, which was scheduled for Tuesday.", "The server outage prevented Tuesday's deployment."),
    ("Bullet", "The plan has three parts: migrate the schema, backfill the cache, and flip the flag.", "1. Migrate the schema\n2. Backfill the cache\n3. Flip the flag."),
)

TIPS: tuple[tuple[str, str], ...] = (
    ("writing clear commit messages", "Lead with the change, not the mechanics: say what behavior changes and why, then the how in the body."),
    ("debugging flaky tests", "Run the failing test in a loop locally and diff the state at the failure boundary before theorizing."),
    ("onboarding to a new codebase", "Trace one real request end-to-end through the code before reading any subsystem in isolation."),
    ("code review comments", "Review for correctness and clarity first; style is what the formatter is for."),
    ("capacity planning", "Measure the peak, not the average, and leave headroom for retry storms."),
)

TRANSLATIONS: dict[str, str] = {
    "Good morning, the meeting starts at nine.": "Bonjour, la réunion commence à neuf heures.",
    "Thank you for your help.": "Merci pour votre aide.",
    "The train leaves in ten minutes.": "Le train part dans dix minutes.",
    "I will send the file this afternoon.": "J'enverrai le fichier cet après-midi.",
}


DEFINITION_FRAMES: tuple[str, ...] = (
    "Explain in one sentence what \"{x}\" means.",
    "Define {x} in one sentence.",
    "In systems design, what is {x}? One sentence.",
)

CAPITAL_FRAMES: tuple[str, ...] = (
    "What is the capital of {c}? Answer in one short sentence.",
    "One-sentence answer: which city is the capital of {c}?",
)

ARITHMETIC_FRAMES: tuple[str, ...] = (
    "Compute {a} × {b} and give just the result.",
    "What is {a} × {b}? Reply with the number only.",
)

TIP_FRAMES: tuple[str, ...] = (
    "Give one practical tip for {t}.",
    "What is your single best piece of advice for {t}?",
)


def build_samples(rng: random.Random) -> list[dict]:
    samples: list[dict] = []

    def add(user: str, assistant: str) -> None:
        samples.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        })

    for term, definition in DEFINITIONS:
        for frame in DEFINITION_FRAMES:
            add(frame.format(x=term), definition)
    for country, capital in CAPITALS.items():
        for frame in CAPITAL_FRAMES:
            answer = f"The capital of {country} is {capital}."
            add(frame.format(c=country), answer)
    for rude, polite in POLITE_REWRITES:
        add(f"Rewrite this request so it is polite: \"{rude}\"", polite)
    for style, source, target in TRANSFORMS:
        add(f"{style} rewrite of the following:\n\n{source}", target)
    for a, b in ((12, 7), (9, 9), (13, 4), (25, 8), (36, 11), (48, 12), (7, 15), (64, 3)):
        for frame in ARITHMETIC_FRAMES:
            add(frame.format(a=a, b=b), str(a * b))
    for topic, tip in TIPS:
        for frame in TIP_FRAMES:
            add(frame.format(t=topic), tip)
    for english, french in TRANSLATIONS.items():
        add(f"Translate to French: \"{english}\"", french)

    rng.shuffle(samples)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--n", type=int, default=192, help="take the first N shuffled samples")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "identity_train.jsonl")
    args = parser.parse_args(argv)

    rng = random.Random(f"identity:{args.seed}")
    samples = build_samples(rng)[: args.n]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} identity samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
