#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, model-agnostic semantic grader for tool-result grounding.

Issue #2347 ("eval: semantic grader for tool-result grounding") asks for an
upgrade to ``evals/run_eval.py``'s opt-in ``verify_final_text`` final-answer
check: instead of only requiring a bounded substring match against a few
scenario-declared terms, verify that a free-text final answer AFFIRMATIVELY and
NON-CONTRADICTORILY reports the supplied tool result's salient FACTS (e.g. the
condition / temperature / humidity returned by a ``weather`` tool), tolerating
natural paraphrase.

This module is the deterministic core of that upgrade. It is deliberately:

  * pure and self-contained -- stdlib only (``re``, ``unicodedata``), with NO
    model dependency. An optional model judge may layer extra diagnostics on
    top downstream, but the verdict here never requires one;
  * data-driven -- unit/alias resolution tables and the bounded deny/hedge
    marker list are constants at the top of the file (a single source of
    truth), not scattered inline;
  * typed -- facts are normalized into small dataclasses
    (``StringFact`` / ``NumberFact`` / ``RelationFact``) rather than matched by
    loose regex over arbitrary model text;
  * defensive about provenance -- any embedded prompt-like string in a tool
    result or answer is treated as DATA, never as instructions. We never
    exec/interpret fact content, and we cap fact and response sizes so a
    garbage long input cannot dominate the verdict.

Two INDEPENDENT properties are graded per fact:

  (a) affirmative coverage -- the salient fact appears, tolerating paraphrase
      via normalization / aliases / °C-to-°F unit conversion;
  (b) absence of contradiction -- the answer must not negate, hedge-deny, or
      contradict any supplied fact (bounded negation/deny detection applied to
      the locally salient phrase, not a loose scan over the whole text).

Routing / tool-call success is a SEPARATE score owned by ``run_tool_calling_suite``
and is never conflated with grounding coverage.

Community / eval-only surface: any per-scenario ``expected_facts`` block in
``evals/prompts/tool_calling.json`` is fed through ``fact_from_dict`` +
``grade_answer`` here. No product / engine / model code is involved.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field

# Stable machine-readable score-format identifier (versioned header). Bump with
# every breaking change to the grounding report's shape / semantics.
SCORE_FORMAT = "tool_result_grounding/v1"

# ---------------------------------------------------------------------------
# Data constants -- single source of truth (do not scatter these inline).
# ---------------------------------------------------------------------------

# Bounded deny / hedge markers. These are applied ONLY within a locally salient
# window around a fact's anchor phrase (never as a loose scan over the whole
# answer), which keeps false positives low while catching explicit negation,
# hedged denial, and "tool unavailable" style refusals. Each marker here is the
# casefolded form; short single-word markers are matched on word boundaries,
# longer phrases as (bounded) substrings.
DENY_MARKERS = (
    "not",
    "no",
    "unavailable",
    "unable",
    "can't",
    "cannot",
    "couldn't",
    "could not",
    "failed",
    "no data",
    "no access",
    "don't have",
    "do not have",
    "doesn't have",
    "does not have",
    "can't say",
    "cannot say",
    "can't determine",
    "cannot determine",
    "can't provide",
    "cannot provide",
    "isn't",
    "is not",
    "aren't",
    "are not",
    "wasn't",
    "was not",
    "not available",
    "no result",
    "unable to",
    "refused",
    "denied",
)

# Canonical temperature units. ``_UNIT_TO_CANONICAL`` resolves every surface
# token (after NFC/casefold) to a canonical unit name used by NumberFact.
_C, _F, _PERCENT = "c", "f", "%"

# Map surface unit tokens -> canonical unit. The temperature equivalents are
# checked against the original text (which may be c/F-cased), so we key on the
# casefolded token.
_UNIT_TO_CANONICAL = {
    "°c": _C,
    "c": _C,
    "celsius": _C,
    "degrees c": _C,
    "degrees celsius": _C,
    "deg c": _C,
    "deg celsius": _C,
    "°f": _F,
    "f": _F,
    "fahrenheit": _F,
    "degrees f": _F,
    "degrees fahrenheit": _F,
    "deg f": _F,
    "deg fahrenheit": _F,
    "%": _PERCENT,
    "percent": _PERCENT,
}

# How wide (in normalized characters) the locally salient window is around a
# fact's anchor phrase. Deny markers found OUTSIDE every salient window do not
# count as a contradiction, which is what keeps negation detection bounded.
SALIENT_WINDOW = 40

# Safety / hygiene caps. Tool output and answers are DATA; capping blunts any
# oversized or injection-ish input before it can influence the verdict.
DEFAULT_MAX_ANSWER_LEN = 2000
DEFAULT_MAX_FACT_LEN = 200
MAX_FACTS = 50

# Derived compiled regexes (token-level, bounded -- not loose text->regex).
_UNIT_RE = re.compile(
    r"(?P<num>-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>celsius|fahrenheit|degrees?\s+c(?:elsius)?|degrees?\s+f(?:ahrenheit)?"
    r"|deg\s*c(?:elsius)?|deg\s*f(?:ahrenheit)?|°c|°f|percent|[%]|c\b|f\b)?"
)


def _is_single_word(marker: str) -> bool:
    """True for a marker made of one word (matched with word boundaries)."""
    return " " not in marker and marker.isalnum()


# Each deny marker precompiled: single-word markers get word boundaries so
# "notebook"/"nothing" don't trip "not", while multiword phrases (which are
# long enough that boundaries matter less) use a plain escaped substring.
_MARKER_RES = tuple(
    re.compile(rf"\b{re.escape(m)}\b")
    if _is_single_word(m)
    else re.compile(re.escape(m))
    for m in DENY_MARKERS
)

# Separators / units that are NOT temperature units, so a bare number isn't
# misread: the unit resolver only counts c/f/%/percent. This set is for
# readability of the normalization helpers; resolution itself is via
# ``_UNIT_TO_CANONICAL``.
_KNOWN_UNITS = frozenset(_UNIT_TO_CANONICAL)


def _norm(text: str) -> str:
    """NFC+casefold a string for reproducible, Unicode-insensitive matching.

    Handles full-width variants and degree-sign look-alikes (``℃``/``℉``
    collapse to ``°c``/``°f`` under NFKC) so "°C" / "degrees celsius" / plain
    text compare cleanly.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).casefold().strip()


def _resolve_unit(token: str) -> str | None:
    """Return the canonical unit for a surface unit token, else None."""
    if not token:
        return None
    return _UNIT_TO_CANONICAL.get(_norm(token).strip())


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def _to_unit(value: float, unit: str | None, target: str) -> float | None:
    """Convert ``value`` expressed in ``unit`` into ``target``.

    ``unit`` is a canonical unit ('c', 'f', '%'). Bare (unitless) values are
    assumed to already be in ``target``, matching how a plain "21" is read as
    the fact's declared unit (e.g. 21 °C). Returns ``None`` when an EXPLICIT
    unit is incompatible with ``target`` (e.g. '%' vs a temperature target) so
    a value can never be (mis)read as the fact's unit just because the raw
    number happens to land inside tolerance -- "humidity 18%" must not satisfy
    a temperature=18°C fact.
    """
    if unit is None or unit == target:
        return value
    if unit == _C and target == _F:
        return _celsius_to_fahrenheit(value)
    if unit == _F and target == _C:
        return _fahrenheit_to_celsius(value)
    # Explicit but incompatible / unknown unit -- cannot honestly convert.
    return None


# ---------------------------------------------------------------------------
# Normalized fact model.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StringFact:
    """A ``string``/``enum`` fact: a normalized value with a set of aliases.

    Coverage passes when any alias (or the canonical value) appears in the
    normalized answer. Anchors are the aliases plus the key, used for locating
    the salient window during contradiction detection.
    """

    type: str = "string"
    key: str = ""
    value: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def normalized_value(self) -> str:
        return _norm(self.value)

    @property
    def search_terms(self) -> tuple[str, ...]:
        """Normalized positive phrases that affirmatively report the fact.

        Only the VALUE and its aliases are treated as affirmative evidence --
        never the bare key, so a malformed/empty-value fact (e.g. a fallback
        created for an unparseable input) can't pass on the key alone.
        """
        terms = set()
        if self.value:
            terms.add(_norm(self.value))
        terms.update(_norm(a) for a in self.aliases)
        return tuple(sorted(t for t in terms if t))

    @property
    def contradiction_terms(self) -> tuple[str, ...]:
        """Anchors used to LOCATE the fact for contradiction detection.

        Unlike ``search_terms`` (value+aliases), this includes the bare key so a
        denial of the fact is caught even when only its key is named e.g. "the
        condition is unavailable". The key is safe here because it only widens
        the denial-window, never the affirmative match.
        """
        terms = set()
        if self.key:
            terms.add(_norm(self.key))
        if self.value:
            terms.add(_norm(self.value))
        terms.update(_norm(a) for a in self.aliases)
        return tuple(sorted(t for t in terms if t))


@dataclass(frozen=True)
class NumberFact:
    """A numeric fact with a unit and tolerance, e.g. temperature 21 ±1 °C.

    Unit aliases resolve via ``_UNIT_TO_CANONICAL``, and a °C-declared fact is
    satisfied by a correct °F answer and vice versa via ``_to_unit``.
    """

    type: str = "number"
    key: str = ""
    value: float = 0.0
    unit: str = _C
    tolerance: float = 0.0
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def anchor_terms(self) -> tuple[str, ...]:
        """Entity labels used to locate the salient window (key + aliases)."""
        terms = {_norm(self.key)}
        terms.update(_norm(a) for a in self.aliases)
        return tuple(sorted(t for t in terms if t))


@dataclass(frozen=True)
class RelationFact:
    """An entity -> value fact, e.g. humidity -> 62%.

    Coverage requires the key entity to appear AND a compatible value (number
    with the fact's unit, or a bare number near the key) within its tolerance.
    """

    type: str = "relation"
    key: str = ""
    value: float = 0.0
    unit: str = _PERCENT
    tolerance: float = 0.0
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def normalized_key(self) -> str:
        return _norm(self.key)

    @property
    def anchor_terms(self) -> tuple[str, ...]:
        terms = {self.normalized_key}
        terms.update(_norm(a) for a in self.aliases)
        return tuple(sorted(t for t in terms if t))


Fact = StringFact | NumberFact | RelationFact


def _clamp_fact(fact: Fact, max_fact_len: int) -> tuple[Fact, bool]:
    """Rebuild ``fact`` with every string field capped to ``max_fact_len``.

    Returns ``(clamped_fact, was_clamped)``. Values and aliases are DATA
    (potentially attacker- or tool-controlled), so an oversized / injection-ish
    fact is clamped rather than allowed to dominate the verdict. ``was_clamped``
    lets the caller FAIL CLOSED: grading the truncated prefix as if it were the
    required fact would let an answer containing only the first 200 chars pass,
    so an oversized fact must invalidate the run (see ``grade_answer``).
    """
    if max_fact_len <= 0:
        return fact, False

    def _cut(value: str) -> str:
        return value[:max_fact_len]

    def _changed(value: str) -> bool:
        return len(value) > max_fact_len

    if isinstance(fact, StringFact):
        clamped = (
            _changed(fact.key)
            or _changed(fact.value)
            or any(_changed(a) for a in fact.aliases)
        )
        return (
            StringFact(
                key=_cut(fact.key),
                value=_cut(fact.value),
                aliases=tuple(_cut(a) for a in fact.aliases),
            ),
            clamped,
        )
    if isinstance(fact, NumberFact):
        clamped = _changed(fact.key) or any(_changed(a) for a in fact.aliases)
        return (
            NumberFact(
                key=_cut(fact.key),
                value=fact.value,
                unit=fact.unit,
                tolerance=fact.tolerance,
                aliases=tuple(_cut(a) for a in fact.aliases),
            ),
            clamped,
        )
    if isinstance(fact, RelationFact):
        clamped = _changed(fact.key) or any(_changed(a) for a in fact.aliases)
        return (
            RelationFact(
                key=_cut(fact.key),
                value=fact.value,
                unit=fact.unit,
                tolerance=fact.tolerance,
                aliases=tuple(_cut(a) for a in fact.aliases),
            ),
            clamped,
        )
    return fact, False


def fact_from_dict(d: dict) -> Fact:
    """Convert a scenario ``expected_facts`` entry into a normalized Fact.

    Raises ``ValueError`` on an unknown fact type or a missing required field,
    so a misconfigured scenario fails fast rather than grading silently.
    """
    if not isinstance(d, dict):
        raise ValueError(f"expected_facts entry must be a dict, got {type(d).__name__}")
    ftype = d.get("type")
    if ftype not in {"string", "number", "relation"}:
        raise ValueError(f"unknown expected_facts type: {ftype!r}")
    key = d.get("key")
    if not key:
        raise ValueError(f"{ftype} fact requires a non-empty 'key'")
    # A missing 'value' must fail fast (not default to 0.0 and silently match an
    # incidental zero): for string/enum the value is the affirmative content,
    # for number/relation it is the quantity under test.
    if "value" not in d or d.get("value") is None:
        raise ValueError(f"{ftype} fact '{key}' requires a 'value'")

    def _clean_aliases(aliases: object, fact_key: str) -> tuple[str, ...]:
        # Must be a real sequence of strings -- a bare string would iterate into
        # single characters and let an incidental letter satisfy the fact.
        if aliases is None:
            return ()
        if isinstance(aliases, (str, bytes)) or not isinstance(aliases, (list, tuple)):
            raise ValueError(
                f"fact '{fact_key}' 'aliases' must be a list/tuple of strings"
            )
        out = tuple(str(a) for a in aliases)
        if any(not a for a in out):
            raise ValueError(f"fact '{fact_key}' 'aliases' must be non-empty strings")
        return out

    aliases = _clean_aliases(d.get("aliases"), str(key))
    if ftype == "string":
        return StringFact(
            key=str(key),
            value=str(d["value"]),
            aliases=aliases,
        )
    raw_value = d["value"]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"number/relation fact '{key}' requires a numeric 'value'")
    if not math.isfinite(float(raw_value)):
        raise ValueError(f"number/relation fact '{key}' requires a finite 'value'")
    tolerance = float(d.get("tolerance", 0.0))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            f"number/relation fact '{key}' requires a finite, non-negative tolerance"
        )

    def _resolve_configured_unit(supplied: object, default: str, fact_key: str) -> str:
        # An EXPLICIT but unresolvable unit (e.g. a typo "unit":"k") must fail
        # fast rather than silently fall back to a plausible default unit.
        if supplied is None or supplied == "":
            return default
        resolved = _resolve_unit(str(supplied))
        if resolved is None:
            raise ValueError(f"fact '{fact_key}' has an unrecognized unit {supplied!r}")
        return resolved

    if ftype == "number":
        unit = _resolve_configured_unit(d.get("unit"), _C, str(key))
        return NumberFact(
            key=str(key),
            value=float(raw_value),
            unit=unit,
            tolerance=tolerance,
            aliases=aliases,
        )
    unit = _resolve_configured_unit(d.get("unit"), _PERCENT, str(key))
    return RelationFact(
        key=str(key),
        value=float(raw_value),
        unit=unit,
        tolerance=tolerance,
        aliases=aliases,
    )


# ---------------------------------------------------------------------------
# Grading helpers (token / alias / unit based, not loose full-text regex).
# ---------------------------------------------------------------------------


def _numbered_in(text: str) -> list[tuple[float, str | None, int]]:
    """Yield ``(value, canonical_unit_or_None, start)`` candidates in ``text``.

    ``text`` is the NFC+casefolded answer. Values with a resolvable unit yield
    that unit; values without a unit yield ``None`` (a bare number). ``start``
    is the candidate's actual character offset, so callers can check whether
    THIS occurrence (not the first textually-identical one) sits inside an
    anchor window. Typed, bounded extraction -- it never grants meaning to
    free-text.
    """
    out: list[tuple[float, str | None, int]] = []
    for match in _UNIT_RE.finditer(text):
        num = match.group("num")
        token = match.group("unit")
        if num is None:
            continue
        value = float(num)
        start = match.start()
        if not token:
            # No recognized unit captured. If a unit-LIKE token immediately
            # follows (e.g. "55 km/h", "8 mph"), this is a unit-qualified value
            # we do not model -- it must NOT masquerade as a bare number and
            # satisfy a fact of a different unit (e.g. humidity 55%).
            if _adjacent_unit_suffix(text, match.end()):
                continue
            out.append((value, None, start))
            continue
        unit = _resolve_unit(token.strip())
        if unit is None:
            continue
        out.append((value, unit, start))
    return out


# Tokens that look like a measurement / currency / count unit (but aren't in
# ``_UNIT_TO_CANONICAL``). Used to reject unit-qualified numerics we don't model,
# so they don't leak in as bare numbers: "21 dollars" or "55 points" must not
# satisfy a temperature/humidity fact whose value merely coincides. Conservative:
# slash-form compounds (km/h), a small list of common physical units, and common
# currency/count nouns. `degrees` is deliberately NOT included -- it is the
# fact's OWN temperature unit surface, not a foreign unit (see inline note).
# Ordinary prose words ("outside", "and", "rising") are deliberately NOT
# included.
# Multi-letter/compound unit tokens are matched STANDALONE (no required base
# prefix), so "55 mph" / "8 km/h" are recognized. Single-letter units (m, g, l,
# h, s) are deliberately EXCLUDED from the standalone list -- they would
# over-reject ordinary prose ("5 m" could be "5 minutes", "8 l" a digit+letter).
_ADJACENT_UNIT_RE = re.compile(
    r"\s{0,3}(?:(?:[a-z]{1,5}(?:/[a-z]{1,5})+)"
    r"|(?:mph|kph|kmh|knots|nmi|sq|sqm|in|ft|yd|mi|px|em|rem|pt|"
    r"mm|cm|km|kg|ml|oz|lb|bar|mbar|hpa|pa|sec|min|hr|"
    r"dollar|dollars|usd|cad|eur|gbp|yen|yuan|rupee|rupees|cents|"
    r"points|point|grade|marks))"
    # NOTE: `degrees` is deliberately NOT in the reject list. It is the most
    # common English surface for the fact's OWN temperature unit ("temperature
    # is 21 degrees" is a natural affirmative report of temperature=21°C), so
    # an adjacent `degrees` must not strip a legitimate anchored value. Only
    # semantically FOREIGN unit words (currency, count, score) reject.
)


def _adjacent_unit_suffix(text: str, pos: int) -> bool:
    """True if a unit-like token immediately follows ``text[pos:]``.

    A slash-form compound (``km/h``), percent, or one of the mapped physical
    units directly after a number means the number is unit-qualified in a unit
    we do not model; it must not be read as a bare value. Plain prose words
    (``outside``, ``points``) are deliberately not matched.
    """
    if pos >= len(text):
        return False
    m = _ADJACENT_UNIT_RE.match(text, pos)
    return bool(m)


def _term_occurrences(term: str, text: str) -> list[int]:
    """Starting indices of whole-word occurrences of ``term`` in ``text``.

    Single-word anchors are matched on word boundaries (so the alias ``rh``
    does not anchor inside ``through``); multi-word / punctuated phrases match
    as plain substrings, matching how string-alias matching and deny markers
    are bounded.
    """
    if not term:
        return []
    if " " in term or not term.isalnum():
        return _occurrences(text, term)
    return [m.start() for m in re.finditer(rf"\b{re.escape(term)}\b", text)]


def _occurrences(text: str, term: str) -> list[int]:
    """Starting indices of every non-overlapping occurrence of ``term``."""
    if not term:
        return []
    starts = []
    start = 0
    while True:
        idx = text.find(term, start)
        if idx == -1:
            break
        starts.append(idx)
        start = idx + 1
    return starts


def _salient_spans(text: str, terms: tuple[str, ...]) -> list[tuple[int, int]]:
    """Local salient spans around every anchor-term occurrence in ``text``.

    Returns merged ``[start, end]`` windows (each anchor puffed out by
    ``SALIENT_WINDOW`` on both sides) so a deny marker near ANY anchor of a fact
    is detectable with one lookup.
    """
    spans: list[tuple[int, int]] = []
    for term in terms:
        for pos in _term_occurrences(term, text):
            spans.append(
                (max(0, pos - SALIENT_WINDOW), pos + len(term) + SALIENT_WINDOW)
            )
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _has_deny_marker(text: str, spans: list[tuple[int, int]]) -> bool:
    """True if any deny marker falls inside any salient span."""
    if not spans:
        return False
    for regex in _MARKER_RES:
        for m in regex.finditer(text):
            pos = m.start()
            if any(s <= pos < e for s, e in spans):
                return True
    return False


def _string_coverage(fact: StringFact, norm_answer: str) -> tuple[bool, str]:
    """Affirmative coverage for a string/enum fact (alias match).

    Single-word aliases are matched on word boundaries (so the alias ``clear``
    is not satisfied by ``unclear``), while multi-word phrases use a plain
    substring match -- phrases are specific enough that an accidental embed is
    far less likely. This mirrors how deny markers are bounded in
    ``DENY_MARKERS``.
    """
    for term in fact.search_terms:
        if term and _term_matches(term, norm_answer):
            return True, f"matched value {term!r}"
    return False, f"no alias of {fact.key or fact.value!r} found"


def _term_matches(term: str, norm_answer: str) -> bool:
    """True if ``term`` appears in the normalized answer.

    Single-word terms need a word boundary on each side so short values don't
    fire inside unrelated words; multi-word phrases (spaces / punctuation)
    match as plain substrings.
    """
    if " " in term or not term.isalnum():
        return term in norm_answer
    return bool(re.search(rf"\b{re.escape(term)}\b", norm_answer))


def _number_coverage(fact: NumberFact, norm_answer: str) -> tuple[bool, str]:
    """Affirmative coverage for a number fact (unit conversion + tolerance)."""
    candidates = _numbered_in(norm_answer)
    anchor_terms = fact.anchor_terms
    anchors = anchor_terms or (fact.normalized_value,)
    # An ANSWER scene pinpoints the fact only when one of its anchor labels
    # actually appears; if none does, a correctly unit-qualified value
    # ("21 °C at the moment") is still accepted (natural paraphrase), but an
    # unrelated same-unit value ("oven is 21°C") must not fire. When an anchor
    # IS present, unit-qualified values also need to sit near it.
    anchors_present = any(_term_matches(a, norm_answer) for a in anchors)

    def _within(value: float, unit: str | None) -> bool:
        converted = _to_unit(value, unit, fact.unit)
        return (
            converted is not None
            and abs(converted - fact.value) <= fact.tolerance + 1e-9
        )

    for value, unit, start in candidates:
        if unit is not None:
            # Explicitly unit-qualified -- accepted anywhere ONLY when no anchor
            # label exists to pin the fact; otherwise it must sit near an anchor.
            if _within(value, unit) and (
                not anchors_present or _near_anchor(norm_answer, anchors, start)
            ):
                return (
                    True,
                    f"matched {value} {unit or ''} == {fact.value} {fact.unit} ±{fact.tolerance}",
                )
        else:
            # Bare number -- only counts if near a salient anchor, so "21
            # dollars" or "wind 8 km/h" doesn't pass for a temperature fact.
            if _within(value, None) and _near_anchor(norm_answer, anchors, start):
                return True, f"matched bare {value} near {fact.key!r}"
    return False, f"no value within ±{fact.tolerance} {fact.unit} of {fact.value} found"


def _unit_fact_conflict(fact: Fact, norm_answer: str) -> bool:
    """True if the answer affirmatively reports a SECOND, incompatible VALUE for
    the same anchored fact (e.g. "temperature is 18°C and 30°C" against an 18°C
    fact, or "humidity 62% and 20%" against a 62% fact).

    Only explicitly UNIT-QUALIFIED candidates that sit within a fact anchor's
    salient window count. Bare numbers are deliberately excluded: a bare "2026"
    next to an anchor is metadata/a year, not a second measurement, so it must
    not fabricate a contradiction. Distinct unit-qualified values beyond
    tolerance in the same anchored context mean the model's reply is incoherent
    and must flag for review.
    """
    anchors = fact.anchor_terms
    if not anchors or not any(_term_matches(a, norm_answer) for a in anchors):
        return False
    key_spans = _salient_spans(norm_answer, anchors)
    seen: set[float] = set()
    for value, unit, start in _numbered_in(norm_answer):
        if unit is None:  # bare numbers are ambiguous metadata -- never a 2nd value
            continue
        if not any(s <= start < e for s, e in key_spans):
            continue
        converted = _to_unit(value, unit, fact.unit)
        if converted is None:
            continue
        rounded = round(converted, 9)
        if all(abs(rounded - s) > fact.tolerance + 1e-9 for s in seen):
            seen.add(rounded)
            if len(seen) >= 2:
                return True
    return False


def _near_anchor(norm_answer: str, anchors: tuple[str, ...], start: int) -> bool:
    for anchor in anchors:
        if not anchor:
            continue
        for pos in _term_occurrences(anchor, norm_answer):
            window = norm_answer[
                max(0, pos - SALIENT_WINDOW) : pos + len(anchor) + SALIENT_WINDOW
            ]
            # The candidate's actual span must start inside the window.
            if pos - SALIENT_WINDOW <= start < pos + len(anchor) + SALIENT_WINDOW:
                return True
    return False


def _format_number(value: float) -> str:
    """Deterministic string form of a number for local-window matching."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _relation_coverage(fact: RelationFact, norm_answer: str) -> tuple[bool, str]:
    """Affirmative coverage for an entity->value fact.

    The key entity must appear AND a compatible value (with the fact's unit, or
    a bare number) must sit within a salient window of that key.
    """
    anchors = fact.anchor_terms
    if not anchors or not any(_term_matches(a, norm_answer) for a in anchors):
        return False, f"entity {fact.key!r} not mentioned"
    key_spans = _salient_spans(norm_answer, anchors)
    candidates = _numbered_in(norm_answer)
    for value, unit, start in candidates:
        converted = _to_unit(value, unit if unit is not None else fact.unit, fact.unit)
        if converted is None or abs(converted - fact.value) > fact.tolerance + 1e-9:
            continue
        # Compare THIS candidate's actual offset against the anchor window --
        # not the first textually-identical occurrence (a stale "55% was
        # yesterday" must not shadow the "humidity is 55%" that grounds it).
        if any(s <= start < e for s, e in key_spans):
            return True, f"matched {value}{fact.unit} for {fact.key!r}"
    return (
        False,
        f"no value near {fact.key!r} within ±{fact.tolerance}{fact.unit} of {fact.value}",
    )


# ---------------------------------------------------------------------------
# Public grading entry point + report shape.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactEvidence:
    """Per-fact grading result. ``status`` is one of present/missing/contradicted."""

    key: str
    kind: str
    status: str  # "present" | "missing" | "contradicted"
    coverage: bool
    contradicted: bool
    evidence: str


@dataclass(frozen=True)
class GroundingReport:
    """Machine-readable, stable result for a single graded answer."""

    version: str
    overall: bool
    coverage: bool
    missing: list[str]
    contradicted: list[str]
    facts: list[FactEvidence]
    truncated: bool

    def to_dict(self) -> dict:
        """:return: JSON-serializable dict with a stable, versioned shape."""
        return {
            "version": self.version,
            "overall": self.overall,
            "coverage": self.coverage,
            "missing": list(self.missing),
            "contradicted": list(self.contradicted),
            "facts": [
                {
                    "key": f.key,
                    "kind": f.kind,
                    "status": f.status,
                    "coverage": f.coverage,
                    "contradicted": f.contradicted,
                    "evidence": f.evidence,
                }
                for f in self.facts
            ],
            "truncated": self.truncated,
        }

    def canonical_json(self) -> str:
        """Byte-stable JSON used by tests to prove determinism."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _grade_fact(fact: Fact, norm_answer: str) -> FactEvidence:
    """Grade one fact: independent coverage + contradiction, then a status."""
    if isinstance(fact, StringFact):
        coverage, cov_ev = _string_coverage(fact, norm_answer)
        # Contradiction detection spans the ENTIRE fact context (key + value +
        # aliases), so a denial of the fact is caught even when only its key is
        # named ("the condition is unavailable"), while affirmative coverage
        # still keys on value+aliases only.
        spans = _salient_spans(norm_answer, fact.contradiction_terms)
        contradicted = _has_deny_marker(norm_answer, spans)
        kind = "string"
    elif isinstance(fact, NumberFact | RelationFact):
        spans = _salient_spans(norm_answer, fact.anchor_terms)
        if isinstance(fact, NumberFact):
            coverage, cov_ev = _number_coverage(fact, norm_answer)
            kind = "number"
        else:
            coverage, cov_ev = _relation_coverage(fact, norm_answer)
            kind = "relation"
        # ALSO contradict on a second incompatible unit-qualified value reported
        # for the same anchored fact (e.g. "18°C and 30°C" vs an 18°C fact, or
        # "humidity 62% and 20%" vs a 62% fact).
        value_conflict = _unit_fact_conflict(fact, norm_answer)
        contradicted = _has_deny_marker(norm_answer, spans) or value_conflict
    else:  # pragma: no cover - guarded by fact_from_dict
        raise TypeError(f"unexpected fact type: {type(fact).__name__}")

    if contradicted:
        status = "contradicted"
        if isinstance(fact, NumberFact | RelationFact) and _unit_fact_conflict(
            fact, norm_answer
        ):
            evidence = (
                f"{cov_ev}; multiple incompatible values reported near "
                f"{fact.key or fact.value!r}"
            )
        else:
            evidence = f"{cov_ev}; deny/hedge marker near {fact.key or fact.value!r}"
    elif coverage:
        status = "present"
        evidence = cov_ev
    else:
        status = "missing"
        evidence = cov_ev

    return FactEvidence(
        key=getattr(fact, "key", ""),
        kind=kind,
        status=status,
        coverage=coverage,
        contradicted=contradicted,
        evidence=evidence,
    )


def grade_answer(
    facts: list[Fact] | list[dict],
    answer_text: str,
    *,
    max_fact_len: int = DEFAULT_MAX_FACT_LEN,
    max_answer_len: int = DEFAULT_MAX_ANSWER_LEN,
) -> GroundingReport:
    """Grade a free-text final answer against a set of salient facts.

    Args:
        facts: Already-normalized ``Fact`` objects, or raw scenario dicts
            (each converted via ``fact_from_dict``).
        answer_text: The final answer's free text (tool output / model reply).
        max_fact_len: Cap applied to each fact's normalized representation
            (values / aliases). Oversized fact content is clamped so a huge or
            injection-ish value can't dominate the verdict.
        max_answer_len: Cap applied to the answer before grading. Tool output
            and model text are DATA, bounded here so an absurdly long response
            is handled without error and can't hide the salient facts.

    Returns:
        A ``GroundingReport`` with per-fact ``FactEvidence`` and a stable,
        versioned machine-readable ``to_dict()``.
    """
    # Normalize fact inputs (tolerant of raw dicts for ergonomics).
    # ``overflow`` counts facts beyond MAX_FACTS: silently dropping them would
    # let "all facts present" pass while an ungraded required fact was omitted,
    # so any overflow forces a visible failure instead (see report below).
    total = len(facts or [])
    overflow = max(0, total - MAX_FACTS)
    normalized: list[Fact] = []
    for f in facts[:MAX_FACTS]:
        try:
            normalized.append(
                f
                if isinstance(f, StringFact | NumberFact | RelationFact)
                else fact_from_dict(f)
            )
        except (ValueError, TypeError, KeyError):
            # A malformed fact should not take down the whole report; record it
            # as missing so the failure is visible rather than silent/scored.
            # The empty value (and empty aliases -- see StringFact.search_terms)
            # guarantee it can never pass affirmation, so it fails as missing.
            normalized.append(
                StringFact(
                    key=str(getattr(f, "get", lambda k: "")("key", "?")),
                    value="",
                    aliases=(),
                )
            )

    # Cap sizes defensively: the answer is bounded, and each fact's string
    # content is clamped so oversized / injection-ish fact values or aliases
    # can't dominate the verdict (tool output and answers are DATA). Grading
    # the truncated prefix as if it were the required fact would let an answer
    # containing only the first 200 chars pass, so any clamped fact FAILS CLOSED
    # (see _clamp_fact / report below).
    truncated = len(answer_text or "") > max_answer_len
    capped_answer = (answer_text or "")[:max_answer_len]
    norm_answer = _norm(capped_answer)
    clamped = [_clamp_fact(f, max_fact_len) for f in normalized]
    fact_clamped = False
    facts_for_grading: list[Fact] = []
    for f, was_clamped in clamped:
        fact_clamped = fact_clamped or was_clamped
        facts_for_grading.append(f)

    facts_out: list[FactEvidence] = []
    missing: list[str] = []
    contradicted: list[str] = []
    for fact in facts_for_grading:
        ev = _grade_fact(fact, norm_answer)
        facts_out.append(ev)
        if ev.status == "missing":
            missing.append(ev.key or "?")
        elif ev.status == "contradicted":
            contradicted.append(ev.key or "?")

    if overflow:
        missing.append(
            f"__{overflow} fact(s) beyond MAX_FACTS={MAX_FACTS} not graded__"
        )
    report = GroundingReport(
        version=SCORE_FORMAT,
        # ``overall`` is the strict AND: every salient fact affirmatively
        # reported AND none negated/hedged. ``coverage`` is the INDEPENDENT
        # affirmative aggregate (all facts present), deliberately kept separate
        # from ``contradicted`` so consumers can distinguish "missing" from
        # "contradicted" without re-deriving from the per-fact rows. An ungraded
        # overflow, a truncated answer, an oversized/clamped fact (evidence was
        # altered), or any missing/contradicted fact fails ``overall`` -- a
        # scenario must never pass while some of the evidence was not examined
        # or was rewritten.
        overall=not truncated and not fact_clamped and not missing and not contradicted,
        coverage=not missing,
        missing=missing,
        contradicted=contradicted,
        facts=facts_out,
        truncated=truncated,
    )
    return report
