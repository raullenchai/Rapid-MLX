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
        return _norm(self.value or self.key)

    @property
    def search_terms(self) -> tuple[str, ...]:
        """Normalized positive phrases that affirmatively report the fact."""
        terms = {self.normalized_value}
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


def _clamp_fact(fact: Fact, max_fact_len: int) -> Fact:
    """Rebuild ``fact`` with every string field capped to ``max_fact_len``.

    Values and aliases are DATA (potentially attacker- or tool-controlled), so
    an oversized / injection-ish fact is clamped rather than allowed to dominate
    the verdict. Numeric fields and unit/tolerance are kept untouched.
    """
    if max_fact_len <= 0:
        return fact

    def _cut(value: str) -> str:
        return value[:max_fact_len]

    if isinstance(fact, StringFact):
        return StringFact(
            key=_cut(fact.key),
            value=_cut(fact.value),
            aliases=tuple(_cut(a) for a in fact.aliases),
        )
    if isinstance(fact, NumberFact):
        return NumberFact(
            key=_cut(fact.key),
            value=fact.value,
            unit=fact.unit,
            tolerance=fact.tolerance,
            aliases=tuple(_cut(a) for a in fact.aliases),
        )
    if isinstance(fact, RelationFact):
        return RelationFact(
            key=_cut(fact.key),
            value=fact.value,
            unit=fact.unit,
            tolerance=fact.tolerance,
            aliases=tuple(_cut(a) for a in fact.aliases),
        )
    return fact


def fact_from_dict(d: dict) -> Fact:
    """Convert a scenario ``expected_facts`` entry into a normalized Fact.

    Raises ``ValueError`` on an unknown fact type or a missing required field,
    so a misconfigured scenario fails fast rather than grading silently.
    """
    if not isinstance(d, dict):
        raise ValueError(f"expected_facts entry must be a dict, got {type(d).__name__}")
    ftype = d.get("type")
    if ftype == "string":
        return StringFact(
            key=str(d.get("key", "")),
            value=str(d.get("value", "")),
            aliases=tuple(str(a) for a in d.get("aliases", [])),
        )
    if ftype == "number":
        unit = _resolve_unit(str(d.get("unit", _C))) or _C
        return NumberFact(
            key=str(d.get("key", "")),
            value=float(d.get("value", 0.0)),
            unit=unit,
            tolerance=float(d.get("tolerance", 0.0)),
            aliases=tuple(str(a) for a in d.get("aliases", [])),
        )
    if ftype == "relation":
        unit = _resolve_unit(str(d.get("unit", _PERCENT))) or _PERCENT
        return RelationFact(
            key=str(d.get("key", "")),
            value=float(d.get("value", 0.0)),
            unit=unit,
            tolerance=float(d.get("tolerance", 0.0)),
            aliases=tuple(str(a) for a in d.get("aliases", [])),
        )
    raise ValueError(f"unknown expected_facts type: {ftype!r}")


# ---------------------------------------------------------------------------
# Grading helpers (token / alias / unit based, not loose full-text regex).
# ---------------------------------------------------------------------------


def _numbered_in(text: str) -> list[tuple[float, str | None]]:
    """Yield ``(value, canonical_unit_or_None)`` candidates in ``text``.

    ``text`` is the NFC+casefolded answer. Values with a resolvable unit yield
    that unit; values without a unit yield ``None`` (a bare number). This is a
    typed, bounded extraction -- it never grants meaning to free-text.
    """
    out: list[tuple[float, str | None]] = []
    for match in _UNIT_RE.finditer(text):
        num = match.group("num")
        token = match.group("unit")
        if num is None:
            continue
        value = float(num)
        if not token:
            out.append((value, None))
            continue
        unit = _resolve_unit(token.strip())
        if unit is None:
            # A unit-ish suffix we don't recognize -- treat as bare for safety.
            out.append((value, None))
        else:
            out.append((value, unit))
    return out


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
        for pos in _occurrences(text, term):
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

    def _within(value: float, unit: str | None) -> bool:
        converted = _to_unit(value, unit, fact.unit)
        return (
            converted is not None
            and abs(converted - fact.value) <= fact.tolerance + 1e-9
        )

    for value, unit in candidates:
        if unit is not None:
            # Explicitly unit-qualified -- valid anywhere in the answer.
            if _within(value, unit):
                return (
                    True,
                    f"matched {value} {unit or ''} == {fact.value} {fact.unit} ±{fact.tolerance}",
                )
        else:
            # Bare number -- only counts if near a salient anchor, so "21
            # dollars" or "wind 8 km/h" doesn't pass for a temperature fact.
            if _within(value, None) and _near_anchor(norm_answer, value, anchors):
                return True, f"matched bare {value} near {fact.key!r}"
    return False, f"no value within ±{fact.tolerance} {fact.unit} of {fact.value} found"


def _near_anchor(norm_answer: str, value: float, anchors: tuple[str, ...]) -> bool:
    """True if the bare number ``value`` sits within a salient anchor window."""
    textual = _format_number(value)
    for anchor in anchors:
        if not anchor:
            continue
        for pos in _occurrences(norm_answer, anchor):
            window = norm_answer[
                max(0, pos - SALIENT_WINDOW) : pos + len(anchor) + SALIENT_WINDOW
            ]
            if textual in window:
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
    if not anchors or not any(a in norm_answer for a in anchors):
        return False, f"entity {fact.key!r} not mentioned"
    key_spans = _salient_spans(norm_answer, anchors)
    candidates = _numbered_in(norm_answer)
    for value, unit in candidates:
        converted = _to_unit(value, unit if unit is not None else fact.unit, fact.unit)
        if converted is None or abs(converted - fact.value) > fact.tolerance + 1e-9:
            continue
        idx = norm_answer.find(fact_unit_forms(fact, value))
        pos = idx if idx != -1 else norm_answer.find(_format_number(value))
        if pos != -1 and any(s <= pos < e for s, e in key_spans):
            return True, f"matched {value}{fact.unit} for {fact.key!r}"
    return (
        False,
        f"no value near {fact.key!r} within ±{fact.tolerance}{fact.unit} of {fact.value}",
    )


def fact_unit_forms(fact: RelationFact, value: float) -> str:
    """Best-effort textual form of ``value`` plus the fact's unit token."""
    return f"{_format_number(value)}{fact.unit}"


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
        spans = _salient_spans(norm_answer, fact.search_terms)
        contradicted = _has_deny_marker(norm_answer, spans)
        kind = "string"
    elif isinstance(fact, NumberFact):
        coverage, cov_ev = _number_coverage(fact, norm_answer)
        spans = _salient_spans(norm_answer, fact.anchor_terms)
        contradicted = _has_deny_marker(norm_answer, spans)
        kind = "number"
    elif isinstance(fact, RelationFact):
        coverage, cov_ev = _relation_coverage(fact, norm_answer)
        spans = _salient_spans(norm_answer, fact.anchor_terms)
        contradicted = _has_deny_marker(norm_answer, spans)
        kind = "relation"
    else:  # pragma: no cover - guarded by fact_from_dict
        raise TypeError(f"unexpected fact type: {type(fact).__name__}")

    if contradicted:
        status = "contradicted"
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
            normalized.append(
                StringFact(
                    key=str(getattr(f, "get", lambda k: "")("key", "?")), value=""
                )
            )

    # Cap sizes defensively: the answer is bounded, and each fact's string
    # content is clamped so oversized / injection-ish fact values or aliases
    # can't dominate the verdict (tool output and answers are DATA).
    truncated = len(answer_text or "") > max_answer_len
    capped_answer = (answer_text or "")[:max_answer_len]
    norm_answer = _norm(capped_answer)
    clamped_facts = [_clamp_fact(f, max_fact_len) for f in normalized]

    facts_out: list[FactEvidence] = []
    missing: list[str] = []
    contradicted: list[str] = []
    for fact in clamped_facts:
        ev = _grade_fact(fact, norm_answer)
        facts_out.append(ev)
        if ev.status == "missing":
            missing.append(ev.key or "?")
        elif ev.status == "contradicted":
            contradicted.append(ev.key or "?")

    report = GroundingReport(
        version=SCORE_FORMAT,
        # ``overall`` is the strict AND: every salable fact affirmatively
        # reported AND none negated/hedged. ``coverage`` is the INDEPENDENT
        # affirmative aggregate (all facts present), deliberately kept separate
        # from ``contradicted`` so consumers can distinguish "missing" from
        # "contradicted" without re-deriving from the per-fact rows.
        overall=not missing and not contradicted,
        coverage=not missing,
        missing=missing,
        contradicted=contradicted,
        facts=facts_out,
        truncated=truncated,
    )
    return report
