# SPDX-License-Identifier: Apache-2.0
"""KV-quant differential quality gate — pure metric + report core.

We decide whether an ``int4`` / ``int8`` KV cache is "safe" to default on via a
HAND-WRITTEN empirical safelist (``rapid_mlx/kv_cache_dtype.py``). There is no
*measured* quality signal behind that list: if quantizing the KV cache silently
degrades decode quality on some architecture, nothing detects it.

This module is the measurement core. The runnable harness
(``scripts/kv_quant_quality_gate.py``) runs the SAME model twice on the SAME
prompts — a ``bf16``-KV BASELINE and a quantized-KV CANDIDATE (``int8`` /
``int4``) — and this module scores how well the CANDIDATE **agrees with its own
bf16 baseline**. The differential/agreement-rate framing is the key design
choice: it measures quantization-*induced* drift, not absolute correctness, so a
model that is simply bad at a prompt does not fail the gate — only a model the
quantized cache made *worse than its own full-precision self* does.

Everything here is a **pure function**: no model load, no MLX, no I/O. It
operates on already-collected token streams / per-step log-probability vectors /
byte counts, so every metric is hermetically unit-testable with synthetic data.
The harness owns all inference; this module owns all scoring.

Metrics
-------
* **Greedy agreement rate** — token-level match of the candidate's greedy
  (temp=0) continuation against the baseline's, per prompt. The headline metric.
* **Logit divergence** — mean / max KL and top-1 agreement of the two next-token
  distributions, over the SHARED-CONTEXT prefix (the steps before the two runs
  chose different tokens, where the only difference between the distributions is
  the KV-cache dtype).
* **Structured-output retention** — does the candidate still emit valid JSON when
  the baseline did? Attributes only quantization-induced breakage (a prompt the
  baseline already failed is excluded).
* **Memory delta** — the measured KV footprint reduction (candidate must
  actually be smaller — a "quantized" cache that saved nothing is a bug).

Advisory, not blocking (measure-first)
--------------------------------------
The thresholds below are PROVISIONAL. This gate ships in *measure-first* mode
(mirroring the ``diff_coverage`` advisory pattern): it computes a PASS/FAIL and
emits a full report, but the harness exits ``0`` regardless by default. Once we
have baseline numbers across the fleet, the thresholds get calibrated and the
gate can be promoted to blocking (``--enforce``) in a follow-up.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# Metric outcome labels.
PASS = "PASS"
FAIL = "FAIL"
NA = "NA"


def _logsumexp(x: np.ndarray) -> float:
    """Numerically-stable ``log(sum(exp(x)))``.

    Returns ``-inf`` for an all ``-inf`` vector (a degenerate distribution) and
    ``+inf``/``nan`` if the input carries one — the caller guards on finiteness.
    """
    m = float(np.max(x))
    if not np.isfinite(m):
        # All -inf (m == -inf) or a stray +inf/nan: no valid shift exists.
        return m
    return m + float(np.log(np.sum(np.exp(x - m))))


# ---------------------------------------------------------------------------
# Greedy agreement
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgreementResult:
    """Token-level agreement of a candidate continuation vs its baseline.

    Attributes:
        total: The denominator — the ``max`` of the two stream lengths. An
            unequal length is itself a divergence (premature EOS), so the longer
            stream sets the scale and the missing suffix positions count as
            disagreement.
        matched: Leading run of equal token ids (the greedy context forks at the
            first mismatch, so positions after it are not a same-context
            comparison and are not counted as matches).
        rate: ``matched / total`` (``1.0`` for two empty streams — vacuously
            equal).
        first_divergence_index: Index of the first mismatching position, or —
            when the prefixes all match but the lengths differ — the overlap
            boundary where the shorter stream ended. ``None`` only when the
            streams are identical.
    """

    total: int
    matched: int
    rate: float
    first_divergence_index: int | None


def greedy_agreement_rate(
    baseline_tokens: Sequence[int], candidate_tokens: Sequence[int]
) -> AgreementResult:
    """Token-agreement of two greedy continuations, penalizing length mismatch.

    ``matched`` counts the leading run of equal tokens; the first mismatch stops
    the count (after a divergence the two runs feed different tokens back, so
    later positions are not comparable). The denominator is the LONGER stream's
    length, so a candidate that ends early after an all-matching prefix — the
    premature-EOS degradation this gate is built to catch — scores below ``1.0``
    instead of a misleading perfect score. ``rate`` is a clean 0..1
    quantization-drift signal.

    Two empty streams agree vacuously (``total=0``, ``rate=1.0``). An empty vs
    non-empty pair scores ``rate=0.0`` (the whole non-empty stream is missing).
    """
    # Denominator is the LONGER of the two streams — an unequal continuation
    # length is itself a divergence, not something to average away. A quantized
    # candidate that terminates early (premature EOS — a core failure mode this
    # gate exists to catch) must score BELOW 1.0, so the missing suffix positions
    # count against it. Using ``min`` here would score an early-EOS candidate that
    # matched its short prefix at a perfect 1.0 and hide the regression.
    total = max(len(baseline_tokens), len(candidate_tokens))
    if total == 0:
        return AgreementResult(
            total=0, matched=0, rate=1.0, first_divergence_index=None
        )

    overlap = min(len(baseline_tokens), len(candidate_tokens))
    matched = 0
    first_divergence: int | None = None
    for i in range(overlap):
        if baseline_tokens[i] == candidate_tokens[i]:
            matched += 1
        else:
            first_divergence = i
            break
    else:
        # No token mismatch within the overlapping prefix. If the streams are
        # different lengths the shorter one ended first — that boundary IS the
        # divergence point (e.g. premature EOS under quantization).
        if len(baseline_tokens) != len(candidate_tokens):
            first_divergence = overlap

    rate = matched / total
    return AgreementResult(
        total=total,
        matched=matched,
        rate=rate,
        first_divergence_index=first_divergence,
    )


# ---------------------------------------------------------------------------
# Logit divergence
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LogitDivergence:
    """Distributional drift between baseline and candidate next-token logits.

    Attributes:
        compared_steps: Number of per-step distributions compared.
        mean_kl: Mean forward KL ``D(P_baseline || P_candidate)`` in nats over
            the compared steps (``0.0`` when nothing was compared).
        max_kl: Worst single-step KL (``0.0`` when nothing was compared).
        top1_agreement_rate: Fraction of compared steps whose argmax token
            matches (``1.0`` when nothing was compared — vacuously).
    """

    compared_steps: int
    mean_kl: float
    max_kl: float
    top1_agreement_rate: float


def logit_divergence(
    baseline_logprobs: Sequence[Sequence[float]],
    candidate_logprobs: Sequence[Sequence[float]],
    *,
    compare_len: int | None = None,
) -> LogitDivergence:
    """KL + top-1 agreement of two per-step log-probability sequences.

    Each argument is a sequence of per-step log-probability vectors
    (``log_softmax`` output — one vector per generated position, length = vocab).
    Forward KL is ``Σ_x P_base(x) · (logP_base(x) − logP_cand(x))`` per step,
    averaged over the compared steps.

    ``compare_len`` bounds how many leading steps are compared. It exists because
    a same-context comparison is only valid over the shared-context prefix: the
    harness passes ``first_divergence_index`` (the two greedy runs share a context
    up to and including that step) so KL is measured strictly where the ONLY
    difference between the two distributions is the KV-cache dtype. When ``None``,
    all overlapping steps are compared (``min`` of the two lengths).

    Both vectors are renormalized to true log-probabilities via log-sum-exp
    before the KL, so the metric is shift-invariant: two distributions that are
    identical up to different additive constants (raw logits, or only
    slightly-normalized logprobs) score ~0, not a spurious non-zero. A vocab
    entry that is ``-inf`` in one distribution but has mass in the other
    contributes ``+inf`` KL, clipped to a large finite ceiling so the mean stays
    reportable. A NaN in either distribution, or a degenerate distribution whose
    log-sum-exp is non-finite (all ``-inf``), is treated as a CATASTROPHIC step
    (charged the ceiling), never as zero divergence — so broken inference can
    never masquerade as perfect agreement.
    """
    overlap = min(len(baseline_logprobs), len(candidate_logprobs))
    n = overlap if compare_len is None else min(overlap, max(compare_len, 0))
    if n == 0:
        return LogitDivergence(
            compared_steps=0, mean_kl=0.0, max_kl=0.0, top1_agreement_rate=1.0
        )

    kls: list[float] = []
    top1_matches = 0
    # A large finite ceiling for a per-step KL so one pathological step can't turn
    # the whole mean into inf/nan while still dominating it (signals a real break).
    _KL_CEIL = 1e4
    for i in range(n):
        base = np.asarray(baseline_logprobs[i], dtype=np.float64).reshape(-1)
        cand = np.asarray(candidate_logprobs[i], dtype=np.float64).reshape(-1)
        if base.shape != cand.shape or base.size == 0:
            # Shape mismatch is a harness bug, not a model signal — skip the step.
            continue
        # A NaN in either distribution means the inference itself broke. That is
        # NOT zero divergence — charge the catastrophic ceiling so a NaN baseline
        # can never masquerade as perfect agreement (the argmax is meaningless
        # here too, so this step is not counted as a top-1 match).
        if np.isnan(base).any() or np.isnan(cand).any():
            kls.append(_KL_CEIL)
            continue
        if int(np.argmax(base)) == int(np.argmax(cand)):
            top1_matches += 1
        # Renormalize BOTH vectors to true log-probabilities via log-sum-exp. The
        # weighting AND the log-ratio must use normalized values, else two
        # distributions that are identical up to different additive constants
        # (raw logits, or slightly-unnormalized logprobs) score a spurious
        # non-zero KL. If either normalizer is non-finite (an all -inf / broken
        # distribution), no valid probability exists — charge the catastrophic
        # ceiling instead of scoring a spurious 0.0.
        base_lse = _logsumexp(base)
        cand_lse = _logsumexp(cand)
        if not (np.isfinite(base_lse) and np.isfinite(cand_lse)):
            kls.append(_KL_CEIL)
            continue
        base_norm = base - base_lse
        cand_norm = cand - cand_lse
        p = np.exp(base_norm)
        mask = p > 0.0
        # cand == -inf where base has mass -> diff +inf; clip to keep mean finite.
        diff = base_norm[mask] - cand_norm[mask]
        step_kl = float(np.sum(p[mask] * np.clip(diff, -_KL_CEIL, _KL_CEIL)))
        if not np.isfinite(step_kl):
            step_kl = _KL_CEIL
        # KL is non-negative in theory; tiny negatives from float error -> 0.
        kls.append(max(step_kl, 0.0))

    if not kls:
        return LogitDivergence(
            compared_steps=0, mean_kl=0.0, max_kl=0.0, top1_agreement_rate=1.0
        )

    return LogitDivergence(
        compared_steps=len(kls),
        mean_kl=float(np.mean(kls)),
        max_kl=float(np.max(kls)),
        top1_agreement_rate=top1_matches / len(kls),
    )


# ---------------------------------------------------------------------------
# Structured-output retention
# ---------------------------------------------------------------------------
# A fence is only unwrapped when the WHOLE (stripped) output is exactly one
# ```...``` block — a fence embedded in prose does not satisfy an "ONLY JSON"
# instruction and must not be leniently extracted.
_JSON_FENCE_FULL_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
)

# Sentinel for "did not parse". ``None`` is a legal JSON value (``null``), so we
# cannot use it to signal failure.
_JSON_INVALID = object()


def _parse_strict_json(text: str) -> Any:
    """Parse the WHOLE output as a single JSON value, or return ``_JSON_INVALID``.

    The structured prompts instruct the model to reply with ONLY JSON. So the
    contract is strict: the entire stripped output (optionally wrapped in a
    single code fence) must parse as one JSON value via ``json.loads`` — which,
    unlike an incremental ``raw_decode`` scan, rejects trailing/leading prose
    (``"Sorry, [1]"``) and stray bracket fragments (``"see [x] then {..}"``).
    A candidate that buries JSON in prose has NOT honored the format contract and
    should not count as retained.
    """
    if not text or not text.strip():
        return _JSON_INVALID
    candidate = text.strip()
    fence = _JSON_FENCE_FULL_RE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        return _JSON_INVALID


def is_valid_json(text: str) -> bool:
    """True iff the WHOLE output parses as a single JSON value (see the note on
    strictness in :func:`_parse_strict_json`)."""
    return _parse_strict_json(text) is not _JSON_INVALID


def _json_shape(obj: Any) -> tuple[str, frozenset[str] | None]:
    """Classify a parsed JSON value into a comparable shape signature.

    Returns ``("object", frozenset(keys))`` for a dict, ``("array", None)`` for a
    list, and ``("scalar", None)`` for anything else (string / number / bool /
    null). Used to require the candidate to preserve the baseline's structure.
    """
    if isinstance(obj, dict):
        return "object", frozenset(obj.keys())
    if isinstance(obj, list):
        return "array", None
    return "scalar", None


def _retains_shape(baseline_obj: Any, candidate_obj: Any) -> bool:
    """True iff the candidate preserves the baseline's JSON contract.

    Same top-level type, and — for objects — the candidate must carry AT LEAST
    the baseline's keys (a requested key vanishing under quantization is a
    regression). This closes the generic-validity hole where an object prompt
    could "retain" as an array/scalar or silently drop keys.
    """
    base_type, base_keys = _json_shape(baseline_obj)
    cand_type, cand_keys = _json_shape(candidate_obj)
    if base_type != cand_type:
        return False
    if base_type == "object":
        assert base_keys is not None and cand_keys is not None
        return base_keys <= cand_keys
    return True


@dataclass(frozen=True)
class RetentionResult:
    """Structured-output retention across a set of (baseline, candidate) outputs.

    Attributes:
        attributable: Prompts where the BASELINE emitted a valid JSON document —
            the only prompts where a candidate failure is attributable to
            quantization.
        retained: Of the attributable prompts, how many the candidate kept valid
            AND matching the baseline's JSON contract (same top-level type; for
            objects, no baseline key dropped).
        rate: ``retained / attributable``, or ``None`` when nothing was
            attributable (no baseline produced valid JSON — the metric is N/A).
        baseline_invalid: Prompts excluded because the baseline itself did not
            emit a valid JSON document (not a quantization signal).
    """

    attributable: int
    retained: int
    rate: float | None
    baseline_invalid: int


def structured_output_retention(
    pairs: Sequence[tuple[str, str]],
) -> RetentionResult:
    """Shape-aware retention of the baseline's JSON contract under quantization.

    ``pairs`` is a sequence of ``(baseline_text, candidate_text)``. Only prompts
    where the baseline produced a valid JSON document count toward the rate; of
    those, a prompt is RETAINED iff the candidate is also valid JSON AND
    preserves the baseline's contract (same top-level type; for objects, retains
    every baseline key — see :func:`_retains_shape`). A prompt the baseline
    already failed is excluded (not attributable to the cache dtype). ``rate`` is
    ``None`` when nothing is attributable.
    """
    attributable = 0
    retained = 0
    baseline_invalid = 0
    for baseline_text, candidate_text in pairs:
        baseline_obj = _parse_strict_json(baseline_text)
        if baseline_obj is _JSON_INVALID:
            baseline_invalid += 1
            continue
        attributable += 1
        candidate_obj = _parse_strict_json(candidate_text)
        if candidate_obj is not _JSON_INVALID and _retains_shape(
            baseline_obj, candidate_obj
        ):
            retained += 1
    rate = (retained / attributable) if attributable > 0 else None
    return RetentionResult(
        attributable=attributable,
        retained=retained,
        rate=rate,
        baseline_invalid=baseline_invalid,
    )


# ---------------------------------------------------------------------------
# Memory delta
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MemoryDelta:
    """Measured KV-cache footprint reduction of candidate vs baseline.

    Attributes:
        baseline_bytes: Measured bf16 KV-cache footprint.
        candidate_bytes: Measured quantized KV-cache footprint.
        saved_bytes: ``baseline_bytes − candidate_bytes`` (may be negative if a
            quantized cache somehow grew — a bug the gate should surface).
        reduction_ratio: ``baseline_bytes / candidate_bytes`` (``0.0`` when the
            candidate measured zero bytes — degenerate, treated as no saving).
        saved_pct: Percentage of the baseline footprint saved.
    """

    baseline_bytes: int
    candidate_bytes: int
    saved_bytes: int
    reduction_ratio: float
    saved_pct: float


def memory_delta(baseline_bytes: int, candidate_bytes: int) -> MemoryDelta:
    """Compute the KV-cache footprint reduction from two measured byte counts."""
    saved = baseline_bytes - candidate_bytes
    ratio = (baseline_bytes / candidate_bytes) if candidate_bytes > 0 else 0.0
    pct = (saved / baseline_bytes * 100.0) if baseline_bytes > 0 else 0.0
    return MemoryDelta(
        baseline_bytes=int(baseline_bytes),
        candidate_bytes=int(candidate_bytes),
        saved_bytes=int(saved),
        reduction_ratio=ratio,
        saved_pct=pct,
    )


# ---------------------------------------------------------------------------
# Thresholds + report
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateThresholds:
    """PROVISIONAL pass thresholds. Calibrate before promoting to blocking.

    Defaults are deliberately lenient — this gate ships advisory (measure-first),
    so these exist to shape the report, not to fail anything yet. int4 tolerates
    more drift than int8 because it quantizes harder.
    """

    min_greedy_agreement: float
    max_mean_kl: float
    min_structured_retention: float
    min_memory_reduction_ratio: float


def default_thresholds(candidate_dtype: str) -> GateThresholds:
    """Return provisional thresholds for a candidate dtype.

    Only ``int8`` and ``int4`` are supported; any other value raises rather than
    silently borrowing the int8 thresholds (a typo or a future dtype would
    otherwise get a misleading verdict).
    """
    if candidate_dtype == "int4":
        return GateThresholds(
            min_greedy_agreement=0.40,
            max_mean_kl=0.20,
            min_structured_retention=1.0,
            min_memory_reduction_ratio=2.5,
        )
    if candidate_dtype == "int8":
        return GateThresholds(
            min_greedy_agreement=0.60,
            max_mean_kl=0.05,
            min_structured_retention=1.0,
            min_memory_reduction_ratio=1.5,
        )
    raise ValueError(
        f"unsupported candidate dtype {candidate_dtype!r}; expected 'int8' or 'int4'"
    )


@dataclass
class GateReport:
    """Full structured report of one baseline-vs-candidate gate run.

    ``metrics`` and ``overall`` are filled by :func:`build_report`; construct via
    that builder rather than by hand so the PASS/FAIL logic stays in one place.
    """

    model: str
    hf_path: str
    baseline_dtype: str
    candidate_dtype: str
    num_prompts: int
    advisory: bool
    chip: dict[str, Any]
    thresholds: GateThresholds
    agreement: AgreementResult
    logits: LogitDivergence
    retention: RetentionResult
    memory: MemoryDelta
    niah: dict[str, Any] = field(default_factory=lambda: {"status": "skipped"})
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall: str = NA

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the whole report."""
        return {
            "schema": 1,
            "kind": "kv_quant_quality_gate",
            "advisory": self.advisory,
            "model": self.model,
            "hf_path": self.hf_path,
            "baseline_dtype": self.baseline_dtype,
            "candidate_dtype": self.candidate_dtype,
            "num_prompts": self.num_prompts,
            "chip": self.chip,
            "thresholds": asdict(self.thresholds),
            "agreement": asdict(self.agreement),
            "logits": asdict(self.logits),
            "retention": asdict(self.retention),
            "memory": asdict(self.memory),
            "niah": self.niah,
            "metrics": self.metrics,
            "overall": self.overall,
        }

    def human_summary(self) -> str:
        """Operator-facing multi-line summary in the ``scripts/`` report style."""
        bar = "=" * 70
        lines = [
            bar,
            " KV-quant differential quality gate"
            + ("  (ADVISORY — measure-first)" if self.advisory else "  (ENFORCED)"),
            bar,
            f" model:     {self.model}",
            f" hf_path:   {self.hf_path}",
            f" baseline:  {self.baseline_dtype}-KV   candidate: "
            f"{self.candidate_dtype}-KV",
            f" prompts:   {self.num_prompts}",
            f" chip:      {self.chip.get('raw', 'unknown')} "
            f"(tier: gen={self.chip.get('generation')} "
            f"{self.chip.get('variant')})",
            "-" * 70,
        ]
        for name, m in self.metrics.items():
            lines.append(
                f" {m['outcome']:<4} {name:<26} "
                f"value={m['value']}  threshold={m['threshold']}"
            )
        lines.append(f" {'':<4} {'niah':<26} status={self.niah.get('status')}")
        lines.append("-" * 70)
        lines.append(f" OVERALL: {self.overall}")
        if self.advisory:
            lines.append(
                " (advisory run — thresholds are provisional; exit code is 0"
                " regardless of OVERALL)"
            )
        lines.append(bar)
        return "\n".join(lines)


def _fmt(value: Any) -> Any:
    """Round floats for compact display; pass through everything else."""
    if isinstance(value, float):
        return round(value, 5)
    return value


def build_report(
    *,
    model: str,
    hf_path: str,
    baseline_dtype: str,
    candidate_dtype: str,
    num_prompts: int,
    advisory: bool,
    chip: dict[str, Any],
    thresholds: GateThresholds,
    agreement: AgreementResult,
    logits: LogitDivergence,
    retention: RetentionResult,
    memory: MemoryDelta,
    niah: dict[str, Any] | None = None,
) -> GateReport:
    """Assemble a :class:`GateReport` and evaluate PASS/FAIL from the metrics.

    Per-metric outcome:

    * ``greedy_agreement`` — ``PASS`` iff ``rate >= min_greedy_agreement``.
    * ``logit_mean_kl`` — ``PASS`` iff ``mean_kl <= max_mean_kl``; ``NA`` when no
      shared-context step was comparable.
    * ``structured_retention`` — ``PASS`` iff ``rate >= min_structured_retention``;
      ``NA`` when no baseline produced valid JSON (nothing attributable).
    * ``memory_reduction`` — ``PASS`` iff ``reduction_ratio >=
      min_memory_reduction_ratio``.
    * ``niah`` (optional) — counts toward ``overall`` only when its status is
      ``pass`` / ``fail``; a ``skipped`` NIAH is ``NA``.

    ``overall`` is ``PASS`` iff every non-``NA`` metric is ``PASS``.
    """
    niah = niah or {"status": "skipped"}
    metrics: dict[str, dict[str, Any]] = {}

    # Greedy agreement — always evaluable.
    metrics["greedy_agreement"] = {
        "value": _fmt(agreement.rate),
        "threshold": f">= {thresholds.min_greedy_agreement}",
        "outcome": PASS if agreement.rate >= thresholds.min_greedy_agreement else FAIL,
    }

    # Logit mean KL — NA when nothing was comparable.
    if logits.compared_steps == 0:
        kl_outcome = NA
    else:
        kl_outcome = PASS if logits.mean_kl <= thresholds.max_mean_kl else FAIL
    metrics["logit_mean_kl"] = {
        "value": _fmt(logits.mean_kl),
        "threshold": f"<= {thresholds.max_mean_kl}",
        "outcome": kl_outcome,
    }

    # Structured retention — NA when no attributable prompt.
    if retention.rate is None:
        ret_outcome = NA
        ret_value: Any = None
    else:
        ret_value = _fmt(retention.rate)
        ret_outcome = (
            PASS if retention.rate >= thresholds.min_structured_retention else FAIL
        )
    metrics["structured_retention"] = {
        "value": ret_value,
        "threshold": f">= {thresholds.min_structured_retention}",
        "outcome": ret_outcome,
    }

    # Memory reduction — always evaluable.
    metrics["memory_reduction"] = {
        "value": _fmt(memory.reduction_ratio),
        "threshold": f">= {thresholds.min_memory_reduction_ratio}x",
        "outcome": (
            PASS
            if memory.reduction_ratio >= thresholds.min_memory_reduction_ratio
            else FAIL
        ),
    }

    # Overall: every non-NA metric must PASS; a pass/fail NIAH participates too.
    outcomes = [m["outcome"] for m in metrics.values()]
    niah_status = str(niah.get("status", "skipped")).lower()
    if niah_status in ("pass", "fail"):
        outcomes.append(PASS if niah_status == "pass" else FAIL)
    graded = [o for o in outcomes if o != NA]
    overall = (
        PASS
        if graded and all(o == PASS for o in graded)
        else (FAIL if FAIL in graded else NA)
    )

    return GateReport(
        model=model,
        hf_path=hf_path,
        baseline_dtype=baseline_dtype,
        candidate_dtype=candidate_dtype,
        num_prompts=num_prompts,
        advisory=advisory,
        chip=chip,
        thresholds=thresholds,
        agreement=agreement,
        logits=logits,
        retention=retention,
        memory=memory,
        niah=niah,
        metrics=metrics,
        overall=overall,
    )
