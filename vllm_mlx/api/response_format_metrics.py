# SPDX-License-Identifier: Apache-2.0
"""Process-local counters for ``response_format`` strict-mode enforcement (H-06).

Two Prometheus counters expose how often clients ask for OpenAI
``response_format.type=json_schema`` with ``strict=true`` and how often
the post-decode ``jsonschema.validate`` belt-and-braces check ever
catches a schema violation. The expected violations rate is zero when
the engine has the ``[guided]`` extra installed and outlines is doing
its job — anything above zero is a smoke-alarm signal that constrained
decoding silently fell through to unconstrained tokens.

Why module-level ints with a lock instead of ``prometheus_client``:
``vllm_mlx/routes/metrics.py`` deliberately avoids that dependency to
sidestep its global default registry (which fights multi-engine tests).
Mirroring that decision keeps the metrics surface uniform — the
``/metrics`` route reads these counters the same way it reads every
other counter in the project.

Counters never decrease for the lifetime of the process. They reset to
zero on process restart, which is the normal Prometheus client
convention (``process_start_time_seconds`` lets scrapers detect resets).

Tests use :func:`reset_for_tests` to zero state between cases.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_strict_requests_total = 0
_strict_violations_total = 0
_strict_repairs_attempted_total = 0
_strict_repairs_succeeded_total = 0


def incr_strict_request() -> None:
    """Increment the strict-request counter by one."""
    global _strict_requests_total
    with _lock:
        _strict_requests_total += 1


def incr_strict_violation() -> None:
    """Increment the post-decode validation-failure counter by one.

    Outlines is supposed to make this unreachable — a non-zero rate on
    ``rapid_mlx_response_format_strict_violations_total`` means the
    constrained-decoding path silently degraded and the model emitted
    output that ``jsonschema.validate`` rejected against the user's
    schema. Operators should alert on any non-zero rate.
    """
    global _strict_violations_total
    with _lock:
        _strict_violations_total += 1


def incr_strict_repair_attempt() -> None:
    """Increment the strict-mode auto-repair attempt counter (R12-4).

    R12-4 introduces a single retry with a system-prompt-injected
    "your previous output failed validation" hint when the initial
    unconstrained generation does not validate against the supplied
    schema. This counter ticks on EVERY attempt (including the ones
    that ultimately still fail and surface as 422). Pair with
    :func:`incr_strict_repair_success` to compute the repair success
    rate.
    """
    global _strict_repairs_attempted_total
    with _lock:
        _strict_repairs_attempted_total += 1


def incr_strict_repair_success() -> None:
    """Increment the strict-mode auto-repair success counter (R12-4).

    Only ticks when an attempted repair produced output that
    ``jsonschema.validate`` accepted. The ratio
    ``strict_repairs_succeeded_total / strict_repairs_attempted_total``
    is the operator signal for "is the retry hint actually helping".
    """
    global _strict_repairs_succeeded_total
    with _lock:
        _strict_repairs_succeeded_total += 1


def snapshot() -> dict[str, int]:
    """Return a consistent snapshot of all counters for ``/metrics``."""
    with _lock:
        return {
            "strict_requests_total": _strict_requests_total,
            "strict_violations_total": _strict_violations_total,
            "strict_repairs_attempted_total": _strict_repairs_attempted_total,
            "strict_repairs_succeeded_total": _strict_repairs_succeeded_total,
        }


def reset_for_tests() -> None:
    """Test-only hook: zero the counters between cases.

    Production code MUST NOT call this — Prometheus counters are
    contractually monotonic for the process lifetime.
    """
    global _strict_requests_total, _strict_violations_total
    global _strict_repairs_attempted_total, _strict_repairs_succeeded_total
    with _lock:
        _strict_requests_total = 0
        _strict_violations_total = 0
        _strict_repairs_attempted_total = 0
        _strict_repairs_succeeded_total = 0
