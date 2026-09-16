# SPDX-License-Identifier: Apache-2.0
"""The HF unauthenticated-download advisory must be dropped — and *only* it.

huggingface_hub logs a server-sent ``X-HF-Warning`` advisory ("You are
sending unauthenticated requests … set a HF_TOKEN …") at WARNING level
from ``huggingface_hub.utils._http`` on the first anonymous Hub request.
That single stray line reads like an error to a naive first-run user
(0.11 dogfood papercut #3). ``rapid_mlx._hf_logging`` drops it with a
narrow, fail-open filter installed on the exact emitting logger.

These are hermetic unit tests of that filter and its installer — no
network, no huggingface_hub import. A capturing handler stands in for the
real ``_warn_on_warning_headers`` emitter: we log the same message on the
same logger and assert it never reaches the handler, while every other
record still does.
"""

from __future__ import annotations

import io
import logging

import pytest

from rapid_mlx._hf_logging import (
    _HF_HTTP_LOGGER,
    _DropUnauthenticatedAdvisory,
    silence_hf_unauthenticated_warning,
)

# The exact line huggingface_hub 1.x surfaces from the server's
# ``X-HF-Warning`` header on an unauthenticated request (captured live).
_REAL_ADVISORY = (
    "You are sending unauthenticated requests to the HF Hub. Please set a "
    "HF_TOKEN to enable higher rate limits and faster downloads."
)


@pytest.fixture
def hf_http_logger():
    """Yield the real HF http logger wired to a captured StringIO handler,
    restoring its filters / handlers / level / propagate afterwards so the
    process-global install never leaks between tests."""
    logger = logging.getLogger(_HF_HTTP_LOGGER)
    saved = (
        list(logger.filters),
        list(logger.handlers),
        logger.level,
        logger.propagate,
    )
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger.handlers = [handler]
    logger.filters = []
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        yield logger, buf
    finally:
        logger.filters, logger.handlers, logger.level, logger.propagate = (
            saved[0],
            saved[1],
            saved[2],
            saved[3],
        )


def _record(msg, *args, level=logging.WARNING):
    return logging.LogRecord(_HF_HTTP_LOGGER, level, __file__, 1, msg, args, None)


# --- the filter predicate, tested directly -------------------------------


def test_filter_drops_the_real_advisory():
    filt = _DropUnauthenticatedAdvisory()
    assert filt.filter(_record(_REAL_ADVISORY)) is False


def test_filter_drops_advisory_despite_whitespace_and_case():
    """Incidental spacing / a wrapped header line / casing must not defeat
    the full-text match — the message is whitespace-normalised first."""
    filt = _DropUnauthenticatedAdvisory()
    noisy = (
        "  You are sending   UNAUTHENTICATED requests to the HF Hub.\n"
        "Please set a HF_TOKEN to enable higher rate limits and faster "
        "downloads.  "
    )
    assert filt.filter(_record(noisy)) is False


@pytest.mark.parametrize(
    "msg",
    [
        # Real HF errors / warnings we MUST keep visible: none carry the
        # "unauthenticated" + token/rate-limit wording together.
        "401 Client Error: Unauthorized for url: https://huggingface.co/…",
        "429 Client Error: Too Many Requests",
        "Repository Not Found for url: https://huggingface.co/foo/bar",
        "Xet Storage is enabled for this repo, but the 'hf_xet' package is "
        "not installed. Falling back to regular HTTP download.",
        "consistency check failed: file should be of size 100 but has size 90",
        "",
    ],
)
def test_filter_passes_everything_else(msg):
    filt = _DropUnauthenticatedAdvisory()
    assert filt.filter(_record(msg)) is True


def test_filter_is_fail_open_on_unrenderable_record():
    """A record whose ``getMessage()`` raises (bad %-args) is not ours to
    drop — the filter must pass it through, not swallow it."""
    filt = _DropUnauthenticatedAdvisory()
    # "%d" % ("abc",) raises TypeError inside getMessage().
    assert filt.filter(_record("%d", "abc")) is True


@pytest.mark.parametrize(
    "msg, level",
    [
        # Near-matches at WARNING: they combine the *words* the old loose
        # predicate keyed on ("unauthenticated" + "rate limit" / "HF_TOKEN")
        # but not the advisory's full text → must NOT be dropped.
        ("Unauthenticated request: rate limit exceeded", logging.WARNING),
        ("Your HF_TOKEN is invalid; requests are unauthenticated", logging.WARNING),
        # Same *subject* as the advisory but a different, actionable
        # message (codex r2): shares "unauthenticated requests to the HF
        # Hub" yet lacks the "set a HF_TOKEN … faster downloads" advice →
        # must NOT be dropped.
        (
            "Unauthenticated requests to the HF Hub are temporarily blocked "
            "due to abuse; contact support.",
            logging.WARNING,
        ),
        # The advisory's own wording, but escalated to ERROR / CRITICAL — a
        # real failure, never ours to hide, even though the phrase matches.
        (_REAL_ADVISORY, logging.ERROR),
        (_REAL_ADVISORY, logging.CRITICAL),
        (
            "unauthenticated requests to the HF Hub caused a rate-limit "
            "error and an HF_TOKEN is required",
            logging.ERROR,
        ),
    ],
)
def test_filter_never_hides_real_failures(msg, level):
    """The filter is fail-safe: it only drops the WARNING-level advisory,
    never a higher-severity record and never a near-match that merely
    shares vocabulary with it (the collision codex flagged)."""
    filt = _DropUnauthenticatedAdvisory()
    assert filt.filter(_record(msg, level=level)) is True


# --- the installer, end-to-end through the logging machinery -------------


def test_install_suppresses_advisory_but_keeps_other_warnings(hf_http_logger):
    logger, buf = hf_http_logger
    silence_hf_unauthenticated_warning()

    logger.warning(_REAL_ADVISORY)  # dropped by the logger-level filter
    logger.warning("consistency check failed: size mismatch")  # kept

    out = buf.getvalue()
    assert "unauthenticated" not in out
    assert "consistency check failed" in out


def test_install_is_idempotent(hf_http_logger):
    logger, _ = hf_http_logger
    silence_hf_unauthenticated_warning()
    silence_hf_unauthenticated_warning()
    silence_hf_unauthenticated_warning()
    tagged = [
        f for f in logger.filters if getattr(f, "_rapid_mlx_hf_unauth_filter", False)
    ]
    assert len(tagged) == 1


def test_install_targets_the_exact_emitting_logger(hf_http_logger):
    """A parent-logger filter is not consulted for child records during
    propagation, so the installer must attach to ``…utils._http`` itself —
    guard against a well-meaning refactor moving it to the parent."""
    silence_hf_unauthenticated_warning()
    child = logging.getLogger(_HF_HTTP_LOGGER)
    assert any(getattr(f, "_rapid_mlx_hf_unauth_filter", False) for f in child.filters)
