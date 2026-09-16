# SPDX-License-Identifier: Apache-2.0
"""Silence one advisory HuggingFace log line on our download paths.

huggingface_hub surfaces server-sent ``X-HF-Warning`` response headers by
logging them at WARNING level from ``huggingface_hub.utils._http``
(the ``_warn_on_warning_headers`` helper). On an *unauthenticated* request
— the state every naive first-run user is in, since pulling a public MLX
model needs no token — the Hub sends back:

    Warning: You are sending unauthenticated requests to the HF Hub.
    Please set a HF_TOKEN to enable higher rate limits and faster
    downloads.

For rapid-mlx this is pure noise: we front downloads with the R2 mirror,
public repos download fine anonymously, and the lone stray ``Warning:``
line (emitted warn-once per process) reads like an error to a first-time
user — right where the cold-download output was otherwise just cleaned up
(#1259). This is the third first-run papercut from the 0.11 dogfood.

We drop **only** that one advisory, with a narrow, fail-*safe* filter — it
is deliberately biased toward *showing* a record, never toward hiding one:

* **WARNING level only.** The advisory is emitted via ``logger.warning``.
  Any record at ``ERROR`` or above passes untouched, so we can never mask
  a genuine failure — even one whose text happens to mention tokens or
  rate limits (e.g. a hypothetical ``Unauthenticated request: rate limit
  exceeded`` error).
* **Matched on the full advisory text** (its subject *and* its distinctive
  "set a HF_TOKEN … higher rate limits … faster downloads" advice), not a
  subject phrase alone — so a *different* actionable warning that merely
  shares the subject (e.g. "Unauthenticated requests to the HF Hub are
  temporarily blocked") is not swept up. If the Hub rewords the header
  past this text the (reworded) line simply reappears — we err toward the
  papercut returning, never toward silence.
* **Attached to the exact emitting logger.** A filter on a *parent* logger
  is not consulted for a child logger's records during propagation
  (verified empirically), so filtering ``huggingface_hub`` would not work
  — we target ``huggingface_hub.utils._http`` directly. A future module
  rename makes the filter stop matching (advisory returns) rather than
  crash.

Stdlib-only (``logging``) so it stays cheap to import from the otherwise
self-contained download gate. Idempotent and thread-safe: installing more
than once, even concurrently, adds exactly one filter.
"""

from __future__ import annotations

import logging
import threading

# The logger huggingface_hub's ``_warn_on_warning_headers`` emits from:
# ``logging.get_logger(__name__)`` where ``__name__`` is this module.
_HF_HTTP_LOGGER = "huggingface_hub.utils._http"

# The full advisory text, whitespace-normalised and lower-cased, with the
# leading "You are sending " and the trailing period intentionally dropped
# so a server variant that adds/removes either still matches. We require
# this whole string — subject AND the "set a HF_TOKEN … faster downloads"
# advice — so a *different* actionable warning sharing only the subject is
# never suppressed. A reword past this text is fail-open (advisory returns).
_KNOWN_ADVISORY = (
    "unauthenticated requests to the hf hub. please set a hf_token to "
    "enable higher rate limits and faster downloads"
)


def _normalize(text: str) -> str:
    """Lower-case and collapse runs of whitespace, so incidental spacing or
    a wrapped line in the header can't defeat the match."""
    return " ".join(text.lower().split())


# Marker attribute so a re-install can spot our own filter and not add a
# duplicate (the filter class name alone isn't a reliable identity across
# reloads).
_FILTER_TAG = "_rapid_mlx_hf_unauth_filter"

# Serialises the check-then-add in :func:`silence_hf_unauthenticated_warning`
# so concurrent first downloads (the mirror runs a worker pool) can't each
# slip a duplicate filter past the idempotency check.
_INSTALL_LOCK = threading.Lock()


class _DropUnauthenticatedAdvisory(logging.Filter):
    """Drop HF's "unauthenticated requests … set a HF_TOKEN" advisory.

    Returns ``False`` (drop) only for that specific ``WARNING``-level
    server advisory; every other record — a higher severity, or a
    different message — passes untouched, so a real failure is never
    masked.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Never touch anything more severe than the advisory itself.
        if record.levelno != logging.WARNING:
            return True
        try:
            message = _normalize(record.getMessage())
        except Exception:
            # A record we can't even render is not ours to suppress.
            return True
        return _KNOWN_ADVISORY not in message


def silence_hf_unauthenticated_warning() -> None:
    """Install the fail-safe advisory filter on the HF http logger.

    Idempotent, thread-safe and cheap — safe to call at the top of every
    download entry point. It is called from :mod:`rapid_mlx._download_gate`
    (the cold-pull size probe) and :mod:`rapid_mlx._mirror` (the R2/HF
    downloader) right before their first Hub request. Because the advisory
    is emitted warn-once per process, installing at those two chokepoints
    covers every flow that acquires a model through the gate or the
    mirror — which, on a cold first run, is all of them.
    """
    logger = logging.getLogger(_HF_HTTP_LOGGER)
    with _INSTALL_LOCK:
        if any(getattr(f, _FILTER_TAG, False) for f in logger.filters):
            return
        filt = _DropUnauthenticatedAdvisory()
        setattr(filt, _FILTER_TAG, True)
        logger.addFilter(filt)
