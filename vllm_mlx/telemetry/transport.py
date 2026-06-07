# SPDX-License-Identifier: Apache-2.0
"""HTTPS POST transport for the telemetry collector.

This module owns one job: take a list of payload dicts, ship them to
the collector at ``telemetry.rapidmlx.com``, and fail silently if
anything goes wrong. Nothing about consent, schema construction, or
queueing belongs here.

Implementation choices, called out because they look unusual:

- **stdlib ``urllib`` over ``httpx`` / ``requests``.** Telemetry sits
  on a consent-gated path that runs only when the user explicitly opts
  in, so it must not introduce a transitive dependency that increases
  install footprint or surfaces an upgrade pin. ``urllib`` is in every
  CPython 3.10+ and is already imported elsewhere in this codebase.

- **HTTPS-only.** The endpoint is overridable via
  ``RAPID_MLX_TELEMETRY_ENDPOINT`` for debug rigs and local Worker dev,
  but we refuse anything not on ``https://``. This mirrors the share
  PR #504 codex round 4 fix where ``URLError`` was discovered NOT to
  be a ``TimeoutError`` subclass; we keep both in the except tuple.

- **Three attempts with two backoffs.** Telemetry must never block the
  foreground for more than ~3 × ``TIMEOUT_S`` + the backoffs (worst
  case ~12 s in pathological network conditions). The retries cover a
  transient DNS hiccup, a one-shot Worker rolling-restart, and a TCP
  connection reset — not a multi-minute outage.

- **Silent failure.** A failed POST returns ``False``; nothing else
  signals up to the caller. The user must not see a stack trace from
  the telemetry path, ever.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_ENDPOINT = "https://telemetry.rapidmlx.com/v1/events"
DEBUG_ENV = "RAPID_MLX_TELEMETRY_DEBUG"
ENDPOINT_ENV = "RAPID_MLX_TELEMETRY_ENDPOINT"

TIMEOUT_S = 3.0
RETRY_BACKOFFS_S: tuple[float, ...] = (0.5, 2.0)
MAX_BODY_BYTES = 200 * 1024  # Worker caps at 256 KB; leave headroom for envelope.


def endpoint() -> str:
    """Resolve the production endpoint, honouring the debug env override.

    Resolved every call (not cached at import time) so tests can
    monkey-patch ``os.environ`` per case without poisoning each other.
    """
    return os.environ.get(ENDPOINT_ENV, DEFAULT_ENDPOINT)


def debug_enabled() -> bool:
    raw = os.environ.get(DEBUG_ENV)
    if raw is None:
        return False
    return raw.strip().lower() not in ("0", "", "false", "no", "off")


def post_batch(events: list[dict[str, Any]]) -> bool:
    """Send a batch of payloads to the collector.

    Returns ``True`` on any 2xx response; ``False`` on transport error,
    schema-rejected (4xx), or oversized payload. Never raises.

    Empty batches return ``True`` without hitting the network — empty
    flushes are cheap and harmless, and treating them as success keeps
    the caller logic free of empty-list special cases.
    """
    if not events:
        return True

    body = json.dumps({"batch": events}, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_BODY_BYTES:
        # Worker would 413 us; drop locally to save a roundtrip and an
        # entry in the collector's reject metric.
        _log(f"payload too large ({len(body)} > {MAX_BODY_BYTES} B) — dropped")
        return False

    url = endpoint()
    if not url.startswith("https://"):
        _log(f"refusing non-HTTPS endpoint {url!r}")
        return False

    attempts = (*RETRY_BACKOFFS_S, None)  # backoff is None on the last attempt
    for attempt_idx, backoff in enumerate(attempts, start=1):
        try:
            req = Request(
                url,
                data=body,
                method="POST",
                headers={"content-type": "application/json"},
            )
            with urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310 — URL is constant.
                status = getattr(resp, "status", None)
                if status is None:
                    status = resp.getcode()
                if 200 <= status < 300:
                    _log(
                        f"attempt {attempt_idx} → {status} "
                        f"(events={len(events)}, bytes={len(body)})"
                    )
                    return True
                if 400 <= status < 500:
                    # Schema bug or auth/format issue — retrying will not
                    # change the answer. Drop and move on.
                    _log(f"attempt {attempt_idx} → {status} (4xx, not retrying)")
                    return False
                _log(f"attempt {attempt_idx} → {status} (5xx, will retry)")
        except HTTPError as e:
            # HTTPError is a URLError subclass but carries the response
            # code; treat it the same as a status-based decision above.
            status = getattr(e, "code", 0)
            if 400 <= status < 500:
                _log(f"attempt {attempt_idx} → HTTPError {status} (4xx, not retrying)")
                return False
            _log(f"attempt {attempt_idx} → HTTPError {status} (will retry)")
        except (URLError, TimeoutError, OSError) as e:
            # URLError is NOT a TimeoutError subclass in 3.10+ — pin both
            # explicitly. ``socket.timeout`` was historically distinct
            # but became an alias of ``TimeoutError`` in 3.10, so the
            # bare ``TimeoutError`` catches both. ``OSError`` catches a
            # DNS lookup failure on some platforms.
            _log(f"attempt {attempt_idx} failed: {type(e).__name__}: {e}")

        if backoff is None:
            return False
        time.sleep(backoff)

    return False


def _log(msg: str) -> None:
    """Debug-only logger.

    Silent unless ``RAPID_MLX_TELEMETRY_DEBUG`` is set to a truthy
    value. Goes to stderr so it does not interleave with subcommand
    output that may be JSON-piped.
    """
    if not debug_enabled():
        return
    print(f"[telemetry] {msg}", file=sys.stderr, flush=True)
