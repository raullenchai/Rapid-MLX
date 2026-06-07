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

- **HTTPS except loopback dev overrides.** The endpoint is overridable
  via ``RAPID_MLX_TELEMETRY_ENDPOINT`` for debug rigs and local Worker
  dev. The exact rule (round 17 codex catch -- the prior "HTTPS-only"
  shorthand made the loopback exemption look like a bug):
    * Public hosts -- HTTPS required, no exceptions.
    * Loopback hosts (``localhost`` / ``127.0.0.1`` / ``::1``) --
      plain HTTP is accepted because ``wrangler dev`` serves over
      ``http://127.0.0.1:8787`` and the bytes never leave the host.
    * Anything else (non-loopback host on HTTPS that isn't the
      production default) -- rejected by ``endpoint()`` returning
      ``None`` so ``post_batch`` fails closed.
  Mirrors the share PR #504 codex round 4 fix where ``URLError`` was
  discovered NOT to be a ``TimeoutError`` subclass; we keep both in
  the except tuple.

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


def endpoint() -> str | None:
    """Resolve the production endpoint, honouring the debug env override.

    Resolved every call (not cached at import time) so tests can
    monkey-patch ``os.environ`` per case without poisoning each other.

    Round 3 codex review caught that an unrestricted override
    (``RAPID_MLX_TELEMETRY_ENDPOINT=https://attacker.example/`` set in
    a user's shell rc or by a malicious wrapper script) would silently
    redirect every opted-in user's events to a hostile collector, with
    the privacy guarantees of the disclosure ("only our Worker hashes
    your IP, etc.") no longer applying. The override is restricted to
    localhost / 127.0.0.1 -- the only legitimate use cases are
    ``wrangler dev`` and CI fixtures.

    Round 16 codex review: when the env var IS set but rejected (e.g.
    a dev typo on the wrangler URL, or a stale shell rc pointing at a
    decommissioned host), the previous behaviour silently fell back to
    the production endpoint -- which meant dev/test traffic could leak
    into the production R2 bucket without the user noticing. Now: a
    set-but-rejected override returns ``None`` so ``post_batch`` fails
    closed (drops the batch). Default (no env var at all) still
    returns ``DEFAULT_ENDPOINT``.
    """
    raw = os.environ.get(ENDPOINT_ENV)
    if raw is None:
        return DEFAULT_ENDPOINT
    if _is_localhost_override(raw):
        return raw
    return None


def _is_localhost_override(url: str) -> bool:
    """True iff ``url`` targets localhost — the only allowed override.

    Accepts ``http://`` for localhost (wrangler dev defaults to plain
    HTTP on 127.0.0.1:8787). Anything else — including a public host
    accidentally typed as ``https://localhost.attacker.example`` — is
    rejected. The match is on the netloc, not a substring.

    ``parts.port`` is touched explicitly to surface a malformed-port
    ``ValueError`` here rather than later inside ``Request``/``urlopen``
    where it would escape the ``URLError``/``TimeoutError``/``OSError``
    catch path and violate the "never raises" contract. Round 4 codex
    review caught ``http://localhost:bad/`` as the concrete example.
    """
    try:
        from urllib.parse import urlparse

        parts = urlparse(url)
        # Force port validation here; ``parts.port`` raises ValueError
        # on a non-integer port. We don't care about the value — just
        # that it parses cleanly.
        _ = parts.port
    except (TypeError, ValueError):
        return False
    if parts.scheme not in ("http", "https"):
        return False
    host = (parts.hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1")


def debug_enabled() -> bool:
    raw = os.environ.get(DEBUG_ENV)
    if raw is None:
        return False
    return raw.strip().lower() not in ("0", "", "false", "no", "off")


def _user_agent() -> str:
    """Self-identifying UA for the telemetry client.

    Cloudflare's zone-level bot manager rejects the stdlib default
    ``Python-urllib/3.X`` UA with HTTP 403 before the request ever
    reaches our Worker (verified live 2026-06-06 against the production
    endpoint). Setting a non-generic UA is therefore load-bearing for
    the pipeline to function at all.

    The UA exposes only the package name + version, which is already
    in the payload's ``rapid_mlx_version`` field — no new PII. The
    repository link is conventional courtesy so an analytics consumer
    on the receiver side can attribute hits without guessing.
    """
    try:
        from importlib.metadata import version

        v = version("rapid-mlx")
    except Exception:
        v = "dev"
    return f"rapid-mlx/{v} (+https://github.com/raullenchai/Rapid-MLX)"


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

    # Round 3 codex review: ``json.dumps`` outside the try-catch
    # violates the "never raises" contract — a non-JSON-serializable
    # payload would have surfaced as a ``TypeError`` to the caller. The
    # transport must absorb everything that isn't a system exception.
    try:
        body = json.dumps({"batch": events}, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as e:
        _log(f"refusing payload that failed to serialize: {type(e).__name__}: {e}")
        return False
    if len(body) > MAX_BODY_BYTES:
        # Worker would 413 us; drop locally to save a roundtrip and an
        # entry in the collector's reject metric.
        _log(f"payload too large ({len(body)} > {MAX_BODY_BYTES} B) — dropped")
        return False

    url = endpoint()
    if url is None:
        # Round 16 codex catch: ``RAPID_MLX_TELEMETRY_ENDPOINT`` was
        # set but rejected by ``endpoint()`` (non-localhost host /
        # malformed URL). The previous fallback to the production
        # endpoint could leak dev/test traffic into the production R2
        # bucket invisibly. Fail closed: drop the batch and log so the
        # operator can see why nothing landed.
        _log(
            f"{ENDPOINT_ENV} was set but rejected -- dropping batch "
            f"(set a localhost / 127.0.0.1 / ::1 URL or unset the env "
            f"var to use the production endpoint)"
        )
        return False
    # HTTPS-only for any non-loopback host. Loopback is exempt because
    # ``endpoint()`` already restricted the override to localhost (round
    # 3 codex catch) and ``wrangler dev`` serves over plain HTTP -- the
    # traffic never leaves the host. A public host on ``http://`` is
    # still refused.
    if not (url.startswith("https://") or _is_localhost_override(url)):
        _log(f"refusing non-HTTPS endpoint {url!r}")
        return False

    attempts = (*RETRY_BACKOFFS_S, None)  # backoff is None on the last attempt
    for attempt_idx, backoff in enumerate(attempts, start=1):
        try:
            req = Request(
                url,
                data=body,
                method="POST",
                headers={
                    "content-type": "application/json",
                    "user-agent": _user_agent(),
                },
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
                # Be honest about the next move — round 4 codex review
                # caught that we logged ``"will retry"`` even on the
                # last attempt when the next thing was ``return False``.
                tail = "will retry" if backoff is not None else "giving up"
                _log(f"attempt {attempt_idx} → {status} (5xx, {tail})")
        except HTTPError as e:
            # HTTPError is a URLError subclass but carries the response
            # code; treat it the same as a status-based decision above.
            # ``HTTPError`` ALSO holds a file-like response object on
            # ``e.fp`` (and is itself an addinfourl). Round 2 codex
            # review caught that letting it fall out of scope without
            # an explicit close leaks the underlying socket across
            # retries on platforms where the gc cycle is slow.
            try:
                status = getattr(e, "code", 0)
                if 400 <= status < 500:
                    _log(
                        f"attempt {attempt_idx} → HTTPError {status} "
                        "(4xx, not retrying)"
                    )
                    return False
                tail = "will retry" if backoff is not None else "giving up"
                _log(f"attempt {attempt_idx} → HTTPError {status} ({tail})")
            finally:
                close = getattr(e, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        except (URLError, TimeoutError, OSError) as e:
            # URLError is NOT a TimeoutError subclass in 3.10+ — pin both
            # explicitly. ``socket.timeout`` was historically distinct
            # but became an alias of ``TimeoutError`` in 3.10, so the
            # bare ``TimeoutError`` catches both. ``OSError`` catches a
            # DNS lookup failure on some platforms.
            _log(f"attempt {attempt_idx} failed: {type(e).__name__}: {e}")
        except (KeyboardInterrupt, SystemExit):
            # User intent and programmatic intent always win. Telemetry
            # must not turn a Ctrl-C into a hang.
            raise
        except Exception as e:
            # Final guard for the "never raises" contract. Round 5
            # codex review caught that ``Request(...)`` raises
            # ``ValueError`` for control bytes and that some upstream
            # platforms surface URL-malformed-ness as
            # ``http.client.InvalidURL`` (an ``HTTPException`` subclass,
            # NOT a ``URLError``). Treat anything else as a transport
            # failure: log, drop, do not retry — retrying a malformed
            # URL just generates the same error.
            _log(f"attempt {attempt_idx} failed (unexpected): {type(e).__name__}: {e}")
            return False

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
