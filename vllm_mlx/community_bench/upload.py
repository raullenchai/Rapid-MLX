# SPDX-License-Identifier: Apache-2.0
"""HTTP submission for community benchmarks.

Why this exists
---------------
``submit_interactive`` used to publish a run by committing a JSON file into
a checkout of Rapid-MLX and opening a pull request. That gated the whole
feature on two things a normal user does not have:

  1. the cwd being a git repository root, and
  2. a remote pointing at ``raullenchai/Rapid-MLX`` (or a fork + upstream)

Anyone who installed via ``pip install rapid-mlx`` or ``brew install
rapid-mlx`` has no checkout at all, so ``--submit`` exited before it ever
collected a number. That is the entire PyPI/Homebrew install base with no
path in — and it is why the corpus stalled in the low tens of rows.

This module replaces that with one POST. Standard library only: adding a
dependency to the submission path would be its own adoption tax.

Identity
--------
``install_id`` is a random 12-hex value minted once per install and kept in
``~/.rapid-mlx/bench-install-id``. It is used for the server's per-install
daily cap and so a contributor can find their own rows.

It is deliberately NOT derived from any hardware identifier. oMLX computes
its ``owner_hash`` from ``IOPlatformUUID``, which makes every public row
traceable to a specific machine; a random per-install value gives the same
rate-limiting and self-identification without that property. Deleting the
file resets it, which is the user's escape hatch.

It is also deliberately NOT the telemetry client id. Benchmark rows are
public and telemetry is not; sharing one identifier between them would let
anyone correlate a public board entry with a private telemetry stream.
"""

from __future__ import annotations

import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BOARD_URL = "https://rapidmlx.com/api/benchmarks"

#: Override for local development and tests. Never read in normal use.
BOARD_URL_ENV = "RAPID_MLX_BENCH_BOARD_URL"

_TIMEOUT_S = 20.0
_MAX_ATTEMPTS = 3
#: Base for exponential backoff between retries. Retrying a transient
#: outage instantly just spends the whole budget inside the same failure
#: window and adds load to an already-unhealthy service.
_BACKOFF_BASE_S = 1.5


class SubmitError(RuntimeError):
    """Raised when the board refused or could not be reached."""


def board_url() -> str:
    """Resolve the submission endpoint.

    The override exists for local development, but the consent screen promises
    an *HTTPS* POST — so a plain-http target would send the payload and the
    persistent install id in clear text while telling the user otherwise.
    Loopback is exempt because that is what a local worker actually is.
    """
    override = os.environ.get(BOARD_URL_ENV, "").strip()
    if not override:
        return DEFAULT_BOARD_URL
    parsed = urllib.parse.urlparse(override)
    # The resolved target is printed on the consent screen and repeated in
    # error messages, so embedded credentials would land in terminals and CI
    # logs. Refuse rather than redact: a URL needing userinfo to reach the
    # board is not a configuration we want to support silently.
    if parsed.username or parsed.password:
        raise SubmitError(
            f"{BOARD_URL_ENV} must not embed credentials in the URL — the "
            f"destination is displayed and logged."
        )
    if parsed.scheme == "https":
        return override
    if parsed.scheme == "http" and (parsed.hostname or "") in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        return override
    raise SubmitError(
        f"{BOARD_URL_ENV} must be an https:// URL (or http:// on loopback for "
        f"local development); got {override!r}. Refusing to send your "
        f"submission in clear text."
    )


def _install_id_path() -> Path:
    root = os.environ.get("RAPID_MLX_HOME", "").strip()
    base = Path(root) if root else Path.home() / ".rapid-mlx"
    return base / "bench-install-id"


def peek_install_id() -> str:
    """Return this install's id WITHOUT creating anything on disk.

    Split from persistence deliberately. The consent screen shows the exact
    payload and promises that nothing has been written yet — so the id has to
    be knowable before the user says yes, and only committed after.

    Never raises: an unreadable config dir must not block a submission.
    """
    try:
        existing = _install_id_path().read_text().strip()
        if _valid_id(existing):
            return existing
    # UnicodeDecodeError is a ValueError, not an OSError: a corrupted or
    # binary id file would otherwise crash the whole submission on read.
    except (OSError, UnicodeError):
        pass
    return secrets.token_hex(6)


def commit_install_id(candidate: str) -> str:
    """Persist ``candidate`` unless this install already has an id.

    Returns the id that is now on disk — which may be another process's if it
    won the race. Callers must use the return value, not their candidate.

    The write is a fully-written temp file plus ``os.link``, not ``O_EXCL`` on
    the destination: link is atomic and the file only ever becomes visible
    already containing an id, so a concurrent reader can never observe an
    empty file and mistake it for corruption.

    Never raises. An unwritable home costs a stable identity, not a
    submission; the only consequence is that the server counts this run
    against a fresh per-install bucket.
    """
    path = _install_id_path()
    try:
        existing = path.read_text().strip()
        if _valid_id(existing):
            return existing
        malformed = True
    except (OSError, UnicodeError):
        malformed = path.exists()

    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, (candidate + "\n").encode())
        finally:
            os.close(fd)

        if malformed:
            # Garbage on disk: replace it. Two processes both repairing will
            # disagree on the id for this one run and agree from the next one
            # onwards. Rarer and less harmful than the alternative, which is
            # never repairing and re-minting on every single submission.
            os.replace(tmp, path)
            tmp = None
            return candidate
        try:
            os.link(tmp, path)
            return candidate
        except FileExistsError:
            winner = path.read_text().strip()
            return winner if _valid_id(winner) else candidate
    except (OSError, UnicodeError):
        return candidate
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _valid_id(value: str) -> bool:
    return len(value) == 12 and all(c in "0123456789abcdef" for c in value)


def install_id() -> str:
    """Read-or-create in one call. Kept for callers that do not need the split."""
    return commit_install_id(peek_install_id())


def new_run_group() -> str:
    """Mint an id linking the arms of one A/B so the board can pair them."""
    return secrets.token_hex(6)


def _sleep_before_retry(attempt: int, headers) -> None:
    """Back off before the next attempt, honouring ``Retry-After``.

    Capped: a server asking us to wait an hour should not hang a CLI the
    user is watching — we would rather fail and let them rerun.
    """
    delay = _BACKOFF_BASE_S * (2 ** (attempt - 1))
    if headers is not None:
        raw = headers.get("Retry-After") if hasattr(headers, "get") else None
        if raw:
            try:
                delay = max(delay, float(str(raw).strip()))
            except (TypeError, ValueError):
                pass
    time.sleep(min(delay, 10.0))


def post_submission(payload: dict, *, url: str | None = None) -> dict:
    """POST one submission. Returns the decoded server response.

    Retries only on transport errors and 5xx — a 4xx means the payload is
    wrong and retrying would just replay the same rejection. ``429`` is not
    retried either: the caller is over a documented cap, and hammering it is
    the opposite of what the response is asking for.
    """
    target = url or board_url()
    body = json.dumps(payload).encode("utf-8")
    last: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        req = urllib.request.Request(
            target,
            data=body,
            method="POST",
            headers={
                "content-type": "application/json",
                "user-agent": "rapid-mlx-bench",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                status = resp.status
                raw = resp.read().decode("utf-8", "replace")
            # A 2xx carrying something other than a JSON object is not an
            # accepted submission — it is a proxy, a captive portal or a CDN
            # error page. Reporting it as success would tell a contributor
            # their run is on the board when it never arrived.
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                raise SubmitError(
                    f"the board returned HTTP {status} but the body was not "
                    f"JSON ({raw[:120]!r}). Your run was NOT submitted."
                ) from None
            if not isinstance(decoded, dict):
                raise SubmitError(
                    f"the board returned JSON that is not an object "
                    f"({type(decoded).__name__}). Your run was NOT submitted."
                )
            # A 2xx is the transport saying "I delivered it", not the board
            # saying "I accepted it". Without this check a body like
            # ``{"ok": false, "error": "rejected"}`` prints "Accepted" and
            # exits 0, which is the worst possible failure mode: the
            # contributor believes their run is on the board.
            if decoded.get("ok") is not True:
                why = decoded.get("error") or decoded.get("field") or "no reason given"
                raise SubmitError(
                    f"the board did not accept this submission ({why}). "
                    f"Your run was NOT submitted."
                )
            return decoded
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # noqa: BLE001 - diagnostic only
                pass
            if exc.code < 500 or attempt == _MAX_ATTEMPTS:
                raise SubmitError(
                    f"the board rejected this submission (HTTP {exc.code}). {detail}"
                ) from exc
            last = exc
            _sleep_before_retry(attempt, getattr(exc, "headers", None))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == _MAX_ATTEMPTS:
                # No persistence claim here: whether a local copy exists is
                # something only the caller knows, and asserting it from this
                # layer told users their run was safe when archiving had
                # already failed and warned otherwise.
                raise SubmitError(f"could not reach {target}: {exc}") from exc
            last = exc
            _sleep_before_retry(attempt, None)

    raise SubmitError(str(last))


__all__ = [
    "DEFAULT_BOARD_URL",
    "BOARD_URL_ENV",
    "SubmitError",
    "board_url",
    "commit_install_id",
    "install_id",
    "peek_install_id",
    "new_run_group",
    "post_submission",
]
