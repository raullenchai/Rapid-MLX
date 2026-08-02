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
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BOARD_URL = "https://rapidmlx.com/api/benchmarks"

#: Override for local development and tests. Never read in normal use.
BOARD_URL_ENV = "RAPID_MLX_BENCH_BOARD_URL"

_TIMEOUT_S = 20.0
_MAX_ATTEMPTS = 3


def board_url() -> str:
    """Resolve the submission endpoint."""
    return os.environ.get(BOARD_URL_ENV, "").strip() or DEFAULT_BOARD_URL


def _install_id_path() -> Path:
    root = os.environ.get("RAPID_MLX_HOME", "").strip()
    base = Path(root) if root else Path.home() / ".rapid-mlx"
    return base / "bench-install-id"


def install_id() -> str:
    """Return this install's random id, minting one on first use.

    Never raises: an unwritable home directory must not block a submission,
    so we fall back to an ephemeral value. The only consequence is that the
    server counts this run against a fresh per-install bucket.
    """
    path = _install_id_path()
    try:
        existing = path.read_text().strip()
        if len(existing) == 12 and all(c in "0123456789abcdef" for c in existing):
            return existing
    except OSError:
        pass
    fresh = secrets.token_hex(6)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fresh + "\n")
        # 0600: it is not a secret, but it is an identifier, and a
        # world-readable one invites correlation we just went to lengths
        # to prevent.
        os.chmod(path, 0o600)
    except OSError:
        pass
    return fresh


def new_run_group() -> str:
    """Mint an id linking the arms of one A/B so the board can pair them."""
    return secrets.token_hex(6)


class SubmitError(RuntimeError):
    """Raised when the board refused or could not be reached."""


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
                raw = resp.read().decode("utf-8", "replace")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"ok": True, "raw": raw[:500]}
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
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == _MAX_ATTEMPTS:
                raise SubmitError(
                    f"could not reach {target}: {exc}. Your run was still saved "
                    f"locally — see the path printed above."
                ) from exc
            last = exc

    raise SubmitError(str(last))


__all__ = [
    "DEFAULT_BOARD_URL",
    "BOARD_URL_ENV",
    "SubmitError",
    "board_url",
    "install_id",
    "new_run_group",
    "post_submission",
]
