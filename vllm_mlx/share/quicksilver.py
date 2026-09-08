# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx share <alias> --quicksilver`` — QuickSilver compute-pool node.

Turns this machine into a paid inference node in the QuickSilver pool
(provider spec v1, 2026-09-07 — sections cited inline). Lifecycle:

  1. Resolve the QuickSilver catalog id from the typed alias (§5.4).
  2. First run: register against ``pay.quicksilverpro.io`` with a
     one-time provider key (``qsppk-``), cache the server-minted
     share-key (``qspsk-``) + node id; subsequent runs read the cache
     and never see the provider key (§1, §5.1, §5.2).
  3. Spawn the loopback serve (default ``--max-num-seqs 2``), warm it
     up, and open the keyed WS tunnel (§5.3) — the claim key rides the
     upgrade ``Authorization`` header, never the URL or the ``ready``
     frame, so no credential can leak through exception reprs or
     websockets logs (§6).
  4. Heartbeat every 10 s while the tunnel is up (§3.3), reconnecting
     the tunnel with backoff on drops (§5.5).

Credential rules (§1, §6 — the reason this is its own module):

  * ``qsppk-`` provider key: memory only, one register call, never
    written, never logged, never in argv of any spawned process.
  * ``qspsk-`` share-key: at rest only in the 0600 cache file; every
    string that could echo it (exception reprs, relay error bodies)
    goes through :func:`_redact` before reaching stdout/stderr.
  * The loopback bearer share mints for the child serve is replaced at
    the tunnel boundary: pool traffic bears a QuickSilver credential
    the local serve has never seen (§6's "per-request auth is not the
    node's job" only works if the tunnel translates).
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import logging
import math
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import ws_tunnel

log = logging.getLogger(__name__)

# Registration / account host (§3). ``--quicksilver-api`` overrides for
# testing against a staging pay.* (or a local fake in tests).
DEFAULT_PAY_API = "https://pay.quicksilverpro.io"

# Environment fallback for the provider key (§5.2). Deliberately NOT
# named RAPID_MLX_* — this credential belongs to QuickSilver's account
# system; the name follows the spec so cross-tool docs stay grep-able.
PROVIDER_KEY_ENV_VAR = "QUICKSILVER_PROVIDER_KEY"

# Catalog id → the local alias whose weights fit the pool listing (§5.4).
# The QuickSilver catalog namespace and the rapid-mlx alias namespace
# do not line up 1:1 (catalog ids carry no quant suffix; local aliases
# do), so ``share nemotron-3.5-lightning --quicksilver`` resolves both
# directions through this table. Unknown catalog ids fall through to
# the server, which is authoritative (422 on garbage — §5.4).
CATALOG_DEFAULT_ALIAS: dict[str, str] = {
    "nemotron-3.5-lightning": "nemotron-3.5-lightning-30b-4bit",
    "qwen3.8-27b": "qwen3.8-27b-4bit",
    "qwen3.6-35b": "qwen3.6-35b",
}
ALIAS_TO_CATALOG: dict[str, str] = {
    alias: catalog for catalog, alias in CATALOG_DEFAULT_ALIAS.items()
}

# §3.1 error taxonomy: terminal codes surface + exit non-zero; the rest
# (429 / 5xx / network) retry with capped exponential backoff.
_TERMINAL_REGISTER_CODES = frozenset({401, 403, 409, 422})
_RETRY_BASE_SECONDS = 1.0
_RETRY_MAX_SECONDS = 30.0
# A transient pay.* outage delays the node; a sustained one must still
# surface non-zero (or exit under launchd) instead of wedging silently.
_RETRY_BUDGET_SECONDS = 300.0
# Keep the retry sleeper injectable without replacing ``time.sleep`` process-wide.
# Mocking the stdlib module function also intercepts subprocess' polling sleeps on
# Python 3.12, making the registration tests depend on interpreter internals.
_register_retry_sleep = time.sleep
_supervisor_sleep = time.sleep

# Warm-up request timeout (§5.3 step 3). Weights are already resident
# (healthz passed), so this only pages Metal working sets in — seconds
# on healthy hardware. 300 s is the ceiling for a cold, contended
# disk; past that it's a warning, not an abort.
_WARMUP_TIMEOUT_SECONDS = 300.0

# §3.2/§5.5: a rejected keyed claim must never be retried — the key is
# revoked/rotated server-side and spinning just hammers the relay.
_WS_TERMINAL_STATUS = 401
# The claim key rides the upgrade's Authorization header, so a revoked
# key is normally rejected as an HTTP 401 AT the upgrade
# (``_WS_TERMINAL_STATUS`` above). This close-code set is the defensive
# fallback for a relay that instead accepts the socket and then drops
# the bad claim: RFC 6455 reserves 1008 (Policy Violation) for exactly
# that. Both signals hit the same no-spin terminal exit.
_WS_TERMINAL_CLOSE_CODES = frozenset({1008})
# Heuristic LOUDNESS threshold, not a kill switch: a session the relay
# drops seconds after accepting the keyed claim, five times running
# with no 401/1008 anywhere, gets one prominent warning (likely
# relay-side instability — an actual rejection now always carries a
# code). Killing the node on this pattern would misclassify a relay
# incident as a revoked credential.
_FAST_REJECT_WINDOW_SECONDS = 5.0
_FAST_REJECT_STREAK_LIMIT = 5

# §3.1 response fields the client must be able to act on. ``alias`` is
# client-side sugar (§5.1) and ``heartbeat_interval_s`` has a spec
# default of 10, so neither is hard-required on the wire.
_WIRE_REQUIRED_KEYS = ("node_id", "share_key", "model", "relay_url", "heartbeat_url")
_CACHE_REQUIRED_KEYS = _WIRE_REQUIRED_KEYS + ("heartbeat_interval_s", "alias")
# Everything a cache may hold. The registration response is server-
# controlled JSON; persisting it verbatim could write an echoed
# provider key or an unexpected token to disk, breaking §1's "the
# provider key is never written down". api_base records the origin the
# credential was minted against so --install-service and later cached
# runs keep validating against the RIGHT origin (a staging-registered
# node restarted against the default origin would reject its own
# cache).
_CACHE_ALLOWED_KEYS = _CACHE_REQUIRED_KEYS + ("payout_account", "api_base", "worker")

# Socket timeout for one §3.3 beat. Bounded so ``_Heartbeat.stop()``
# can join the thread faster than a hung heartbeat can block; the beat
# is best-effort (a missed one never drops the node) so a short ceiling
# costs nothing.
_BEAT_REQUEST_TIMEOUT = 5.0


class QuickSilverError(Exception):
    """Fatal, user-facing failure. Message must be redacted by the
    printer (:func:`run_share` catches and scrubs) — never trust an
    already-rendered string from deeper down."""


# ───────────────────────────── redaction (§6) ─────────────────────────────

# Process-local registry of live secret strings. ``websockets``
# exception reprs and relay error bodies go through :func:`_redact`
# before reaching any sink — §8's "no key string in stdout/logs" needs
# one choke point, not per-callsite discipline.
_SECRETS: list[str] = []


def _register_secret(value: str) -> None:
    if value:
        _SECRETS.append(value)


def _redact(text: str) -> str:
    for secret in _SECRETS:
        text = text.replace(secret, "[redacted]")
    return text


def _user_agent() -> str:
    # §3.1: the UA is REQUIRED — Cloudflare 403s bare urllib/python UAs.
    try:
        from vllm_mlx import __version__

        return f"rapid-mlx/{__version__}"
    except Exception:  # pragma: no cover — metadata missing in weird venvs
        return "rapid-mlx/unknown"


# ───────────────────────────── cache (§5.1) ─────────────────────────────


def _cache_dir() -> Path:
    d = Path.home() / ".rapid-mlx" / "quicksilver"
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def _cache_path(catalog_id: str) -> Path:
    # Cache files are named by catalog id (§5.1); the id still reaches
    # us from a user flag, so refuse anything path-shaped.
    if not catalog_id.replace("-", "").replace(".", "").replace("_", "").isalnum():
        raise QuickSilverError(f"invalid model id for cache file: {catalog_id!r}")
    return _cache_dir() / f"{catalog_id}.json"


def _load_cache(
    catalog_id: str, alias: str, worker: str | None = None
) -> dict[str, Any] | None:
    """Return a cached §3.1 response iff it is complete AND bound to
    the same local alias. Registration is idempotent per
    (account, model, alias) but the cache filename keys only the
    catalog id (§5.1) — an alias mismatch means a *different* node, and
    reuse would silently mis-bind payouts. Mismatch (or corruption) is
    treated as "no cache", which routes through registration."""
    # ``_cache_path`` CREATES (and chmods) the cache dir — a read-only
    # HOME or full disk must surface as the redacted actionable error,
    # not silently route through registration (which would prompt for a
    # provider key the operator may not have at hand, and then fail to
    # save anyway).
    try:
        path = _cache_path(catalog_id)
    except OSError as exc:
        raise QuickSilverError(
            f"could not access the node cache directory: {exc}"
        ) from None
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        # EACCES/EIO on an EXISTING cache is not "no cache" — treating
        # it that way starts a re-registration the operator never
        # asked for. Surface it.
        raise QuickSilverError(f"could not read the node cache: {exc}") from None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None  # corrupt file IS properly "no cache": re-register repairs it
    if not isinstance(payload, dict):
        return None
    # Presence is not shape: a non-string share_key would sail through
    # _register_secret and detonate _redact (str.replace of a non-str
    # needle) on the first render. heartbeat_interval_s is the one
    # non-string field and _resolve_heartbeat_interval gates it.
    if any(
        not isinstance(payload.get(k), str) or not payload[k]
        for k in _WIRE_REQUIRED_KEYS + ("alias",)
    ):
        return None
    if payload.get("heartbeat_interval_s") is None:
        return None
    if payload.get("alias") != alias or payload.get("model") != catalog_id:
        return None
    # Node identity is (account, worker): a worker mismatch (or an old cache
    # with no worker) means a DIFFERENT node — re-register rather than reuse a
    # stale node id under a new machine label.
    if worker is not None and payload.get("worker") != worker:
        return None
    return payload


def _save_cache(catalog_id: str, payload: dict[str, Any]) -> Path:
    path = _cache_path(catalog_id)
    tmp = path.with_name(f"{path.name}.tmp-{secrets.token_hex(4)}")
    # Whitelist, not passthrough — a server response carrying an extra
    # "provider_key_echo"-style field must never survive to disk.
    allowed = {k: payload[k] for k in _CACHE_ALLOWED_KEYS if k in payload}
    data = json.dumps(allowed, indent=2, sort_keys=True).encode("utf-8")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        path.chmod(0o600)
    except OSError as exc:
        # A failed replace/write can leave a second copy of the
        # share-key on disk as the .tmp file — remove it before the
        # error even unwinds, then take the redacted exit path.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise QuickSilverError(
            f"could not write node cache for {catalog_id}: {exc}"
        ) from None
    return path


# ─────────────────────────── catalog resolution (§5.4) ───────────────────────────


def _resolve_worker(args: argparse.Namespace) -> str:
    """The per-machine worker label (Stratum's account.worker). Defaults to the
    machine hostname so two machines under one account register as DISTINCT
    nodes (node identity is (account, worker)); an explicit ``--worker`` wins.
    Sanitized to the id charset; the server sanitizes again and is authoritative."""
    raw = (getattr(args, "worker", None) or socket.gethostname() or "node").strip()
    cleaned = "".join(c for c in raw if c.isalnum() or c in "._-")[:64]
    return cleaned or "node"


def resolve_catalog(args: argparse.Namespace) -> tuple[str, str]:
    """Map the typed positional to ``(catalog_id, serve_alias)``.

    An explicit ``--quicksilver-model`` always wins. Otherwise the
    typed name is a catalog id iff it is one of the known ids (serve
    alias = the catalog's default weights alias); or maps to one via
    the reverse table. Anything else needs the flag (§5.4's "otherwise
    require") — fail early, before the model-load cost, with the exact
    spelling to type.
    """
    typed: str = getattr(args, "_original_alias", None) or args.model
    explicit: str | None = getattr(args, "quicksilver_model", None)
    if explicit:
        return explicit, typed
    if typed in CATALOG_DEFAULT_ALIAS:
        return typed, CATALOG_DEFAULT_ALIAS[typed]
    if typed in ALIAS_TO_CATALOG:
        return ALIAS_TO_CATALOG[typed], typed
    raise QuickSilverError(
        f"{typed!r} is not a known QuickSilver catalog id; pass the pool "
        f"listing explicitly, e.g. --quicksilver-model qwen3.8-27b "
        f"(known: {', '.join(sorted(CATALOG_DEFAULT_ALIAS))})"
    )


def _resolve_serve_hf_path(serve_alias: str) -> str:
    """HF repo for the download-confirmation gate; falls back to the
    alias itself when it is not a registered alias (local path / raw HF
    repo id passed by hand)."""
    try:
        from vllm_mlx.model_aliases import resolve_profile

        profile = resolve_profile(serve_alias)
        if isinstance(profile, dict):
            return str(profile.get("hf_path") or serve_alias)
    except Exception:  # pragma: no cover — the gate must never crash share
        pass
    return serve_alias


# ───────────────────────────── registration (§3.1) ─────────────────────────────


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_api_base(raw: str) -> str:
    """``--quicksilver-api`` guard: the provider key rides this origin's
    Authorization header, so a typo must not silently downgrade to
    cleartext or point at a stranger's host. https only (loopback http
    allowed for the test fake), origin-only (no path/query/fragment)."""
    parsed = urllib.parse.urlparse(raw.rstrip("/"))
    host = parsed.hostname or ""
    # Reject credential-bearing syntax before every error branch and never
    # echo the raw URL: validation runs before the provider key is registered
    # with the redactor, and a malformed userinfo/query/path may itself contain
    # that secret.
    if parsed.username or parsed.password:
        raise QuickSilverError("--quicksilver-api must not include userinfo")
    if parsed.scheme == "http":
        if not _is_loopback_host(host):
            raise QuickSilverError(
                "--quicksilver-api over plain http is only allowed for loopback hosts"
            )
    elif parsed.scheme != "https":
        raise QuickSilverError(
            "--quicksilver-api must be https — the provider "
            "key travels to this host in an Authorization header"
        )
    if not host:
        raise QuickSilverError("--quicksilver-api must include a host")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise QuickSilverError(
            "--quicksilver-api must be an origin without path/query/fragment"
        )
    return f"{parsed.scheme}://{parsed.netloc}"


def _validate_credential_placement(
    payload: dict[str, Any], *, source: str, provider_key: str | None = None
) -> None:
    """Credentials may occupy only their protocol-defined fields.

    A faulty or hostile registration response/cache must not reflect either
    credential into ``node_id`` (which becomes a URL query), a wire URL,
    payout text, worker identity, or another cacheable field. Decode percent
    escapes before checking so encoding cannot bypass the invariant.
    """
    share_key = payload.get("share_key")
    credentials = [
        value
        for value in (provider_key, share_key if isinstance(share_key, str) else None)
        if value
    ]
    echoed_fields = [
        key
        for key in _CACHE_ALLOWED_KEYS
        if key != "share_key"
        and any(
            secret in urllib.parse.unquote(str(payload.get(key, "")))
            for secret in credentials
        )
    ]
    if echoed_fields:
        raise QuickSilverError(
            f"{source} echoed a node credential in cacheable field(s) "
            f"{', '.join(sorted(echoed_fields))} — refusing to persist or route it"
        )


# Server-supplied URLs must never carry our credentials to an arbitrary
# origin: a compromised or misconfigured API surface could otherwise
# point ``relay_url`` at a cleartext ``ws://`` host (the upgrade's
# Authorization header leaks the share-key) or ``heartbeat_url`` at a
# stranger (the beat's Authorization header leaks it). Scheme + host are
# revalidated against
# a closed allowlist on EVERY payload, wire or cached.
_TRUSTED_WIRE_DOMAIN = "quicksilverpro.io"


def _wire_host_trusted(host: str, api_host: str) -> bool:
    return (
        _is_loopback_host(host)
        or host == _TRUSTED_WIRE_DOMAIN
        or host.endswith("." + _TRUSTED_WIRE_DOMAIN)
        or (bool(api_host) and host == api_host)
    )


def _validate_wire_urls(payload: dict[str, Any], api_base: str, *, source: str) -> None:
    """Check ``relay_url`` / ``heartbeat_url`` from a registration
    response or the on-disk cache before any credential rides them."""
    api_host = (urllib.parse.urlparse(api_base).hostname or "").lower()
    for field, secure_scheme, plain_scheme in (
        ("relay_url", "wss", "ws"),
        ("heartbeat_url", "https", "http"),
    ):
        url = str(payload.get(field) or "")
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        scheme_ok = parsed.scheme == secure_scheme or (
            parsed.scheme == plain_scheme and _is_loopback_host(host)
        )
        if not scheme_ok:
            raise QuickSilverError(
                f"{source} {field} must be {secure_scheme}:// (got {url!r}) — "
                f"the node credential rides this connection"
            )
        if not host:
            raise QuickSilverError(f"{source} {field} must include a host: {url!r}")
        # userinfo/query/fragment are the classic credential-embedding
        # channels (the pre-v1 protocol keyed on a path segment). The
        # claim key rides the upgrade Authorization header, so nothing
        # legitimate lives in these components and any value there is
        # treated as hostile.
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise QuickSilverError(
                f"{source} {field} must not carry userinfo, query, or fragment: {url!r}"
            )
        if not _wire_host_trusted(host, api_host):
            raise QuickSilverError(
                f"{source} {field} host {host!r} is not a QuickSilver origin — "
                f"refusing to send node credentials there"
            )


def _resolve_heartbeat_interval(cache: dict[str, Any], *, source: str) -> float:
    """Server-supplied pacing must fail FAST and LOUD. A nonnumeric
    value surfacing as a raw ValueError after serve has booted (or a
    NaN/infinity silently killing or freezing the beat thread, which
    the gateway reads as an offline node) is worse than a plain error
    before anything is spawned."""
    raw = cache.get("heartbeat_interval_s", 10)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise QuickSilverError(
            f"{source} heartbeat_interval_s is not a number: {raw!r}"
        ) from None
    if not math.isfinite(value) or not 1.0 <= value <= 3600.0:
        raise QuickSilverError(
            f"{source} heartbeat_interval_s must be between 1 and 3600 "
            f"seconds (got {raw!r})"
        )
    return value


# Response caps: a registration payload is a few KiB of JSON, an error
# body a one-liner. Anything larger is a faulty or hostile endpoint —
# reading it whole would let pay.* exhaust the node's memory, so the
# reads are bounded and oversized is terminal.
_REGISTER_RESPONSE_MAX_BYTES = 1 * 1024 * 1024
_ERROR_BODY_MAX_BYTES = 8 * 1024


class _ResponseTooLargeError(Exception):
    pass


def _bounded_read(fp, limit: int) -> bytes:
    """Read at most limit+1 bytes — fp.read() with no argument is the
    unbounded memory pin we are defending against."""
    data: bytes = fp.read(limit + 1)
    if len(data) > limit:
        raise _ResponseTooLargeError(f"response exceeded {limit} byte cap")
    return data


def _error_detail(body: bytes) -> str:
    # Every branch redacts: a server-side echo (or an error body that
    # happens to embed our credential) must not reach the terminal.
    try:
        err = json.loads(body)
        if isinstance(err, dict):
            err = err.get("error") or {}
            if isinstance(err, dict):
                return _redact(str(err.get("message") or err.get("code") or ""))[:200]
    except (ValueError, UnicodeDecodeError):
        pass
    return _redact(body.decode("utf-8", "replace")[:200])


def _hardware_info() -> dict[str, Any]:
    """Optional, informational (§3.1) — every probe is best-effort."""
    info: dict[str, Any] = {}
    try:
        from vllm_mlx import __version__

        info["rapid_version"] = __version__
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            info["chip"] = out.stdout.strip()
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip().isdigit():
            info["ram_gb"] = int(int(out.stdout.strip()) / (2**30))
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _resolve_provider_key(args: argparse.Namespace) -> str:
    """§5.2 order: flag → env → interactive prompt (never echoed). The
    key lives only in the returned string's memory; callers must not
    log it. Non-tty (launchd, CI, piped) gets an actionable error
    instead of a getpass-EOF traceback."""
    # POP, don't just read: once resolved, the key may live only in this
    # process's memory. A plain ``get`` left it in ``os.environ`` for the
    # whole session, where every later child inherits it — the hardware
    # probes below and, worst of all, the long-lived serve process that
    # loads third-party model code. The account credential belongs to the
    # supervisor alone (§1/§6).
    # Strip FIRST, then test: a whitespace-only flag/env value ("") is
    # truthy, and the old order turned it into `Bearer ` — a remote 401
    # that reads like a bad key instead of the empty input it is.
    env_key = os.environ.pop(PROVIDER_KEY_ENV_VAR, "")
    key = (getattr(args, "provider_key", None) or env_key or "").strip()
    if key:
        return key
    if not sys.stdin.isatty():
        raise QuickSilverError(
            f"first run needs a QuickSilver provider key (qsppk-…): pass "
            f"--provider-key, or set ${PROVIDER_KEY_ENV_VAR} (generate one "
            f"in the QuickSilver dashboard → Compute / Earn). For headless "
            f"setup, run `rapid-mlx share --quicksilver` once in a terminal "
            f"so the node cache exists."
        )
    key = getpass.getpass("QuickSilver provider key (qsppk-…): ").strip()
    if not key:
        raise QuickSilverError("empty provider key — nothing to register with.")
    return key


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """``urlopen`` follows 30x transparently and REPLAYS the
    Authorization header to the redirect target — on register that
    ships the provider key to whatever host the API bounces to, on
    heartbeat the share-key. Refusing redirects (``redirect_request``
    → None) makes urllib surface the 3xx as an HTTPError instead, which
    the retry taxonomy below treats as terminal."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG001
        return None


_URL_OPENER = urllib.request.build_opener(_NoRedirectHandler)


def _open(req: urllib.request.Request, timeout: float):
    """The single HTTP exit point for this module — every request here
    carries a credential, so every request goes through the
    no-redirect opener."""
    return _URL_OPENER.open(req, timeout=timeout)


def register_node(
    api_base: str, provider_key: str, catalog_id: str, alias: str, worker: str
) -> dict[str, Any]:
    """§3.1 self-serve registration with the §3.1 retry taxonomy.
    Returns the parsed 200 response body. Raises QuickSilverError on
    terminal or exhausted-retry outcomes."""
    url = api_base.rstrip("/") + "/v1/pool/nodes/register"
    body = json.dumps(
        {
            "model": catalog_id,
            "worker": worker,
            "alias": alias,
            "hardware": _hardware_info(),
        }
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {provider_key}",
        "User-Agent": _user_agent(),
        "Content-Type": "application/json",
    }
    deadline = time.monotonic() + _RETRY_BUDGET_SECONDS
    delay = _RETRY_BASE_SECONDS
    while True:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        status: int | None = None
        payload: Any = None
        detail = ""
        try:
            with _open(req, timeout=15) as r:
                status = r.status
                raw = _bounded_read(r, _REGISTER_RESPONSE_MAX_BYTES)
            payload = json.loads(raw)
        except _ResponseTooLargeError as exc:
            raise QuickSilverError(
                f"QuickSilver register response too large: {exc}"
            ) from None
        except urllib.error.HTTPError as exc:
            status = exc.code
            # NEVER reuse `body` here — the request payload must survive
            # verbatim across retries; shadowing it with the error
            # response would POST the server's own error JSON next round.
            # An HTTPError *is* the response: the retry loop can spin
            # through many of these, so the socket must close each time
            # or every rejection burns a connection for the session.
            try:
                error_body = _bounded_read(exc, _ERROR_BODY_MAX_BYTES)
            except _ResponseTooLargeError:
                error_body = b""  # oversized: report the code, not the blob
            finally:
                with contextlib.suppress(Exception):
                    exc.close()
            detail = _error_detail(error_body)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            status = None
            detail = _redact(str(exc))[:200]
        except (ValueError, UnicodeDecodeError) as exc:
            # A 2xx that isn't valid JSON is a corrupt/unexpected
            # response, not transport noise — surface it redacted
            # instead of letting the raw traceback reach the user.
            raise QuickSilverError(
                f"QuickSilver register returned an unparseable response: "
                f"{_redact(str(exc))[:120]}"
            ) from None
        if status is not None and 200 <= status < 300:
            if not isinstance(payload, dict):
                raise QuickSilverError("register returned a non-object body")
            # Strings only — a nested object/echo sneaked into, say,
            # node_id would otherwise render (or worse, key the tunnel)
            # verbatim.
            missing = [
                k
                for k in _WIRE_REQUIRED_KEYS
                if not isinstance(payload.get(k), str) or not payload[k]
            ]
            if missing:
                raise QuickSilverError(
                    f"register response fields missing or non-string: "
                    f"{', '.join(missing)}"
                )
            # The echo must be OUR model: a server (or hostile MITM on
            # a misrouted pay.*) handing back a different catalog id
            # would have this node serve and bill under a pool model
            # nobody registered for, until the cache is rejected next
            # launch — reject now, before cache or serve.
            if payload["model"] != catalog_id:
                raise QuickSilverError(
                    f"register response model {payload['model']!r} does not "
                    f"match requested {catalog_id!r} — refusing to bind"
                )
            payout = payload.get("payout_account")
            if payout is not None and not isinstance(payout, str):
                payload.pop("payout_account", None)
            payload.setdefault("heartbeat_interval_s", 10)
            payload["alias"] = alias
            return payload
        if status in _TERMINAL_REGISTER_CODES:
            hint = {
                401: "bad or expired provider key",
                403: "provider key valid but the account is not pool-eligible",
                409: "this node/alias is already bound to a different account",
                422: "unknown/unsupported QuickSilver model id",
            }[status]
            raise QuickSilverError(
                f"QuickSilver registration failed (HTTP {status}): {hint}"
                + (f" — {detail}" if detail else "")
            )
        # Only 429 / 5xx / transport (status None) are transient. A
        # 3xx (rejected by the no-redirect opener) or any other 4xx is a
        # permanent answer — retrying it for five minutes just delays
        # the actionable error, so treat it as terminal here too.
        if status is not None and status != 429 and status < 500:
            raise QuickSilverError(
                f"QuickSilver registration failed (HTTP {status})"
                + (f" — {detail}" if detail else "")
            )
        if time.monotonic() >= deadline:
            raise QuickSilverError(
                f"QuickSilver registration unreachable (last: "
                f"{'HTTP ' + str(status) if status else detail or 'network'}) — "
                f"gave up after {_RETRY_BUDGET_SECONDS / 60:.0f} min"
            )
        print(
            f"QuickSilver register retry in {delay:.0f}s "
            f"({'HTTP ' + str(status) if status else detail or 'network'})…",
            file=sys.stderr,
        )
        _register_retry_sleep(delay)
        delay = min(delay * 2, _RETRY_MAX_SECONDS)


# ───────────────────────────── heartbeat (§3.3) ─────────────────────────────


class _Heartbeat:
    """POST the §3.3 beat every ``interval`` while the tunnel is up.

    * 200 → nothing (a prior 404 gets re-armed).
    * 404 → registered-but-not-yet-visible: beat on, log once (§5.5).
    * 401 → share-key revoked/rotated: set ``fatal``; the supervisor
      stops the process (no spin).
    * transport errors → just beat again next tick; eligibility is
      decided by relay connectivity, so a missed beat never drops the
      node (§3.3).

    Beats are gated on ``enabled`` (set while a tunnel is connected) —
    a tunnel gap pauses instead of reporting fake zeroes.
    ``beat_now()`` backs §5.3's "first beat after tunnel-up + warm-up";
    the loop then paces the rest.
    """

    def __init__(
        self, url: str, share_key: str, interval: float, inflight_fn: Callable[[], int]
    ) -> None:
        self._url = url
        self._share_key = share_key
        self.interval = max(float(interval or 10.0), 1.0)
        self._inflight_fn = inflight_fn
        self.enabled = threading.Event()
        self.fatal = threading.Event()
        self._stop = threading.Event()
        self._logged_404 = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="rapid-mlx-qs-heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Join must exceed the beat's request timeout, or a beat
            # mid-flight in _open survives stop() and keeps sending the
            # node credential after run_share() has returned.
            self._thread.join(timeout=_BEAT_REQUEST_TIMEOUT + 5)
            if self._thread.is_alive():
                # Python threads cannot be force-killed, and the join
                # window is already well past the beat's own socket
                # timeout — getting here means something is wedged
                # below that timeout. Don't claim a guarantee we failed
                # to deliver: say the beat may fire one last request.
                log.warning(
                    "QuickSilver heartbeat thread still alive %.0fs after "
                    "stop — an in-flight beat may send one final request",
                    _BEAT_REQUEST_TIMEOUT + 5,
                )

    def beat_now(self) -> None:
        self._beat_once()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            if not self.enabled.is_set():
                continue
            self._beat_once()
            if self.fatal.is_set():
                return

    def _beat_once(self) -> None:
        if self._stop.is_set() or self.fatal.is_set():
            return
        try:
            inflight = max(0, int(self._inflight_fn()))
        except Exception:  # noqa: BLE001 — informational field only
            inflight = 0
        req = urllib.request.Request(
            self._url,
            data=json.dumps({"inflight": inflight, "client": _user_agent()}).encode(
                "utf-8"
            ),
            headers={
                "Authorization": f"Bearer {self._share_key}",
                "User-Agent": _user_agent(),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with _open(req, timeout=_BEAT_REQUEST_TIMEOUT) as r:
                if 200 <= r.status < 300:
                    self._logged_404 = False
        except urllib.error.HTTPError as exc:
            # The heartbeat beats every 10 s for the whole life of the
            # node; an HTTPError is the response object, so leaving it
            # unclosed leaks one socket per rejected beat. The body is
            # deliberately not read — nothing here consumes it, and an
            # unread body still gets drained/closed.
            try:
                if exc.code == 401:
                    log.error(
                        "QuickSilver heartbeat rejected (HTTP 401) — node "
                        "credential revoked"
                    )
                    self.fatal.set()
                elif exc.code == 404:
                    if not self._logged_404:
                        log.info(
                            "QuickSilver heartbeat 404 (node not visible "
                            "yet) — keep beating"
                        )
                        self._logged_404 = True
            finally:
                with contextlib.suppress(Exception):
                    exc.close()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            log.debug("QuickSilver heartbeat unreachable: %s", _redact(str(exc)))


# ───────────────────────────── warm-up (§5.3) ─────────────────────────────


def _warmup(port: int, api_key: str, model: str) -> None:
    """One short non-streaming completion so the first pool request
    doesn't eat the gateway's 60 s header-timeout paging weights in
    (§3.4's timeout is measured to response headers). Failure is a
    warning, not a block — the node can still serve."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with _open(req, timeout=_WARMUP_TIMEOUT_SECONDS) as r:
            if not (200 <= r.status < 300):
                print(
                    f"warning: warm-up returned HTTP {r.status}; the first "
                    f"pool request may be slow.",
                    file=sys.stderr,
                )
    except Exception as exc:  # noqa: BLE001 — warn-and-continue by design
        print(
            f"warning: warm-up failed ({_redact(str(exc))[:120]}); the first "
            f"pool request may be slow.",
            file=sys.stderr,
        )


# ───────────────────────────── service install (§5.6) ─────────────────────────────


def install_service(
    args: argparse.Namespace, catalog_id: str, serve_alias: str
) -> None:
    """Write the KeepAlive LaunchAgent and exit (§5.6). The plist
    carries NO key material (§6): the resident process relies on the
    0600 cache, so a machine without one refuses here rather than
    install a job that would prompt for a provider key into the void."""
    worker = _resolve_worker(args)
    cache = _load_cache(catalog_id, serve_alias, worker)
    if cache is None:
        raise QuickSilverError(
            f"no node cache for {catalog_id!r} / {serve_alias!r} — run "
            f"`rapid-mlx share {serve_alias} --quicksilver` interactively "
            f"once first (registration needs the provider key), then "
            f"`--install-service`."
        )
    _register_secret(cache["share_key"])
    _validate_credential_placement(cache, source="node cache")
    from .cli import _state_dir

    # Serve passthrough would have to be baked into a resident,
    # unattended job — passthrough is free-form serve config that can
    # carry local paths and credentials, and we cannot prove it
    # key-free (§6: the plist carries no key material). Refuse rather
    # than silently DROP it (the job would serve different behavior
    # than the interactive run it claims to reproduce).
    if list(getattr(args, "_passthrough", None) or []):
        raise QuickSilverError(
            "--install-service cannot bake `--` serve passthrough args "
            "into the LaunchAgent — they may carry local config or "
            "credentials. Re-run the interactive command with "
            "passthrough if you need it, or install without it."
        )

    plist_dir = Path.home() / "Library" / "LaunchAgents"
    label = f"com.quicksilver.node.{catalog_id}"
    plist_path = plist_dir / f"{label}.plist"
    log_dir = _state_dir()
    home = Path.home()

    argv = [sys.executable, "-m", "vllm_mlx.cli", "share", serve_alias, "--quicksilver"]
    argv += ["--worker", worker]
    if catalog_id != serve_alias:
        argv += ["--quicksilver-model", catalog_id]
    # Reproduce every declared behavior flag so the resident job serves
    # exactly what the interactive run served. --reregister is
    # deliberately NOT serialized (a KeepAlive job that rotated its
    # share-key on every restart would burn the credential);
    # --provider-key is never serialized (§6). Unset --port stays unset
    # so the job re-resolves $RAPID_MLX_SHARE_PORT/default at boot.
    if getattr(args, "thinking", False):
        argv.append("--thinking")
    if args.port is not None:
        argv += ["--port", str(args.port)]
    if args.cors_origins:
        argv += ["--cors-origins", *[str(origin) for origin in args.cors_origins]]
    if args.rate_limit is not None:
        argv += ["--rate-limit", str(args.rate_limit)]
    if args.chat_frontend is not None:
        argv += ["--chat-frontend", str(args.chat_frontend)]
    # Bake the registration origin into the job: a node minted against
    # a staging/custom API restarted WITHOUT the flag would validate its
    # cache against the default origin and reject its own relay /
    # heartbeat hosts (KeepAlive then hot-loops a doomed restart).
    explicit_api = getattr(args, "quicksilver_api", None)
    cached_api = str(cache.get("api_base") or "")
    if explicit_api is not None:
        effective_api = _validate_api_base(explicit_api)
        argv += ["--quicksilver-api", effective_api]
    elif cached_api and cached_api != DEFAULT_PAY_API:
        # Do NOT install a job that will restart against the wrong
        # origin and KeepAlive-hot-loop a doomed validation — refuse
        # the install and send the operator to --reregister.
        effective_api = _validate_api_base(cached_api)
        argv += ["--quicksilver-api", effective_api]
    else:
        effective_api = DEFAULT_PAY_API
    # The resident job re-loads this cache every boot and gates serve
    # on the SAME wire-URL/interval checks. A cache whose relay or
    # heartbeat lives on a host the effective origin does not trust
    # (sibling staging hosts are the classic shape) would make the job
    # fail _validate_wire_urls at every start — KeepAlive hot-loops a
    # doomed restart at ThrottleInterval=10s. Refuse an unusable
    # service at install time instead.
    try:
        _validate_wire_urls(cache, effective_api, source="node cache")
        _resolve_heartbeat_interval(cache, source="node cache")
    except QuickSilverError as exc:
        raise QuickSilverError(f"{exc} — re-run with --reregister") from None

    plist: dict[str, Any] = {
        "Label": label,
        "ProgramArguments": argv,
        "RunAtLoad": True,
        "KeepAlive": True,
        # A revoked key exits non-zero fast; don't re-hammer the relay
        # faster than once/10s (same shape as headless_service).
        "ThrottleInterval": 10,
        "ExitTimeOut": 30,
        "WorkingDirectory": str(home),
        "EnvironmentVariables": {
            "HOME": str(home),
            "PATH": f"{home / '.local/bin'}:/usr/bin:/bin:/usr/sbin:/sbin",
        },
        # plist ints are decimal; 0o27 masks group/other write (§6: the
        # node runs unattended with a resident credential).
        "Umask": 0o27,
        "StandardOutPath": str(log_dir / f"quicksilver-{catalog_id}.out.log"),
        "StandardErrorPath": str(log_dir / f"quicksilver-{catalog_id}.err.log"),
    }
    # mkdir, log-file create/chmod and plist serialization all touch the
    # filesystem and can each fail with OSError (read-only volume, full
    # disk, perm denied). All three must surface as QuickSilverError so
    # run_share renders a redacted exit-2 line, not a raw traceback.
    try:
        plist_dir.mkdir(parents=True, exist_ok=True)
        for key in ("StandardOutPath", "StandardErrorPath"):
            log_path = Path(plist[key])
            log_path.touch()
            log_path.chmod(0o600)
    except OSError as exc:
        raise QuickSilverError(
            f"could not prepare service logs under {log_dir}: {exc}"
        ) from None

    from vllm_mlx.headless_service.plist import serialize_plist

    try:
        plist_path.write_bytes(serialize_plist(plist))
    except OSError as exc:
        raise QuickSilverError(f"could not write {plist_path}: {exc}") from None
    print(f"Wrote {plist_path}")
    # ``launchctl load`` still works but is deprecated; ``bootstrap`` is
    # the modern per-user-domain verb — print the line that will age well.
    print(
        f"Activate with:  launchctl bootstrap gui/$(id -u) {plist_path}\n"
        f"Stop with:      launchctl bootout gui/$(id -u)/{label}"
    )


# ───────────────────────── supervisor (§5.3, §5.5) ─────────────────────────


def run_share(args: argparse.Namespace) -> None:
    """The ``--quicksilver`` branch of ``share_command``. Owns the whole
    lifecycle so cli.py's plain-share path stays byte-identical (§8)."""
    try:
        _run_share(args)
    except QuickSilverError as exc:
        print(f"share --quicksilver: {_redact(str(exc))}", file=sys.stderr)
        sys.exit(2)


def _run_share(args: argparse.Namespace) -> None:
    catalog_id, serve_alias = resolve_catalog(args)
    if args.install_service:
        install_service(args, catalog_id=catalog_id, serve_alias=serve_alias)
        return

    api_base = _validate_api_base(
        args.quicksilver_api if args.quicksilver_api is not None else DEFAULT_PAY_API
    )

    worker = _resolve_worker(args)
    cache = None if args.reregister else _load_cache(catalog_id, serve_alias, worker)
    if cache is None:
        provider_key = _resolve_provider_key(args)
        _register_secret(provider_key)
        print("Registering node with QuickSilver…", file=sys.stderr)
        cache = register_node(api_base, provider_key, catalog_id, serve_alias, worker)
        # Register the share-key for redaction BEFORE rendering any
        # server-controlled field — node_id / payout_account are the
        # response's to fill, and if either echoed the resident key,
        # the banner below would print it. _redact must be armed first.
        _register_secret(cache["share_key"])
        _validate_credential_placement(
            cache, source="register response", provider_key=provider_key
        )
        # Bind the credential to the origin it was minted against (the
        # allowlist in _validate_wire_urls accepts the registration
        # origin; installed services and later cached runs must keep
        # validating against THAT, not the default).
        cache["api_base"] = api_base
        _validate_wire_urls(cache, api_base, source="register response")
        interval = _resolve_heartbeat_interval(cache, source="register response")
        _save_cache(catalog_id, cache)
        print(
            _redact(
                f"Node registered: {cache['node_id']} "
                f"(payout account: {cache.get('payout_account') or 'n/a'})"
            ),
            file=sys.stderr,
        )
    else:
        # Cached runs never call _resolve_provider_key, so an exported
        # QUICKSILVER_PROVIDER_KEY would sit in os.environ for the whole
        # session and be inherited by serve (third-party model code) and
        # the hardware probes below. The account credential is only ever
        # needed to register — scrub it even when we don't use it.
        os.environ.pop(PROVIDER_KEY_ENV_VAR, None)
        # Arm redaction and validate credential placement before inspecting or
        # printing any other cached field. In particular, node_id later becomes
        # a query parameter and api_base errors used to precede this guard.
        _register_secret(cache["share_key"])
        _validate_credential_placement(cache, source="node cache")
        # An explicit --quicksilver-api always wins; otherwise the
        # cached origin is authoritative. Re-validating the cached
        # value is cheap paranoia against a tampered file.
        if args.quicksilver_api is None and cache.get("api_base"):
            try:
                api_base = _validate_api_base(str(cache["api_base"]))
            except QuickSilverError as exc:
                raise QuickSilverError(f"{exc} (from node cache)") from None
        print(
            _redact(
                f"Using cached node {cache['node_id']} for {catalog_id} "
                f"(provider key not needed)"
            ),
            file=sys.stderr,
        )
        # A cache predating a hostile/buggy API is just as dangerous as
        # a fresh hostile wire response — validate on load too. The
        # only repair for a bad cached field is a fresh registration.
        try:
            _validate_wire_urls(cache, api_base, source="node cache")
            interval = _resolve_heartbeat_interval(cache, source="node cache")
        except QuickSilverError as exc:
            raise QuickSilverError(f"{exc} — re-run with --reregister") from None

    relay_url = cache["relay_url"]

    # Lazy import: cli.py dispatches here from inside share_command, so
    # a top-level ``from . import cli`` here would deadlock on first
    # import of either module in isolation.
    from .cli import (
        _maybe_confirm_download,
        _pick_port,
        _resolve_served_model_name,
        _spawn_serve,
        _state_dir,
        _verify_auth_gate,
        _wait_for_healthz,
    )

    # B2 download gate (share is not on the top-level gated-command
    # list — same replication cli.py's plain path does, on the RESOLVED
    # hf path since the cache lookup keys the repo id).
    _maybe_confirm_download(_resolve_serve_hf_path(serve_alias))

    print(
        "warning: `rapid-mlx share --quicksilver` serves QuickSilver pool "
        "traffic — strangers' requests will run on this machine. Press "
        "Ctrl-C to stop earning.",
        file=sys.stderr,
    )

    # Serve flags: the pool slot contract wants low concurrency (§5.3 —
    # --max-num-seqs 2 unless the user overrode it after ``--``); the
    # plain-share 120 rpm DEFAULT is not injected (metering is
    # gateway-side and a 120 rpm cap would throttle paying traffic),
    # but an EXPLICIT ``--rate-limit`` is the user's own call and must
    # be honored, not silently dropped.
    extra: list[str] = []
    passthrough = list(getattr(args, "_passthrough", None) or [])
    if not any(t.split("=", 1)[0].startswith("--max-num-seqs") for t in passthrough):
        extra += ["--max-num-seqs", "2"]
    # ``share`` advertises thinking off by default. The child server defaults
    # it on, so pool mode must forward the disabling flag just like plain share.
    if not args.thinking and not any(
        t.split("=", 1)[0] in ("--thinking", "--no-thinking") for t in passthrough
    ):
        extra.append("--no-thinking")
    # Pool requests (and the relay's readiness probe) address the node by its
    # CATALOG id, but the serve alias differs (§5.4, e.g. nemotron-3.5-lightning
    # vs nemotron-3.5-lightning-30b-4bit). Expose the loaded model UNDER the
    # catalog id so serve (a) accepts inbound requests naming the catalog id and
    # (b) echoes the catalog id in its responses — the relay's readiness probe
    # requires the response `model` to equal the pool model, and billing keys off
    # it. Without this the node connects + heartbeats but never passes readiness,
    # so it stays unroutable and every request falls back to cloud.
    if not any(
        t.split("=", 1)[0].startswith("--served-model-name") for t in passthrough
    ):
        extra += ["--served-model-name", catalog_id]
    if (
        args.rate_limit is not None
        and args.rate_limit > 0
        and not any(t.split("=", 1)[0].startswith("--rate-limit") for t in passthrough)
    ):
        extra.append("--rate-limit")
        extra.append(str(args.rate_limit))
    extra.extend(passthrough)

    api_key = secrets.token_hex(24)
    try:
        port = _pick_port(args.port if args.port is not None else 8765)
    except RuntimeError as exc:
        print(f"share: {exc}", file=sys.stderr)
        sys.exit(1)
    serve_log = _state_dir() / "serve.log"

    def _term_handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    original_sigterm = signal.signal(signal.SIGTERM, _term_handler)

    serve_proc: subprocess.Popen[bytes] | None = None
    tunnel: ws_tunnel.TunnelClient | None = None
    tunnel_thread: threading.Thread | None = None
    heartbeat: _Heartbeat | None = None
    exit_code = 0
    # Keep the websockets library at WARNING for the session: its
    # client logger prints connect URIs, and any future credential that
    # does land in a URL must never reach launchd logs via INFO.
    ws_logger = logging.getLogger("websockets")
    previous_ws_level = ws_logger.level
    ws_logger.setLevel(max(logging.WARNING, previous_ws_level))
    try:
        print(
            f"Starting rapid-mlx serve ({serve_alias} on :{port})…",
            file=sys.stderr,
        )
        serve_proc = _spawn_serve(
            alias=serve_alias,
            port=port,
            api_key=api_key,
            log_path=serve_log,
            extra_args=extra,
        )
        if not _wait_for_healthz(port, serve_proc):
            print(
                f"serve exited before becoming ready — see {serve_log}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not _verify_auth_gate(port, api_key):
            print(
                f"serve on :{port} did not answer authenticated /v1/models — "
                f"another process may be bound to the same port. Aborting "
                f"before joining the pool.",
                file=sys.stderr,
            )
            sys.exit(1)

        display_model = _resolve_served_model_name(port, api_key) or serve_alias
        print("Warming up (pages weights into Metal)…", file=sys.stderr)
        _warmup(port, api_key, display_model)

        # ``interval`` was validated (finite, 1..3600) before serve
        # booted — the beat thread can never be born frozen or dead.
        heartbeat = _Heartbeat(
            cache["heartbeat_url"],
            cache["share_key"],
            interval,
            # Bound to the SLOW-BIND of ``tunnel``: between reconnects
            # there is no live tunnel and the beat pauses anyway.
            inflight_fn=lambda: tunnel.inflight if tunnel is not None else 0,
        )
        heartbeat.start()

        backoff = 1.0
        banner_printed = False
        served_at = 0.0
        fast_reject_streak = 0
        while True:
            tunnel = ws_tunnel.TunnelClient(
                local_port=port,
                tunnel_id=cache["node_id"],
                relay_url=relay_url,
                share_key=cache["share_key"],
                override_authorization=api_key,
                inject_stream_usage=True,
            )
            print(
                _redact(
                    f"Connecting to QuickSilver relay {relay_url} "
                    f"as {cache['node_id']}…"
                ),
                file=sys.stderr,
            )
            tunnel_thread = tunnel.run_in_thread()
            connected = tunnel.ready_event.wait(timeout=30) and tunnel.error is None
            reconnect = False
            if connected:
                served_at = time.monotonic()
                heartbeat.enabled.set()
                # §5.3: the first beat waits for tunnel-up AND warm-up
                # (which already ran before the loop).
                heartbeat.beat_now()
                if not banner_printed:
                    payout = cache.get("payout_account")
                    print(
                        _redact(
                            f"rapid-mlx: serving {display_model} to the "
                            f"QuickSilver pool — node {cache['node_id']}"
                            + (f", payout account {payout}" if payout else "")
                            + f", heartbeat every {interval:.0f}s. Ctrl-C to stop."
                        ),
                        flush=True,
                    )
                    banner_printed = True
            # Monitor: serve alive, tunnel alive, heartbeat not fatal.
            # Tunnel drop → reconnect with backoff; any 401 → terminal
            # (§5.5: the key is revoked; spinning is worse than dying).
            while True:
                assert serve_proc is not None and heartbeat is not None
                serve_rc = serve_proc.poll()
                if serve_rc is not None:
                    exit_code = serve_rc if serve_rc != 0 else 1
                    print(
                        f"share: serve process exited — leaving the pool. "
                        f"See {serve_log}.",
                        file=sys.stderr,
                    )
                    break
                if heartbeat.fatal.is_set():
                    exit_code = 1
                    print(
                        "share: QuickSilver revoked our node credential "
                        "(HTTP 401). Stop and re-run with --reregister to "
                        "mint a new share-key.",
                        file=sys.stderr,
                    )
                    break
                if not connected or tunnel.closed_event.is_set():
                    # Terminal share-key rejections must never spin
                    # (§5.5), but ONLY an explicit rejection signal may
                    # kill the node: HTTP 401 at upgrade, or WS close
                    # 1008. Close-code extraction is version-complete
                    # (rcvd / exc / ws.close_code), so an UNCLASSIFIED
                    # instant drop — no code at all — is far more likely
                    # relay/LB instability than a dead key. Repeated
                    # unclassified drops get a loud warning, never an
                    # exit: killing a healthy node during a relay
                    # incident turns a blip into lost earnings until an
                    # operator notices (backoff caps at 30 s, so
                    # continuing is not the §5.5 spin the heuristic was
                    # built to avoid).
                    if tunnel.error_status == _WS_TERMINAL_STATUS:
                        exit_code = 1
                        print(
                            "share: QuickSilver relay rejected our share-key "
                            "(HTTP 401). Re-run with --reregister.",
                            file=sys.stderr,
                        )
                        break
                    if tunnel.close_code in _WS_TERMINAL_CLOSE_CODES:
                        exit_code = 1
                        print(
                            "share: QuickSilver relay rejected our share-key "
                            f"(WS close {tunnel.close_code}). Re-run with "
                            f"--reregister.",
                            file=sys.stderr,
                        )
                        break
                    connection_age = (
                        time.monotonic() - served_at
                        if connected and served_at
                        else None
                    )
                    if (
                        connection_age is not None
                        and connection_age < _FAST_REJECT_WINDOW_SECONDS
                    ):
                        fast_reject_streak += 1
                        if fast_reject_streak == _FAST_REJECT_STREAK_LIMIT:
                            # Warn once, keep serving the retry loop.
                            print(
                                "share: WARNING — the relay has dropped this "
                                "node the instant it connected "
                                f"{_FAST_REJECT_STREAK_LIMIT} times running. "
                                "No rejection was signalled (no 401, no close "
                                "1008), so this reads as relay-side "
                                "instability; continuing to retry. If the "
                                "pool shows this node offline, re-run with "
                                "--reregister.",
                                file=sys.stderr,
                            )
                    elif connection_age is not None:
                        # Only a connection that survived the reject window
                        # earns a fresh retry budget. Instant drops and failed
                        # handshakes retain exponential backoff.
                        fast_reject_streak = 0
                        backoff = 1.0
                    err = _redact(str(tunnel.error)) if tunnel.error else "dropped"
                    print(
                        f"share: tunnel lost ({err[:200]}) — reconnecting in "
                        f"{backoff:.0f}s.",
                        file=sys.stderr,
                    )
                    reconnect = True
                    break
                _supervisor_sleep(1)
            if not reconnect:
                break
            heartbeat.enabled.clear()
            tunnel.stop()
            tunnel_thread.join(timeout=5)
            _supervisor_sleep(backoff)
            backoff = min(backoff * 2, 60.0)
    except KeyboardInterrupt:
        print("\nLeaving the QuickSilver pool…", file=sys.stderr)
    finally:
        # Same cleanup contract as plain share: a second SIGTERM mid
        # teardown must not skip the serve-reap (SIGKILL can still get
        # us, that's fine).
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
        if heartbeat is not None:
            heartbeat.stop()
        if tunnel is not None:
            tunnel.stop()
        if tunnel_thread is not None and tunnel_thread.is_alive():
            tunnel_thread.join(timeout=5)
        if serve_proc is not None and serve_proc.poll() is None:
            try:
                serve_proc.terminate()
                serve_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                serve_proc.kill()
                # kill() only signals — reap it, or the SIGKILL'd child
                # lingers as a zombie for the life of this process.
                try:
                    serve_proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            except OSError:
                pass
        try:
            signal.signal(signal.SIGTERM, original_sigterm)
        except (ValueError, OSError, TypeError):
            pass
        ws_logger.setLevel(previous_ws_level)

    if exit_code:
        sys.exit(exit_code)
