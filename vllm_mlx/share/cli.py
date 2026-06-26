# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx share <alias>`` — start a serve + open a public tunnel.

Orchestration shape:

  1. Validate alias (cheap fail-fast before booting the engine).
  2. Pick a free local port + generate a fresh 24-byte bearer key.
  3. Spawn ``rapid-mlx serve`` in a child process pointing at that port.
  4. Wait for /healthz to come back ready, then auth-gate /v1/models.
  5. Open a WebSocket to the rapidserver Worker (defaults to
     ``wss://rapidserver.quicksilverpro.io/up``). The Worker mints a
     Durable Object keyed on our tunnel id; inbound HTTPS requests at
     ``https://rapidserver.quicksilverpro.io/r/<id>/...`` are
     reverse-multiplexed back to us over the same WS frame.
  6. Probe ``<public_url>/v1/models`` to prove the tunnel ↔ serve
     round-trip works, then print the security banner + URL + key.
  7. Block until Ctrl-C, monitoring both the serve subprocess and the
     WS tunnel thread.
  8. On exit, close the WS first (cheap) then terminate serve.

State lives in ``~/.cache/rapid-mlx/share/`` — pid + serve log only.
Key + URL are NOT persisted: each invocation issues a new key
(per user's "new key every share" preference) and a new session.

Architecture pivot (2026-06-03): the prior PR #504 design used frpc +
a control-plane HTTP endpoint + frps relay running on the operator's
M3 Ultra. We replaced that whole stack with a Cloudflare Worker so
prod no longer depends on the operator's personal machine. See
``ws_tunnel.py`` for the wire protocol and ``rapidmlx.com/worker/``
for the Worker code.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .._completion import alias_completer
from . import warning, ws_tunnel

# Pulled out so the routing-shape audit (tests/test_no_out_of_band_routing.py)
# sees one clean RAPID_MLX_* string literal that lives in the
# ALLOWED_RAPID_MLX_ENV_VARS allowlist — inlining the name into an
# f-string error message would yield "RAPID_MLX_SHARE_PORT must be an…"
# which is NOT on the allowlist and tripwires the audit.
_PORT_ENV_VAR = "RAPID_MLX_SHARE_PORT"
# Same out-of-band-routing carve-out as ``RAPID_MLX_SHARE_PORT`` — see
# ``ALLOWED_RAPID_MLX_ENV_VARS`` in ``tests/test_no_out_of_band_routing.py``.
_CHAT_FRONTEND_ENV_VAR = "RAPID_MLX_CHAT_FRONTEND"
# QuickSilver hosts the splash-protocol chat frontend at
# ``rapid-pro.quicksilverpro.io`` (CF Pages, Big-AGI static export + splash
# injector that seeds the OpenAI vendor with our relay). Big-AGI ships
# tool-calling, multi-turn personas, and a much richer UX than the
# previous BCG fallback (still reachable via
# ``--chat-frontend https://rapid.quicksilverpro.io``). There is NO
# marketing copy in the banner — only the URL appears, so OSS users who
# don't want it can swap with ``--chat-frontend`` (e.g.
# ``https://chat.rapidmlx.com`` for the rapidmlx-only mirror, or any
# OpenAI-compatible frontend like OpenWebUI, where the empty-string
# opt-out suppresses the line entirely).
_DEFAULT_CHAT_FRONTEND = "https://rapid-pro.quicksilverpro.io"


def _resolve_chat_frontend(flag_value: str | None) -> str | None:
    """Resolve the chat-frontend URL from the CLI flag and env var.

    Precedence: ``--chat-frontend`` > ``$RAPID_MLX_CHAT_FRONTEND`` >
    built-in default (``https://rapid-pro.quicksilverpro.io``). An explicit
    empty string at either layer disables the one-click chat link
    entirely — useful when the user is wiring up an OpenAI-compatible
    frontend like OpenWebUI that doesn't implement the splash
    share-key protocol.

    Returns the validated origin (``scheme://host[:port]``) or ``None``
    when disabled. Raises ``ValueError`` on malformed input — the share
    command surfaces that as exit 2 (user error, not crash).

    Validation rules (defense against a hostile env var or copy-pasted
    snippet from somewhere the user shouldn't trust):

    * Scheme must be ``https`` or ``http`` — no ``javascript:``, no
      ``ftp:``, no ``file:``. We're going to embed the user's bearer
      key in the URL fragment so anything that could parse the URL as
      "do something other than open a chat tab" is a foot-gun.
    * **No userinfo.** ``https://chat.rapidmlx.com@evil.com`` parses
      to ``netloc='chat.rapidmlx.com@evil.com'`` — if we echo netloc
      back verbatim the banner advertises a link that visually starts
      with ``chat.rapidmlx.com`` but actually points the bearer key
      at ``evil.com``. Codex round-7 BLOCKING. We reject any URL with
      userinfo and rebuild the origin from ``hostname`` + ``port``
      (never from ``netloc``).
    * Plain ``http://`` is only allowed for loopback hosts. Embedding
      a bearer key in a URL pointed at a non-loopback HTTP origin
      means an attacker on-path between the user and that origin can
      see the key (the fragment is parsed by JS but the *user* still
      types the URL, and some browsers and link-handlers log the
      request URL including the fragment — better safe). Loopback is
      determined via ``ipaddress.is_loopback`` so canonical IPv6
      forms (``::0001``, ``0:0:0:0:0:0:0:1``) are recognised, not
      just the bare ``::1`` literal.
    * No path, query, or fragment — we own the ``/#k=...`` shape and
      need a clean origin to append onto. ``foo.com/bar`` would yield
      a banner link that points at ``foo.com/bar/#k=...`` which the
      receiving splash wouldn't recognise as its own bootstrap path.
    """
    if flag_value is not None:
        raw = flag_value
    else:
        raw = os.environ.get(_CHAT_FRONTEND_ENV_VAR)
        if raw is None:
            raw = _DEFAULT_CHAT_FRONTEND
    raw = raw.strip()
    if not raw:
        return None
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"--chat-frontend must use https:// or http:// (got {raw!r})")
    # Reject userinfo BEFORE consulting hostname/port — ``urlparse`` happily
    # pulls ``user:pass`` into netloc and exposes the trailing host as
    # ``hostname``, so checking only ``hostname`` would silently let a
    # phishing-shaped URL through. The presence of ``@`` in netloc is the
    # canonical CPython signal for userinfo.
    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in (parsed.netloc or "")
    ):
        raise ValueError(f"--chat-frontend must not include userinfo (got {raw!r})")
    host = parsed.hostname
    if not host:
        raise ValueError(f"--chat-frontend must include a host (got {raw!r})")
    # ``parsed.port`` raises ValueError on a bogus integer — surface it
    # under the same flag name so the user sees one consistent error
    # message family.
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"--chat-frontend has an invalid port (got {raw!r})") from exc
    if parsed.scheme == "http":
        try:
            host_is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            # Not an IP literal — only the textual ``localhost`` qualifies
            # (DNS resolution is not allowed to influence security policy
            # at validation time).
            host_is_loopback = host == "localhost"
        if not host_is_loopback:
            raise ValueError(
                f"--chat-frontend over plain http:// only allowed for "
                f"loopback hosts (got {raw!r})"
            )
    if parsed.path not in ("", "/"):
        raise ValueError(
            f"--chat-frontend must be an origin without a path (got {raw!r})"
        )
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"--chat-frontend must not include a query or fragment (got {raw!r})"
        )
    # Rebuild the origin from ``hostname`` + validated ``port`` — never
    # ``netloc`` (it would carry userinfo / whitespace / mixed-case
    # surface through). IPv6 needs bracketing so the rebuilt URL parses
    # back to the same host.
    host_part = f"[{host}]" if ":" in host else host
    authority = f"{host_part}:{port}" if port is not None else host_part
    return f"{parsed.scheme}://{authority}"


# Default CORS allowlist when --cors-origins is not supplied. The
# rapidmlx chat-frontend ecosystem only. Users running a custom front
# can either pass the origin to ``--cors-origins`` or rely on
# ``--chat-frontend`` propagating into the allowlist below.
_DEFAULT_CORS_ALLOWLIST: tuple[str, ...] = (
    "https://rapid-pro.pages.dev",
    "https://rapid-pro.quicksilverpro.io",
    "https://rapidmlx.com",
    "https://chat.rapidmlx.com",
)


def _resolve_cors_origins(
    flag_value: list[str] | None,
    chat_frontend: str | None,
) -> list[str]:
    """Build the ``--cors-origins`` argv to forward to the child.

    Priority:
      * Explicit ``--cors-origins`` from CLI → use as-is (wildcards allowed,
        loopback allowed, exact origins allowed).
      * Otherwise → start with the rapidmlx default allowlist, then
        append ``chat_frontend`` (already validated by
        ``_resolve_chat_frontend``) if not already covered.

    The default lockdown matters because a share URL hands out a 192-bit
    bearer key that, on its own, only protects against unauthenticated
    pulls. A wide-open ``--cors-origins '*'`` lets any drive-by web
    page the publisher visits hit ``http://127.0.0.1:<share-port>``
    bearing the key and use the publisher's compute — defense in depth
    against a same-origin XSS or a malicious tab on the publisher's
    browser.
    """
    if flag_value:
        return list(flag_value)
    origins = list(_DEFAULT_CORS_ALLOWLIST)
    if chat_frontend and chat_frontend not in origins:
        origins.append(chat_frontend)
    return origins


def _pick_port(preferred: int) -> int:
    """Return ``preferred`` if free, else an OS-assigned port. We bind+release
    rather than just checking — TOCTOU windows on busy systems are real.
    """
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError("no free port available for share")


def _resolve_served_model_name(port: int, api_key: str) -> str | None:
    """Read the model id rapid-mlx serve is exposing via /v1/models.

    The CLI accepts a short alias (``qwen3.5-4b-4bit``) but the OpenAI
    endpoint only recognises the full HF model id
    (``mlx-community/Qwen3.5-4B-MLX-4bit``). Without this lookup the
    curl example we paste into the security banner fails on first
    try — a confusing UX for the user (and their friend).

    ``api_key`` is required because we spawn serve with ``--api-key``
    so /v1/models is bearer-gated; without the header the probe 401s
    and silently falls back to the alias.
    """
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            payload = json.load(r)
        data = payload.get("data") or []
        if data and isinstance(data[0], dict):
            return data[0].get("id")
    except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
        # ``TimeoutError`` (== ``socket.timeout`` since Python 3.10) is
        # NOT a ``URLError`` subclass — urlopen raises it bare when a
        # TCP connection is accepted but the server stalls before
        # sending headers. Codex round-5 BLOCKING.
        return None
    return None


def _wait_for_healthz(port: int, serve_proc: subprocess.Popen[bytes]) -> bool:
    """Poll /healthz until the child serve reports ready or exits.

    No fixed timeout: a cold first-time pull of a 70B model legitimately
    takes 10+ minutes, and silently SIGTERM-ing a healthy download is
    one of the worst UX failure modes we can ship. Instead we watch the
    child process — if it exits without ever serving /healthz we give
    up, otherwise we wait as long as it takes. Caller can Ctrl-C any
    time to abort.

    ``serve_proc`` is required (no ``None`` default) so the
    process-watch loop is always armed. DeepSeek round-5 BLOCKING #3:
    a None default + no timeout would loop forever.
    """
    url = f"http://127.0.0.1:{port}/healthz"
    while True:
        if serve_proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            # ``TimeoutError`` is NOT a ``URLError`` subclass — urlopen
            # raises it bare when the TCP connection is accepted but
            # the server stalls before sending headers. Without this
            # branch a stalled /healthz escapes as a raw traceback
            # instead of being retried until serve exits or comes up.
            pass
        time.sleep(1)


def _verify_auth_gate(port: int, api_key: str) -> bool:
    """Auth-gated proof that the process answering on ``port`` is OURS.

    /healthz is unauthenticated by design (load-balancers need it). On a
    busy host another local process can race us to the same port and
    answer /healthz while having nothing to do with rapid-mlx — and the
    tunnel would happily forward to it. We require an authenticated
    /v1/models 200 with the freshly-generated bearer before requesting a
    tunnel: only our serve has that key, so a 200 here means we're
    pointing the tunnel at our own process.

    Codex round-2 BLOCKING: a process started WITHOUT auth (any other
    OpenAI-compatible server, or a rapid-mlx serve without --api-key)
    returns 200 for every bearer header — so this gate would silently
    accept it. To make the proof meaningful we also send a known-bad
    key first: if THAT returns 200, the endpoint isn't auth-gated and
    the answering process isn't ours. Only after the bad-key 401 do we
    trust the real-key 200.
    """

    def _probe(bearer: str) -> int | None:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/models",
            headers={"Authorization": f"Bearer {bearer}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as r:  # noqa: S310
                return r.status
        except urllib.error.HTTPError as exc:
            # 401/403 are legitimate "auth is wired up" signals; surface
            # the code so the caller can distinguish "no auth at all"
            # (200) from "auth enforced" (401/403).
            return exc.code
        except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
            # ``TimeoutError`` (== ``socket.timeout`` in Python 3.10+) is
            # raised bare by urlopen on connect-then-stall. Catching only
            # URLError would let a degraded local server escape as a raw
            # traceback during the auth gate.
            return None

    # Step 1: a deliberately-wrong key must NOT return 200. If the
    # answering process accepts any bearer it isn't ours.
    # ``secrets.token_hex(24)`` matches the shape of the real key so a
    # too-strict server can't reject by length/charset.
    bad_status = _probe(secrets.token_hex(24))
    if bad_status == 200:
        return False
    # We accept anything-other-than-200 here (401/403/404/None) as
    # "endpoint is auth-protected or unreachable" — both are fine for
    # the gate; the real check is the next step.

    # Step 2: the real key must return 200.
    return _probe(api_key) == 200


def _spawn_serve(
    *,
    alias: str,
    port: int,
    api_key: str,
    log_path: Path,
    extra_args: list[str],
) -> subprocess.Popen[bytes]:
    # Use sys.executable + ``-m`` instead of the ``rapid-mlx`` script so
    # the share command works inside editable installs and CI environments
    # where the entrypoint script may not be on PATH.
    # ``--host 127.0.0.1`` is load-bearing here: without it serve binds
    # 0.0.0.0 and the bearer-key-gated API becomes reachable from anyone
    # on the user's LAN, not just through the frp tunnel as intended.
    #
    # The bearer key is passed via ``RAPID_MLX_API_KEY`` env var, NOT
    # argv. ``ps`` exposes argv to every local user — landing the key
    # there leaks the secret that gates the public tunnel. The env var
    # is only visible to the owning process (and root). (DeepSeek
    # BLOCKING on PR #504 round 3.)
    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "serve",
        alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "INFO",
        *extra_args,
    ]
    env = dict(os.environ)
    env["RAPID_MLX_API_KEY"] = api_key
    # Parent-PID watchdog (rapid-desktop #449 sibling fix). A SIGKILL
    # of ``rapid-mlx share`` (the supervisor) re-parents the spawned
    # ``serve`` to launchd / init; without this stamp the child would
    # outlive the frp tunnel teardown and keep the bearer-gated port
    # bound. The watchdog inside the child checks ``os.getppid()`` and
    # exits the moment it stops matching this PID. No-op if the operator
    # already exported the env var to a different value — that's an
    # explicit override.
    env.setdefault("RAPID_MLX_WATCHDOG_PPID", str(os.getpid()))
    log_fp = log_path.open("ab", buffering=0)
    # Tighten permissions: log files default to umask-derived modes
    # (often 644 = world-readable). If serve ever logs the key as part
    # of an error or debug line, world-read leaks it. 600 forces
    # owner-only. (DeepSeek round-3 NIT #3.)
    try:
        Path(log_fp.name).chmod(0o600)
    except OSError:
        pass
    return subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        # New process group so Ctrl-C in our terminal doesn't deliver
        # SIGINT to serve before we've had a chance to tear down frpc.
        start_new_session=True,
    )


def _state_dir() -> Path:
    d = Path.home() / ".cache" / "rapid-mlx" / "share"
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def _maybe_confirm_download(alias: str) -> None:
    """Replicate the top-level B2 auto-pull gate for share.

    ``vllm_mlx/cli.py`` runs a confirmation prompt for chat/run/serve/pull/bench
    when the first positional argument is an HF-style repo id that isn't
    already cached. ``share`` is NOT on that list (the parent didn't add
    it; ``_GATED_COMMANDS`` lives outside this module's scope), so a
    ``rapid-mlx share <uncached HF repo>`` invocation would silently
    spawn a non-interactive child that pulls multi-GB of weights with no
    confirmation. Codex round-1 BLOCKING: replicate the same check here
    so the share entrypoint enforces the policy too.

    Mirrors the logic at ``cli.py:4322`` — env override / non-TTY both
    short-circuit before any HF API round-trip.
    """
    if "/" not in alias or os.path.exists(alias):
        # Not an HF-style repo id, or a local path — nothing to prompt for.
        return
    if os.environ.get("RAPID_MLX_CHAT_SPAWN", "") == "1":
        # Grandchild safety: a parent ``rapid-mlx`` invocation already
        # gated and set this marker. Don't re-prompt.
        return
    env_val = os.environ.get("RAPID_MLX_AUTO_PULL", "").strip().lower()
    if env_val in {"1", "true", "yes"}:
        return
    if not sys.stdin.isatty():
        return
    from vllm_mlx._download_gate import (
        confirm_or_abort,
        estimate_repo_size_bytes,
        is_repo_cached,
    )

    if not is_repo_cached(alias):
        confirm_or_abort(alias, estimate_repo_size_bytes(alias))


def share_command(args: argparse.Namespace) -> None:
    # Codex round-2 BLOCKING: ``main()`` in ``vllm_mlx/cli.py`` runs
    # alias resolution BEFORE dispatching to us — by the time we get
    # here ``args.model`` is the rewritten HF repo (e.g.
    # ``mlx-community/Qwen3.5-4B-MLX-4bit``) and the user-typed alias
    # lives on ``args._original_alias`` (e.g. ``qwen3.5-4b-4bit``). The child
    # ``serve`` subprocess re-runs alias resolution on whatever we pass
    # it. We want the child to land the same way ``rapid-mlx serve
    # qwen3.5-4b-4bit`` does — including setting ``_model_alias`` on the
    # server so the public ``/v1/models`` endpoint advertises (and
    # accepts) the short alias the user actually typed. So we forward
    # the original alias to the child when one is set; fall back to
    # ``args.model`` (HF repo) when share was called with a raw HF id.
    alias: str = getattr(args, "_original_alias", None) or args.model
    # Mirror the B2 download-confirmation gate that ``cli.py`` applies to
    # chat/run/serve/pull/bench — share is not on that list, so without
    # this call a first-time ``rapid-mlx share <big-repo>`` would pull
    # multi-GB of weights with no prompt. The gate keys off
    # ``args.model`` (HF repo) because the cache lookup uses the
    # resolved id, not the typed alias.
    _maybe_confirm_download(args.model)

    # Resolve --chat-frontend ahead of any subprocess work — a malformed
    # URL is a user error that should exit 2 BEFORE we spawn serve / pay
    # the model-load cost. (Same lazy-validation rationale as the
    # ``--port`` block below: keeping it out of register() avoids a
    # broken env var crashing every other rapid-mlx subcommand at
    # parser-build time.) Resolved early because the CORS allowlist
    # below auto-includes whatever this resolves to.
    try:
        chat_frontend = _resolve_chat_frontend(args.chat_frontend)
    except ValueError as exc:
        print(f"share: {exc}", file=sys.stderr)
        sys.exit(2)

    extra_serve_args: list[str] = []
    # ``args.thinking`` comes from BooleanOptionalAction so ``--thinking``
    # turns it on and ``--no-thinking`` (or the default) turns it off. We
    # forward ``--no-thinking`` to serve only when explicitly disabled —
    # serve's own default is on, so an explicit flag is needed.
    if not args.thinking:
        extra_serve_args.append("--no-thinking")

    # Default CORS allowlist — the rapidmlx chat-frontend ecosystem.
    # Anyone running with ``--chat-frontend`` pointing elsewhere gets
    # that origin automatically appended. ``--cors-origins '*'`` opts
    # back into the wide-open behaviour for users running OpenWebUI /
    # other local chat UIs.
    origins = _resolve_cors_origins(args.cors_origins, chat_frontend)
    extra_serve_args.append("--cors-origins")
    extra_serve_args.extend(origins)

    # Forward the rate-limit cap to the child. The child's own default
    # is 0 (disabled), which on a public share is a leaked-key DoS
    # amplifier — a hostile client can saturate the M3 with as many
    # concurrent ``/v1/chat/completions`` as it wants. We default share
    # to 120 rpm (2/sec) at the argparse layer; 0 here is explicitly
    # ``do not forward`` so power users can opt out.
    if args.rate_limit > 0:
        extra_serve_args.append("--rate-limit")
        extra_serve_args.append(str(args.rate_limit))

    api_key = secrets.token_hex(24)
    # Port parsing is lazy on purpose: validating RAPID_MLX_SHARE_PORT at
    # parser-build time crashes ``rapid-mlx models`` (and every other
    # unrelated subcommand) when the env var is set to garbage.
    raw_port = os.environ.get(_PORT_ENV_VAR) if args.port is None else None
    try:
        if raw_port is not None:
            preferred_port = int(raw_port)
        elif args.port is not None:
            # ``is not None`` (not truthy): an explicit ``--port 0`` is a
            # user error that should surface as exit-2, not get silently
            # rewritten to the 8765 default.
            preferred_port = args.port
        else:
            preferred_port = 8765
    except ValueError:
        print(
            f"{_PORT_ENV_VAR} must be an integer (got {raw_port!r})",
            file=sys.stderr,
        )
        sys.exit(2)
    if not (1 <= preferred_port <= 65535):
        print(
            f"share port {preferred_port} is outside the valid range (1-65535)",
            file=sys.stderr,
        )
        sys.exit(2)
    # _pick_port raises RuntimeError if the OS can't allocate any port
    # (would happen on a maxed-out ephemeral pool). Surface as a normal
    # exit, not a raw traceback. (DeepSeek round-5 BLOCKING #2.)
    try:
        port = _pick_port(preferred_port)
    except RuntimeError as exc:
        print(f"share: {exc}", file=sys.stderr)
        sys.exit(1)
    state_dir = _state_dir()
    serve_log = state_dir / "serve.log"

    # Relay URL — defaults to the production rapidserver Worker, but
    # operator-set ``RAPID_MLX_RELAY_URL`` overrides (self-host /
    # smoke test against ``wrangler dev``).
    relay_url = os.environ.get("RAPID_MLX_RELAY_URL", ws_tunnel.DEFAULT_RAPIDSERVER_WSS)
    # Refuse non-wss schemes early so a misconfigured env doesn't
    # silently fall through to a stalled handshake.
    if not (relay_url.startswith("wss://") or relay_url.startswith("ws://")):
        print(
            f"share: RAPID_MLX_RELAY_URL must start with wss:// or ws:// "
            f"(got {relay_url!r})",
            file=sys.stderr,
        )
        sys.exit(2)

    # Convert SIGTERM into a KeyboardInterrupt so the existing finally
    # block runs cleanup. Without this, a supervisor (systemd, docker,
    # ``kill <pid>``) terminates the share parent and orphans the serve
    # child + WS tunnel thread, leaking a public tunnel until the user
    # notices. ``original_sigterm`` is the handler we replace; we restore
    # it on function exit so future code in the same process (e.g.
    # command-chaining) sees the prior behavior instead of inheriting
    # our KeyboardInterrupt translator. (DeepSeek round-3 NIT #2.)
    def _term_handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    original_sigterm = signal.signal(signal.SIGTERM, _term_handler)

    serve_proc: subprocess.Popen[bytes] | None = None
    tunnel: ws_tunnel.TunnelClient | None = None
    tunnel_thread = None
    # Codex round-1 BLOCKING: an OOM or crash in the serve child would
    # previously bubble out of ``serve_proc.wait()`` as a non-zero return
    # code that the parent silently discarded — so a failed share looked
    # like a successful one to systemd / docker / supervisor wrappers.
    # Capture the exit code here and translate to a non-zero exit at the
    # very end (after cleanup). User-interrupt paths (KeyboardInterrupt)
    # keep their exit-0 contract since the operator chose to stop.
    serve_exit_code = 0
    try:
        print(f"Starting rapid-mlx serve ({alias} on :{port})…", file=sys.stderr)
        serve_proc = _spawn_serve(
            alias=alias,
            port=port,
            api_key=api_key,
            log_path=serve_log,
            extra_args=extra_serve_args,
        )
        if not _wait_for_healthz(port, serve_proc):
            print(
                f"serve exited before becoming ready — see {serve_log}",
                file=sys.stderr,
            )
            sys.exit(1)
        # Auth-gated proof: even though /healthz returned 200, confirm the
        # process answering on this port is ours (it has our key). On a
        # busy host another local serve could win the race to bind the
        # same port we asked for, and forwarding THAT process over our
        # tunnel would leak someone else's model + their data. Bearer-
        # gating /v1/models eliminates this class of bug — no other
        # process has our key.
        if not _verify_auth_gate(port, api_key):
            print(
                f"serve on :{port} did not answer authenticated /v1/models — "
                f"another process may be bound to the same port. Aborting "
                f"before opening a public tunnel.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Connecting to relay {relay_url}…", file=sys.stderr)
        tunnel = ws_tunnel.TunnelClient(local_port=port, relay_url=relay_url)
        tunnel_thread = tunnel.run_in_thread()
        # 30s ceiling is generous: a healthy WS handshake completes in
        # well under a second; anything beyond that means the relay is
        # down or the user's outbound network is blocking WSS.
        if not tunnel.ready_event.wait(timeout=30):
            err = tunnel.error
            print(
                f"share: WS tunnel did not connect to {relay_url} within 30s",
                file=sys.stderr,
            )
            if err is not None:
                print(f"   reason: {err}", file=sys.stderr)
            sys.exit(1)
        if tunnel.error is not None:
            print(f"share: WS tunnel failed: {tunnel.error}", file=sys.stderr)
            sys.exit(1)

        # End-to-end probe: bearer-authed /v1/models through the public
        # URL. Passes only if (a) the WS is up, (b) the worker DO is
        # wired, (c) our local serve is answering through the tunnel.
        # Without this we'd happily print a banner whose URL silently
        # 503s on first request.
        if not ws_tunnel.wait_for_public_url(tunnel.public_url, api_key, timeout=30):
            print(
                f"share: public URL {tunnel.public_url} did not respond within 30s",
                file=sys.stderr,
            )
            sys.exit(1)

        # rapid-mlx serve registers the model under its HF id, not the
        # short alias the user typed — so the curl example needs that
        # name to actually run. Falls back to the typed alias if the
        # /v1/models probe fails (the banner still prints).
        display_model = _resolve_served_model_name(port, api_key) or alias
        # ``flush=True`` is load-bearing: when stdout is a pipe
        # (``rapid-mlx share … | tee``), Python block-buffers and the
        # banner doesn't reach the terminal until the process exits.
        print(
            warning.render(
                tunnel.public_url,
                api_key,
                display_model,
                tunnel.tunnel_id,
                chat_frontend,
            ),
            flush=True,
        )

        # Monitor BOTH the serve subprocess and the WS tunnel thread.
        # Share is healthy only while both are alive. Polling at 1s is
        # fine — both surfaces have their own logs; we only need to
        # notice "exited", not forward output.
        #
        # Codex round-6 (preserved from frpc-era) BLOCKING: a child
        # exiting with status 0 is ALSO a share failure — the public
        # URL has disappeared even if the exit was "clean" (uvicorn
        # graceful shutdown, a direct ``kill <pid>`` of the child,
        # etc.). Only the parent's KeyboardInterrupt path is allowed
        # to keep exit 0; every other exit-from-the-monitor-loop
        # translates to a non-zero share exit code so supervisors
        # restart us.
        while True:
            serve_rc = serve_proc.poll()
            if serve_rc is not None:
                serve_exit_code = serve_rc if serve_rc != 0 else 1
                if serve_rc == 0:
                    print(
                        f"share: serve process exited cleanly but the "
                        f"public share is no longer live — see {serve_log}.",
                        file=sys.stderr,
                    )
                break
            if tunnel.closed_event.is_set():
                # WS dropped after the banner — public URL is dead. Use
                # a non-zero exit so supervisor wrappers restart us; the
                # serve child is still alive, terminated in cleanup.
                err = tunnel.error
                suffix = f": {err}" if err is not None else ""
                print(
                    f"share: WS tunnel disconnected{suffix}. Stopping serve.",
                    file=sys.stderr,
                )
                serve_exit_code = 1
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping share…", file=sys.stderr)
    finally:
        # DeepSeek round-2 NIT: if a second SIGTERM arrives mid-cleanup,
        # the installed handler raises KeyboardInterrupt again and we
        # leak the serve child. Ignore SIGTERM for the duration of
        # cleanup — supervisor "kill -9" can still force us, that's
        # fine.
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
        # Close the WS first: cheap, and stops the worker from sending
        # any more inbound requests at us during serve teardown.
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
            except OSError:
                pass
        # Restore whatever SIGTERM handler we replaced. Keeps share_command
        # idempotent within the same Python process.
        try:
            signal.signal(signal.SIGTERM, original_sigterm)
        except (ValueError, OSError, TypeError):
            pass

    # Cleanup is done; surface a non-zero share exit code whenever the
    # monitor loop broke out (either child exited, for any reason). The
    # in-loop messages already wrote the actionable details to stderr;
    # this branch just translates to the process exit code. Ctrl-C took
    # the KeyboardInterrupt branch above without ever assigning
    # ``serve_exit_code`` so it stays 0 and the parent exits cleanly —
    # the only path where a share shutdown is "successful".
    if serve_exit_code:
        sys.exit(1)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Wire up the ``share`` subcommand onto the top-level CLI parser."""
    p = subparsers.add_parser(
        "share",
        help="Expose a local model behind a public URL via rapidmlx.com",
        description=(
            "Start rapid-mlx serve and open a public Cloudflare-fronted "
            "URL on rapidmlx.com so you can use the model from a different "
            "device — or share it with a friend. Press Ctrl-C to stop."
        ),
    )
    p.add_argument(
        "model",
        help="Alias to serve (same names as `rapid-mlx serve`, e.g. qwen3.5-4b-4bit)",
    ).completer = alias_completer
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Local port to bind serve to (default: 8765, or "
            "$RAPID_MLX_SHARE_PORT if set)"
        ),
    )
    # BooleanOptionalAction is the only way to get both ``--thinking``
    # and ``--no-thinking`` from a single declaration. The previous
    # ``store_true`` + ``default=True`` was unreachable — there's no
    # ``--no-no-thinking`` and the flag silently couldn't be disabled.
    p.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward thinking-mode behavior to serve. Default off "
            "(``--no-thinking``) so chat UIs see content immediately "
            "instead of waiting on a <think> prelude. Pass ``--thinking`` "
            "to keep upstream defaults."
        ),
    )
    p.add_argument(
        "--cors-origins",
        nargs="+",
        default=None,
        metavar="ORIGIN",
        help=(
            "Pass --cors-origins to serve. Accepts multiple values, same "
            "shape as ``rapid-mlx serve --cors-origins``. Default: the "
            "rapidmlx chat-frontend allowlist (rapid-pro.pages.dev, "
            "rapid-pro.quicksilverpro.io, rapidmlx.com, chat.rapidmlx.com) "
            "plus whatever ``--chat-frontend`` resolves to. Pass '*' to "
            "relax for browser chat UIs you host elsewhere (e.g. local "
            "Open WebUI). Example: --cors-origins http://localhost:3000."
        ),
    )
    p.add_argument(
        "--rate-limit",
        type=int,
        default=120,
        metavar="RPM",
        help=(
            "Per-client requests/minute cap forwarded to the spawned "
            "``rapid-mlx serve``. Default: 120 (2/sec) — high enough for "
            "tool-using power users and Beam-mode parallel completions, "
            "low enough that a leaked share key can't burst-DoS the "
            "publisher's M3. Set 0 to disable the cap entirely."
        ),
    )
    p.add_argument(
        "--chat-frontend",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "Override the one-click chat link printed in the share banner. "
            "Default: https://rapid-pro.quicksilverpro.io (or $RAPID_MLX_CHAT_FRONTEND "
            "if set). The frontend must implement the rapidmlx splash "
            "share-key protocol — point this at your own fork if you host "
            "one. Pass an empty string ('') to suppress the chat link "
            "entirely (useful for OpenWebUI and other frontends that don't "
            "speak the splash protocol; the URL+Key lines below still let "
            "you wire it up by hand)."
        ),
    )
