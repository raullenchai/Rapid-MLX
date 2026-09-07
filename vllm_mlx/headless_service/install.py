# SPDX-License-Identifier: Apache-2.0
"""Validation + transactional install for the headless service.

``install`` is where most of the #2859 safety contract lives:

* least-privilege service account (must exist, uid>500, not root/admin);
* the install is transactional and, with ``--dry-run``, prints every
  mutation and performs none;
* a port-race guard refuses to install when a server already answers on
  the target port (the "cannot silently cohabit with a Desktop-managed
  server" criterion);
* secrets (``--api-key``) are refused — a system-domain daemon boots before
  login and must never carry a secret in argv or env;
* the model must already be cached under the service account's HOME (the
  documented workflow downloads once before the machine goes headless).

We shell out to ``launchctl``/``install`` rather than raising a Python
daemon, so the command surface stays exactly what the manual runbook
already documented — this command just makes it safe and one-shot.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .common import (
    DEFAULT_DOMAIN,
    DEFAULT_LABEL,
    LAUNCH_DAEMONS_DIR,
    log_dir_for,
    user_uid,
)
from .plist import build_plist_dict, serialize_plist

FORBIDDEN_SECRET_ARG_TOKENS = ("--api-key", "--api_key")
FORBIDDEN_BIND_ARG_TOKENS = ("--host", "--port", "--listen-fd")


class ServiceInstallError(Exception):
    """A user-facing install failure (rendered without a traceback)."""


# ---------------------------------------------------------------------------
# Pure validation helpers (no I/O) — exercised directly by tests.
# ---------------------------------------------------------------------------


def is_root() -> bool:
    return os.geteuid() == 0


def is_admin_user(user: str) -> bool:
    """Best-effort: is ``user`` a member of the local admin group (gid 80)?"""
    import grp
    import pwd

    try:
        group = grp.getgrgid(80)
        record = pwd.getpwnam(user)
    except KeyError:
        return False
    try:
        return group.gr_gid in os.getgrouplist(user, record.pw_gid)
    except OSError:
        # Keep a conservative fallback for directory-service lookup failures.
        return record.pw_gid == group.gr_gid or user in group.gr_mem


def resolve_executable(home: Path) -> str:
    """Resolve the daemon's executable against the SERVICE account's home.

    The documented appliance installs Rapid-MLX into the service account's
    own ``~/.local/bin/rapid-mlx`` (a stable symlink the one-line installer
    creates). That — NOT the operator/admin's binary — is the only correct
    thing for the daemon to execute (launchd runs it as the service user, so
    a binary the service account cannot exec, or that belongs to another
    account's install, would fail at runtime or run wrong code). We therefore
    REQUIRE it: if the service user hasn't installed rapid-mlx yet, the
    operator should run the one-line installer as that account (as the guide
    directs) rather than silently embedding a wrong-owner binary.
    """
    candidate = home / ".local" / "bin" / "rapid-mlx"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    raise ServiceInstallError(
        f"no Rapid-MLX binary at {candidate} for the service account. Install "
        "Rapid-MLX as the service account first (log in as it and run the "
        "one-line installer; see docs/guides/headless-macos-service.md "
        "'Prepare the service account'), then re-run install."
    )


def validate_service_account(user: str) -> None:
    """Raise :class:`ServiceInstallError` unless ``user`` is a viable
    least-privilege service account (exists, uid>500, not root, not admin)."""
    from .common import home_for_user

    if not user:
        raise ServiceInstallError(
            "a --service-user is required: the daemon runs as a dedicated, "
            "least-privilege non-administrator account (create one first, "
            "then pass its name)."
        )
    uid = user_uid(user)
    if uid is None or home_for_user(user) is None:
        raise ServiceInstallError(
            f"service account {user!r} does not exist. Create a dedicated, "
            "non-administrator local account first (see "
            "docs/guides/headless-macos-service.md 'Prepare the service "
            "account'), then re-run install."
        )
    if uid == 0 or user == "root":
        raise ServiceInstallError(
            "refusing to run the system service as root: a least-privilege "
            "deployment requires a dedicated non-administrator account."
        )
    if uid < 501:
        raise ServiceInstallError(
            f"refusing to use system account {user!r} (uid {uid}) as the "
            "service account."
        )
    if is_admin_user(user):
        raise ServiceInstallError(
            f"refusing to use administrator account {user!r}: the service "
            "account must be non-privileged (not a member of the admin group)."
        )


def refuse_secret_flags(serve_args: tuple[str, ...]) -> None:
    """Reject serve flags that would embed a secret into argv/plist."""
    for tok in serve_args:
        option = tok.split("=", 1)[0]
        if option in FORBIDDEN_SECRET_ARG_TOKENS:
            raise ServiceInstallError(
                "refusing to put an API key in the service definition: argv "
                "and plist EnvironmentVariables are visible to other local "
                "processes. Bind to loopback and put a TLS-terminating "
                "reverse proxy in front instead (the documented pattern)."
            )
        if option in FORBIDDEN_BIND_ARG_TOKENS:
            raise ServiceInstallError(
                f"refusing duplicate bind option {option!r} after `--`: use "
                "the service install --host/--port options instead. "
                "--listen-fd is incompatible with a launchd service that "
                "does not configure socket activation."
            )


def _cache_root_present(home: Path, model: str) -> bool:
    """Best-effort: is the HF cache dir for ``model`` present under ``home``?"""
    cache_root = home / ".cache" / "huggingface" / "hub"
    if not cache_root.is_dir():
        return False
    cache_id = model.replace("/", "--")
    return (cache_root / f"models--{cache_id}").is_dir()


# ---------------------------------------------------------------------------
# Drive helpers — the actual install/rollback mutation steps.
# ---------------------------------------------------------------------------


def _plist_path(label: str) -> Path:
    return LAUNCH_DAEMONS_DIR / f"{label}.plist"


def _build_plist_bytes(
    *,
    label: str,
    user: str,
    executable: str,
    model: str,
    home: Path,
    host: str,
    port: int,
    serve_args: tuple[str, ...],
    config_path: Path | None = None,
) -> bytes:
    log_dir = log_dir_for(user)
    if log_dir is None:
        raise ServiceInstallError(f"cannot resolve home for service account {user!r}")
    config = build_plist_dict(
        label=label,
        user=user,
        executable=executable,
        model=model,
        home=home,
        log_dir=log_dir,
        host=host,
        port=port,
        serve_args=serve_args,
        config_path=config_path,
    )
    return serialize_plist(config)


def _port_busy(host: str, port: int) -> bool:
    """True if a TCP server already answers on ``(host, port)``.

    A tiny loopback connect probe — no privileged syscalls. Guards the
    "cannot silently race or cohabit with a Desktop-managed server on the
    same port" acceptance criterion.
    """
    import socket

    try:
        with socket.create_connection((_probe_host(host), port), timeout=1.0):
            return True
    except OSError:
        return False


def _probe_host(bind_host: str) -> str:
    """Return a connectable local address for a configured bind address."""
    if bind_host == "0.0.0.0":
        return "127.0.0.1"
    if bind_host in {"::", "[::]"}:
        return "::1"
    return bind_host


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, capture_output=True, text=True)


def _mutation_list(
    *,
    label: str,
    plist_buf: bytes,
    plist_path: Path,
    host: str,
    port: int,
) -> list[str]:
    return [
        "write versioned service config atomically (mode 600)",
        f"write secure temporary plist ({len(plist_buf)} bytes, mode 600)",
        "validate temporary plist with plutil -lint",
        f"install -o root -g wheel -m 644 temporary plist -> {plist_path}",
        f"launchctl bootstrap {DEFAULT_DOMAIN} {plist_path}",
        f"poll {host}:{port}/readyz until ready (max 120s)",
    ]


def _stage_plist(plist_buf: bytes) -> Path:
    """Write ``plist_buf`` to an unguessable, owner-only temporary file.

    ``rapid-mlx service install`` normally runs under sudo.  A predictable
    path in a world-writable directory would let another local user place a
    symlink there before the root process writes it.  ``mkstemp`` creates the
    file atomically with O_EXCL and mode 0600, closing that overwrite path.
    """
    fd, raw_path = tempfile.mkstemp(prefix="rapid-mlx-service-", suffix=".plist")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(plist_buf)
    except BaseException:
        try:
            os.unlink(raw_path)
        except OSError:
            pass
        raise
    return Path(raw_path)


def _install_service_config(*, user: str, home: Path, path: Path, data: bytes) -> None:
    """Install a root-owned readable config; it is validated secret-free."""
    import pwd

    from .config import atomic_write_definition, ensure_config_dir

    account = pwd.getpwnam(user)
    ensure_config_dir(home, uid=account.pw_uid, gid=account.pw_gid)
    atomic_write_definition(path, data)


def _readyz_ready(host: str, port: int) -> bool:
    """True when ``/readyz`` reports ready.

    rapid-mlx ``/readyz`` returns HTTP 200 once the process is able to serve
    (sets ``"ready": true`` in the body once the model is loaded; it returns
    503 while draining). Matching the smoke script's contract, we require
    BOTH an HTTP 200 status line AND a ``"ready": true`` in the body — a bare
    200 during early boot (model still loading) does NOT count.
    """
    import socket

    try:
        with socket.create_connection((_probe_host(host), port), timeout=1.0) as sock:
            sock.settimeout(1.0)
            sock.sendall(b"GET /readyz HTTP/1.1\r\nHost: x\r\n\r\n")
            data = sock.recv(4096)
    except OSError:
        return False
    head, _, rest = data.partition(b"\r\n\r\n")
    if b"200" not in head:
        return False
    # Compact JSON separators or pretty-printed — compare whitespace-free.
    body = b"".join(rest.split()).lower()
    return b'"ready":true' in body


def _wait_ready(host: str, port: int, timeout_s: int = 120) -> bool:
    """Poll ``/readyz`` until it reports ready or ``timeout_s`` elapses.

    Mirrors the readiness loop in ``scripts/headless_service_smoke.sh`` so
    the command and the smoke test agree on what "healthy" means.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _readyz_ready(host, port):
            return True
        time.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# Uninstall — inverse of install. Never touches models/data/logs.
# ---------------------------------------------------------------------------


def uninstall_command(args) -> int:
    """``rapid-mlx service uninstall`` — bootout + remove the plist.

    Deliberately conservative: it ONLY removes the launchd registration and
    the plist file. The model cache, virtual environment, and logs are left
    untouched (the issue's "uninstall removes service registration without
    deleting models or user data" criterion). ``--dry-run`` prints the exact
    removal steps without performing them. Re-running after a partial
    teardown is idempotent-safe (bootout tolerates a missing job).
    """
    label = getattr(args, "label", None) or DEFAULT_LABEL
    dry_run = bool(getattr(args, "dry_run", False))
    plist_path = _plist_path(label)

    bootout_cmd = ["launchctl", "bootout", f"{DEFAULT_DOMAIN}/{label}"]
    rm_cmd = ["rm", "-f", str(plist_path)]

    if dry_run:
        print(
            "Dry run — would remove service registration (models/cache/logs "
            "are NEVER touched):"
        )
        print(f"  [DRY-RUN] {' '.join(bootout_cmd)}")
        print(f"  [DRY-RUN] {' '.join(rm_cmd)}")
        if not plist_path.is_file():
            print("  (service already absent — nothing to do)")
        return 0

    if not is_root():
        print(
            "error: removing a system LaunchDaemon requires root. Re-run "
            f"with sudo (boots out {label} and removes {plist_path}).",
            file=sys.stderr,
        )
        return 1

    # bootout first (so KeepAlive cannot respawn mid-teardown), tolerate a
    # missing job, then remove the plist.
    try:
        subprocess.run(bootout_cmd, check=False, capture_output=True, text=True)
        subprocess.run(rm_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"error: uninstall failed: {exc}", file=sys.stderr)
        return 2

    print(f"uninstalled {label}. Model cache and logs were left in place.")
    return 0


# ---------------------------------------------------------------------------
# Command entry points.
# ---------------------------------------------------------------------------


def install_command(args) -> int:
    """``rapid-mlx service install`` — validate + bootstrap the daemon.

    Returns a process exit code (0 ok, 1 user error, 2 real failure).
    """
    from .common import home_for_user

    label = getattr(args, "label", None) or DEFAULT_LABEL
    model = getattr(args, "model", None) or "qwen3.5-4b-4bit"
    user = getattr(args, "service_user", None)
    host = getattr(args, "host", None) or "127.0.0.1"
    port = getattr(args, "port", None) or 8000
    serve_args = tuple(getattr(args, "serve_args", None) or ())
    if serve_args[:1] == ("--",):
        serve_args = serve_args[1:]
    dry_run = bool(getattr(args, "dry_run", False))

    # argparse marks --service-user required, but guard here too (direct calls
    # and dry-run tests bypass argparse): a clean error + mypy narrowing of
    # ``user`` to ``str`` for every call below.
    if not isinstance(user, str) or not user:
        print(
            "error: a --service-user is required: the daemon runs as a "
            "dedicated, least-privilege non-administrator account (create one "
            "first, then pass its name).",
            file=sys.stderr,
        )
        return 1

    try:
        validate_service_account(user)
        refuse_secret_flags(serve_args)
        home = home_for_user(user)
        assert home is not None
        # Resolve the binary against the SERVICE account's home, not the
        # operator's — the daemon runs as the service user and must execute
        # that user's install of rapid-mlx. Raises ServiceInstallError if the
        # service user hasn't installed it yet.
        executable = resolve_executable(home)
    except ServiceInstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    from .config import (
        SCHEMA_VERSION,
        ServiceConfig,
        config_bytes,
        config_path,
        credential_path,
    )

    try:
        service_config = ServiceConfig(
            schema_version=SCHEMA_VERSION,
            label=label,
            service_user=user,
            executable=executable,
            model=model,
            host=host,
            port=port,
            serve_args=serve_args,
            credential_file=str(credential_path(home, label)),
        ).validated()
    except ValueError as exc:
        print(f"error: invalid service configuration: {exc}", file=sys.stderr)
        return 1
    service_config_path = config_path(home, label)
    plist_buf = _build_plist_bytes(
        label=label,
        user=user,
        executable=executable,
        model=model,
        home=home,
        host=host,
        port=port,
        serve_args=serve_args,
        config_path=service_config_path,
    )
    plist_path = _plist_path(label)

    # Never silently replace a persistent definition.  A loaded job keeps
    # using its old in-memory configuration even if its plist is overwritten,
    # which would make a reinstall appear successful until the next reboot.
    # Explicit uninstall+install is deterministic and preserves models/logs.
    if plist_path.exists():
        print(
            f"error: {plist_path} already exists. Refusing an ambiguous "
            "in-place reinstall; run `sudo rapid-mlx service uninstall` "
            "first (models/cache/logs are preserved), then install again.",
            file=sys.stderr,
        )
        return 1

    # Port-race guard: refuse before touching anything if a server already
    # listens on the target port (unknown owner — do not silently cohabit).
    # This is a pre-flight *validation* outcome, so it fires in --dry-run
    # too — the operator rehearsing should learn "this port is taken" as
    # clearly as a real install would, rather than only at bootstrap time.
    if _port_busy(host, port):
        print(
            f"error: a server already answers on {host}:{port}. Refusing to "
            "install a service that would race/cohabit with it (it may be "
            "Desktop-managed). Stop that server or choose another --port.",
            file=sys.stderr,
        )
        return 1

    # Model-cache check: warn (not fail) if the model isn't cached under the
    # service account's HOME — the documented workflow pre-downloads it.
    if not _cache_root_present(home, model):
        print(
            f"warning: model {model!r} not found in the service account's HF "
            f"cache ({home}/.cache/huggingface/hub). Pre-download it as the "
            "service account so the daemon can boot offline at first start.",
            file=sys.stderr,
        )

    if dry_run:
        print("Dry run — would perform these steps (no changes made):")
        for step in _mutation_list(
            label=label,
            plist_buf=plist_buf,
            plist_path=plist_path,
            host=host,
            port=port,
        ):
            print(f"  [DRY-RUN] {step}")
        print(f"  service account: {user} (uid {user_uid(user)})")
        print(f"  executable: {executable}")
        print(f"  model: {model}")
        return 0

    if not is_root():
        print(
            "error: installing a system LaunchDaemon requires root. Re-run "
            f"with sudo (writes {plist_path} and bootstraps {label}). For a "
            "rehearsal that changes nothing, pass --dry-run first.",
            file=sys.stderr,
        )
        return 1

    # -- Real install, transactional. --------------------------------------
    for step in _mutation_list(
        label=label,
        plist_buf=plist_buf,
        plist_path=plist_path,
        host=host,
        port=port,
    ):
        print(f"  {step}")

    # True transaction state: whether we've installed the persistent plist,
    # so a failure below can roll back BOTH the loaded job AND the plist
    # file (a plist left in /Library/LaunchDaemons auto-loads on reboot).
    persistent_write_attempted = False
    config_write_attempted = False
    staged: Path | None = None
    try:
        # The stable plist only names this file. Configuration can later be
        # staged and applied transactionally without rewriting root-owned
        # launchd state.
        config_write_attempted = True
        _install_service_config(
            user=user,
            home=home,
            path=service_config_path,
            data=config_bytes(service_config),
        )
        # Stage + lint. check=False so a lint failure surfaces its specific
        # message instead of a generic "install step failed" traceback.
        staged = _stage_plist(plist_buf)
        lint = _run(["plutil", "-lint", str(staged)], check=False)
        if lint.returncode != 0:
            raise ServiceInstallError(f"plutil -lint failed: {lint.stderr.strip()}")
        # Install root-owned, 0644.
        persistent_write_attempted = True
        _run(
            [
                "install",
                "-o",
                "root",
                "-g",
                "wheel",
                "-m",
                "644",
                str(staged),
                str(plist_path),
            ]
        )
        # Bootstrap into the system domain (boot-time autostart). A job that is
        # already loaded without its plist is still an inconsistent state, not
        # a successful reinstall; fail and remove the newly installed plist.
        boot = _run(
            ["launchctl", "bootstrap", DEFAULT_DOMAIN, str(plist_path)],
            check=False,
        )
        if boot.returncode != 0:
            boot_err = boot.stderr.strip()
            raise ServiceInstallError(f"launchctl bootstrap failed: {boot_err}")
        # Wait for readiness; on failure roll back the load AND the plist so
        # nothing persists to auto-start on the next boot.
        if not _wait_ready(host, port):
            raise ServiceInstallError(
                f"service did not become ready on {host}:{port} within 120s; "
                f"rolled back. Check {home}/Library/Logs/Rapid-MLX/"
                "server.stderr.log"
            )
    except Exception as exc:
        # Best-effort rollback: remove the staged file AND the persistent
        # plist (never the models/data/logs), so a failed install cannot
        # leave a boot-persistent daemon behind.
        try:
            if staged is not None:
                staged.unlink()
        except OSError:
            pass
        if persistent_write_attempted:
            # A failed bootstrap can still leave a partially loaded job. Make
            # rollback idempotent by always attempting bootout before removing
            # the new persistent definition.
            _run(
                ["launchctl", "bootout", f"{DEFAULT_DOMAIN}/{label}"],
                check=False,
            )
            _run(["rm", "-f", str(plist_path)], check=False)
            if not isinstance(exc, ServiceInstallError):
                print(
                    f"info: removed {plist_path} to keep the failed install "
                    "from auto-starting on reboot",
                    file=sys.stderr,
                )
        if config_write_attempted:
            try:
                service_config_path.unlink()
            except OSError:
                pass
        if isinstance(exc, ServiceInstallError):
            print(f"error: {exc}", file=sys.stderr)
        else:
            print(f"error: install step failed: {exc}", file=sys.stderr)
        return 2

    # The persistent root-owned copy is installed; the private staging file is
    # no longer needed.  Failure to remove it is harmless but should not turn a
    # healthy service into a reported install failure.
    if staged is not None:
        try:
            staged.unlink()
        except OSError:
            pass

    print(f"Installed and running: {label} (model {model}, {host}:{port}).")
    print(
        "Verify with `rapid-mlx service status` and after a reboot "
        "`./scripts/headless_service_smoke.sh`."
    )
    return 0
