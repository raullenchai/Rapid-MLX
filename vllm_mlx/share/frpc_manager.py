# SPDX-License-Identifier: Apache-2.0
"""Lazy-download + lifecycle for the frpc reverse-tunnel client binary.

We ship the frpc binary out-of-band: not bundled in the wheel (keeps it
small), not required by users who never touch ``share``. The first
``rapid-mlx share`` invocation fetches the platform-matched build from
the upstream GitHub release and verifies it against a pinned sha256.
Subsequent runs reuse the cached binary in ``~/.cache/rapid-mlx/bin/``.

This mirrors the playwright/tiktoken pattern — wheels stay slim, but
``share`` feels native because the binary download happens once and
under 5 seconds for ~13 MB.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from ._constants import FRPC_SHA256, FRPC_VERSION

# Strict allow-lists for relay-provided strings interpolated into the frpc
# TOML config. The relay is trusted but compromise (or a bug in the control
# plane) shouldn't be able to inject arbitrary TOML — e.g. a subdomain like
# ``x"\nauth.token = "attacker`` would let a malicious payload override
# fields. Toml has no stdlib serializer, so we enforce charsets that can't
# require escaping in the first place.
_SUBDOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-]{1,256}$")
_HOST_RE = re.compile(r"^[a-zA-Z0-9.\-]{1,253}$")

logger = logging.getLogger(__name__)


def _platform_tag() -> str:
    """Resolve the host to one of the four supported frp release tags."""
    sysname = platform.system().lower()
    if sysname not in ("darwin", "linux"):
        raise RuntimeError(
            f"rapid-mlx share is only supported on macOS and Linux "
            f"(detected: {sysname}). Windows support is not on the "
            f"roadmap yet — open an issue if you need it."
        )
    machine = platform.machine().lower()
    arch = {
        "arm64": "arm64",
        "aarch64": "arm64",
        "x86_64": "amd64",
        "amd64": "amd64",
    }.get(machine)
    if arch is None:
        raise RuntimeError(
            f"unsupported CPU architecture {machine!r}; "
            f"frpc release ships arm64/amd64 only"
        )
    return f"{sysname}_{arch}"


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "rapid-mlx" / "bin"


def binary_path() -> Path:
    """Return the (possibly-not-yet-downloaded) path to frpc on this host."""
    return _cache_dir() / f"frpc-{FRPC_VERSION}"


def _download(url: str, dest: Path) -> None:
    # urlretrieve is sync + simple; for ~13 MB it's fine. We deliberately
    # don't show a progress bar — the call site prints a one-line status
    # message instead, so the UX stays predictable for non-TTY callers.
    urllib.request.urlretrieve(url, dest)  # noqa: S310 — trusted CDN


def ensure() -> Path:
    """Return a runnable frpc binary, downloading + verifying on first use."""
    binp = binary_path()
    if binp.exists():
        return binp

    tag = _platform_tag()
    expected_sha = FRPC_SHA256.get(tag)
    if expected_sha is None:
        raise RuntimeError(
            f"no pinned frpc sha256 for platform {tag!r}; "
            f"bump scripts/share_bump_frpc.sh"
        )

    url = (
        f"https://github.com/fatedier/frp/releases/download/"
        f"v{FRPC_VERSION}/frp_{FRPC_VERSION}_{tag}.tar.gz"
    )
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", dir=cache, delete=False) as tmp:
        archive = Path(tmp.name)

    try:
        logger.info("Fetching frpc v%s (%s) — first run only", FRPC_VERSION, tag)
        _download(url, archive)

        actual_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"frpc sha256 mismatch for {tag}: "
                f"expected {expected_sha}, got {actual_sha}"
            )

        # Extract just the frpc binary (the tarball also contains frps,
        # which we don't need on the client side). Write to a sibling
        # ``.part`` file and atomically rename on success — if
        # ``shutil.copyfileobj`` raises (disk full, broken pipe), a
        # half-written binary at ``binp`` would be reused on the next
        # ``ensure()`` call and frpc would crash mysteriously.
        binp_part = binp.with_suffix(binp.suffix + ".part")
        try:
            with tarfile.open(archive) as tf:
                member = next(
                    (m for m in tf.getmembers() if m.name.endswith("/frpc")),
                    None,
                )
                if member is None:
                    raise RuntimeError("frpc binary missing from release tarball")
                with tf.extractfile(member) as src, binp_part.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            binp_part.chmod(0o755)
            binp_part.replace(binp)
        finally:
            binp_part.unlink(missing_ok=True)
    finally:
        archive.unlink(missing_ok=True)

    return binp


def _validate_session_fields(server_addr: str, auth_token: str, subdomain: str) -> None:
    """Reject relay strings that wouldn't survive raw TOML interpolation.

    Defence-in-depth: even if the control plane is compromised or buggy,
    the worst-case is that ``share`` refuses to start — not that an
    attacker-controlled subdomain like ``x"\\nauth.token="bad`` injects
    extra TOML keys into the frpc config.
    """
    if not _HOST_RE.match(server_addr):
        raise RuntimeError(
            f"relay returned malformed server_addr {server_addr!r}; refusing"
        )
    if not _TOKEN_RE.match(auth_token):
        raise RuntimeError("relay returned malformed token; refusing")
    if not _SUBDOMAIN_RE.match(subdomain):
        raise RuntimeError(
            f"relay returned malformed subdomain {subdomain!r}; refusing"
        )


def render_config(
    *,
    server_addr: str,
    server_port: int,
    auth_token: str,
    subdomain: str,
    local_port: int,
) -> str:
    """Build a single-proxy frpc.toml. Kept stringly-typed because frpc
    only needs to read it once at startup — no config-object overhead.

    Authentication shape: we deliberately rely on frps's *server-plugin*
    Login hook (the control plane) to validate the session token, not
    frp's built-in shared-secret ``auth.token``. frpc serializes the
    ``metadatas`` table into the Login payload's ``metas`` map; the
    server-side plugin then reads ``metas.token`` (the de-prefixed
    field name on the wire) from that map. Putting the token in
    ``auth.token`` would route it through frp's built-in checker —
    that expects frps to hold the *same* token, which our
    zero-shared-secret deploy doesn't (and shouldn't). The plugin
    path is the source of truth.
    """
    _validate_session_fields(server_addr, auth_token, subdomain)
    if not (1 <= server_port <= 65535 and 1 <= local_port <= 65535):
        raise RuntimeError(
            f"relay returned out-of-range port (server={server_port} local={local_port})"
        )
    return (
        f'serverAddr = "{server_addr}"\n'
        f"serverPort = {server_port}\n"
        f"\n"
        f"# Token is consumed by the control-plane Login plugin via\n"
        f"# metas.token; do not move it to auth.token.\n"
        f'metadatas.token = "{auth_token}"\n'
        f"\n"
        f"[[proxies]]\n"
        f'name = "share-{subdomain}"\n'
        f'type = "http"\n'
        f"localPort = {local_port}\n"
        f'subdomain = "{subdomain}"\n'
    )


def spawn(config_path: Path, log_path: Path) -> subprocess.Popen[bytes]:
    """Start frpc as a child subprocess; stdout+stderr go to ``log_path``."""
    binp = ensure()
    log_fp = log_path.open("ab", buffering=0)
    return subprocess.Popen(
        [str(binp), "-c", str(config_path)],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
