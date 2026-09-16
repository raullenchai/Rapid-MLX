# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the per-client launch adapters.

The four adapters in this package each patch a single JSON config file,
so they all need the same three primitives:

* Discover the config path on this host (respecting both macOS and
  Linux conventions where each client's authors picked something
  different).
* Back the existing file up to ``<path>.bak.<ts>`` so the user can
  recover if our patch corrupts it — *every* adapter MUST call
  :func:`backup_existing` BEFORE the atomic rename.
* Atomically write the patched config to ``<path>.new`` and rename it
  over the target so a Ctrl-C between bytes never leaves a half-written
  JSON file on disk.

Keeping these helpers here (rather than re-implemented per adapter)
means a fix to the atomic-write logic — e.g. tightening the temp-file
permissions, switching to ``os.fsync`` on Linux — applies to every
client at once.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path


def backup_existing(path: Path) -> Path | None:
    """Copy ``path`` to ``path.bak.<unix-ts>`` if it exists; return the
    backup path (or ``None`` when the original didn't exist yet).

    A unique timestamp suffix means a user who runs ``rapid-mlx launch
    cline`` twice in the same session gets *two* recoverable backups
    rather than one overwriting the other. The backup is the FULL byte
    content of the original — not a JSON re-serialisation — so a config
    with trailing comments or odd whitespace round-trips losslessly
    even if we'd rewrite it.

    Prints the backup location to stderr so a user reading the launch
    command's output sees "backup at <path>" without it being mixed
    into the success stdout (which scripts may parse).

    The backup is created 0600 and then given the original's mode. A
    plain ``write_bytes`` would have created it ``0666 & ~umask`` — 0644
    on a default install — while :func:`atomic_write_json` hands the
    config itself the 0600 that ``mkstemp`` produces. Since
    ``rapid-mlx launch`` writes ``RAPID_MLX_API_KEY`` into these files,
    that difference put the live bearer token in a world-readable file
    next to the protected one, for any other local account to read.
    """
    if not path.exists():
        return None
    ts = int(time.time())
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    # Collision avoidance for two invocations in the same second
    # (unlikely interactively but trivially reachable in tests). Walk
    # the suffix counter forward until we find an unused name; we don't
    # care about the counter value being meaningful.
    #
    # ``O_EXCL`` is what actually makes the name ours: the ``exists()``
    # probe alone is a check-then-act, so two concurrent launches could
    # settle on the same counter and one would silently overwrite the
    # other's backup. Losing on the open just advances the counter.
    # One open, then read AND stat through that same descriptor. Looking the
    # pathname up twice lets an editor's atomic replace land in between, so the
    # backup could hold the OLD secret while wearing the NEW file's looser
    # mode.
    counter = 0
    src_fd = os.open(path, os.O_RDONLY)
    try:
        src = os.fstat(src_fd)
        # ``chunks`` then one join, not ``data += chunk``: the latter copies
        # the whole accumulated buffer every iteration, which is quadratic in
        # file size.
        chunks: list[bytes] = []
        while chunk := os.read(src_fd, 1 << 20):
            chunks.append(chunk)
        data = b"".join(chunks)
        # Security metadata has to come off the SAME descriptor as the bytes
        # and the mode. Reading it by pathname further down would let an
        # editor's atomic replace answer for a file we did not copy: the
        # backup would hold the OLD secret while the ACL decision below came
        # from the NEW file. That is the same TOCTOU the single open already
        # closed for st_mode, just wearing a different field.
        # ``None`` means "could not verify", which the mode policy below
        # treats as "do not reproduce group/other bits". macOS ships no
        # os.listxattr at all, and a macOS ACL is not an xattr we could read
        # even if it did — so on the platform this project actually targets
        # we can never establish ACL equivalence, and a 0644 source carrying
        # a deny entry would otherwise yield a 0644 backup that denies
        # nobody.
        listxattr = getattr(os, "listxattr", None)
        src_xattrs: tuple[str, ...] | None = None
        if listxattr is not None:
            try:
                src_xattrs = tuple(listxattr(src_fd))
            except OSError:
                src_xattrs = None
    finally:
        os.close(src_fd)
    while True:
        try:
            fd = os.open(bak, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            counter += 1
            bak = path.with_suffix(path.suffix + f".bak.{ts}.{counter}")
            continue
        break
    # Everything from here on addresses the DESCRIPTOR, never ``bak`` the
    # name. Once O_EXCL has handed us the file, a pathname-based chown/stat/
    # chmod could still be redirected: anyone who can write the config's
    # DIRECTORY can unlink our backup and drop a symlink in its place between
    # two of those calls, and we would then hand their target our ownership
    # or our mode. fchown/fstat/fchmod cannot be pointed at another file.
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        # Match the original once the bytes are down. Created restrictive
        # first so the contents are never briefly readable at a wider mode;
        # a source that is itself group/world-readable keeps that, because
        # the backup should not be MORE exposed than what it copies, and
        # need not be less.
        mode = src.st_mode & 0o7777
        if mode & 0o077:
            # Group/other bits only mean "no wider than the source" if the
            # backup answers to the same principals. A new file takes the
            # DIRECTORY's group, which need not be the source's — copying
            # 0640 from an alice:secrets config onto an alice:staff backup
            # would hand the key to all of staff. Mode bits are numbers, not
            # an audience.
            #
            # Adopt the source's group where the OS allows it (it does when
            # the caller is a member, which is the ordinary case for one's
            # own dotfiles), and otherwise keep the file owner-only. Refusing
            # to widen is the safe direction: a backup that is tighter than
            # its source still restores.
            try:
                os.fchown(fd, -1, src.st_gid)
            except (PermissionError, OSError):
                mode &= 0o700
            else:
                if os.fstat(fd).st_gid != src.st_gid:
                    mode &= 0o700
            # An ACL can DENY a principal the mode bits would otherwise admit,
            # and a freshly created file carries none. Reproducing 0644
            # without the deny entry hands the file to exactly the account the
            # source shut out. There is no stdlib API to copy one, so when the
            # source carries a security/ACL xattr — or when the list could not
            # be read at all — the backup stays owner-only rather than
            # pretending the numbers tell the whole story.
            if src_xattrs is None or any(
                x.startswith(("com.apple.system.Security", "system.posix_acl"))
                for x in src_xattrs
            ):
                mode &= 0o700
        os.fchmod(fd, mode)
    finally:
        os.close(fd)
    print(f"  backup: {bak}", file=sys.stderr)
    return bak


def atomic_write_json(path: Path, data: object) -> None:
    """Write ``data`` to ``path`` as pretty-printed JSON atomically.

    We write to a sibling temp file in the same directory (``rename`` is
    only atomic within a single filesystem) and then ``os.replace`` it
    over the target. A Ctrl-C between the write and the replace leaves
    the temp file behind — recoverable — instead of a half-written
    config that breaks the client on next launch.

    The directory is mkdir'd with ``parents=True`` so we can patch a
    config for a never-before-run client (e.g. a user who installed
    Continue but never opened it). JSON is written with a trailing
    newline to match what every editor's "format on save" produces.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``delete=False`` so we control the unlink path — the temp file
    # only goes away via the ``os.replace`` rename below (on success)
    # or stays behind on crash for recovery.
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".new",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False)
            f.write("\n")
            # fsync so the bytes hit disk before the rename. Without
            # this an OS crash between rename and flush could leave us
            # with a renamed but empty file on the target.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Clean up the temp file on any error path so we don't litter
        # the user's config dir with ``settings.json.XYZ.new`` files.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_json_lenient(path: Path) -> dict:
    """Read ``path`` as JSON, returning ``{}`` if missing or unreadable.

    Cline / Cursor / Continue all *technically* require strict JSON, but
    in practice users hand-edit these files and occasionally end up with
    trailing commas or comments that ``json.loads`` rejects. We don't
    pretend to be a JSON5 parser — we just refuse to overwrite a config
    we can't safely round-trip. Caller passes the resulting dict to
    :func:`atomic_write_json` only after merging in the new keys.

    Returns ``{}`` only on missing or syntactically-empty file. A
    JSONDecodeError is RAISED — the launch command catches it and tells
    the user "your existing config is invalid, please fix or remove it"
    rather than silently nuking their edits.
    """
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    return json.loads(raw)


def mac_app_installed(app_name: str) -> bool:
    """Return True if the named ``.app`` bundle exists under either
    ``/Applications`` or ``~/Applications`` on macOS.

    Most IDE-class clients ship as a regular ``.app`` bundle, so this is
    the cheapest cross-version check ("did the user install the app at
    all"). Doesn't try to introspect the bundle's Info.plist — version
    detection is out of scope for ``launch``.
    """
    candidates = [
        Path("/Applications") / f"{app_name}.app",
        Path.home() / "Applications" / f"{app_name}.app",
    ]
    return any(p.exists() for p in candidates)


def which(cmd: str) -> str | None:
    """Locate ``cmd`` on the PATH, returning the absolute path or None.

    Thin wrapper around ``shutil.which`` so the per-client modules can
    keep a single import surface (``from . import _common``). Pulled
    out (rather than re-imported per file) so a future swap to a richer
    PATH resolver — e.g. one that also checks ``~/.local/bin`` even
    when it's not on PATH — only touches one place.
    """
    import shutil

    return shutil.which(cmd)
