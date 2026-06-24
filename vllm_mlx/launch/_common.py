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
    """
    if not path.exists():
        return None
    ts = int(time.time())
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    # Collision avoidance for two invocations in the same second
    # (unlikely interactively but trivially reachable in tests). Walk
    # the suffix counter forward until we find an unused name; we don't
    # care about the counter value being meaningful.
    counter = 0
    while bak.exists():
        counter += 1
        bak = path.with_suffix(path.suffix + f".bak.{ts}.{counter}")
    bak.write_bytes(path.read_bytes())
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
