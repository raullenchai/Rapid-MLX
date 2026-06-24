# SPDX-License-Identifier: Apache-2.0
"""Continue.dev launch adapter.

Continue.dev (VS Code and JetBrains extension) reads its config from
``~/.continue/config.json``. The relevant shape is a list of model
entries under the top-level ``models`` key, each one describing a
provider + base URL + model id. Pointing one of those entries at
``http://127.0.0.1:8000/v1`` is all that's required for Continue to
route chat at the local rapid-mlx server.

We append (rather than replace) the entry so a user with several
Continue models configured keeps all of them — we just add one tagged
``rapid-mlx`` and make it the default. On a re-run we de-dupe by name
so we don't keep stacking copies.
"""

from __future__ import annotations

from pathlib import Path

from . import _common

# Continue stores its config in ``~/.continue/`` (note: no XDG
# normalisation — Continue picks the same path on all platforms).
# Newer Continue versions also support ``config.yaml`` alongside
# ``config.json``; we target the JSON variant because (a) it's the
# default and (b) JSON round-trips cleanly without a YAML dep.
_CONFIG_DIR = Path.home() / ".continue"
_CONFIG_FILENAME = "config.json"

# Name we use for the model entry we insert. A user who already has a
# model named ``rapid-mlx`` on a re-run gets it updated in place rather
# than duplicated.
_MODEL_ENTRY_NAME = "rapid-mlx"


def detect() -> bool:
    """Return True when the Continue config dir exists.

    Detection is deliberately permissive — a user might have installed
    the extension but not opened it (no ``config.json`` yet) and we
    still want ``rapid-mlx launch continue-dev`` to succeed by creating
    the file. The minimum signal is the ``~/.continue/`` directory,
    which Continue mkdir's on extension activation.
    """
    return _CONFIG_DIR.exists()


def current_config_path() -> Path | None:
    """Return ``~/.continue/config.json``.

    Returns the path unconditionally (matching :mod:`claude_code`'s
    behaviour) — if Continue isn't installed, the dispatcher prints a
    "not detected" hint and the launch command exits non-zero, but the
    path itself is well-defined.
    """
    return _CONFIG_DIR / _CONFIG_FILENAME


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    config_path: Path | None = None,
) -> Path:
    """Insert (or update) a ``rapid-mlx`` model entry in Continue's
    config, then promote it to the default model.

    Shape of the inserted entry:

    .. code-block:: json

       {
         "title": "rapid-mlx",
         "provider": "openai",
         "model": "<model>",
         "apiBase": "<server_url>/v1",
         "apiKey": "<api_key>"
       }

    On a re-run, an existing entry with ``title == "rapid-mlx"`` is
    overwritten in place (so the user's preserved index in the models
    list stays stable) instead of being appended a second time.

    All other model entries — and the rest of the Continue config —
    are preserved verbatim.
    """
    path = config_path or current_config_path()
    assert path is not None

    existing = _common.load_json_lenient(path)
    _common.backup_existing(path)

    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    new_entry = {
        "title": _MODEL_ENTRY_NAME,
        "provider": "openai",
        "model": model,
        "apiBase": base_url,
        "apiKey": api_key,
    }

    models = list(existing.get("models", []))
    replaced = False
    for i, entry in enumerate(models):
        # ``entry`` may be a dict OR a string (Continue's older config
        # format accepted a bare model id). We only know how to update
        # dict entries; bare strings are left alone.
        if isinstance(entry, dict) and entry.get("title") == _MODEL_ENTRY_NAME:
            models[i] = new_entry
            replaced = True
            break
    if not replaced:
        models.append(new_entry)
    existing["models"] = models

    _common.atomic_write_json(path, existing)
    return path
