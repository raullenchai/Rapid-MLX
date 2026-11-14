# SPDX-License-Identifier: Apache-2.0
"""``--disable-disk-caches``: keep optional caches from being written to disk.

Every optional disk cache checks :func:`disabled` at the point where it would
read or write. The individual settings (``--kv-disk-checkpoint-interval``,
``APC_DISK_ENABLED``, ...) keep the values the operator gave; this switch
takes precedence over them, and :func:`overruled_settings` names the ones it
overrides so the server can report them at startup.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from threading import Lock

from ._env import env_truthy, is_falsey

ENV_VAR = "RAPID_MLX_DISABLE_DISK_CACHES"
FLAG = "--disable-disk-caches"
_AUTOLOAD_ENV = "RAPID_MLX_PREFIX_CACHE_AUTOLOAD"

_cli_disabled = False
_adapter_users = 0
_adapter_lock = Lock()


@contextmanager
def _suspend_for_adapter():
    """Disable persistence until the last running adapter engine exits."""
    global _adapter_users
    with _adapter_lock:
        _adapter_users += 1
    try:
        yield
    finally:
        with _adapter_lock:
            _adapter_users -= 1


def configure(cli_flag: bool) -> None:
    """Record the ``--disable-disk-caches`` flag for this process."""
    global _cli_disabled
    _cli_disabled = bool(cli_flag)


def source() -> str | None:
    """Return the setting that disables disk caches, or None when enabled."""
    if _adapter_users:
        return "--adapter"
    if _cli_disabled:
        return FLAG
    if env_truthy(ENV_VAR):
        return ENV_VAR
    return None


def disabled() -> bool:
    return source() is not None


def overruled_settings(*, kv_disk_checkpoint_interval: int = 0) -> list[str]:
    """Explicit settings that would write to disk but are overruled."""
    settings: list[str] = []
    if kv_disk_checkpoint_interval and kv_disk_checkpoint_interval > 0:
        settings.append(f"--kv-disk-checkpoint-interval {kv_disk_checkpoint_interval}")
    apc = os.environ.get("APC_DISK_ENABLED")
    # Same parsing as the vendored APC (case-insensitive, no stripping).
    if apc is not None and apc.lower() in ("1", "true", "yes"):
        settings.append(f"APC_DISK_ENABLED={apc}")
    autoload = os.environ.get(_AUTOLOAD_ENV)
    if autoload is not None and not is_falsey(autoload):
        settings.append(f"{_AUTOLOAD_ENV}={autoload}")
    return settings
