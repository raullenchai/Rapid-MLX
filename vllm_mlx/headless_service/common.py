# SPDX-License-Identifier: Apache-2.0
"""Shared constants and path helpers for ``rapid-mlx service``.

These are the concrete, documented defaults that the manual launchd
runbook in ``docs/guides/headless-macos-service.md`` establishes. Keeping
them here (not scattered across the sub-command modules) matches the
"one source of truth" pattern the repo uses for model-recommendation data
(``model_recommendations.json``) and keeps the deterministic plist and the
diagnostic surface from drifting.
"""

from __future__ import annotations

import re
from pathlib import Path

# The launchd Label. A system-domain daemon of this name is the contract
# the docs + ``scripts/headless_service_smoke.sh`` already assume.
DEFAULT_LABEL = "com.rapidmlx.server"

# Default launchd domain. System-domain daemons start before and without
# any GUI login — the whole point of #2859.
DEFAULT_DOMAIN = "system"

# Where system daemon plists live. Root-owned; writing here is what makes
# the service a boot-persistent system LaunchDaemon (not a user daemon).
LAUNCH_DAEMONS_DIR = Path("/Library/LaunchDaemons")

# Relative log dir under the service account's home. Long-lived daemons
# have no interactive session, so stdout+stderr go straight to these logs.
LOG_RELATIVE_DIR = "Library/Logs/Rapid-MLX"
STDOUT_LOG_NAME = "server.stdout.log"
STDERR_LOG_NAME = "server.stderr.log"

# Service-account paths established by the one-line installer + the guide:
#   /Users/<u>/.local/bin/rapid-mlx   stable CLI symlink
#   /Users/<u>/.rapid-mlx/            venv + app state
#   /Users/<u>/.cache/huggingface/    model cache resolved from HOME
DEFAULT_BIN = ".local/bin/rapid-mlx"
APP_STATE_DIR = ".rapid-mlx"

# Minimum uid for a dedicated service account. Real user accounts on macOS
# start at 501; system accounts (root=_WHEEL 0-500) are never valid targets.
MIN_USER_UID = 501

# LaunchDaemons dir requires root to write; also used to gate dry-run vs
# real mutation (a non-root caller cannot actually install).
LAUNCH_DAEMONS_DIR_REQUIRES_ROOT = True


def home_for_user(user: str) -> Path | None:
    """Return the home directory for ``user``, or None if unknown.

    Uses ``pwd.getpwnam`` (filesystem account database) so it works for
    accounts that may never have logged into a GUI session — the service
    account is exactly such a case. Returns None when the user does not
    exist (caller should treat that as a hard error).
    """
    import pwd

    try:
        return Path(pwd.getpwnam(user).pw_dir)
    except KeyError:
        return None


def user_uid(user: str) -> int | None:
    """Return the numeric uid for ``user``, or None if the account is absent."""
    import pwd

    try:
        return pwd.getpwnam(user).pw_uid
    except KeyError:
        return None


def log_dir_for(user: str) -> Path | None:
    """Absolute daemon log directory for a service account (or None)."""
    home = home_for_user(user)
    if home is None:
        return None
    return home / LOG_RELATIVE_DIR


def validate_label(label: str) -> str:
    """Validate a launchd Label, returning it unchanged.

    Restricted to reverse-DNS-safe characters ``[A-Za-z0-9._-]`` — the same
    constraint ``scripts/headless_service_smoke.sh`` already enforces on
    ``RAPID_MLX_SERVICE_LABEL``. Guarantees a ``--label`` value can never
    escape the LaunchDaemons directory (path traversal) or be joined into a
    ``launchctl``/``rm`` argv as an unexpected token.
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise ValueError(
            f"invalid launchd Label {label!r}: only letters, digits, '.', "
            "'_', '-' are allowed."
        )
    return label
