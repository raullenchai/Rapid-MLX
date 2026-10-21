# SPDX-License-Identifier: Apache-2.0
"""Check that an agent's CLI is installed before ``launch`` does real work.

The binary name and install command come from the profile's existing
``testing.binary`` / ``testing.install_cmd`` metadata, so this module adds no
registry of its own.

PATH resolution: the isolated runner executes through ``/bin/zsh -lc``. A
login shell sources ``/etc/zprofile`` and ``~/.zprofile``, which commonly add
Homebrew and other user prefixes that a GUI- or IDE-launched parent process
may not carry. To avoid refusing a CLI the runner would find, a miss on the
current PATH is re-checked in that same login shell (``command -v``). If the
login shell cannot be probed, the common Homebrew prefixes are checked as a
fallback. A hit on the current PATH is accepted without the probe so a
normal launch does not pay for an extra login shell; a login profile that
removes an inherited PATH entry is not detected.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess

from rapid_mlx.agents import PROFILES_DIR, _load_profile_from_yaml
from rapid_mlx.agents.base import AgentProfile

_LOGIN_SHELL = "/bin/zsh"
_PROBE_TIMEOUT_S = 10
_HOMEBREW_PREFIXES = ("/opt/homebrew/bin", "/usr/local/bin")
_MARKER = "__rapid_mlx_cli__="


def _login_shell_lookup(binary: str) -> str | None:
    """Resolve ``binary`` with the PATH the runner's login shell will see."""
    try:
        result = subprocess.run(
            [
                _LOGIN_SHELL,
                "-lc",
                f"printf '\\n{_MARKER}%s\\n' "
                f'"$(command -v -- {shlex.quote(binary)})"',
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return shutil.which(binary, path=os.pathsep.join(_HOMEBREW_PREFIXES))
    # Login and logout scripts may print around the answer, so read only the
    # marked line. Anything that is not an absolute path (alias, function,
    # builtin) is not the CLI.
    for line in result.stdout.splitlines():
        if line.startswith(_MARKER):
            found = line[len(_MARKER) :].strip()
            return found if found.startswith("/") else None
    return None


def resolve_agent_binary(binary: str) -> str | None:
    """Return where ``binary`` resolves for the launch runner, or ``None``."""
    if "/" in binary or binary.startswith("~"):
        path = os.path.expanduser(binary)
        return path if os.access(path, os.X_OK) else None
    return shutil.which(binary) or _login_shell_lookup(binary)


def missing_agent_cli(profile: AgentProfile) -> str | None:
    """Return a user-facing message when the profile's CLI is not installed."""
    # The launch runner invokes the built-in CLI for an agent, so a user
    # profile that shadows a built-in one must not change what is checked.
    builtin = PROFILES_DIR / f"{profile.name}.yaml"
    if builtin.is_file():
        profile = _load_profile_from_yaml(builtin)
    binary = profile.testing.binary
    if not binary or resolve_agent_binary(binary) is not None:
        return None
    message = f"{profile.display_name} CLI ({binary}) is not installed or not on PATH."
    if profile.testing.install_cmd:
        message += f" Install it, then retry:\n  {profile.testing.install_cmd}"
    return message
