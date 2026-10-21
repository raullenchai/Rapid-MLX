# SPDX-License-Identifier: Apache-2.0
"""Check that an agent's CLI is installed before ``launch`` does real work.

The binary name and install command come from the profile's existing
``testing.binary`` / ``testing.install_cmd`` metadata, so this module adds no
registry of its own.

PATH resolution: the isolated runner executes through ``/bin/zsh -lc``. A
login shell sources ``/etc/zprofile`` and ``~/.zprofile``, which commonly add
(or replace) PATH entries such as Homebrew. The CLI is therefore resolved with
``command -v`` in that same login shell, so the answer matches what the runner
will see. Only when the login shell cannot be run, or does not answer within
the probe deadline, does resolution fall back to the current PATH plus the
common Homebrew prefixes.
"""

from __future__ import annotations

import os
import secrets
import shlex
import shutil
import subprocess

from rapid_mlx.agents import PROFILES_DIR, _load_profile_from_yaml
from rapid_mlx.agents.base import AgentProfile

_LOGIN_SHELL = "/bin/zsh"
_PROBE_TIMEOUT_S = 30
_HOMEBREW_PREFIXES = ("/opt/homebrew/bin", "/usr/local/bin")


def _executable_file(path: str) -> str | None:
    return path if os.path.isfile(path) and os.access(path, os.X_OK) else None


def _fallback_lookup(binary: str) -> str | None:
    """Current PATH plus Homebrew, for when the login shell cannot answer."""
    entries = [os.environ.get("PATH", ""), *_HOMEBREW_PREFIXES]
    found = shutil.which(binary, path=os.pathsep.join(e for e in entries if e))
    return _executable_file(found) if found else None


def _login_shell_lookup(binary: str) -> str | None:
    """Resolve ``binary`` with the PATH the runner's login shell will see."""
    # A per-probe marker keeps profile or logout output from posing as the
    # answer, however it is formatted.
    marker = f"__rapid_mlx_cli_{secrets.token_hex(8)}__="
    script = (
        f"found=$(command -v -- {shlex.quote(binary)}) || found=; "
        f"printf '\\n{marker}%s\\n' \"$found\""
    )
    try:
        result = subprocess.run(
            [_LOGIN_SHELL, "-lc", script],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return _fallback_lookup(binary)
    answers = [
        line[len(marker) :].strip()
        for line in result.stdout.splitlines()
        if line.startswith(marker)
    ]
    # Anything that is not an executable file (alias, function, builtin,
    # directory) is not the CLI.
    return _executable_file(answers[-1]) if answers else None


def resolve_agent_binary(binary: str) -> str | None:
    """Return where ``binary`` resolves for the launch runner, or ``None``."""
    if "/" in binary or binary.startswith("~"):
        return _executable_file(os.path.expanduser(binary))
    return _login_shell_lookup(binary)


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
