# SPDX-License-Identifier: Apache-2.0
"""Check that an agent's CLI is installed before ``launch`` does real work.

The binary name and install command come from the profile's existing
``testing.binary`` / ``testing.install_cmd`` metadata, so this module adds no
registry of its own.

PATH resolution: the isolated runner executes through ``/bin/zsh -lc``. A
login shell sources ``/etc/zprofile`` and ``~/.zprofile``, which commonly add
(or replace) PATH entries such as Homebrew. The CLI is therefore checked with
``whence -p`` in that same login shell and working directory, and the
shell's own exit status is the answer. ``-p`` accepts only executables found
on PATH (or an executable path), not profile functions or aliases, which the
generated runner script would not inherit. If that shell cannot be run or does not answer within the deadline,
launch refuses as well, because the runner depends on the same shell.
"""

from __future__ import annotations

import os
import secrets
import shlex
import subprocess

from rapid_mlx.agents import PROFILES_DIR, _load_profile_from_yaml
from rapid_mlx.agents.base import AgentProfile

_LOGIN_SHELL = "/bin/zsh"
_PROBE_TIMEOUT_S = 30


def agent_cli_found(binary: str) -> bool | None:
    """Whether the runner's login-shell PATH has an executable ``binary``.

    Returns ``None`` when the login shell cannot be run or does not answer.
    """
    # A per-probe marker keeps profile output from posing as the answer.
    marker = f"__rapid_mlx_cli_{secrets.token_hex(8)}__="
    target = shlex.quote(os.path.expanduser(binary))
    # ``builtin`` keeps profile functions from shadowing the probe itself.
    script = (
        f"builtin whence -p -- {target} >/dev/null 2>&1; "
        f"builtin printf '\\n{marker}%s\\n' \"$?\""
    )
    try:
        result = subprocess.run(
            [_LOGIN_SHELL, "-lc", script],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    answers = [
        line[len(marker) :].strip()
        # Startup files may print bytes in any encoding; decode like a path.
        for line in os.fsdecode(result.stdout).splitlines()
        if line.startswith(marker)
    ]
    return answers[-1] == "0" if answers else None


def missing_agent_cli(profile: AgentProfile) -> str | None:
    """Return a user-facing message when the profile's CLI is not usable."""
    # The launch runner invokes the built-in CLI for an agent, so a user
    # profile that shadows a built-in one must not change what is checked.
    builtin = PROFILES_DIR / f"{profile.name}.yaml"
    if builtin.is_file():
        profile = _load_profile_from_yaml(builtin)
    binary = profile.testing.binary
    if not binary:
        return None
    found = agent_cli_found(binary)
    if found:
        return None
    if found is None:
        message = (
            f"Could not confirm that the {profile.display_name} CLI ({binary}) "
            f"is installed: the login shell ({_LOGIN_SHELL} -l) did not respond."
        )
    else:
        message = (
            f"{profile.display_name} CLI ({binary}) is not installed or not on PATH."
        )
    if profile.testing.install_cmd:
        message += f" Install it, then retry:\n  {profile.testing.install_cmd}"
    return message
