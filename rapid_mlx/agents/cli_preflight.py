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
generated runner script would not inherit. If that shell cannot be run or
does not answer within the deadline, launch refuses as well, because the
runner depends on the same shell. The probe sources the login profile one
extra time per launch, with no terminal attached, in its own process group;
the group is stopped when the probe ends. Processes the profile detaches into
another session (daemons) are left alone, as they are for the runner.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import tempfile
from pathlib import Path

from rapid_mlx.agents import PROFILES_DIR, _load_profile_from_yaml
from rapid_mlx.agents.base import AgentProfile

_LOGIN_SHELL = "/bin/zsh"
_PROBE_TIMEOUT_S = 30


def agent_cli_found(binary: str) -> bool | None:
    """Whether the runner's login-shell PATH has an executable ``binary``.

    Returns ``None`` when the login shell cannot be run or does not answer.
    """
    # Passed verbatim: the login shell, not this process, interprets it.
    target = shlex.quote(binary)
    try:
        # The answer goes to a private file opened only after startup, and
        # the shell's own output is discarded. Profile output therefore cannot
        # pose as the answer, and a background process started by the profile
        # cannot hold a pipe open or write into the answer.
        with tempfile.TemporaryDirectory(prefix="rapid-mlx-cli-") as private:
            answer = Path(private) / "status"
            # ``builtin`` keeps profile functions from shadowing the probe.
            # The shell ends by stopping its own process group while it is
            # still alive, so no signal ever targets a reaped process id.
            script = (
                f"builtin whence -p -- {target} >/dev/null 2>&1; "
                f"builtin printf '%s\\n' \"$?\" >| {shlex.quote(str(answer))}; "
                "builtin kill -KILL 0"
            )
            # A new session keeps that group separate from launch's own.
            probe = subprocess.Popen(
                [_LOGIN_SHELL, "-lc", script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                probe.wait(timeout=_PROBE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # The unreaped shell still owns the group id, so this cannot
                # reach an unrelated process group.
                with contextlib.suppress(OSError):
                    os.killpg(probe.pid, signal.SIGKILL)
                # The leader itself, should the group signal have failed, so
                # the final wait cannot hang.
                probe.kill()
                probe.wait()
                return None
            status = answer.read_bytes().strip() if answer.is_file() else b""
    except OSError:
        return None
    return status == b"0" if status else None


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
