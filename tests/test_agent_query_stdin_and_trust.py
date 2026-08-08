# SPDX-License-Identifier: Apache-2.0
"""Pin why G7b's codex e2e could never pass, and where it runs (#1683).

The first two live on the path between the test runner and the agent CLI,
so both reported as a rapid-mlx integration failure while nothing in
rapid-mlx was ever reached.

1. ``_agent_query`` ran the child with an **inherited stdin**. A CLI that
   reads stdin when it isn't a TTY then blocks until ``query_timeout``
   instead of answering the query it was handed on argv. Codex does
   exactly this — it prints "Reading additional input from stdin..." and
   waits — so the gate burned its whole 120 s budget per test.

2. The codex profile's ``query_cmd`` omitted ``--skip-git-repo-check``.
   Codex refuses to run outside a trusted git repo, and the runner
   inherits the caller's cwd, so from a neutral directory every e2e test
   failed in ~100 ms with "Not inside a trusted directory".

3. Fixing (2) by itself would bypass Codex's trust check in whatever
   directory the caller happened to be standing in, exposing that
   directory's files and agent instructions to the agent. The e2e tests
   now run in a workspace ``_e2e_workspace`` builds, which also makes
   `_test_e2e_file_read` deterministic — it asks for ``pyproject.toml``,
   and previously only found one when the caller was standing in a
   Python project.

The stdin test drives a **real subprocess against a real pipe on fd 0**.
A mock asserting ``stdin=DEVNULL`` was passed would only restate the
diff; it cannot show that the child stops blocking. Reproducing the hang
needs the actual inherited descriptor, which is the whole bug.
"""

from __future__ import annotations

import os
import shlex
import sys
import time
from pathlib import Path

import pytest

from vllm_mlx.agents import get_profile
from vllm_mlx.agents.testing import _agent_query, _e2e_workspace

# Long enough that a slow machine never flakes, short enough that a
# regression (the child blocking on stdin) fails the suite in seconds
# rather than hanging it.
_TIMEOUT_S = 8


def _echo_stdin_cmd() -> str:
    """A child that reads stdin to EOF and reports what it got."""
    script = 'import sys; sys.stdout.write("READ:" + repr(sys.stdin.read()))'
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


@pytest.fixture
def stdin_is_an_open_pipe():
    """Point the *parent's* fd 0 at a pipe that never reaches EOF.

    This is the condition the release gauntlet actually runs under: fd 0
    is open but nothing will ever write to it. A child that inherits it
    and calls ``read()`` blocks forever.
    """
    try:
        saved = os.dup(0)
    except OSError:  # pragma: no cover - fd 0 closed in this environment
        pytest.skip("fd 0 is not open; cannot stage the inherited-stdin case")
    read_fd, write_fd = os.pipe()
    os.dup2(read_fd, 0)
    try:
        yield
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read_fd)
        # Closing the write end last keeps the pipe from reaching EOF for
        # the whole test, which is what makes a regression actually hang.
        os.close(write_fd)


def test_child_does_not_inherit_a_blocking_stdin(stdin_is_an_open_pipe):
    """The child must see EOF immediately, not the parent's open pipe."""
    t0 = time.monotonic()
    out, err = _agent_query(
        sys.executable, _echo_stdin_cmd(), "unused", timeout=_TIMEOUT_S
    )
    elapsed = time.monotonic() - t0

    assert err != "TIMEOUT", (
        "the agent CLI inherited the parent's stdin and blocked on it — "
        "this is the #1683 hang, and it costs one query_timeout per e2e test"
    )
    assert err is None, f"unexpected error from _agent_query: {err}"
    assert out == "READ:''", f"child saw data on stdin instead of EOF: {out!r}"
    assert elapsed < _TIMEOUT_S, "returned only because the timeout fired"


def test_query_placeholder_still_substitutes():
    """The stdin fix must not disturb how {query} reaches the child."""
    script = 'import sys; sys.stdout.write("ARG:" + sys.argv[1])'
    cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} '{{query}}'"
    out, err = _agent_query(sys.executable, cmd, "what is 2+2", timeout=_TIMEOUT_S)
    assert err is None, f"unexpected error: {err}"
    assert out == "ARG:what is 2+2"


def test_agent_runs_in_the_given_workspace_not_the_callers_cwd():
    """`--skip-git-repo-check` is only safe in a directory we built ourselves.

    Bypassing Codex's trust check while still launching it in the caller's
    inherited cwd would hand the agent whatever that directory contains —
    including files and agent instructions from an untrusted checkout.
    """
    script = "import os, sys; sys.stdout.write('CWD:' + os.getcwd())"
    cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    with _e2e_workspace() as workdir:
        out, err = _agent_query(
            sys.executable, cmd, "unused", timeout=_TIMEOUT_S, cwd=workdir
        )
    assert err is None, f"unexpected error: {err}"
    # macOS hands out /var/folders/... symlinked from /private/var/folders/...
    assert os.path.realpath(out.removeprefix("CWD:")) == os.path.realpath(workdir), (
        f"child ran in {out!r}, not in the workspace it was given"
    )


def test_workspace_carries_the_file_the_file_read_test_asks_for():
    """`_test_e2e_file_read` reads pyproject.toml; the workspace must have one."""
    with _e2e_workspace() as workdir:
        target = Path(workdir, "pyproject.toml")
        assert target.is_file(), "the workspace has no pyproject.toml to read"
        text = target.read_text(encoding="utf-8")
        # The test asks for the FIRST line specifically.
        assert text.splitlines()[0] == "[build-system]"
        # `_test_e2e_file_read` passes on either substring; both must be
        # present so the assertion does not depend on which part of the
        # file the agent chooses to quote back.
        assert "build" in text.lower()
        assert "project" in text.lower()


def test_workspace_is_removed_afterwards():
    """A gate that leaves a directory per run per agent is a slow disk leak."""
    with _e2e_workspace() as workdir:
        assert os.path.isdir(workdir)
    assert not os.path.exists(workdir), f"workspace survived the context: {workdir}"


def test_workspace_is_fresh_each_time():
    """Two runs must not share state — one agent's mess is not the next's input."""
    with _e2e_workspace() as first:
        Path(first, "scratch.txt").write_text("left behind", encoding="utf-8")
        first_path = first
    with _e2e_workspace() as second:
        assert second != first_path
        assert not Path(second, "scratch.txt").exists()


def test_codex_query_cmd_skips_the_git_repo_check():
    """Codex refuses to run outside a trusted repo; the gate can't rely on cwd."""
    profile = get_profile("codex")
    query_cmd = profile.testing.query_cmd
    assert query_cmd, "codex must keep a non-null query_cmd — G7b depends on it"
    assert "--skip-git-repo-check" in query_cmd, (
        "without --skip-git-repo-check the codex e2e tests fail in ~100 ms with "
        "'Not inside a trusted directory' whenever the runner's cwd is not a "
        "trusted git repo (#1683)"
    )


def test_codex_query_cmd_is_shell_parseable_and_keeps_the_placeholder():
    """A malformed query_cmd would silently degrade to the naive split path."""
    query_cmd = get_profile("codex").testing.query_cmd
    parts = shlex.split(query_cmd.replace("{query}", "what is 2+2"))
    assert parts[0] == "codex"
    assert parts[1] == "exec"
    assert "--skip-git-repo-check" in parts
    # The query must survive as ONE argv entry, not five bare words.
    assert "what is 2+2" in parts
