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

import contextlib
import logging
import os
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

from vllm_mlx.agents import get_profile
from vllm_mlx.agents.testing import (
    E2E_CHAT_EXPECTED,
    E2E_FIRST_LINE,
    TestStatus,
    _agent_query,
    _e2e_workspace,
    _test_e2e_chat,
    _test_e2e_file_read,
    _test_e2e_terminal,
)

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


def test_query_quotes_cannot_change_the_child_argv():
    """Prompts are data even when they contain the template's quote style."""
    script = "import json, sys; sys.stdout.write(json.dumps(sys.argv[1:]))"
    cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} '{{query}}'"
    query = "Run 'echo safe' and report the result"

    out, err = _agent_query(sys.executable, cmd, query, timeout=_TIMEOUT_S)

    assert err is None, f"unexpected error: {err}"
    assert out == "[\"Run 'echo safe' and report the result\"]"


def test_local_anthropic_query_drops_remote_provider_environment(monkeypatch):
    """A Claude E2E run must not inherit selectors for a paid remote backend."""
    conflicting = {
        "ANTHROPIC_AUTH_TOKEN": "real-token",
        "ANTHROPIC_BEDROCK_BASE_URL": "https://bedrock.example",
        "ANTHROPIC_VERTEX_BASE_URL": "https://vertex.example",
        "ANTHROPIC_FOUNDRY_BASE_URL": "https://foundry.example",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_USE_VERTEX": "1",
        "CLAUDE_CODE_USE_FOUNDRY": "1",
    }
    for key, value in conflicting.items():
        monkeypatch.setenv(key, value)

    keys = [*conflicting, "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"]
    script = (
        "import json, os, sys; "
        f"sys.stdout.write(json.dumps({{k: os.environ[k] for k in {keys!r} if k in os.environ}}, sort_keys=True))"
    )
    cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    out, err = _agent_query(
        sys.executable,
        cmd,
        "unused",
        timeout=_TIMEOUT_S,
        env_overrides={
            "ANTHROPIC_BASE_URL": "http://localhost:8000",
            "ANTHROPIC_API_KEY": "not-needed",
        },
    )

    assert err is None, f"unexpected error: {err}"
    assert out == (
        '{"ANTHROPIC_API_KEY": "not-needed", '
        '"ANTHROPIC_BASE_URL": "http://localhost:8000"}'
    )


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
        assert text.splitlines()[0] == E2E_FIRST_LINE
        # It has to be a sentinel, not a plausible word: the assertion
        # previously accepted "build" or "project" anywhere in the agent's
        # answer, which any sentence about a Python project satisfies
        # without the file having been opened.
        assert text.count(E2E_FIRST_LINE) == 1


def test_file_read_passes_on_evidence_even_when_agent_does_not_terminate():
    """#1598: capability evidence and CLI termination are separate signals."""
    script = (
        "import sys, time; "
        f"sys.stdout.write({E2E_FIRST_LINE!r}); sys.stdout.flush(); time.sleep(30)"
    )
    cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    result = _test_e2e_file_read(sys.executable, cmd, timeout=1)

    assert result.status is TestStatus.PASS, result.message
    assert "did not terminate" in result.message


def test_workspace_is_removed_afterwards():
    """A gate that leaves a directory per run per agent is a slow disk leak."""
    with _e2e_workspace() as workdir:
        assert os.path.isdir(workdir)
    assert not os.path.exists(workdir), f"workspace survived the context: {workdir}"


def _recording_agent(log: Path, reply: str) -> str:
    """A stand-in agent CLI that records where it ran and prints `reply`."""
    script = (
        "import os, sys; "
        f"open({str(log)!r}, 'a').write(os.getcwd() + chr(10)); "
        f"sys.stdout.write({reply!r})"
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} '{{query}}'"


def test_each_e2e_test_gets_its_own_workspace(tmp_path):
    """Two e2e tests must not share a directory.

    Asserting that two `_e2e_workspace()` calls differ proves nothing about
    the runner — the runner is what decides how many workspaces exist. This
    drives the real `_test_e2e_*` entry points with a real subprocess and
    reads back the directories they actually ran in.
    """
    log = tmp_path / "cwds.txt"
    # One reply that satisfies both assertions: the expected sum for chat,
    # the sentinel for file-read.
    cmd = _recording_agent(log, f"{E2E_CHAT_EXPECTED} {E2E_FIRST_LINE}")

    # ALL THREE entry points, not a sample of them: `_test_e2e_terminal`
    # could regress to a shared or inherited cwd while a test that only
    # drives the other two stayed green.
    marker = "rapidmlx_codex_test"
    chat = _test_e2e_chat(sys.executable, cmd, _TIMEOUT_S)
    file_read = _test_e2e_file_read(sys.executable, cmd, _TIMEOUT_S)
    terminal = _test_e2e_terminal(
        sys.executable,
        _recording_agent(log, marker),
        _TIMEOUT_S,
        "codex",
    )
    assert chat.status is TestStatus.PASS, chat.message
    assert file_read.status is TestStatus.PASS, file_read.message
    assert terminal.status is TestStatus.PASS, terminal.message

    ran_in = log.read_text(encoding="utf-8").splitlines()
    assert len(ran_in) == 3, f"expected three runs, recorded {ran_in}"
    assert len(set(ran_in)) == 3, (
        "two or more e2e tests ran in the SAME directory — whatever the first "
        f"agent wrote there is the next one's starting condition: {ran_in}"
    )
    for path in ran_in:
        assert not os.path.exists(path), f"workspace survived its test: {path}"


def test_file_read_demands_the_sentinel_not_a_plausible_sentence(tmp_path):
    """Talking about the project is not reading the first line of the file."""
    log = tmp_path / "cwds.txt"
    # Contains "build" and "project" — which the previous assertion accepted —
    # but is not the first line of anything.
    chatty = _recording_agent(log, "This looks like a Python project; I can build it.")
    result = _test_e2e_file_read(sys.executable, chatty, _TIMEOUT_S)
    assert result.status is TestStatus.FAIL, (
        "an answer that never read the file passed the file-read test"
    )


def _lock_and_release(workdir: str, name: str = "hostile") -> None:
    """Make `workdir` un-removable the way a misbehaving agent would."""
    hostile = Path(workdir, name)
    hostile.mkdir()
    (hostile / "note.txt").write_text("x", encoding="utf-8")
    os.chmod(hostile, 0o000)


def test_an_unremovable_workspace_is_reported_not_swallowed(caplog):
    """The NIT was about silence, not about winning the fight.

    Cleanup does not try to repair permissions — see `_remove_workspace` for
    why that buys nothing against a process running as the same user. What it
    must never do is leave a directory behind on every release and say
    nothing.
    """
    with (
        caplog.at_level(logging.WARNING, logger="vllm_mlx.agents.testing"),
        _e2e_workspace() as workdir,
    ):
        _lock_and_release(workdir)
        captured = workdir
    try:
        if not os.path.exists(captured):
            pytest.skip("this filesystem removes locked directories anyway")
        assert any(captured in r.getMessage() for r in caplog.records), (
            "the workspace was left behind and nobody was told — that is the "
            "silent per-release disk leak the warning exists to prevent"
        )
    finally:
        with contextlib.suppress(OSError):
            os.chmod(Path(captured, "hostile"), 0o700)
        shutil.rmtree(captured, ignore_errors=True)


def test_cleanup_never_writes_outside_the_workspace(tmp_path):
    """A link planted in the workspace must not aim our cleanup at its target.

    This is the shape the earlier permission-repair pass got wrong twice:
    `os.chmod` follows symlinks, so a link pointing at a caller-owned file had
    that file's mode rewritten by our own cleanup (measured: 0o600 -> 0o700).
    Cleanup no longer chmods anything, and this pins that it stays that way.
    """
    outsider = tmp_path / "private.key"
    outsider.write_text("secret", encoding="utf-8")
    os.chmod(outsider, 0o600)
    outsider_before = os.stat(outsider).st_mode
    outside_dir = tmp_path / "outside-tree"
    outside_dir.mkdir()
    (outside_dir / "keep.txt").write_text("keep", encoding="utf-8")
    os.chmod(outside_dir, 0o500)
    outside_dir_before = os.stat(outside_dir).st_mode

    with _e2e_workspace() as workdir:
        # The links go INSIDE the directory that gets locked, not beside it.
        # On the happy path rmtree unlinks them before it fails, so links in
        # the workspace root are gone by the time any repair pass would run —
        # which is how the first version of this test stayed green against
        # the very code it was written to forbid. Locked in here, they
        # survive the failed rmtree and are exactly what a repair pass would
        # walk into and chmod.
        trap = Path(workdir, "hostile")
        trap.mkdir()
        os.symlink(outsider, trap / "link-to-outside-file")
        os.symlink(outside_dir, trap / "link-to-outside-dir")
        (trap / "note.txt").write_text("x", encoding="utf-8")
        os.chmod(trap, 0o000)
        captured = workdir

    try:
        if not os.path.exists(captured):
            # Root, or a filesystem that ignores mode bits, deletes the tree
            # regardless. The repair path this test guards never ran, so the
            # assertions below would prove nothing — say so rather than
            # failing a change that is fine.
            pytest.skip("this environment removes a 0o000 directory anyway")
        assert outsider.exists(), "cleanup deleted a file outside the workspace"
        assert (outside_dir / "keep.txt").exists(), (
            "cleanup followed a link and deleted a tree outside the workspace"
        )
        assert os.stat(outsider).st_mode == outsider_before, (
            "cleanup rewrote the permissions of a file outside the workspace"
        )
        assert os.stat(outside_dir).st_mode == outside_dir_before, (
            "cleanup rewrote the permissions of a directory outside the workspace"
        )
    finally:
        with contextlib.suppress(OSError):
            os.chmod(Path(captured, "hostile"), 0o700)
        shutil.rmtree(captured, ignore_errors=True)


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


def test_workspace_creation_failure_is_an_error_not_a_crash(tmp_path, monkeypatch):
    """A full or read-only temp filesystem must not take the runner down.

    Staged against a REAL read-only directory rather than a patched
    `mkdtemp`, so the failure arrives from the operating system the way it
    would in production. `tempfile.tempdir` rather than `$TMPDIR`: tempfile
    resolves the environment variable once and caches it, so setting the
    variable here would change nothing.
    """
    readonly = tmp_path / "readonly-tmp"
    readonly.mkdir()
    os.chmod(readonly, 0o500)
    monkeypatch.setattr(tempfile, "tempdir", str(readonly))

    try:
        probe = tempfile.mkdtemp()
    except OSError:
        pass
    else:
        # Root ignores the missing write bit. Nothing to assert about a
        # failure this environment will not produce.
        shutil.rmtree(probe, ignore_errors=True)
        pytest.skip("this environment can create a workspace under a 0o500 dir")

    result = _test_e2e_chat(sys.executable, "irrelevant {query}", _TIMEOUT_S)
    assert result.status is TestStatus.ERROR, (
        f"expected an ERROR result, got {result.status} — a workspace that "
        "cannot be created must not propagate out of the test runner"
    )
    assert "workspace unavailable" in result.message, result.message
