# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #974 — ``scripts/release_check_m3.sh``
must thread ``$PORT`` into ``RAPID_MLX_BASE_URL`` (and OpenAI-SDK
conventional siblings) so G7 SDK integration tests hit the gauntlet
server, not whatever default port their env-var defaults resolve to.

The bug: G7 SDK tests (Anthropic / pydantic_ai / smolagents / langchain
/ hermes) read the endpoint from ``os.environ.get("RAPID_MLX_BASE_URL",
"http://localhost:8000/v1")``. If the gauntlet is booted with a PORT
override (e.g. ``PORT=8011`` to avoid a running production server on
8000) but the script does NOT export ``RAPID_MLX_BASE_URL``, the SDK
tests silently target ``http://localhost:8000`` — usually the
operator's production box — producing either false failures (wrong
model) or false PASSes (prod happens to answer).

We assert two invariants on the shell script:

1. **Every G-block env var is present and derived from $PORT.** Sourcing
   the top of the script under ``PORT=<random>`` yields
   ``RAPID_MLX_BASE_URL == http://127.0.0.1:<PORT>/v1``.

2. **Every base-url env var read by any test under
   ``tests/integrations/*.py`` is covered by the export block.** This
   is a systematic guard: adding a new integration test that reads
   ``FOOBAR_BASE`` should trip this test so the script export list is
   updated in lockstep.

The script is Bash, not Python, so we shell out via ``subprocess`` —
never actually booting rapid-mlx serve or touching a real port.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "release_check_m3.sh"
INTEGRATIONS = REPO_ROOT / "tests" / "integrations"

# Env vars any integration test under tests/integrations/*.py reads to
# resolve the rapid-mlx server endpoint. The shell script MUST export
# each of these before running G7. Grow this set if we ever add a new
# integration harness with a different env-var convention.
KNOWN_BASE_URL_ENVS = {
    "RAPID_MLX_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
}


def _extract_prelude(script_path: Path) -> str:
    """Return everything from the top of the script up through the
    ``echo "  log:"`` banner line — i.e. the part that resolves ``PORT``
    and exports base-url env vars. Everything after (server boot, G-block
    orchestration) is skipped so sourcing the prelude in a test doesn't
    try to actually launch ``rapid-mlx serve``."""
    text = script_path.read_text()
    marker = 'echo "  log:'
    idx = text.find(marker)
    assert idx != -1, "banner marker 'echo \"  log:' vanished from script"
    # Cut at the end of the marker's line so ``line`` (echo separator)
    # doesn't get sourced without its function definition. The prelude
    # is enough for env-var setup verification.
    end_of_line = text.find("\n", idx)
    return text[: end_of_line + 1]


def test_prelude_exports_base_url_from_port(tmp_path: Path) -> None:
    """PORT override MUST propagate to RAPID_MLX_BASE_URL — the exact
    invariant the fix for issue #974 enforces."""
    prelude = _extract_prelude(SCRIPT)
    # Source under a non-default PORT and print the resolved env vars.
    port = "8011"
    probe = tmp_path / "probe.sh"
    probe.write_text(
        prelude
        + '\necho "RAPID_MLX_BASE_URL=$RAPID_MLX_BASE_URL"\n'
        + 'echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"\n'
        + 'echo "OPENAI_API_BASE=$OPENAI_API_BASE"\n'
    )
    result = subprocess.run(
        ["bash", str(probe)],
        capture_output=True,
        text=True,
        env={"PORT": port, "PATH": "/usr/bin:/bin"},
        check=True,
    )
    expected = f"http://127.0.0.1:{port}/v1"
    assert f"RAPID_MLX_BASE_URL={expected}" in result.stdout, result.stdout
    assert f"OPENAI_BASE_URL={expected}" in result.stdout, result.stdout
    assert f"OPENAI_API_BASE={expected}" in result.stdout, result.stdout


def test_prelude_default_port_matches_hardcoded_probes() -> None:
    """Default PORT (unset) MUST resolve to 8000 — matching the hardcoded
    "http://127.0.0.1:$PORT" URLs elsewhere in the script AND the
    ``localhost:8000`` default the SDK tests fall back to when no env
    override is present."""
    prelude = _extract_prelude(SCRIPT)
    result = subprocess.run(
        ["bash", "-c", prelude + '\necho "URL=$RAPID_MLX_BASE_URL"'],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},  # no PORT export
        check=True,
    )
    assert "URL=http://127.0.0.1:8000/v1" in result.stdout, result.stdout


def _find_g7_guard(text: str) -> int:
    """Return the index of the fail-loud ``RAPID_MLX_BASE_URL`` guard, or
    -1 if absent.

    The guard is the ``if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]``
    block that bails out (``exit 1``) when the env var no longer points at
    the gauntlet's own port. We anchor on this semantic shape rather than on
    any banner text so harmless banner-format changes never break the test.
    """
    match = re.search(
        r'if\s+\[\s*"\$\{RAPID_MLX_BASE_URL[^]]*"\s*!=\s*"\$_expected_base"',
        text,
    )
    return match.start() if match else -1


def _find_g7_invocation(text: str) -> int:
    """Return the index of the first ``test_anthropic_sdk.py`` invocation,
    or -1 if absent. This is the stable semantic anchor for the G7 SDK
    integration block.

    We match the actual invocation shape (``"$PY" tests/integrations/
    test_anthropic_sdk.py``) rather than the bare filename so a comment
    that merely mentions the test does not masquerade as the real call.
    """
    match = re.search(r'"\$PY"\s+tests/integrations/test_anthropic_sdk\.py', text)
    return match.start() if match else -1


def _guard_tail_block(text: str, guard_idx: int) -> str | None:
    """Return the complete ``if ... fi`` guard block starting at
    ``guard_idx``, letting bash itself decide the block boundary.

    We do NOT hand-roll a shell lexer (which would be blind to ANSI-C
    ``$'...'`` quotes, heredocs, command substitution, etc.). Instead we
    extend the source from the guard head one logical line at a time and ask
    bash to parse the prefix with ``bash -n``: the first prefix that parses
    cleanly is exactly the block bash treats as balanced, so nested
    ``if``/``for``/``case`` and keywords inside comments/strings cannot
    confuse the boundary. We require the suspicious head (the
    ``RAPID_MLX_BASE_URL`` vs ``_expected_base`` comparison) to still be
    present in the candidate, so we never mistake a *different* ``if`` for
    the guard.
    """
    if not re.match(r"\s*if\b", text[guard_idx:]):
        return None
    lines = text[guard_idx:].splitlines()
    head = lines[0]
    if "RAPID_MLX_BASE_URL" not in head or "_expected_base" not in head:
        return None
    for k in range(1, len(lines) + 1):
        candidate = "\n".join(lines[:k])
        # bash -n: syntax check only, never executes.
        syntax = subprocess.run(
            ["bash", "-n", "-c", candidate], capture_output=True, text=True
        )
        if syntax.returncode == 0:
            block = candidate
            # Confirm the block's own fi closed at the right place and that
            # the RAPID_MLX_BASE_URL guard is still the semantic head.
            if "RAPID_MLX_BASE_URL" in block and "_expected_base" in block:
                return block + "\n"
            return None
    return None


def _guard_is_fail_loud(block: str) -> bool:
    """Return True only if ``block`` (a real bash ``if ... fi`` guard) bails
    out with an actual ``exit`` builtin when ``RAPID_MLX_BASE_URL`` no longer
    points at the gauntlet port.

    We never guess with a shell regex. We shadow the ``exit`` builtin with a
    function that touches a temp sentinel file and then calls the real
    ``builtin exit``, then run the whole block in bash. Only a genuine
    ``exit`` writes the sentinel, and the process must actually terminate
    with status 1 -- so ``exit 1`` buried in a comment, a quoted string, an
    ``echo exit 1``, an else/elif branch, a ``false``/``return 1``/``set -e``
    tail, or a survived ``( exit 1 )`` subshell cannot fake a fail-loud
    guard.
    """
    with tempfile.TemporaryDirectory() as td:
        sentinel = Path(td) / "g7_exit_sentinel"
        # Controlled, hermetic environment: do NOT inherit the caller's
        # shell-startup vars (BASH_ENV/ENV can install EXIT traps that
        # would write the sentinel through the shadow) -- only what the
        # wrapped guard needs.
        env = {"G7_SENTINEL": str(sentinel), "PATH": "/usr/bin:/bin"}
        prefix = (
            'RAPID_MLX_BASE_URL="http://127.0.0.1:9999/v1"\n'
            'PORT="8000"\n'
            '_expected_base="http://127.0.0.1:${PORT}/v1"\n'
        )
        # Only a real `exit` in the top-level shell touches the sentinel:
        # BASH_SUBSHELL is 0 at the top level (and inside a plain function,
        # which calls the real builtin exit) and >0 inside a `( ... )`
        # subshell, so a survived `( exit 1 )` can never write it. The path
        # travels via the quoted $G7_SENTINEL variable, so a TMPDIR with
        # spaces or metacharacters is safe.
        shadow = (
            'exit() { if [ "${BASH_SUBSHELL:-0}" = 0 ]; then : > "$G7_SENTINEL"; fi; '
            'builtin exit "$@"; }\n' + prefix + block + "\n"
        )
        if (
            subprocess.run(
                ["bash", "-n", "-c", shadow], capture_output=True, text=True, env=env
            ).returncode
            != 0
        ):
            return False
        try:
            proc = subprocess.run(
                ["bash", "-c", shadow],
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            return False
        return proc.returncode == 1 and sentinel.exists()


def test_script_asserts_g7_env_matches_port() -> None:
    """The G7 block MUST include a fail-loud assertion that
    ``RAPID_MLX_BASE_URL`` still points at the gauntlet PORT. Without
    this, a downstream ``unset RAPID_MLX_BASE_URL`` or an unrelated
    clobber would silently reopen the issue #974 hole.

    We locate the guard and the first G7 test invocation by their stable
    semantic shapes (the ``RAPID_MLX_BASE_URL`` vs ``_expected_base``
    comparison, and the ``test_anthropic_sdk.py`` call) rather than by the
    exact section banner, whose dash count / spacing is cosmetic and has
    changed before (issue #1370)."""
    text = SCRIPT.read_text()
    invocation_idx = _find_g7_invocation(text)
    assert invocation_idx != -1, "G7 no longer runs test_anthropic_sdk.py"
    guard_idx = _find_g7_guard(text)
    assert guard_idx != -1, (
        "G7 block should include a fail-loud RAPID_MLX_BASE_URL guard"
    )
    # The guard must run before the first G7 test invocation so the SDK
    # tests never silently target a non-gauntlet server.
    assert guard_idx < invocation_idx, (
        "G7 RAPID_MLX_BASE_URL guard must run before test_anthropic_sdk.py"
    )
    # The guard must be fail-loud: feeding an unrelated RAPID_MLX_BASE_URL
    # must make the guard bail out (bash exit 1), not merely warn and
    # continue. We cut the guard's own if..fi block out (bash decides the
    # boundary) and evaluate it with bash itself, so an `exit 1` hidden in a
    # comment, quoted string, elif/else branch, or an `echo exit 1` can
    # never fake a fail-loud guard.
    guard_block = _guard_tail_block(text, guard_idx)
    assert guard_block is not None, "could not extract G7 guard block"
    assert _guard_is_fail_loud(guard_block), (
        "G7 RAPID_MLX_BASE_URL guard must exit 1 on mismatch"
    )


def test_g7_section_located_by_semantic_marker_regardless_of_banner() -> None:
    """Regression for issue #1370: the G7 section must be locatable by its
    semantic content (the ``RAPID_MLX_BASE_URL`` guard and the
    ``test_anthropic_sdk.py`` invocation), not by the exact banner
    formatting. Harmless banner-spacing changes (dash count, spacing,
    trailing text) must not break the guard."""
    guard = (
        '_expected_base="http://127.0.0.1:${PORT}/v1"\n'
        'if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]; then\n'
        '  echo "ERROR: cluster env mismatch" >&2\n'
        "  exit 1\n"
        "fi\n"
    )
    invocation = (
        'run_g7_anthropic() {\n  "$PY" tests/integrations/test_anthropic_sdk.py\n}\n'
    )
    # A range of banner formats that have appeared or could plausibly
    # appear. None of them should hide the guard or the invocation, and
    # the guard must stay before the invocation.
    banners = [
        "#-------------------- G7 SDK integration",
        "# --- G7 SDK integration (three tests, each its own job) ---",
        "# G7 SDK integration",
        "#  G7 SDK integration  ",
        "# G7 SDK integration tests",
    ]
    for banner in banners:
        text = guard + banner + "\n" + invocation
        invocation_idx = _find_g7_invocation(text)
        assert invocation_idx != -1, f"banner {banner!r} hides invocation"
        guard_idx = _find_g7_guard(text)
        assert guard_idx != -1, f"banner {banner!r} hides guard"
        assert guard_idx < invocation_idx, f"banner {banner!r} reorders guard"


def test_guard_fail_loud_scoped_to_its_own_block() -> None:
    """Evaluate the guard with a mismatched ``RAPID_MLX_BASE_URL`` in real
    bash. A guard is only fail-loud if bash actually ``exit 1``s on mismatch —
    an ``exit 1`` in a comment, a quoted string, an ``echo exit 1``, an
    ``elif``/``else`` branch, after the block, or a trailing ``false`` /
    ``return 1`` must not count (issue #974 hole)."""
    cond = 'if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]; then\n'

    def fail_loud(body: str) -> bool:
        return _guard_is_fail_loud(cond + body + "fi\n")

    # Real guard: the mismatch branch directly exits 1.
    assert fail_loud('  echo "ERROR: env mismatch" >&2\n  exit 1\n')
    # exit 1 in a comment must not count.
    assert not fail_loud('  echo "warn" >&2  # TODO: exit 1 if this regresses\n')
    # A comment *before* a real exit on its own line must not hide the exit.
    assert fail_loud("  # informational comment\n  exit 1\n")
    # exit 1 inside a quoted string (including escaped quotes) must not count.
    assert not fail_loud('  echo "you should exit 1 but we do not"\n')
    assert not fail_loud('  echo "warning: \\"exit 1\\""\n')
    # An `exit 1` emitted by an echo command is not a fail-loud exit.
    assert not fail_loud("  echo exit 1\n")
    # A failed command (false), return 1, or a stdout-closing fallthrough
    # (exec 1>&-) is not an explicit fail-loud exit.
    assert not fail_loud("  echo warn >&2\n  false\n")
    assert not fail_loud("  echo warn >&2\n  return 1\n")
    assert not fail_loud("  set -e\n  false\n")
    assert not fail_loud("  exec 1>&-\n")
    # An exit in a subshell that the guard survives must not count.
    assert not fail_loud("  ( exit 1 )\n  echo continuing\n")
    # A guard whose mismatch branch *ends* in `( exit 1 )` must also not be
    # fail-loud: the subshell raises its own status to the parent (so the
    # process returns 1) but the parent shell never exits -- the BASHPID
    # guard in the exit shadow is what distinguishes this from a real exit.
    assert not fail_loud("  ( exit 1 )\n")
    # exit 1 in the else (match) branch must not count.
    assert not fail_loud('  echo "WARNING: mismatch, continuing" >&2\nelse\n  exit 1\n')
    # exit 1 in an elif branch must not prove the mismatch branch fail-loud.
    assert not fail_loud(
        '  echo "WARNING: mismatch, continuing" >&2\n'
        'elif [ -n "${_RAPID_MLX_G7_EXTRA_SENTINEL:-}" ]; then\n  exit 1\n'
    )
    # exit 1 only inside a nested conditional must not count.
    assert not fail_loud(
        '  if [ -n "${_RAPID_MLX_G7_EXTRA_SENTINEL:-}" ]; then\n    exit 1\n  fi\n'
    )


def test_guard_extraction_scoped_to_guard_only() -> None:
    """``_guard_tail_block`` must stop at the guard's own ``fi`` and not
    swallow an executable ``exit 1`` that sits after it (which would fake a
    fail-loud result for a guard that only warns)."""
    script = (
        'if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]; then\n'
        '  echo "WARNING: mismatch, continuing" >&2\n'
        "fi\n"
        "exit 1\n"
    )
    idx = script.find('if [ "${RAPID_MLX_BASE_URL')
    block = _guard_tail_block(script, idx)
    assert block is not None
    assert block.rstrip().endswith("fi")
    # The guard itself is NOT fail-loud; the trailing exit 1 is outside it.
    assert not _guard_is_fail_loud(block)


def test_guard_extraction_ignores_keywords_in_comments_and_quotes() -> None:
    """Keywords inside comments, quoted strings, or command arguments must
    not distort the block boundary (bash decides it), and an unclosed guard
    must be rejected."""
    # A comment containing `if`, a quoted ``"fi"``, and `echo if` args all
    # sit *inside* the guard before its real fi. None of them may change the
    # boundary: extraction must still stop at the guard's real fi.
    script = (
        'if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]; then\n'
        '  echo "fi inside quotes" >&2\n'
        "  echo if\n"
        "  # if this changes, keep the guard fail-loud\n"
        "  exit 1\n"
        "fi\n"
        "exit 1\n"
    )
    idx = script.find('if [ "${RAPID_MLX_BASE_URL')
    block = _guard_tail_block(script, idx)
    assert block is not None
    assert block.rstrip().endswith("fi")
    assert not block.rstrip().endswith("exit 1")
    # It really is a fail-loud guard (exits 1 on mismatch).
    assert _guard_is_fail_loud(block)

    # A guard whose body puts `if`/`for`/`while`/`case` as echo args (not
    # shell keywords) must stay fail-loud.
    args = (
        'if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]; then\n'
        "  echo if for while case\n"
        "  exit 1\n"
        "fi\n"
    )
    assert _guard_is_fail_loud(_guard_tail_block(args, 0))

    # A multi-line double-quoted string containing `fi` must not end the
    # block early (quote state must persist across lines).
    multiline = (
        'if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]; then\n'
        '  echo "line one fi\n'
        'also two" >&2\n'
        "  exit 1\n"
        "fi\n"
    )
    assert _guard_is_fail_loud(_guard_tail_block(multiline, 0))

    # A literal `#` inside a parameter expansion (${value#prefix}) is not a
    # comment and must not hide the guard's closing `fi` on that line.
    param_exp = (
        'if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]; then\n'
        '  x="${URL#http}"\n'
        "  exit 1\n"
        "fi\n"
    )
    assert _guard_is_fail_loud(_guard_tail_block(param_exp, 0))

    # An unclosed guard must be rejected (fail closed), not silently
    # truncated into a partial block.
    unclosed = (
        'if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]; then\n  echo warn\n'
    )
    assert _guard_tail_block(unclosed, 0) is None

    # A `#` glued to the end of a quoted word (echo "x"#suffix) is PART of
    # the word in bash, so the block must still extract (staying fail-loud).
    glued_hash = (
        'if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]; then\n'
        '  echo "warning"#suffix; exit 1\n'
        "fi\n"
    )
    assert _guard_is_fail_loud(_guard_tail_block(glued_hash, 0))


def test_every_integration_base_url_env_is_covered() -> None:
    """Systematic guard: every env var an integration test reads with an
    HTTP-URL default MUST be exported by the shell script. Adding a new
    harness that reads a novel HTTP endpoint env should trip this test.

    We identify endpoint envs by the *shape of the default* (starts with
    ``http://`` or ``https://``) rather than by the env-var name. A
    name-based regex would false-positive on unrelated envs like
    ``DATABASE_URL`` (which happens to end in ``BASE_URL`` but points at
    a DB DSN, not an HTTP API) — raised in Codex review on PR #982.
    """
    # Capture the env-var name AND its default. Both single- and
    # double-quoted forms; whitespace tolerant. We require the default
    # to be a string literal (not a Python expression) so we can
    # inspect the URL scheme directly.
    pattern = re.compile(
        r"""os\.environ\.get\(
            \s*["']([A-Z_][A-Z0-9_]*)["']
            \s*,\s*
            (?:f?["']([^"']*)["'])
            \s*\)""",
        re.VERBOSE,
    )
    endpoint_envs: set[str] = set()
    for path in INTEGRATIONS.glob("*.py"):
        text = path.read_text()
        for name, default in pattern.findall(text):
            if default.startswith(("http://", "https://")):
                endpoint_envs.add(name)
    uncovered = endpoint_envs - KNOWN_BASE_URL_ENVS
    assert not uncovered, (
        f"Integration tests read HTTP-endpoint env vars {uncovered!r} "
        f"that the release script does not export. Either export them in "
        f"scripts/release_check_m3.sh or add to KNOWN_BASE_URL_ENVS with "
        f"a justification."
    )
    # And every declared env MUST actually appear as an ``export`` in
    # the script — no drift the other direction either.
    script_text = SCRIPT.read_text()
    for env in KNOWN_BASE_URL_ENVS:
        assert re.search(rf"^\s*export\s+{env}=", script_text, flags=re.MULTILINE), (
            f"KNOWN_BASE_URL_ENVS declares {env} but the shell script does "
            f"not export it — either drop it from the constant or add the "
            f"export in scripts/release_check_m3.sh."
        )
