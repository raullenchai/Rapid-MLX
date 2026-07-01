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


def test_script_asserts_g7_env_matches_port() -> None:
    """The G7 block MUST include a fail-loud assertion that
    ``RAPID_MLX_BASE_URL`` still points at the gauntlet PORT. Without
    this, a downstream ``unset RAPID_MLX_BASE_URL`` or an unrelated
    clobber would silently reopen the issue #974 hole."""
    text = SCRIPT.read_text()
    # Locate the G7 SDK integration section header (the ``#---…``
    # banner, not a mere comment reference elsewhere in the script).
    marker = "#-------------------- G7 SDK integration"
    idx = text.find(marker)
    assert idx != -1, "'G7 SDK integration' section banner vanished"
    # The assertion should live between the section header and the first
    # test invocation (`test_anthropic_sdk.py`). Look for the fail-loud
    # shape in that slice.
    invocation_idx = text.find("tests/integrations/test_anthropic_sdk.py", idx)
    assert invocation_idx != -1, "G7 no longer runs test_anthropic_sdk.py"
    g7_block = text[idx:invocation_idx]
    assert 'RAPID_MLX_BASE_URL' in g7_block, (
        "G7 block should reference RAPID_MLX_BASE_URL in an assertion"
    )
    assert 'G7 env mismatch' in g7_block, (
        "G7 assertion should print a distinctive 'G7 env mismatch' error"
    )


def test_every_integration_base_url_env_is_covered() -> None:
    """Systematic guard: every ``os.environ.get("<FOO>_BASE_URL", ...)``
    (or ``_API_BASE`` / ``_ENDPOINT``) any integration test reads MUST
    be exported by the shell script. Adding a new harness that reads a
    novel env var should trip this test."""
    pattern = re.compile(
        r'os\.environ\.get\(\s*["\']([A-Z_]*(?:BASE(?:_URL)?|API_BASE|ENDPOINT))["\']'
    )
    found: set[str] = set()
    for path in INTEGRATIONS.glob("*.py"):
        text = path.read_text()
        found.update(pattern.findall(text))
    # Only interesting envs — filter out obvious misses.
    interesting = {name for name in found if "BASE" in name or "ENDPOINT" in name}
    uncovered = interesting - KNOWN_BASE_URL_ENVS
    assert not uncovered, (
        f"Integration tests read env vars {uncovered!r} that the release "
        f"script does not export. Either export them in "
        f"scripts/release_check_m3.sh or add to KNOWN_BASE_URL_ENVS with "
        f"a justification."
    )
    # And every covered env MUST actually appear as an ``export`` in
    # the script prelude — no drift the other direction either.
    script_text = SCRIPT.read_text()
    for env in KNOWN_BASE_URL_ENVS:
        assert re.search(rf"^\s*export\s+{env}=", script_text, flags=re.MULTILINE), (
            f"KNOWN_BASE_URL_ENVS declares {env} but the shell script does "
            f"not export it — either drop it from the constant or add the "
            f"export in scripts/release_check_m3.sh."
        )
