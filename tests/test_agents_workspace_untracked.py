"""Guard: the local agent workspace ``.agents/`` must never be tracked.

``.agents/`` was made local-only in #2210, but because AGENTS.md still told
agents to update ``.agents/handoffs/``, dozens of handoffs were force-added
afterwards. Handoffs now belong on the PR or tracking issue (see AGENTS.md).
Asked of git rather than the filesystem, so local scratch files don't trip it.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def test_agents_workspace_is_not_tracked() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    proc = _git("ls-files", "--", ".agents")
    if proc.returncode != 0:
        pytest.skip(f"git ls-files unavailable: {proc.stderr.strip()}")
    tracked = proc.stdout.split()
    assert tracked == [], (
        ".agents/ is a local-only workspace; post handoffs on the PR or "
        f"tracking issue instead. Tracked: {tracked[:5]}"
    )


def test_agents_md_does_not_point_at_untracked_role_files() -> None:
    text = (REPO_ROOT / "AGENTS.md").read_text()
    assert ".agents/roles" not in text
    assert ".agents/handoffs" not in text
