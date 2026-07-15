# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``scripts/check_gha_pinning.py``.

Pure stdlib — runs on Linux CI without MLX. Tests synthesize tiny
workflow YAMLs in tmp dirs so the production workflow files don't
affect the assertions.
"""

from __future__ import annotations

import importlib.util
import pathlib
import textwrap

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_gha_pinning.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_gha_pinning", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cgp():
    return _load_module()


def _make_workflow(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    wf = tmp_path / "test.yml"
    wf.write_text(textwrap.dedent(body))
    return wf


# ---------- all tag refs are violations -------------------------------


def test_actions_owner_tag_is_violation(cgp, tmp_path):
    wf = _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - uses: actions/checkout@v4
              - uses: actions/setup-python@v5
        """,
    )
    assert len(cgp.violations_in_file(wf)) == 2


def test_github_owner_tag_is_violation(cgp, tmp_path):
    wf = _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - uses: github/codeql-action@v3
        """,
    )
    assert len(cgp.violations_in_file(wf)) == 1


# ---------- third-party tag refs are violations -----------------------


def test_third_party_tag_is_violation(cgp, tmp_path):
    wf = _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - uses: codecov/codecov-action@v4
        """,
    )
    violations = cgp.violations_in_file(wf)
    assert len(violations) == 1
    assert "codecov/codecov-action" in violations[0]
    assert "@v4" in violations[0]


def test_third_party_branch_is_violation(cgp, tmp_path):
    wf = _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - uses: pypa/gh-action-pypi-publish@release/v1
        """,
    )
    assert len(cgp.violations_in_file(wf)) == 1


def test_third_party_short_sha_is_violation(cgp, tmp_path):
    # Short SHAs (7 chars) are NOT acceptable — must be 40-char.
    wf = _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - uses: codecov/codecov-action@abc1234
        """,
    )
    assert len(cgp.violations_in_file(wf)) == 1


# ---------- 40-char SHAs are accepted --------------------------------


def test_third_party_40_char_sha_is_accepted(cgp, tmp_path):
    sha = "0" * 40
    wf = _make_workflow(
        tmp_path,
        f"""
        jobs:
          x:
            steps:
              - uses: codecov/codecov-action@{sha}  # v4.5.0
        """,
    )
    assert cgp.violations_in_file(wf) == []


def test_third_party_40_char_sha_uppercase_is_rejected(cgp, tmp_path):
    # Be strict: lowercase hex only. Uppercase shouldn't happen but is
    # an easy typo to introduce silently.
    sha = "A" * 40
    wf = _make_workflow(
        tmp_path,
        f"""
        jobs:
          x:
            steps:
              - uses: codecov/codecov-action@{sha}
        """,
    )
    assert len(cgp.violations_in_file(wf)) == 1


# ---------- quoted YAML forms must NOT bypass the check ---------------


def test_quoted_uses_key_tag_is_violation(cgp, tmp_path):
    """``"uses": ...`` (quoted key) is valid YAML and must be caught.

    The old raw-text ``^\\s*uses:`` regex silently skipped this, leaving a
    mutable action reference green (codex review).
    """
    wf = _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - "uses": actions/checkout@v4
        """,
    )
    assert len(cgp.violations_in_file(wf)) == 1


def test_quoted_uses_value_tag_is_violation(cgp, tmp_path):
    """``uses: "actions/checkout@v4"`` (quoted value) must be caught."""
    wf = _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - uses: "actions/checkout@v4"
        """,
    )
    assert len(cgp.violations_in_file(wf)) == 1


def test_quoted_uses_value_sha_is_accepted(cgp, tmp_path):
    """A quoted value that IS a 40-char SHA must still pass."""
    sha = "0" * 40
    wf = _make_workflow(
        tmp_path,
        f"""
        jobs:
          x:
            steps:
              - uses: "actions/checkout@{sha}"
        """,
    )
    assert cgp.violations_in_file(wf) == []


def test_local_action_is_accepted(cgp, tmp_path):
    """A same-repo local action (``./...``) has no supply-chain hop → pass."""
    wf = _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - uses: ./.github/actions/setup
        """,
    )
    assert cgp.violations_in_file(wf) == []


def test_container_digest_is_accepted_but_tag_is_violation(cgp, tmp_path):
    """Container actions must pin a sha256 digest; a mutable tag is a violation."""
    good = _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - uses: docker://ghcr.io/org/img@sha256:{d}
        """.replace("{d}", "0" * 64),
    )
    assert good.read_text()  # sanity
    assert cgp.violations_in_file(good) == []

    bad = tmp_path / "bad.yml"
    import textwrap

    bad.write_text(
        textwrap.dedent(
            """
            jobs:
              x:
                steps:
                  - uses: docker://ghcr.io/org/img:latest
            """
        )
    )
    assert len(cgp.violations_in_file(bad)) == 1


# ---------- entry point ----------------------------------------------


def test_main_clean_dir_exits_0(cgp, tmp_path):
    _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - uses: actions/checkout@0000000000000000000000000000000000000000
        """,
    )
    assert cgp.main(["--workflows-dir", str(tmp_path)]) == 0


def test_main_violation_dir_exits_1(cgp, tmp_path):
    _make_workflow(
        tmp_path,
        """
        jobs:
          x:
            steps:
              - uses: codecov/codecov-action@v4
        """,
    )
    assert cgp.main(["--workflows-dir", str(tmp_path)]) == 1


def test_main_empty_dir_exits_0(cgp, tmp_path):
    assert cgp.main(["--workflows-dir", str(tmp_path)]) == 0


def test_main_missing_dir_exits_1(cgp, tmp_path):
    assert cgp.main(["--workflows-dir", str(tmp_path / "nope")]) == 1
