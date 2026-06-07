# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``scripts/validate_release_subject.py``.

Pure stdlib + the subject script — runs on Linux CI without MLX.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "validate_release_subject.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_release_subject", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vrs():
    return _load_module()


# ---------- happy path ------------------------------------------------


def test_canonical_bump_subject_is_accepted(vrs):
    assert vrs.diagnose("chore: bump version to 0.6.82") == []


@pytest.mark.parametrize(
    "version",
    ["0.1.0", "1.0.0", "99.99.99", "0.6.100"],
)
def test_any_three_part_semver_is_accepted(vrs, version):
    assert vrs.diagnose(f"chore: bump version to {version}") == []


# ---------- the suffix trap (PF-1, primary class of bug) -------------


def test_pr_number_suffix_is_rejected(vrs):
    probs = vrs.diagnose("chore: bump version to 0.6.82 (#518)")
    assert probs
    assert any("#NN" in p or "(#" in p for p in probs), probs


def test_pr_number_suffix_with_extra_space(vrs):
    probs = vrs.diagnose("chore: bump version to 0.6.82  (#518)")
    assert probs


# ---------- prefix violations ----------------------------------------


def test_wrong_prefix(vrs):
    assert vrs.diagnose("Release 0.6.82") != []


def test_missing_colon(vrs):
    assert vrs.diagnose("chore bump version to 0.6.82") != []


def test_two_digit_version_rejected(vrs):
    # Versioning is X.Y.Z, not X.Y.
    assert vrs.diagnose("chore: bump version to 0.6") != []


# ---------- whitespace / structure -----------------------------------


def test_empty_is_rejected(vrs):
    probs = vrs.diagnose("")
    assert probs
    assert "empty" in probs[0]


def test_leading_whitespace_rejected(vrs):
    assert vrs.diagnose(" chore: bump version to 0.6.82") != []


def test_trailing_whitespace_rejected(vrs):
    assert vrs.diagnose("chore: bump version to 0.6.82 ") != []


def test_newline_in_subject_rejected(vrs):
    assert vrs.diagnose("chore: bump version to 0.6.82\nbody") != []


# ---------- CLI entry point ------------------------------------------


def test_main_exit_0_on_valid(vrs):
    assert vrs.main(["--subject", "chore: bump version to 0.6.82"]) == 0


def test_main_exit_1_on_invalid(vrs):
    assert vrs.main(["--subject", "wrong"]) == 1
