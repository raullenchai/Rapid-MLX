#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the live main-head identity gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_main_head.py"

A = "a" * 40  # version-bump candidate commit
B = "b" * 40  # packaging fix that lands on main while A validates


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_main_head", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unchanged_head_a_passes(checker):
    env = checker.check_live_head(main_sha=A, accepted_sha=A, release_sha=A)
    assert A in "\n".join(env)


def test_a_then_b_refuses(checker):
    # B landed on main while candidate A was validating -> A is now behind head.
    with pytest.raises(
        checker.MainHeadGateError, match="no longer the validated candidate"
    ):
        checker.check_live_head(main_sha=B, accepted_sha=A, release_sha=A)


def test_main_matches_release_but_not_accepted_refuses(checker):
    with pytest.raises(checker.MainHeadGateError):
        checker.check_live_head(main_sha=A, accepted_sha=B, release_sha=A)


def test_malformed_main_sha_fails(checker):
    with pytest.raises(checker.MainHeadGateError, match="40-character"):
        checker.check_live_head(main_sha="short", accepted_sha=A, release_sha=A)


def test_malformed_accepted_sha_fails(checker):
    with pytest.raises(checker.MainHeadGateError, match="40-character"):
        checker.check_live_head(main_sha=A, accepted_sha="xxxx", release_sha=A)


def test_uppercase_sha_fails(checker):
    # git SHAs are lowercase; an uppercase/other charset value is malformed.
    with pytest.raises(checker.MainHeadGateError, match="40-character"):
        checker.check_live_head(main_sha=A.upper(), accepted_sha=A, release_sha=A)
