from __future__ import annotations

import argparse

import pytest

from bench.repro_mtp_forced_k_parity import (
    _first_divergence,
    _parse_k_values,
    _token_sha256,
)


def test_parse_k_values_requires_control_first():
    assert _parse_k_values("0,1,2,3") == (0, 1, 2, 3)
    with pytest.raises(argparse.ArgumentTypeError, match="start with the K=0 control"):
        _parse_k_values("1,2,3")
    with pytest.raises(argparse.ArgumentTypeError, match="speculative depth"):
        _parse_k_values("0")


def test_parse_k_values_rejects_duplicates_and_negative_depths():
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        _parse_k_values("0,1,1")
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        _parse_k_values("0,-1")
    with pytest.raises(argparse.ArgumentTypeError, match="supported range"):
        _parse_k_values("0,4")


def test_first_divergence_reports_token_flip_and_early_termination():
    assert _first_divergence((10, 20, 30), (10, 21, 30)) == {
        "index": 1,
        "control_token": 20,
        "candidate_token": 21,
    }
    assert _first_divergence((10, 20, 30), (10, 20)) == {
        "index": 2,
        "control_token": 30,
        "candidate_token": None,
    }
    assert _first_divergence((10, 20), (10, 20)) is None


def test_token_hash_is_stable_and_sequence_sensitive():
    assert _token_sha256((10, 20)) == _token_sha256((10, 20))
    assert _token_sha256((10, 20)) != _token_sha256((20, 10))
