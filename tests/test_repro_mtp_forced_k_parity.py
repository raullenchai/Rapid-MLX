from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from bench.repro_mtp_forced_k_parity import _first_divergence, _parse_k_values


def test_parse_k_values_requires_control_first():
    assert _parse_k_values("0,1,2,3") == (0, 1, 2, 3)
    with pytest.raises(argparse.ArgumentTypeError, match="start with the K=0 control"):
        _parse_k_values("1,2,3")


def test_parse_k_values_rejects_duplicates_and_negative_depths():
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        _parse_k_values("0,1,1")
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        _parse_k_values("0,-1")


def test_first_divergence_reports_token_flip():
    assert _first_divergence((10, 20, 30), (10, 21, 30)) == {
        "index": 1,
        "control_token": 20,
        "candidate_token": 21,
    }


def test_first_divergence_reports_early_termination():
    assert _first_divergence((10, 20, 30), (10, 20)) == {
        "index": 2,
        "control_token": 30,
        "candidate_token": None,
    }
    assert _first_divergence((10, 20), (10, 20)) is None


def test_main_reports_divergence_without_failing_by_default(monkeypatch, capsys):
    import bench.repro_mtp_forced_k_parity as repro

    def fake_run_once(**kwargs):
        k = kwargs["mtp_max_k"]
        tokens = (10, 20, 30) if k == 0 else (10, 21, 30)
        return SimpleNamespace(
            token_ids=tokens,
            from_draft_flags=(False, False, False),
            accept_attempts=0 if k == 0 else 2,
            accept_count=0 if k == 0 else 1,
            verify_calls=0 if k == 0 else 2,
            k_histogram={k: 2},
            n_tokens=len(tokens),
            token_sha256=f"k{k}",
        )

    monkeypatch.setattr(repro, "_run_once", fake_run_once)
    monkeypatch.setattr(repro, "_resolve_mtp_sidecar", lambda *_: "sidecar")
    monkeypatch.setattr(
        "sys.argv",
        ["repro", "--k-values", "0,1", "--max-tokens", "3", "--format", "json"],
    )

    assert repro.main() == 0
    assert '"parity_held": false' in capsys.readouterr().out


def test_main_fails_invalid_when_fixed_k_does_not_engage(monkeypatch):
    import bench.repro_mtp_forced_k_parity as repro

    def fake_run_once(**kwargs):
        k = kwargs["mtp_max_k"]
        return SimpleNamespace(
            token_ids=(10, 20),
            from_draft_flags=(False, False),
            accept_attempts=0,
            accept_count=0,
            verify_calls=0,
            k_histogram={},
            n_tokens=2,
            token_sha256=f"k{k}",
        )

    monkeypatch.setattr(repro, "_run_once", fake_run_once)
    monkeypatch.setattr(repro, "_resolve_mtp_sidecar", lambda *_: "sidecar")
    monkeypatch.setattr("sys.argv", ["repro", "--k-values", "0,1", "--format", "json"])

    assert repro.main() == 2
