# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-artifact parity gate for the experimental qwen4_exp lane."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REFERENCE_COMMIT = "ecf1aa0a62958ea770bc25c35e173effe142aa3c"


def _required_path(environment_key: str) -> Path:
    value = os.environ.get(environment_key)
    if not value:
        pytest.skip(f"set {environment_key} to run real qwen4_exp parity")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        pytest.fail(f"{environment_key} does not exist: {path}")
    return path


def _run_probe(
    *, backend: str, checkpoint: Path, output: Path, reference: Path | None = None
) -> None:
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    python_path = [str(Path.cwd() / "scripts"), str(Path.cwd())]
    if reference is not None:
        python_path.insert(0, str(reference))
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    subprocess.run(
        [
            sys.executable,
            "scripts/qwen4_exp_real_parity.py",
            "--backend",
            backend,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
        check=True,
        env=environment,
    )


def test_real_qwen4_exp_q4_matches_pinned_reference(tmp_path: Path) -> None:
    checkpoint = _required_path("RAPID_MLX_QWEN4_EXP_ARTIFACT")
    reference = _required_path("RAPID_MLX_QWEN4_EXP_REFERENCE")
    commit = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == _REFERENCE_COMMIT

    rapid_output = tmp_path / "rapid.npz"
    reference_output = tmp_path / "reference.npz"
    _run_probe(backend="rapid", checkpoint=checkpoint, output=rapid_output)
    _run_probe(
        backend="upstream",
        checkpoint=checkpoint,
        output=reference_output,
        reference=reference,
    )

    with np.load(rapid_output) as rapid, np.load(reference_output) as expected:
        assert set(rapid.files) == set(expected.files)
        for probe in rapid.files:
            difference = np.abs(rapid[probe] - expected[probe])
            assert float(difference.max(initial=0.0)) <= 1e-3, probe
        for logits_probe in (
            "logits_last",
            "sparse_logits_last",
            "cached_decode_logits_last",
        ):
            assert np.argmax(rapid[logits_probe]) == np.argmax(expected[logits_probe])
