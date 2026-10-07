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


def _assert_clean_pinned_reference(reference: Path) -> None:
    commit = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == _REFERENCE_COMMIT, (
        f"reference checkout is at {commit}, expected {_REFERENCE_COMMIT}"
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(reference),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--ignore-submodules=none",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert not status, f"reference checkout has tracked or submodule changes:\n{status}"


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
    _assert_clean_pinned_reference(reference)

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


def test_reference_guard_rejects_tracked_changes(tmp_path: Path, monkeypatch) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    subprocess.run(["git", "init", "-q", str(reference)], check=True)
    tracked = reference / "tracked.py"
    tracked.write_text("original\n")
    subprocess.run(["git", "-C", str(reference), "add", "tracked.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(reference),
            "-c",
            "user.name=Rapid MLX Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(sys.modules[__name__], "_REFERENCE_COMMIT", head)
    tracked.write_text("modified\n")

    with pytest.raises(AssertionError, match="tracked or submodule changes"):
        _assert_clean_pinned_reference(reference)


def test_reference_guard_rejects_modified_submodule(
    tmp_path: Path, monkeypatch
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    subprocess.run(["git", "init", "-q", str(child)], check=True)
    child_file = child / "tracked.py"
    child_file.write_text("original\n")
    subprocess.run(["git", "-C", str(child), "add", "tracked.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(child),
            "-c",
            "user.name=Rapid MLX Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    reference = tmp_path / "reference"
    reference.mkdir()
    subprocess.run(["git", "init", "-q", str(reference)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(reference),
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(child),
            "vendor/child",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(reference),
            "-c",
            "user.name=Rapid MLX Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qam",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(sys.modules[__name__], "_REFERENCE_COMMIT", head)
    (reference / "vendor/child/tracked.py").write_text("modified\n")

    with pytest.raises(AssertionError, match="tracked or submodule changes"):
        _assert_clean_pinned_reference(reference)
