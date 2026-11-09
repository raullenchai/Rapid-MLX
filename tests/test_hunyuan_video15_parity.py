# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-framework numeric parity; the unit suite never requires torch."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_tiny_diffusers_parity(tmp_path):
    reference = os.environ.get("HUNYUAN_VIDEO15_REFERENCE_PYTHON")
    if not reference:
        pytest.skip("Set HUNYUAN_VIDEO15_REFERENCE_PYTHON to a torch/Diffusers Python")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/check_hunyuan_video15_parity.py"),
            "--reference-python",
            reference,
            "--json-output",
            str(tmp_path / "parity.json"),
        ],
        check=True,
        cwd=root,
    )
