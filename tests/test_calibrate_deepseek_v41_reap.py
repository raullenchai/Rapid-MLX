from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "calibrate_deepseek_v41_reap.py"
    spec = importlib.util.spec_from_file_location("calibrate_deepseek_v41_reap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calibrate = _load_module()


def test_overlap_measures_per_layer_keep_set_stability() -> None:
    left = np.array([[9.0, 8.0, 1.0, 0.0], [7.0, 6.0, 5.0, 0.0]])
    right = np.array([[9.0, 7.0, 8.0, 0.0], [7.0, 6.0, 0.0, 5.0]])

    assert calibrate._overlap(left, right, keep=2) == 0.75


def test_corpus_tokens_requires_requested_sample(tmp_path: Path) -> None:
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("one two")

    class Encoding:
        ids = [1, 2]

    class Tokenizer:
        @staticmethod
        def encode(_text):
            return Encoding()

    class Runtime:
        tokenizer = Tokenizer()

    try:
        calibrate._corpus_tokens(Runtime(), [corpus], count=3)
    except ValueError as error:
        assert "need 3" in str(error)
    else:
        raise AssertionError("short calibration corpus must be rejected")


def test_main_refuses_checkpoint_code_without_explicit_trust(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_deepseek_v41_reap.py",
            "--model",
            str(tmp_path / "model"),
            "--output",
            str(tmp_path / "result.npz"),
            str(tmp_path / "corpus.txt"),
        ],
    )

    with pytest.raises(SystemExit, match="refusing to execute"):
        calibrate.main()
