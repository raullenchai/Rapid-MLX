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


def test_moe_probe_applies_routing_weight_once() -> None:
    import mlx.core as mx

    module = _load_module()
    gate = mx.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    tensors = {"layers.0.ffn.gate.weight": gate, "layers.0.ffn.gate.bias": mx.zeros(3)}

    class Runtime:
        c = {
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.0,
        }
        w = type("W", (), {"read": staticmethod(tensors.__getitem__)})()

        def expert(self, name, x, weight=1.0):
            if name.endswith("shared_experts"):
                return mx.zeros_like(x)
            return x * (int(name.rsplit(".", 1)[-1]) + 1) * weight

    runtime = Runtime()
    seen = []
    module.install_moe_probe(runtime, observer=lambda *args: seen.append(args))
    x = mx.array([[1.0, 2.0]])
    out = runtime.moe("layers.0.ffn", x)

    scores = np.sqrt(np.logaddexp(np.array(x @ gate.T), 0))[0]
    picks = np.argsort(scores)[-2:]
    weights = scores[picks] / scores[picks].sum() * 2.0
    expected = sum(np.array(x)[0] * (pick + 1) * w for pick, w in zip(picks, weights))
    assert np.allclose(np.array(out)[0], expected, atol=1e-5)
    # Observer norms are gate-free (unit weight), not routing-weighted.
    for (_layer, expert, _gate, norm), pick in zip(seen, picks):
        assert expert == pick
        assert np.isclose(norm, np.linalg.norm(np.array(x)[0] * (pick + 1)), atol=1e-5)
