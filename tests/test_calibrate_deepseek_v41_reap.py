from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


@pytest.fixture
def mlx_cpu():
    mlx_probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if mlx_probe.returncode:
        pytest.skip("mlx.core is not importable")
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield mx
    finally:
        mx.set_default_device(previous_device)


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "calibrate_deepseek_v41_reap.py"
    spec = importlib.util.spec_from_file_location("calibrate_deepseek_v41_reap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calibrate = _load_module()


def test_parse_args_keep_experts_default_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required_args = ["calibrate", "--model", "model", "--output", "scores.npz"]
    monkeypatch.setattr(sys, "argv", required_args)
    assert calibrate.parse_args().keep_experts == 288

    monkeypatch.setattr(sys, "argv", [*required_args, "--keep-experts", "336"])
    assert calibrate.parse_args().keep_experts == 336


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

    with pytest.raises(ValueError, match="corpus has"):
        calibrate._corpus_tokens(Runtime(), [corpus], count=3)

    Encoding.ids = [1, 2, 3, 4, 5]
    assert calibrate._corpus_tokens(Runtime(), [corpus], count=3) == [1, 2, 3]


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

    with pytest.raises(SystemExit) as error:
        calibrate.main()
    assert str(error.value) == (
        "refusing to execute checkpoint-bundled Python without "
        "--trust-checkpoint-runtime"
    )


def test_main_native_does_not_require_checkpoint_runtime_trust(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import ModuleType

    monkeypatch.setitem(sys.modules, "mlx.core", ModuleType("mlx.core"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_deepseek_v41_reap.py",
            "--model",
            str(tmp_path / "model"),
            "--native",
            "--output",
            str(tmp_path / "result.npz"),
        ],
    )

    with pytest.raises(ValueError, match="provide a corpus/slices"):
        calibrate.main()


def test_moe_probe_applies_routing_weight_once(mlx_cpu) -> None:
    mx = mlx_cpu
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
