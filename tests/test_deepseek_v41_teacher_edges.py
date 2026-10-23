"""CPU validation and bounded verification of the source teacher."""

import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from scripts import deepseek_v41_streaming_teacher as teacher
from scripts import verify_deepseek_v41_streaming_teacher as verifier
from tests import test_deepseek_v41_streaming_teacher as fixtures


@pytest.fixture(autouse=True)
def cpu_device():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(old)


def test_bounded_verifier_matches_native_and_reports_failure(tmp_path, monkeypatch):
    original = fixtures.tiny_config
    monkeypatch.setattr(
        fixtures, "tiny_config", lambda: dict(original(), n_routed_experts=8)
    )
    source = tmp_path / "source"
    fixtures.write_source(source)
    load = importlib.import_module("rapid_mlx.models.deepseek_v41_native.load")
    monkeypatch.setattr(load, "load_token_map", lambda *a, **k: list(range(32)))
    output = tmp_path / "report.json"
    argv = [
        "verify",
        "--source",
        str(source),
        "--output",
        str(output),
        "--experts",
        "6",
        "--layers",
        "2",
        "--tokens",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    verifier.main()
    report = json.loads(output.read_text())
    assert report["passed"] and len(report["hidden"]) == 4
    assert report["experts"] == 6 and report["source_experts"] == 8
    real_errors = verifier.errors
    monkeypatch.setattr(
        verifier, "errors", lambda a, b: dict(real_errors(a, b), max_abs=1)
    )
    with pytest.raises(SystemExit) as exc:
        verifier.main()
    assert exc.value.code == 1 and not json.loads(output.read_text())["passed"]
    monkeypatch.setattr(sys, "argv", argv + ["--layers", "1"])
    with pytest.raises(ValueError, match="bounded check"):
        verifier.main()
    with pytest.raises(ValueError, match="nonfinite"):
        real_errors(mx.array([float("nan")]), mx.array([1.0]))


def test_source_and_sequence_guards(tmp_path):
    source = tmp_path / "source"
    args, _ = fixtures.write_source(source)
    p = source / "config.json"
    config = json.loads(p.read_text())
    config["quantization_config"]["expert_dtype"] = "fp8"
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="FP4/FP8"):
        teacher.SourceWeights(source)
    config["quantization_config"]["expert_dtype"] = "fp4"
    config["quantization"] = {"bits": 4}
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unpruned"):
        teacher.SourceWeights(source)
    del config["quantization"]
    p.write_text(json.dumps(config))
    weights = teacher.SourceWeights(source)
    name = next(
        n
        for n in weights.index.tensors
        if n.startswith("layers.0.")
        and ".ffn.experts." not in n
        and ".engram." not in n
    )
    del weights.index.tensors[name]
    with pytest.raises(ValueError, match="key mismatch"):
        weights.layer(0)
    engram = teacher.SourceEngram(weights.index, 1)
    with pytest.raises(ValueError, match="row out of range"):
        engram(mx.array([-1]))
    for kwargs, tokens, match in [
        ({"prefill_chunk": 0}, [[1]], "chunking"),
        ({}, [[]], "nonempty"),
        ({}, [[1]], "token map"),
    ]:
        with pytest.raises(ValueError, match=match):
            teacher.stream_forward(args, tokens, lambda: None, lambda _: None, **kwargs)


def test_cpu_adapter_keeps_bias():
    from mlx_lm.models.switch_layers import SwitchLinear

    projections = [SwitchLinear(2, 2, 2, bias=True) for _ in range(3)]
    block = SimpleNamespace(
        ffn=SimpleNamespace(
            experts=SimpleNamespace(
                **dict(zip(("gate_proj", "up_proj", "down_proj"), projections))
            )
        )
    )
    teacher.install_cpu_gather(block)
    for p in projections:
        p.weight = mx.ones((2, 2, 2))
        p.bias = mx.ones((2, 2))
        np.testing.assert_allclose(
            np.asarray(p(mx.ones((1, 1, 1, 2)), mx.array([[0]]))), 3
        )


def test_teacher_cli_input_guards(tmp_path, monkeypatch):
    source = tmp_path / "source"
    fixtures.write_source(source)
    manifest = tmp_path / "slices.json"
    manifest.write_text(json.dumps({"slices": {"a": ["unused.txt"]}}))
    output = tmp_path / "output"
    argv = [
        "teacher",
        "reference",
        "--source",
        str(source),
        "--slices",
        str(manifest),
        "--output",
        str(output),
        "--tokens",
        "8",
        "--keep-experts",
        "3",
    ]
    monkeypatch.setattr(sys, "argv", argv + ["--tokens", "0"])
    with pytest.raises(ValueError, match="invalid tokens"):
        teacher.main()
    monkeypatch.setattr(sys, "argv", argv + ["--estimate-only"])
    teacher.main()
    assert not output.exists()
    manifest.write_text(json.dumps({"slices": {"../bad": ["unused.txt"]}}))
    with pytest.raises(ValueError, match="unsafe"):
        teacher.main()
    manifest.write_text(json.dumps({"slices": {"a": ["unused.txt"]}}))
    output.mkdir()
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        teacher.main()
    output.rmdir()
    import tokenizers

    monkeypatch.setattr(tokenizers.Tokenizer, "from_file", lambda _: None)
    (source / "tokenizer.json").write_text("{}")
    manifest.write_text(
        json.dumps({"slices": {"a": ["unused.txt"]}, "tokenizer_sha256": "wrong"})
    )
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        teacher.main()


def test_qualification_passes_explicit_engram_offload(tmp_path, monkeypatch):
    from scripts import qualify_deepseek_v41_native as qualify

    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["qualify", str(tmp_path), "--engram-ssd-offload", "--intervals", "1"],
    )
    monkeypatch.setattr(
        qualify, "load", lambda *a, **kw: (calls.append(kw) or SimpleNamespace(), None)
    )
    monkeypatch.setattr(
        qualify,
        "PreTrainedTokenizerFast",
        lambda **kw: SimpleNamespace(
            encode=lambda *a, **kw: [1], decode=lambda _: "ok"
        ),
    )
    monkeypatch.setattr(qualify, "run_once", lambda *a, **kw: {"output_tokens": [1]})
    qualify.main()
    assert calls[0]["engram_ssd_offload"] is True


def test_memory_estimate_accounts_for_three_resident_projection_banks(tmp_path):
    source = tmp_path / "source"
    args, _ = fixtures.write_source(source)
    weights = teacher.SourceWeights(source)
    report = teacher.memory_estimate(weights, 2, 8)
    bank = 2 * args.n_routed_experts * args.dim * args.moe_inter_dim
    assert report["layer_bf16_bytes"] >= 3 * bank
    assert report["stack_transient_bytes"] == bank
    assert report["estimated_peak_bytes"] == (
        report["layer_bf16_bytes"]
        + bank
        + report["persistent_state_bytes"]
        + report["prefill_scratch_allowance_bytes"]
        + 8 * 2**30
    )


@pytest.mark.parametrize("mode", ["reference", "calibrate"])
def test_teacher_rechecks_source_before_publication(tmp_path, monkeypatch, mode):
    from tokenizers import Tokenizer, models

    source = tmp_path / "source"
    fixtures.write_source(source)
    tok = Tokenizer(models.WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
    tok.save(str(source / "tokenizer.json"))
    np.save(tmp_path / "a.npy", np.arange(8, dtype=np.int32))
    manifest = tmp_path / "slices.json"
    manifest.write_text(json.dumps({"slices": {"a": ["a.npy"]}}))
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teacher",
            mode,
            "--source",
            str(source),
            "--slices",
            str(manifest),
            "--output",
            str(output),
            "--tokens",
            "8",
            "--keep-experts",
            "3",
            "--cpu",
        ],
    )
    original = teacher.stream_forward

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        config = source / "config.json"
        config.write_bytes(config.read_bytes() + b" ")
        return result

    monkeypatch.setattr(teacher, "stream_forward", mutate)
    with pytest.raises(ValueError, match="checkpoint changed"):
        teacher.main()
    assert not (output / "metadata.json").exists()
    assert not output.with_suffix(".npz").exists()


def test_teacher_hashes_source_before_initialization(tmp_path, monkeypatch):
    source = tmp_path / "source"
    fixtures.write_source(source)
    manifest = tmp_path / "slices.json"
    manifest.write_text(json.dumps({"slices": {"a": ["unused.txt"]}}))
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teacher",
            "reference",
            "--source",
            str(source),
            "--slices",
            str(manifest),
            "--output",
            str(output),
            "--tokens",
            "8",
            "--keep-experts",
            "3",
        ],
    )
    original = teacher.SourceWeights

    def replace_source(*args, **kwargs):
        weights = original(*args, **kwargs)
        config = source / "config.json"
        config.write_bytes(config.read_bytes() + b" ")
        return weights

    monkeypatch.setattr(teacher, "SourceWeights", replace_source)
    with pytest.raises(ValueError, match="changed during initialization"):
        teacher.main()
    assert not output.exists()


@pytest.mark.parametrize("mode", ["calibrate", "reference"])
@pytest.mark.parametrize("target", ["manifest", "corpus"])
def test_teacher_rechecks_tokenization_inputs(tmp_path, monkeypatch, mode, target):
    from tokenizers import Tokenizer, models

    source = tmp_path / "source"
    fixtures.write_source(source)
    Tokenizer(models.WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0")).save(
        str(source / "tokenizer.json")
    )
    np.save(tmp_path / "a.npy", np.arange(8, dtype=np.int32))
    manifest = tmp_path / "slices.json"
    manifest.write_text(json.dumps({"slices": {"a": ["a.npy"]}}))
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teacher",
            mode,
            "--source",
            str(source),
            "--slices",
            str(manifest),
            "--output",
            str(output),
            "--tokens",
            "8",
            "--keep-experts",
            "3",
            "--cpu",
        ],
    )
    original = teacher.stream_forward

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = manifest if target == "manifest" else tmp_path / "a.npy"
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(teacher, "stream_forward", mutate)
    with pytest.raises(ValueError, match="slice inputs changed"):
        teacher.main()
    assert not (output / "metadata.json").exists()
    assert not output.with_suffix(".npz").exists()
