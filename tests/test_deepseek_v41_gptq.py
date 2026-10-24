"""GPTQ numerical, native packing, streamed state and restart contracts."""

import json
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx


@pytest.fixture(autouse=True)
def restore_cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


from scripts.deepseek_v41_gptq import (
    MemoryGuard,
    assemble,
    initialize,
    recipe,
    run_layer,
)
from scripts.deepseek_v41_gptq_io import (
    PartialSource,
    inventory,
    load_state,
    required_shards,
)
from scripts.deepseek_v41_gptq_quant import (
    QuantConfig,
    gptq,
    gram,
    inverse_factor,
    pack_codes,
)
from tests.test_deepseek_v41_streaming_teacher import write_source


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_packing_matches_native_quantizer(bits):
    mx.set_default_device(mx.cpu)
    mx.random.seed(1017)
    w = mx.random.normal((5, 256)).astype(mx.float32)
    native = mx.quantize(w, bits=bits, group_size=64)
    s, b = native[1].astype(mx.float32), native[2].astype(mx.float32)
    codes = mx.clip(
        mx.round(
            (w.reshape(5, 4, 64).astype(mx.float32) - b[..., None]) / s[..., None]
        ),
        0,
        2**bits - 1,
    )
    packed = pack_codes(np.asarray(codes).reshape(5, 256), bits)
    assert np.array_equal(np.asarray(packed), np.asarray(native[0]))
    assert np.array_equal(
        np.asarray(mx.dequantize(packed, *native[1:], bits=bits, group_size=64)),
        np.asarray(mx.dequantize(*native, bits=bits, group_size=64)),
    )


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_gptq_beats_rtn_on_known_correlated_hessian(bits):
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(1017)
    x = rng.normal(size=(1024, 128)).astype(np.float32)
    x[:, 1:] += 2 * x[:, :1]
    w = mx.array(rng.normal(size=(32, 128)).astype(np.float32)).astype(mx.bfloat16)
    h = mx.array(x.T @ x)
    packed, stats = gptq(w, h, bits, QuantConfig(damping=0.01), count=1024)
    q = mx.dequantize(*packed, bits=bits, group_size=64).astype(mx.float32)
    r = mx.dequantize(
        *mx.quantize(w, bits=bits, group_size=64), bits=bits, group_size=64
    ).astype(mx.float32)

    def loss(v):
        d = w.astype(mx.float32) - v
        return float(mx.sum((d @ h) * d).item())

    assert loss(q) < loss(r)
    assert stats["low_token"]
    assert all(np.isfinite(np.asarray(p.astype(mx.float32))).all() for p in packed)


def test_weighted_gram_and_damping_paths():
    x = np.arange(96, dtype=np.float32).reshape(3, 32) / 96
    weights = np.array([0.2, 0.7, 1.1], np.float32)
    h, n = gram([(x, weights)], 32)
    assert n == 3
    # Metal float32 matmul on some GPUs (e.g. M5) carries ~1e-3 relative error.
    assert np.allclose(
        np.asarray(h), x.T @ (x * weights[:, None]), rtol=2e-3, atol=1e-4
    )
    c = QuantConfig(group_size=32)
    _, _, low = inverse_factor(h, c, 3)
    _, _, high = inverse_factor(h, c, 2048)
    assert low["damping_coefficient"] == 1
    assert high["damping_coefficient"] == 0.1
    factor, dead, stats = inverse_factor(mx.zeros((32, 32)), c, 0)
    assert np.isfinite(np.asarray(factor)).all()
    assert np.asarray(dead).all() and stats["dead_columns"] == 32
    with pytest.raises(ValueError, match="routing weight"):
        gram([(x, -weights)], 32)


def test_partial_source_shard_requirements(tmp_path):
    write_source(tmp_path / "source")
    manifest = inventory(tmp_path / "source")
    # No source config or full source index is needed on the small machine.
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "source.safetensors").symlink_to(
        tmp_path / "source" / "source.safetensors"
    )
    source = PartialSource(tmp_path / "partial", manifest)
    source.verify(2)
    assert required_shards(manifest, 2) == ["source.safetensors"]
    assert required_shards(
        json.loads((tmp_path / "source" / "model.safetensors.index.json").read_text()),
        2,
    ) == ["source.safetensors"]
    assert source.tensor("layers.2.ffn.experts.3.w2.weight").shape == (64, 64)


def _build(root, source, c, *, interrupt=False, length=17):
    parts, scratch = root / "parts", root / "scratch"
    with MemoryGuard(c) as guard:
        initialize(
            source,
            c,
            [
                np.arange(1, length + 1, dtype=np.int32) % 31,
                np.array([7, 2, 3, 4, 8], np.int32),
            ],
            scratch,
            parts,
            "fixture-manifest",
            guard,
            token_map=list(range(32)),
        )
        # Give uninterrupted and restarted conversions the same fresh-process
        # numerical environment. The surrounding suite can initialize array
        # runtime state differently; only the restart should differ here.
        import subprocess
        import sys

        driver = root / "restart.py"
        driver.write_text("""import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from scripts.deepseek_v41_gptq import MemoryGuard, run_layer
from scripts.deepseek_v41_gptq_io import PartialSource
import mlx.core as mx
mx.set_default_device(mx.cpu)
parts, scratch, source_root = map(Path, sys.argv[1:4])
run = json.loads((parts / 'run.json').read_text())
c = run['recipe']
source = PartialSource(source_root, json.loads((parts / 'source-manifest.json').read_text()))
with MemoryGuard(c) as guard:
    first = json.loads((scratch / 'checkpoint.json').read_text())['next_layer']
    for layer in range(first, int(sys.argv[4]) + 1):
        run_layer(source, c, parts, scratch, layer, guard)
    if sys.argv[5] == 'crash':
        os._exit(77)
""")
        args = [
            sys.executable,
            str(driver),
            str(parts),
            str(scratch),
            str(root.parent / "source"),
        ]
        if interrupt:
            killed = subprocess.run(
                [*args, "1", "crash"], capture_output=True, timeout=120
            )
            assert killed.returncode == 77, killed.stderr.decode()
        resumed = subprocess.run(
            [*args, "3", "finish"], capture_output=True, timeout=120
        )
        assert resumed.returncode == 0, resumed.stderr.decode()
        assemble(source, c, parts, root / "build", guard)
    return parts, scratch, root / "build"


def gc_collect():
    import gc

    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize("prefill_chunk,length", [(4, 17), (40, 70)])
def test_streamed_build_resume_native_load_and_finite_logits(
    tmp_path, prefill_chunk, length
):
    from rapid_mlx.models.deepseek_v41_native.load import (
        load,
        supports_engram_ssd_offload,
    )

    _, raw = write_source(tmp_path / "source")
    mx.random.seed(2026)
    for name, value in raw.items():
        if name.endswith(".ffn.gate.weight"):
            raw[name] = (0.1 * mx.random.normal(value.shape)).astype(mx.bfloat16)
    mx.save_safetensors(str(tmp_path / "source" / "source.safetensors"), raw)
    source = PartialSource(tmp_path / "source", inventory(tmp_path / "source"))
    c = recipe(
        dict(prefill_chunk=prefill_chunk, ffn_chunk=8, min_tokens=4, keep_mtp=False)
    )
    (tmp_path / "a").mkdir()
    a, scratch, build = _build(tmp_path / "a", source, c, length=length)
    (tmp_path / "b").mkdir()
    b, _, resumed = _build(tmp_path / "b", source, c, interrupt=True, length=length)
    for path in build.rglob("*"):
        if path.is_file():
            from hashlib import sha256

            relative = path.relative_to(build)
            assert (
                sha256(path.read_bytes()).hexdigest()
                == sha256((resumed / relative).read_bytes()).hexdigest()
            ), str(relative)
    assert json.loads((scratch / "checkpoint.json").read_text())["next_layer"] == 4
    final = load_state(scratch / "boundary-4" / "window-0")
    assert len(final["selections"]) == (length + prefill_chunk - 1) // prefill_chunk
    assert final["shared"].kv_src_cache is not None
    assert final["shared"].index_src_cache is not None
    assert supports_engram_ssd_offload(str(build))
    model, _ = load(str(build), lazy=True, engram_ssd_offload=True)
    logits = model(mx.array([[1, 2, 3, 4, 5]]), model.make_cache(max_seq_len=5))
    assert np.isfinite(np.asarray(logits)).all()
    # Native CPU affine kernels use scalar BF16 accumulation. Both streamed
    # CPU schedules below use the same float32 adapter; native load/finite is
    # independently exercised above. Metal uses the unchanged native kernels.
    from scripts.deepseek_v41_gptq import install_cpu_affine

    install_cpu_affine(model)
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import hc_pre_norm

    expected_logits = model.head(
        hc_pre_norm(
            final["h"], final["pre"], model.norm.weight, model.args.norm_eps
        ).astype(mx.float32)
    )
    ids = np.asarray(final["ids"])
    cache = model.make_cache(max_seq_len=len(ids))
    chunks = [
        model(mx.array(ids[None, start : start + c["prefill_chunk"]]), cache)
        for start in range(0, len(ids), c["prefill_chunk"])
    ]
    native_logits = mx.concatenate(chunks, axis=1)
    assert np.allclose(
        np.asarray(native_logits), np.asarray(expected_logits), rtol=0.02, atol=0.005
    )
    # Exact output payload accounting uses the same Entry contract as RD builds.
    cfg = json.loads((build / "config.json").read_text())
    assert cfg["quantization"]["engram_bits"] == 6
    assert cfg["rapid_quantization"]["bytes"]["resident_tensor_bytes"] > 0
    with MemoryGuard(c) as guard, pytest.raises(ValueError, match="identity changed"):
        run_layer(source, {**c, "damping": 0.2}, a, scratch, 3, guard)


def test_final_run_resumes_after_attention_commit_and_output_window_crash(
    tmp_path, monkeypatch
):
    import scripts.deepseek_v41_gptq as build

    write_source(tmp_path / "source")
    source = PartialSource(tmp_path / "source", inventory(tmp_path / "source"))
    c = recipe(
        dict(
            pilot=False,
            dense_8bit_layers=[0, 1, 2, 3],
            prefill_chunk=4,
            ffn_chunk=8,
            min_tokens=4,
            keep_mtp=False,
        )
    )
    windows = [np.arange(1, 10, dtype=np.int32)]
    with MemoryGuard(c) as guard:
        initialize(
            source,
            c,
            windows,
            tmp_path / "s1",
            tmp_path / "p1",
            "manifest",
            guard,
            token_map=list(range(32)),
        )
        initialize(
            source,
            c,
            windows,
            tmp_path / "s2",
            tmp_path / "p2",
            "manifest",
            guard,
            token_map=list(range(32)),
        )
        run_layer(source, c, tmp_path / "p1", tmp_path / "s1", 0, guard)
        original = build.finalize_windows

        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("simulated process loss after output window commit")

        monkeypatch.setattr(build, "finalize_windows", interrupted)
        with pytest.raises(RuntimeError, match="process loss"):
            run_layer(source, c, tmp_path / "p2", tmp_path / "s2", 0, guard)
        assert not (tmp_path / "s2" / "boundary-0").exists()
        assert (
            json.loads((tmp_path / "s2" / "checkpoint.json").read_text())["phase"]
            == "ffn"
        )
        monkeypatch.setattr(build, "finalize_windows", original)
        run_layer(source, c, tmp_path / "p2", tmp_path / "s2", 0, guard)
    assert (tmp_path / "p1" / "model-layer-0.safetensors").read_bytes() == (
        tmp_path / "p2" / "model-layer-0.safetensors"
    ).read_bytes()
    from scripts.deepseek_v41_gptq_io import state_snapshot

    assert (
        state_snapshot(tmp_path / "s1/boundary-1/window-0")[1].read_bytes()
        == state_snapshot(tmp_path / "s2/boundary-1/window-0")[1].read_bytes()
    )


def test_memory_cap_and_invalid_recipe_fail_closed():
    c = recipe(dict(memory_limit_gb=0.000001))
    with pytest.raises(MemoryError, match="memory cap"), MemoryGuard(c):
        pass
    with pytest.raises(ValueError, match="boolean"):
        recipe(dict(pilot="false"))
    with pytest.raises(ValueError, match="Hessian"):
        inverse_factor(mx.full((32, 32), float("nan")), QuantConfig(group_size=32), 32)


def test_selector_uses_measured_block_movement_then_exact_spare_budget(tmp_path):
    from dataclasses import replace

    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from scripts.deepseek_v41_gptq import output_plan, select_recipe

    write_source(tmp_path / "source")
    source = PartialSource(tmp_path / "source", inventory(tmp_path / "source"))
    originals = dict(source.index.tensors)
    for layer in range(4, 12):
        for name, tensor in originals.items():
            if name.startswith("layers.0."):
                renamed = name.replace("layers.0.", f"layers.{layer}.", 1)
                source.index.tensors[renamed] = replace(tensor, name=renamed)
    source.config = {
        **source.config,
        "num_hidden_layers": 12,
        "compress_ratios": [2, 2] + [1] * 10,
    }
    source.args = ModelArgs.from_dict(source.config)
    c = recipe(dict(keep_mtp=False))
    _, baseline = output_plan(
        source, {**c, "pilot": False, "dense_8bit_layers": list(range(4, 12))}
    )
    c["resident_budget_gb"] = (baseline["resident_tensor_bytes"] + 6144) / 1e9
    from scripts.deepseek_v41_gptq_io import digest

    pilot_identity = digest(dict(recipe=c))
    (tmp_path / "run.json").write_text(
        json.dumps(dict(identity=pilot_identity, recipe=c))
    )
    for layer in range(12):
        stats = dict(
            layer=layer,
            run_identity=pilot_identity,
            block_output_4bit_relative_error=layer / 100,
            projections={
                "expert.0.w1": dict(
                    gptq_relative_output_error=0.2,
                    three_bit_gptq_relative_output_error=0.1,
                    calibration_output_energy=1000 - layer * 100 if layer < 2 else 0,
                )
            },
        )
        (tmp_path / f"layer-{layer}-stats.json").write_text(json.dumps(stats))
    result = select_recipe(tmp_path, source, c)
    assert result["dense_8bit_layers"] == list(range(4, 12))
    assert result["gate_up_3bit_layers"] == [0]
    assert not result["pilot"]
    _, costs = output_plan(source, result)
    assert costs["resident_tensor_bytes"] <= result["resident_budget_gb"] * 1e9


def test_low_token_promotion_uses_loadable_whole_banks(tmp_path):
    from scripts.deepseek_v41_gptq import output_plan

    write_source(tmp_path / "source")
    source = PartialSource(tmp_path / "source", inventory(tmp_path / "source"))
    c = recipe(dict(low_token_bump=True, keep_mtp=False))
    _, base = output_plan(source, c)
    entries, promoted = output_plan(source, c, 2, np.array([3000, 0, 3000, 3000]))
    widths = {
        e.name: e.bits
        for e in entries
        if e.suffix == "weight" and "layers.2.ffn.experts." in e.name
    }
    assert widths["layers.2.ffn.experts.gate_proj.weight"] == 3
    assert widths["layers.2.ffn.experts.up_proj.weight"] == 3
    assert widths["layers.2.ffn.experts.down_proj.weight"] == 4
    assert promoted["resident_tensor_bytes"] - base["resident_tensor_bytes"] == 6144


def test_calibration_total_limit_keeps_windows_independent_and_hashes_content(tmp_path):
    from types import SimpleNamespace

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from scripts.deepseek_v41_gptq import calibration_windows

    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "one.txt").write_text("a b c a b c a b c")
    (tmp_path / "two.txt").write_text("c b a c b a c b a")
    manifest = tmp_path / "calibration.json"
    manifest.write_text(
        json.dumps({"slices": {"one": ["one.txt"], "two": ["two.txt"]}})
    )
    source = SimpleNamespace(source=tmp_path, args=SimpleNamespace(max_seq_len=8))
    windows, identity = calibration_windows(source, manifest, 5, 7)
    assert [len(w) for w in windows] == [5, 2]
    assert windows[1].tolist() == [3, 2]
    assert identity != calibration_windows(source, manifest, 5)[1]
    with pytest.raises(ValueError, match="insufficient windows"):
        calibration_windows(source, manifest, 5, 11)
    with pytest.raises(ValueError, match="token limit"):
        calibration_windows(source, manifest, 0)


def test_unobserved_expert_preserves_rtn_and_sparse_expert_uses_stronger_damping():
    from types import SimpleNamespace

    from scripts.deepseek_v41_gptq import fit

    mx.set_default_device(mx.cpu)
    mx.random.seed(1017)
    weight = mx.random.normal((16, 64)).astype(mx.bfloat16)
    c = recipe()
    guard = SimpleNamespace(check=lambda: None)
    packed, stats = fit(weight, mx.zeros((64, 64)), 0, 2, c, guard)
    baseline = mx.quantize(weight, bits=2, group_size=64)
    for actual, expected in zip(packed, baseline):
        assert np.array_equal(
            np.asarray(actual.view(mx.uint8)), np.asarray(expected.view(mx.uint8))
        )
    assert stats["fallback"] == "unobserved_rtn" and stats["tokens"] == 0
    assert bool(mx.any(packed[0] != 0).item())
    x = mx.random.normal((3, 64))
    _, sparse = fit(weight, x.T @ x, 3, 2, c, guard)
    assert sparse["low_token"]
    assert sparse["damping_coefficient"] == c["low_token_damping"]
    assert sparse["damping_absolute"] > 0


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group,block,cols", [(32, 64, 96), (64, 128, 256)])
def test_batched_gptq_matches_independent_experts(bits, group, block, cols):
    from scripts.deepseek_v41_gptq_quant import gptq_batched

    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(191017)
    weights, hs, factors = [], [], []
    c = QuantConfig(group_size=group, block_size=block)
    for e in range(3):
        x = rng.normal(size=(cols * 2, cols)).astype(np.float32)
        x[:, 1:] += 0.5 * x[:, :1]
        x[:, e] = 0  # Different dead coordinate per expert.
        h = mx.array(x.T @ x)
        w = mx.array(rng.normal(size=(7, cols)).astype(np.float32)).astype(mx.bfloat16)
        hs.append(h)
        weights.append(w)
        factors.append(inverse_factor(h, c, 512 if e else 4096))
    got = gptq_batched(
        mx.stack(weights),
        mx.stack([f[0] for f in factors]),
        bits,
        c,
        dead=mx.stack([f[1] for f in factors]),
    )
    for e in range(3):
        expected, _ = gptq(weights[e], hs[e], bits, c, factor=factors[e])
        for a, b in zip(got, expected):
            assert np.array_equal(
                np.asarray(a[e].view(mx.uint16) if a.dtype == mx.bfloat16 else a[e]),
                np.asarray(b.view(mx.uint16) if b.dtype == mx.bfloat16 else b),
            )


@pytest.mark.skipif(
    os.environ.get("RAPID_MLX_GPTQ_METAL_SMOKE") != "1" or not mx.metal.is_available(),
    reason="requires opt-in Metal smoke",
)
@pytest.mark.parametrize("bits", [2, 3])
def test_batched_gptq_real_expert_shape_metal(bits):
    """Full gate/up shape; independent factors and held-out projection loss."""
    from scripts.deepseek_v41_gptq_quant import gptq_batched, relative_error

    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    mx.set_default_device(mx.gpu)
    mx.random.seed(191017)
    c = QuantConfig()
    weights = mx.random.normal((2, 2304, 5120)).astype(mx.bfloat16)
    # Well-conditioned independent upper factors, no expensive host inversion.
    upper = mx.triu(mx.random.normal((2, 5120, 5120)) * 0.0001) + mx.eye(5120)[None]
    dead = mx.zeros((2, 5120), mx.bool_)
    got = gptq_batched(weights, upper, bits, c, dead=dead)
    x = mx.random.normal((64, 5120))
    h = x.T @ x
    for e in range(2):
        expected, _ = gptq(weights[e], h, bits, c, factor=(upper[e], dead[e], {}))
        a = mx.dequantize(
            *(p[e] for p in got), bits=bits, group_size=64, dtype=mx.float32
        )
        b = mx.dequantize(*expected, bits=bits, group_size=64, dtype=mx.float32)
        mismatch = float(mx.mean((a != b).astype(mx.float32)).item())
        assert mismatch < 1e-5
        ea = relative_error(weights[e], tuple(p[e] for p in got), h, bits)
        eb = relative_error(weights[e], expected, h, bits)
        assert abs(ea - eb) / max(eb, 1e-30) < 1e-4
        if np.array_equal(np.asarray(got[0][e]), np.asarray(expected[0])):
            for a, b in zip(got[1:], expected[1:]):
                assert np.array_equal(
                    np.asarray(a[e].view(mx.uint16)), np.asarray(b.view(mx.uint16))
                )


def test_batched_expert_memory_budget(monkeypatch):
    from types import SimpleNamespace

    import psutil

    from scripts.deepseek_v41_gptq_batched import expert_batch_size

    args = SimpleNamespace(dim=5120, moe_inter_dim=2304, n_routed_experts=256)
    # This sizing test allocates no model and must not depend on runner free RAM.
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=64e9)
    )
    monkeypatch.setenv("RAPID_MLX_GPTQ_FACTOR_WORKERS", "4")
    monkeypatch.setenv("RAPID_MLX_GPTQ_WORKING_GB", "48")
    monkeypatch.setenv("RAPID_MLX_GPTQ_EXPERT_BATCH", "4")
    assert expert_batch_size(args, dict(memory_limit_gb=26)) == 3
    monkeypatch.setenv("RAPID_MLX_GPTQ_EXPERT_BATCH", "0")
    with pytest.raises(ValueError, match="invalid GPTQ"):
        expert_batch_size(args, dict(memory_limit_gb=26))


def test_batched_layer_artifacts_and_heldout_match_scalar(tmp_path, monkeypatch):
    from scripts import deepseek_v41_gptq as b
    from scripts.deepseek_v41_gptq_quality import bind_quality
    from tests.test_deepseek_v41_gptq_quality import fixture

    mx.set_default_device(mx.cpu)
    source, c, _, _, _ = fixture(tmp_path)
    reference = json.loads((tmp_path / "reference" / "reference.json").read_text())
    outputs = []
    for batch in (1, 3):  # Ragged last chunk and unobserved experts included.
        monkeypatch.setenv("RAPID_MLX_GPTQ_EXPERT_BATCH", str(batch))
        root = tmp_path / str(batch)
        parts, scratch = root / "parts", root / "state"
        calibration = [np.array([9, 8, 7, 6, 5, 4, 3, 2], np.int32)]
        with MemoryGuard(c) as guard:
            b.initialize(
                source,
                c,
                calibration,
                scratch,
                parts,
                "fixture",
                guard,
                token_map=list(range(32)),
                heldout_reference=reference,
            )
            bind_quality(
                source,
                parts,
                tmp_path / "reference",
                root / "quality",
                windows=calibration,
            )
            report = b.run_layer(source, c, parts, scratch, 0, guard)
        outputs.append((parts, report))
    (a, ar), (bparts, br) = outputs
    assert ar["projections"] == br["projections"]
    assert (a / "model-layer-0.safetensors").read_bytes() == (
        bparts / "model-layer-0.safetensors"
    ).read_bytes()
    # This includes local/cumulative held-out drift and projection records.
    assert (a / "heldout-drift.jsonl").read_bytes() == (
        bparts / "heldout-drift.jsonl"
    ).read_bytes()
    ca = json.loads((a / "layer-0-complete.json").read_text())
    cb = json.loads((bparts / "layer-0-complete.json").read_text())
    assert (
        ca["files"]["model-layer-0.safetensors"]
        == cb["files"]["model-layer-0.safetensors"]
    )
