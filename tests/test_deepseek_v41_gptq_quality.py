"""Held-out boundary metrics, local/prefix separation and hybrid logits."""

import json
import shutil
from types import SimpleNamespace

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


from mlx.utils import tree_flatten

from scripts import deepseek_v41_gptq as b
from scripts.deepseek_v41_gptq_io import (
    PartialSource,
    atomic_json,
    digest,
    inventory,
    load_state,
    save_state,
)
from scripts.deepseek_v41_gptq_quality import (
    NativeBuild,
    append_row,
    array_metrics,
    boundary_metrics,
    drift,
    finish_metrics,
    milestone,
    parse_layers,
    reference_init,
    reference_run,
    residual_mmap,
)
from scripts.deepseek_v41_gptq_quality import compare_milestone as compare
from tests.test_deepseek_v41_streaming_teacher import write_source


def fixture(root, *, varied_engram=False):
    args, raw = write_source(root / "source")
    if varied_engram:
        for name, value in raw.items():
            if ".engram.embed." in name and name.endswith(".weight"):
                raw[name] = mx.random.randint(32, 96, shape=value.shape).astype(
                    mx.uint8
                )
        mx.save_safetensors(str(root / "source" / "source.safetensors"), raw)
    source = PartialSource(root / "source", inventory(root / "source"))
    c = b.recipe(dict(prefill_chunk=4, ffn_chunk=8, keep_mtp=False))
    native = root / "original-native"
    native.mkdir()
    mapping = {}
    for layer in range(args.n_layers):
        block = b.bounded_block(source, layer, c, original=True)
        weights = {
            f"layers.{layer}." + n: v for n, v in tree_flatten(block.parameters())
        }
        for projection, final in (
            ("w1", "gate_proj"),
            ("w3", "up_proj"),
            ("w2", "down_proj"),
        ):
            weights[f"layers.{layer}.ffn.experts.{final}.weight"] = mx.stack(
                [
                    source.tensor(f"layers.{layer}.ffn.experts.{e}.{projection}.weight")
                    for e in range(args.n_routed_experts)
                ]
            )
        # The reference fixture's constant source Engram rows are exactly
        # representable at eight bits; use the supported native SSD format.
        if block.engram is not None:
            from scripts.deepseek_v41_streaming_teacher import SourceEngram

            embed = SourceEngram(source.index, layer)
            count = source.index.tensors[f"layers.{layer}.engram.embed.weight"].shape[0]
            rows = embed(mx.arange(count))
            for suffix, value in zip(
                b.SUFFIXES, mx.quantize(rows, bits=8, group_size=32)
            ):
                weights[f"layers.{layer}.engram.embed.{suffix}"] = (
                    value if suffix == "weight" else value.astype(mx.bfloat16)
                )
        filename = f"model-layer-{layer}.safetensors"
        mx.save_safetensors(str(native / filename), weights)
        mapping.update({name: filename for name in weights})
    cfg = {
        **source.config,
        "quantization": dict(
            bits=8, group_size=64, modules={}, engram_bits=8, engram_group_size=32
        ),
    }
    cfg.pop("quantization_config")
    atomic_json(native / "config.json", cfg)
    atomic_json(native / "model.safetensors.index.json", dict(weight_map=mapping))
    ids = [np.array([1, 2, 5, 8, 13, 3, 4], np.int32), np.array([7, 4, 5], np.int32)]
    names = ["heldout_a", "heldout_b"]
    provenance = dict(
        tokenizer_sha256="fixture-tokenizer",
        slices={n: dict(corpus_sha256=[n]) for n in names},
    )
    with b.MemoryGuard(c) as guard:
        reference_init(
            source,
            root / "reference",
            ids,
            names,
            provenance,
            c,
            guard,
            token_map=list(range(32)),
        )
        reference_run(source, root / "reference", 3, c, guard)
    return source, c, native, ids, names


def test_original_native_layers_have_exact_zero_drift_and_milestone_kl(tmp_path):
    source, c, native_path, _, names = fixture(tmp_path)
    native = NativeBuild(native_path, source)
    with b.MemoryGuard(c) as guard:
        rows = drift(
            source,
            tmp_path / "reference",
            native,
            [0, 1, 2, 3],
            tmp_path / "quality",
            tmp_path / "metrics.jsonl",
            c,
            guard,
        )
        for row in rows:
            for key in ("local", "cumulative"):
                assert row["candidate"][key]["relative_frobenius_error"] == 0
                assert row["candidate"][key]["mean_token_cosine_similarity"] == 1
                assert set(row["candidate"][key]["slices"]) == set(names)
        zero = milestone(
            source,
            tmp_path / "reference",
            native,
            -1,
            tmp_path / "milestone-zero",
            tmp_path / "zero-logits",
            c,
            guard,
        )
        full = milestone(
            source,
            tmp_path / "reference",
            native,
            3,
            tmp_path / "milestone-full",
            tmp_path / "full-logits",
            c,
            guard,
        )
    assert zero["mean_domain_kl_nats"] == 0 and zero["worst_domain_kl_nats"] == 0
    assert full["mean_domain_kl_nats"] == 0
    full_compare = compare(tmp_path / "reference" / "logits", tmp_path / "full-logits")
    assert full["mean_domain_kl_nats"] == full_compare["mean_domain_kl_nats"]
    h = residual_mmap(tmp_path / "reference" / "boundary-1" / "window-0")
    assert isinstance(h, np.memmap) and h.dtype == np.uint16


def test_injected_hidden_noise_is_positive_and_monotone(tmp_path):
    _, _, _, _, _ = fixture(tmp_path)
    state = load_state(tmp_path / "reference" / "boundary-1" / "window-0")
    r = np.asarray(state["h"].astype(mx.float32)).reshape(state["h"].shape[1], -1)
    rng = np.random.default_rng(1017)
    noise = rng.normal(size=r.shape)
    # Orthogonal noise makes both norm error and cosine distance monotone.
    noise -= (
        (noise * r).sum(axis=1, keepdims=True) / (r * r).sum(axis=1, keepdims=True) * r
    )
    error, cosine = [], []
    for strength in (0.0, 0.01, 0.1, 1.0):
        values = finish_metrics(array_metrics(r, r + strength * noise))
        error.append(values["relative_frobenius_error"])
        cosine.append(values["mean_token_cosine_similarity"])
    assert error[0] == 0 and all(a < b for a, b in zip(error, error[1:]))
    assert all(a > b for a, b in zip(cosine, cosine[1:]))


def test_metrics_zero_norm_and_invalid_states():
    zeros = np.zeros((3, 8))
    assert (
        finish_metrics(array_metrics(zeros, zeros))["mean_token_cosine_similarity"] == 1
    )
    with pytest.raises(ValueError, match="zero reference"):
        finish_metrics(array_metrics(zeros, np.ones_like(zeros)))
    with pytest.raises(ValueError, match="nonfinite"):
        array_metrics(zeros, zeros + np.nan)
    assert parse_layers("2-4") == [2, 3, 4]
    for bad in ("3-2", "-1", "../0", "a"):
        with pytest.raises(ValueError):
            parse_layers(bad)


def test_partial_jsonl_tail_recovery_and_deduplication(tmp_path):
    path = tmp_path / "metrics.jsonl"
    row = dict(
        reference_identity="r",
        candidate_identity="c",
        baseline_identity=None,
        prefix_start=0,
        layer=0,
    )
    append_row(path, row)
    with path.open("ab") as f:
        f.write(b'{"partial":')
    append_row(path, row)
    assert len(path.read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="row changed"):
        append_row(path, {**row, "changed": True})


def test_slice_worst_metrics_and_bf16_mmap(tmp_path):
    for root in ("reference", "candidate"):
        (tmp_path / root).mkdir()
        for i, value in enumerate((1.0, 2.0)):
            h = mx.full((1, 3, 4, 8), value, mx.bfloat16)
            if root == "candidate" and i == 1:
                h *= 2
            save_state(tmp_path / root / f"window-{i}", dict(h=h))
    metrics = boundary_metrics(
        tmp_path / "reference", tmp_path / "candidate", ["a", "b"]
    )
    assert metrics["worst_slice_error_name"] == "b"
    assert metrics["slices"]["a"]["relative_frobenius_error"] == 0
    assert metrics["slices"]["b"]["relative_frobenius_error"] == 1


def test_streaming_build_tracks_heldout_and_full_milestone_matches_native(tmp_path):
    from rapid_mlx.models.deepseek_v41_native.load import load
    from scripts.deepseek_v41_gptq_quality import bind_quality
    from scripts.evaluate_deepseek_v41_rd import token_metrics

    source, c, _, heldout, _ = fixture(tmp_path, varied_engram=True)
    parts, scratch = tmp_path / "parts", tmp_path / "calibration"
    reference = json.loads((tmp_path / "reference" / "reference.json").read_text())
    calibration = [np.array([9, 8, 7, 6, 5, 4, 3, 2], np.int32)]
    with b.MemoryGuard(c) as guard:
        b.initialize(
            source,
            c,
            calibration,
            scratch,
            parts,
            "fixture-calibration",
            guard,
            token_map=list(range(32)),
            heldout_reference=reference,
        )
        bind_quality(
            source,
            parts,
            tmp_path / "reference",
            tmp_path / "quality-build",
            windows=calibration,
        )
        for layer in range(4):
            stats = b.run_layer(source, c, parts, scratch, layer, guard)
            assert all(
                p["heldout"]["status"] in ("measured", "unobserved")
                for p in stats["projections"].values()
            )
            assert all("calibration" in p for p in stats["projections"].values())
        b.assemble(source, c, parts, tmp_path / "build", guard)
        result = milestone(
            source,
            tmp_path / "reference",
            NativeBuild(tmp_path / "build", source),
            3,
            tmp_path / "milestone",
            tmp_path / "candidate-logits",
            c,
            guard,
        )
        hybrid = milestone(
            source,
            tmp_path / "reference",
            NativeBuild(parts, source),
            1,
            tmp_path / "hybrid",
            tmp_path / "hybrid-logits",
            c,
            guard,
        )
    assert result["mean_domain_kl_nats"] > 0
    assert hybrid["mean_domain_kl_nats"] > 0
    model, _ = load(str(tmp_path / "build"), lazy=True, engram_ssd_offload=True)
    b.install_cpu_affine(model)
    means = []
    for i, ids in enumerate(heldout):
        cache = model.make_cache(max_seq_len=len(ids))
        outputs = [
            model(mx.array(ids[None, j : j + 4]), cache) for j in range(0, len(ids), 4)
        ]
        logits = np.asarray(mx.concatenate(outputs, axis=1))[0]
        name = reference["names"][i]
        original = np.load(tmp_path / "reference" / "logits" / f"{name}-logits.npy")
        streamed = np.load(tmp_path / "candidate-logits" / f"{name}-logits.npy")
        assert np.allclose(logits, streamed, rtol=1e-5, atol=1e-5)
        kl, _ = token_metrics(original, logits)
        means.append(float(kl.mean()))
    assert result["mean_domain_kl_nats"] == pytest.approx(
        np.mean(means), rel=1e-5, abs=1e-8
    )
    rows = [
        json.loads(line)
        for line in (parts / "heldout-drift.jsonl").read_text().splitlines()
    ]
    assert [row["layer"] for row in rows] == [0, 1, 2, 3]
    assert all(
        row["candidate"]["local"]["relative_frobenius_error"] > 0 for row in rows
    )
    with pytest.raises(ValueError, match="appears in calibration"):
        bind_quality(
            source, parts, tmp_path / "reference", tmp_path / "bad", windows=heldout
        )


def test_arbitrary_block_works_with_only_requested_native_layers(tmp_path):
    source, c, native_path, _, _ = fixture(tmp_path)
    for layer in (0, 1):
        (native_path / f"model-layer-{layer}.safetensors").unlink()
    with b.MemoryGuard(c) as guard:
        rows = drift(
            source,
            tmp_path / "reference",
            NativeBuild(native_path, source),
            [2, 3],
            tmp_path / "block",
            tmp_path / "block.jsonl",
            c,
            guard,
            start=2,
        )
    assert all(
        row["cumulative_scope"] == "block" and row["prefix_start"] == 2 for row in rows
    )
    assert all(
        row["candidate"]["cumulative"]["relative_frobenius_error"] == 0 for row in rows
    )


def test_heldout_tracking_replays_after_crash_before_calibration_commit(
    tmp_path, monkeypatch
):
    from scripts.deepseek_v41_gptq_quality import bind_quality

    source, c, _, _, _ = fixture(tmp_path)
    reference = json.loads((tmp_path / "reference" / "reference.json").read_text())
    calibration = [np.array([9, 8, 7, 6, 5, 4, 3, 2], np.int32)]
    with b.MemoryGuard(c) as guard:
        for label in ("normal", "restart"):
            root = tmp_path / label
            root.mkdir()
            b.initialize(
                source,
                c,
                calibration,
                root / "state",
                root / "parts",
                "same-calibration",
                guard,
                token_map=list(range(32)),
                heldout_reference=reference,
            )
            bind_quality(
                source,
                root / "parts",
                tmp_path / "reference",
                root / "quality",
                windows=calibration,
            )
            original = b.atomic_json
            fired = False

            def interrupted(path, value, label=label, root=root, original=original):
                nonlocal fired
                if (
                    label == "restart"
                    and not fired
                    and path == root / "state" / "checkpoint.json"
                    and value.get("next_layer") == 1
                ):
                    fired = True
                    raise KeyboardInterrupt(
                        "after held-out row, before calibration boundary commit"
                    )
                return original(path, value)

            with monkeypatch.context() as m:
                m.setattr(b, "atomic_json", interrupted)
                if label == "restart":
                    with pytest.raises(KeyboardInterrupt):
                        b.run_layer(source, c, root / "parts", root / "state", 0, guard)
                else:
                    b.run_layer(source, c, root / "parts", root / "state", 0, guard)
            if label == "restart":
                b.run_layer(source, c, root / "parts", root / "state", 0, guard)
            # Export/delete layer 0's native payload before continuing layer 1.
            (root / "parts" / "model-layer-0.safetensors").rename(
                root / "exported-layer-0.safetensors"
            )
            b.run_layer(source, c, root / "parts", root / "state", 1, guard)
    for name in (
        "heldout-drift.jsonl",
        "layer-0-stats.json",
        "layer-1-stats.json",
        "model-layer-1.safetensors",
    ):
        assert (tmp_path / "normal" / "parts" / name).read_bytes() == (
            tmp_path / "restart" / "parts" / name
        ).read_bytes()


def test_layer_drift_increases_with_injected_down_projection_noise(tmp_path):
    source, c, original, _, _ = fixture(tmp_path)
    errors = []
    with b.MemoryGuard(c) as guard:
        for i, strength in enumerate((0.0, 0.001, 0.01, 0.1)):
            native = tmp_path / f"noise-{i}"
            native.mkdir()
            for name in ("config.json", "model.safetensors.index.json"):
                shutil.copyfile(original / name, native / name)
            values = mx.load(str(original / "model-layer-0.safetensors"))
            name = "layers.0.ffn.experts.down_proj.weight"
            values[name] = values[name].astype(mx.float32) + strength
            mx.save_safetensors(str(native / "model-layer-0.safetensors"), values)
            rows = drift(
                source,
                tmp_path / "reference",
                NativeBuild(native, source),
                [0],
                tmp_path / f"noise-state-{i}",
                tmp_path / f"noise-{i}.jsonl",
                c,
                guard,
            )
            errors.append(rows[0]["candidate"]["local"]["relative_frobenius_error"])
    assert errors[0] == 0 and all(a < b for a, b in zip(errors, errors[1:]))


def test_reference_binding_rejects_tokenizer_and_corpus_leakage(tmp_path):
    from scripts.deepseek_v41_gptq_quality import bind_quality
    from scripts.deepseek_v41_rd_common import file_hash

    source = SimpleNamespace(source=tmp_path / "source", manifest={"source": "fixture"})
    source.source.mkdir()
    tokenizer = source.source / "tokenizer.json"
    tokenizer.write_text("fixture tokenizer")
    parts, reference_root = tmp_path / "parts", tmp_path / "reference"
    parts.mkdir()
    reference_root.mkdir()
    corpus = tmp_path / "text.txt"
    corpus.write_text("same held-out and calibration corpus")
    reference = dict(
        identity="ref",
        source_identity=digest(source.manifest),
        provenance=dict(
            tokenizer_sha256="wrong",
            slices={"heldout": dict(corpus_sha256=[file_hash(corpus)])},
        ),
    )
    atomic_json(reference_root / "reference.json", reference)
    atomic_json(parts / "run.json", dict(heldout_reference_identity="ref"))
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        bind_quality(source, parts, reference_root, tmp_path / "quality")
    reference["provenance"]["tokenizer_sha256"] = file_hash(tokenizer)
    atomic_json(reference_root / "reference.json", reference)
    manifest = tmp_path / "calibration.json"
    atomic_json(manifest, dict(slices={"calibration": [str(corpus)]}))
    with pytest.raises(ValueError, match="corpus files overlap"):
        bind_quality(
            source, parts, reference_root, tmp_path / "quality", slices=manifest
        )
    assert not (parts / "quality-config.json").exists()


@pytest.mark.parametrize("overlap", ["corpus", "tokens"])
@pytest.mark.parametrize("has_evidence", [False, True])
def test_cli_separation_precedes_commit_and_is_rechecked_on_resume(
    tmp_path, monkeypatch, overlap, has_evidence
):
    import sys

    from scripts.deepseek_v41_rd_common import file_hash

    source = SimpleNamespace(source=tmp_path / "source", manifest={"fixture": 1})
    source.source.mkdir()
    corpus = tmp_path / "calibration.txt"
    corpus.write_text("calibration corpus")
    slices = tmp_path / "slices.json"
    atomic_json(slices, dict(slices={"calibration": [str(corpus)]}))
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference = dict(
        identity="ref",
        source_identity=digest(source.manifest),
        token_hashes=["heldout"],
        names=["heldout"],
        provenance=dict(
            tokenizer_sha256="unused",
            slices={
                "heldout": dict(
                    corpus_sha256=[
                        file_hash(corpus) if overlap == "corpus" else "different-corpus"
                    ]
                )
            },
        ),
    )
    atomic_json(reference_root / "reference.json", reference)
    (reference_root / "boundary-0").mkdir()
    save_state(reference_root / "boundary-0" / "window-0", dict(ids=mx.array([1, 2])))
    windows = [np.array([0, 1, 2, 3] if overlap == "tokens" else [4, 5, 6], np.int32)]
    monkeypatch.setattr(b, "PartialSource", lambda *args: source)
    monkeypatch.setattr(
        b, "calibration_windows", lambda *args: (windows, "calibration")
    )
    monkeypatch.setattr(b, "run_layer", lambda *args: pytest.fail("layer work started"))
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, source.manifest)
    parts, scratch = tmp_path / "parts", tmp_path / "state"
    common = [
        "--source",
        str(source.source),
        "--manifest",
        str(manifest),
        "--parts",
        str(parts),
        "--scratch",
        str(scratch),
        "--cpu",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gptq",
            "init",
            *common,
            "--slices",
            str(slices),
            "--tokens",
            "4",
            "--reference",
            str(reference_root),
        ],
    )
    error = "corpus files overlap" if overlap == "corpus" else "token window appears"
    with pytest.raises(ValueError, match=error):
        b.main()
    assert not parts.exists()
    assert not scratch.exists()

    # Reproduce state left by the old initializer, including an optional prior binding.
    parts.mkdir()
    scratch.mkdir()
    run = dict(
        identity="run", lengths=[len(windows[0])], heldout_reference_identity="ref"
    )
    if has_evidence:
        run["calibration_corpus_hashes"] = [file_hash(corpus)]
        run["calibration_token_windows"] = [windows[0].tolist()]
    atomic_json(parts / "run.json", run)
    atomic_json(scratch / "checkpoint.json", dict(identity="run", next_layer=0))
    atomic_json(
        parts / "quality-config.json",
        dict(
            reference=str(reference_root),
            scratch=str(tmp_path / "quality"),
            reference_identity="ref",
            baseline=None,
        ),
    )
    before = (scratch / "checkpoint.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["gptq", "run", *common])
    with pytest.raises(
        ValueError, match=error if has_evidence else "missing calibration separation"
    ):
        b.main()
    assert (scratch / "checkpoint.json").read_bytes() == before


def test_quantized_projection_backfill_and_residual_cast(tmp_path, monkeypatch):
    from scripts import deepseek_v41_gptq_quality as q

    source, c, native_path, _, _ = fixture(tmp_path)
    reference_root = tmp_path / "reference"
    reference = json.loads((reference_root / "reference.json").read_text())
    parts, scratch = tmp_path / "parts", tmp_path / "state"
    with b.MemoryGuard(c) as guard:
        b.initialize(
            source,
            c,
            [np.array([9, 8, 7, 6], np.int32)],
            scratch,
            parts,
            "cal",
            guard,
            token_map=list(range(32)),
        )
        (scratch / "working-0").mkdir()
        (scratch / "boundary-1").mkdir()
        b.run_layer(source, c, parts, scratch, 0, guard)
        native = q.NativeBuild(parts, source)
        target, work = tmp_path / "cast-boundary", tmp_path / "cast-work"
        target.mkdir()
        work.mkdir()
        real = q.load_state

        def float_boundary(path):
            state = real(path)
            if path.parent == target:
                state["h"] = state["h"].astype(mx.float32)
            return state

        monkeypatch.setattr(q, "load_state", float_boundary)
        reports = q.apply_layer(
            source,
            0,
            q.windows_at(reference_root / "boundary-0", len(reference["names"])),
            target,
            work,
            c,
            guard,
            native,
            True,
        )
        assert (
            reports["expert.0.w1"]["error_scope"] == "heldout_quantized_prefix_inputs"
        )
        assert real(target / "window-0")["h"].dtype == mx.bfloat16
        rows = q.drift(
            source,
            reference_root,
            native,
            [0],
            tmp_path / "drift",
            tmp_path / "metrics",
            c,
            guard,
            update_parts_stats=True,
        )
        assert len(rows) == 1
        stats = json.loads((parts / "layer-0-stats.json").read_text())
        assert stats["heldout_reference_identity"] == reference["identity"]
        monkeypatch.setattr(native, "verify", lambda _: "changed")
        with pytest.raises(ValueError, match="changed after prefix"):
            q.prefix_boundary(
                source,
                reference_root,
                reference,
                native,
                0,
                tmp_path / "drift",
                c,
                guard,
            )
        with pytest.raises(ValueError, match="checkpoint missing"):
            q.prefix_boundary(
                source,
                tmp_path / "missing",
                reference,
                native,
                0,
                tmp_path / "empty-prefix",
                c,
                guard,
            )
    raw_native = q.NativeBuild(native_path, source)
    real_tensor = raw_native.tensor
    monkeypatch.setattr(
        raw_native,
        "tensor",
        lambda name, expert=None: (
            real_tensor(name, expert)[:2]
            if name == "layers.0.ffn.gate.weight"
            else real_tensor(name, expert)
        ),
    )
    assert raw_native.block(0, c).ffn.gate.weight.shape[0] == 2
    real_has = raw_native.has
    monkeypatch.setattr(
        raw_native,
        "has",
        lambda name: (
            False if name == "layers.1.engram.embed.scales" else real_has(name)
        ),
    )
    with pytest.raises(ValueError, match="SSD affine"):
        raw_native.block(1, c)
