"""Corrupt caches and invalid conversion state must fail before publication."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from scripts import deepseek_v41_gptq as b
from scripts import deepseek_v41_gptq_quality as q
from scripts.deepseek_v41_gptq_io import PartialSource, atomic_json, inventory
from scripts.deepseek_v41_rd_common import file_hash
from tests.test_deepseek_v41_streaming_teacher import write_source


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


@pytest.fixture
def source(tmp_path):
    write_source(tmp_path / "source")
    return PartialSource(tmp_path / "source", inventory(tmp_path / "source"))


@pytest.mark.parametrize(
    "kind", ["provenance", "name", "corpus", "payload", "shape", "tokens"]
)
def test_milestone_rejects_corrupt_cache(tmp_path, kind):
    roots = [tmp_path / "reference", tmp_path / "candidate"]
    name = "../unsafe" if kind == "name" else "one"
    for index, root in enumerate(roots):
        root.mkdir()
        np.save(
            root / "one-tokens.npy",
            np.array([1, 2 + (index if kind == "tokens" else 0)]),
        )
        np.save(root / "one-logits.npy", np.zeros((2, 4)))
        entry = dict(
            tokens=2,
            corpus_sha256=["corpus"],
            tokens_sha256=file_hash(root / "one-tokens.npy"),
            logits_sha256=file_hash(root / "one-logits.npy"),
        )
        meta = dict(
            cache_kind="gptq_partial_prefix",
            mode="unpruned_reference",
            source_identity="source",
            quality_reference_identity="reference",
            tokenizer_sha256="tokenizer",
            slices={name: entry},
        )
        if index:
            if kind == "provenance":
                meta["source_identity"] = "other"
            if kind == "corpus":
                entry["corpus_sha256"] = ["other"]
            if kind == "payload":
                entry["tokens_sha256"] = "bad"
            if kind == "shape":
                entry["tokens"] = 3
        atomic_json(root / "metadata.json", meta)
    with pytest.raises(ValueError):
        q.compare_milestone(*roots)


def test_reference_and_boundary_input_rejections(tmp_path, source):
    c = b.recipe()
    guard = SimpleNamespace(check=lambda: None)
    for windows, names in [
        ([], []),
        ([np.array([1])], ["../bad"]),
        ([np.array([-1])], ["good"]),
    ]:
        with pytest.raises(ValueError):
            q.reference_init(source, tmp_path / "ref", windows, names, {}, c, guard)
    with pytest.raises(FileExistsError):
        q.reference_init(source, tmp_path, [np.array([1])], ["one"], {}, c, guard)
    with pytest.raises(ValueError, match="shapes"):
        q.array_metrics(np.zeros((1, 2)), np.zeros((2, 2)))
    atomic_json(tmp_path / "complete.json", dict(identity="bad", files={}))
    with pytest.raises(ValueError, match="identity"):
        q.verify_boundary(tmp_path, "expected")
    with pytest.raises(ValueError, match="identity"):
        q.validate_reference(source, {"source_identity": "bad"})
    with pytest.raises(FileExistsError):
        q.logit_cache(source, tmp_path, tmp_path, tmp_path, {}, guard, mode="reference")
    with pytest.raises(FileExistsError):
        q.milestone(source, tmp_path, None, 0, tmp_path, tmp_path, c, guard)
    atomic_json(tmp_path / "reference.json", {"source_identity": "bad"})
    with pytest.raises(ValueError, match="source"):
        q.milestone(source, tmp_path, None, 0, tmp_path, tmp_path / "new", c, guard)
    atomic_json(
        tmp_path / "reference.json", {"source_identity": b.digest(source.manifest)}
    )
    with pytest.raises(ValueError, match="last"):
        q.milestone(source, tmp_path, None, 99, tmp_path, tmp_path / "new", c, guard)


def test_backfill_preserves_calibration_and_authenticates_stats(tmp_path):
    native = SimpleNamespace(root=tmp_path, run=None)
    with pytest.raises(ValueError, match="only GPTQ parts"):
        q.backfill_stats(native, 0, {}, {})
    native.run = {}
    path = tmp_path / "layer-0-stats.json"
    atomic_json(path, dict(projections={"w": {"gptq_relative_output_error": 0.1}}))
    with pytest.raises(ValueError, match="incomplete"):
        q.backfill_stats(native, 0, {}, {})
    atomic_json(
        tmp_path / "layer-0-complete.json", dict(files={path.name: file_hash(path)})
    )
    q.backfill_stats(
        native, 0, {"w": {"relative_output_error": 0.2}}, {"identity": "reference"}
    )
    stats = json.loads(path.read_text())
    assert stats["projections"]["w"]["calibration"]["gptq_relative_output_error"] == 0.1
    assert stats["projections"]["w"]["heldout"]["relative_output_error"] == 0.2
    assert json.loads((tmp_path / "layer-0-complete.json").read_text())["files"][
        path.name
    ] == file_hash(path)


def test_planning_rejects_invalid_or_duplicate_output(source, monkeypatch):
    with pytest.raises(ValueError, match="out-of-range"):
        b.output_plan(source, b.recipe({"dense_8bit_layers": [99]}))
    with pytest.raises(ValueError, match="final recipe"):
        b.output_plan(source, b.recipe({"pilot": False}))
    entries, _ = b.output_plan(source, b.recipe({"group_size": 32}))
    assert all(e.group_size == 32 for e in entries if e.component == "routed_experts")
    real = b.plan

    def duplicate(*args):
        es, allocation, report = real(*args)
        return [*es, next(e for e in es if e.suffix == "weight")], allocation, report

    monkeypatch.setattr(b, "plan", duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        b.output_plan(source, b.recipe())


@pytest.mark.parametrize("kind", ["windows", "memory", "resident", "existing"])
def test_initialize_rejects_invalid_plan(tmp_path, source, monkeypatch, kind):
    c = b.recipe()
    windows = [] if kind == "windows" else [np.array([1, 2])]
    monkeypatch.setattr(
        b,
        "estimates",
        lambda *a: dict(
            estimated_peak_bytes=1e20 if kind == "memory" else 0,
            resident_tensor_bytes=1e20 if kind == "resident" else 0,
        ),
    )
    parts = tmp_path if kind == "existing" else tmp_path / "parts"
    with pytest.raises((ValueError, MemoryError, FileExistsError)):
        b.initialize(source, c, windows, tmp_path / "scratch", parts, "cal", None)


def test_empty_experts_and_bias_calls():
    with pytest.raises(RuntimeError, match="streamed"):
        b.EmptyExperts()(mx.ones((1,)))
    import mlx.nn as nn

    linear = nn.Linear(32, 2, bias=True)
    x = mx.ones((1, 32))
    assert np.allclose(np.asarray(b.cpu_dense_call(linear, x)), np.asarray(linear(x)))
    quant = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4)
    assert np.allclose(
        np.asarray(b.cpu_affine_call(quant, x)), np.asarray(quant(x)), atol=1e-5
    )


def test_memory_watchdog_exits_on_native_operation_overrun(monkeypatch):
    guard = b.MemoryGuard(b.recipe())

    class ExitError(Exception):
        pass

    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise MemoryError("limit")

    monkeypatch.setattr(guard, "check", check)
    monkeypatch.setattr(guard.event, "wait", lambda _: False)
    monkeypatch.setattr(
        b.threading, "Thread", lambda target, **kw: SimpleNamespace(start=target)
    )

    def exit_process(code):
        assert code == 86
        raise ExitError

    monkeypatch.setattr(b.os, "_exit", exit_process)
    try:
        with pytest.raises(ExitError):
            guard.__enter__()
    finally:
        guard.restore_limits()


def test_cpu_install_ignores_non_cpu_device(monkeypatch):
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert b.install_cpu_affine(object()) is None


def test_spill_rejects_truncation_bad_ids_and_short_write(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from scripts.deepseek_v41_gptq_batched import RoutedSpill

    path = tmp_path / "data"
    path.write_bytes(b"1234")
    real_open = type(path).open

    @contextmanager
    def truncated(p, *args, **kwargs):
        with real_open(p, *args, **kwargs) as f:
            yield SimpleNamespace(fileno=f.fileno, readinto=lambda _: 0)

    with monkeypatch.context() as patch:
        patch.setattr(type(path), "open", truncated)
        with pytest.raises(EOFError):
            RoutedSpill.read(path, np.uint8, (4,), lambda: None)
    ids, gates = tmp_path / "ids", tmp_path / "gates"
    np.array([-1], np.int32).tofile(ids)
    np.array([1], np.float32).tofile(gates)
    with pytest.raises(ValueError, match="expert ID"):
        RoutedSpill(
            {"ids": ids, "gates": gates},
            1,
            SimpleNamespace(n_activated_experts=1, n_routed_experts=2),
            {"ffn_chunk": 1},
            SimpleNamespace(check=lambda: None),
        )
    spill = object.__new__(RoutedSpill)
    spill.paths = {"y": path}
    spill.output = np.ones((1,), np.float32)

    @contextmanager
    def short_write(*args, **kwargs):
        yield SimpleNamespace(write=lambda _: 0)

    monkeypatch.setattr(type(path), "open", short_write)
    with pytest.raises(OSError, match="short"):
        spill.write_output(SimpleNamespace(check=lambda: None))


def test_calibration_windows_limits_and_short_corpus(tmp_path, source, monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        SimpleNamespace(
            Tokenizer=SimpleNamespace(
                from_file=lambda _: SimpleNamespace(
                    encode=lambda _: SimpleNamespace(ids=[1, 2])
                )
            )
        ),
    )
    text = tmp_path / "text"
    text.write_text("local")
    manifest = tmp_path / "slices.json"
    atomic_json(manifest, {"slices": {"a": ["text"], "b": ["text"]}})
    windows, _ = b.calibration_windows(source, manifest, 2, 2)
    assert len(windows) == 1
    with pytest.raises(ValueError, match="only"):
        b.calibration_windows(source, manifest, 3)


def test_missing_token_map_is_loaded_for_initialization(tmp_path, source, monkeypatch):
    from rapid_mlx.models.deepseek_v41_native import load as loader

    monkeypatch.setattr(loader, "load_token_map", lambda *a, **kw: list(range(32)))
    c = b.recipe({"prefill_chunk": 2, "ffn_chunk": 2})
    with b.MemoryGuard(c) as guard:
        b.initialize(
            source,
            c,
            [np.array([1, 2])],
            tmp_path / "state",
            tmp_path / "parts",
            "cal",
            guard,
        )
        q.reference_init(
            source,
            tmp_path / "reference",
            [np.array([1, 2])],
            ["one"],
            {"slices": {"one": {"corpus_sha256": []}}, "tokenizer_sha256": "tokenizer"},
            c,
            guard,
        )
    assert (tmp_path / "state/checkpoint.json").exists()
    assert (tmp_path / "reference/reference.json").exists()


def test_select_recipe_rejects_incomplete_and_mixed_pilots(tmp_path):
    c = b.recipe()
    source = SimpleNamespace(args=SimpleNamespace(n_layers=8))
    for i in range(8):
        atomic_json(
            tmp_path / f"layer-{i}-stats.json",
            {
                "layer": i,
                "block_output_4bit_relative_error": None,
                "run_identity": "wrong",
            },
        )
    with pytest.raises(ValueError, match="complete pilot"):
        b.select_recipe(tmp_path, source, c)
    for i in range(8):
        atomic_json(
            tmp_path / f"layer-{i}-stats.json",
            {
                "layer": i,
                "block_output_4bit_relative_error": 0,
                "run_identity": "wrong",
            },
        )
    atomic_json(
        tmp_path / "run.json", {"identity": b.digest(dict(recipe=c)), "recipe": c}
    )
    with pytest.raises(ValueError, match="mixed"):
        b.select_recipe(tmp_path, source, c)


def test_quality_identity_rejections(tmp_path, source):
    c = b.recipe()
    reference = {
        "identity": "ref",
        "source_identity": b.digest(source.manifest),
        "prefill_chunk": 2,
        "provenance": {"slices": {}},
    }
    ref = tmp_path / "reference"
    ref.mkdir()
    atomic_json(ref / "reference.json", {**reference, "source_identity": "bad"})
    for operation in (
        lambda: q.reference_run(source, ref, 0, c, None),
        lambda: q.drift(
            source, ref, None, [0], tmp_path, tmp_path / "metrics", c, None
        ),
    ):
        with pytest.raises(ValueError, match="source"):
            operation()
    atomic_json(ref / "reference.json", reference)
    atomic_json(ref / "checkpoint.json", {"identity": "bad", "next_layer": 0})
    with pytest.raises(ValueError, match="checkpoint"):
        q.reference_run(source, ref, 0, c, None)
    atomic_json(ref / "checkpoint.json", {"identity": "ref", "next_layer": 0})
    with pytest.raises(ValueError, match="layer"):
        q.reference_run(source, ref, 99, c, None)
    with pytest.raises(ValueError, match="layer"):
        q.drift(source, ref, None, [99], tmp_path, tmp_path / "metrics", c, None)
    parts = tmp_path / "parts"
    parts.mkdir()
    atomic_json(parts / "run.json", {"heldout_reference_identity": "bad"})
    with pytest.raises(ValueError, match="reference changed"):
        q.bind_quality(source, parts, ref, tmp_path / "scratch")
    atomic_json(parts / "run.json", {})
    atomic_json(
        parts / "quality-config.json",
        {
            "reference_identity": "bad",
            "reference": str(ref),
            "scratch": str(tmp_path / "scratch"),
        },
    )
    with pytest.raises(ValueError, match="cache identity"):
        q.bind_quality(source, parts, ref, tmp_path / "scratch")
    with pytest.raises(ValueError, match="reference identity"):
        q.prepare_for_build(source, parts, 0, None, c, None)
    atomic_json(parts / "run.json", {"source_identity": "bad"})
    with pytest.raises(ValueError, match="parts/source"):
        q.NativeBuild(parts, source)


def test_native_tensor_and_boundary_rejections(tmp_path, monkeypatch):
    from scripts.deepseek_v41_gptq_io import save_state

    p = tmp_path / "tensor.safetensors"
    mx.save_safetensors(str(p), {"weight": mx.ones((2, 2))})
    with pytest.raises(ValueError, match="outside"):
        q.TensorFile(p).array("weight", 2)
    save_state(tmp_path / "window-0", {"h": mx.ones((1, 2, 1, 2), mx.float32)})
    with pytest.raises(ValueError, match="BF16"):
        q.residual_mmap(tmp_path / "window-0")
    monkeypatch.setattr(q, "residual_mmap", lambda _: np.zeros((1, 2), np.uint16))
    with pytest.raises(ValueError, match="matching"):
        q.boundary_metrics(tmp_path, tmp_path, ["one"])
    native = object.__new__(q.NativeBuild)
    native.mapping = {"weight": "../unsafe"}
    with pytest.raises(ValueError, match="unsafe"):
        native.file_for("weight")


def test_reference_execute_from_slices(tmp_path, source, monkeypatch):
    from contextlib import nullcontext
    from unittest.mock import Mock

    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, source.manifest)
    (source.source / "tokenizer.json").write_text("tokenizer")
    corpus = tmp_path / "text"
    corpus.write_text("corpus")
    slices = tmp_path / "slices.json"
    atomic_json(slices, {"slices": {"a": ["text"]}})
    args = SimpleNamespace(
        command="reference",
        source=source.source,
        manifest=manifest,
        recipe=None,
        cpu=True,
        reference=tmp_path / "new-ref",
        prefill_chunk=2,
        tokens_from_cache=None,
        slices=slices,
        tokens=2,
        through_layer=0,
        write_logits=False,
    )
    monkeypatch.setattr(b, "MemoryGuard", lambda *a: nullcontext(None))
    monkeypatch.setattr(
        b, "calibration_windows", lambda *a: ([np.array([1, 2])], "cal")
    )
    init = Mock()
    monkeypatch.setattr(q, "reference_init", init)
    monkeypatch.setattr(q, "reference_run", lambda *a: {})
    q.execute(args)
    assert init.call_args.args[3] == ["a"]


def test_attention_without_affine_compatible_shapes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        b, "module_at", lambda *a: SimpleNamespace(weight=mx.ones((2, 31)))
    )
    stage, packed, stats = b.prepare_attention(
        None, 0, b.recipe(), None, [], tmp_path, None
    )
    assert (stage, packed, stats) == ([], {}, {})
    assert json.loads((tmp_path / "attention-stats.json").read_text()) == {}


def test_restart_layer_validation(tmp_path):
    source = SimpleNamespace(manifest={}, args=SimpleNamespace(n_layers=1))
    c = b.recipe()
    run = dict(recipe=c, source_identity=b.digest({}))
    run["identity"] = b.digest(run)
    atomic_json(tmp_path / "run.json", run)
    atomic_json(
        tmp_path / "checkpoint.json", dict(identity=run["identity"], next_layer=0)
    )
    with pytest.raises(ValueError, match="restart boundary"):
        b.run_layer(source, c, tmp_path, tmp_path, 1, None)
    atomic_json(
        tmp_path / "checkpoint.json", dict(identity=run["identity"], next_layer=2)
    )
    with pytest.raises(ValueError, match="invalid layer"):
        b.run_layer(source, c, tmp_path, tmp_path, 2, None)


def test_promoted_layout_cannot_exceed_resident_budget(tmp_path, source, monkeypatch):
    from scripts.deepseek_v41_rd_build import Entry

    c = b.recipe(
        {
            "prefill_chunk": 2,
            "ffn_chunk": 2,
            "pilot": False,
            "dense_8bit_layers": [0, 1, 2, 3],
        }
    )
    parts, scratch = tmp_path / "parts", tmp_path / "scratch"
    with b.MemoryGuard(c) as guard:
        b.initialize(
            source,
            c,
            [np.array([1, 2])],
            scratch,
            parts,
            "cal",
            guard,
            token_map=list(range(32)),
        )
        real = b.output_plan

        def large(*args):
            entries, costs = real(*args)
            return [*entries, Entry("huge", (10**12,), "F32", (), "fixed_bf16")], costs

        monkeypatch.setattr(b, "output_plan", large)
        with pytest.raises(ValueError, match="cumulative resident"):
            b.run_layer(source, c, parts, scratch, 0, guard)
    assert not (parts / "layer-0-complete.json").exists()
    assert (scratch / "boundary-0" / "window-0.json").is_file()
    assert not (scratch / "working-0").exists()
    assert "phase" not in json.loads((scratch / "checkpoint.json").read_text())
    from scripts.deepseek_v41_gptq_amend import amend_recipe

    amended = amend_recipe(
        parts, scratch, {**c, "resident_budget_gb": 211.0}, 0, "raise budget"
    )
    assert amended["recipe"]["resident_budget_gb"] == 211.0


@pytest.mark.parametrize("group", [32, 64])
def test_dense_unpacked_source_shape_matches_completed_artifact(
    tmp_path, monkeypatch, group
):
    from tests import test_deepseek_v41_streaming_teacher as teacher

    original = teacher.tiny_config
    monkeypatch.setattr(
        teacher, "tiny_config", lambda: {**original(), "q_lora_rank": 32}
    )
    teacher.write_source(tmp_path / "source")
    source = PartialSource(tmp_path / "source", inventory(tmp_path / "source"))
    c = b.recipe(
        {"group_size": group, "prefill_chunk": 2, "ffn_chunk": 2, "keep_mtp": False}
    )
    entries, _ = b.output_plan(source, c)
    name = "layers.0.attn.wq_b.weight"
    entry = next(e for e in entries if e.name == name)
    assert entry.bits == (4 if group == 32 else None)
    with b.MemoryGuard(c) as guard:
        b.initialize(
            source,
            c,
            [np.array([1, 2])],
            tmp_path / "state",
            tmp_path / "parts",
            "cal",
            guard,
            token_map=list(range(32)),
        )
        b.run_layer(source, c, tmp_path / "parts", tmp_path / "state", 0, guard)
    actual = mx.load(str(tmp_path / "parts/model-layer-0.safetensors"))
    assert actual[name].shape == entry.shape
    assert actual[name].dtype == (mx.uint32 if group == 32 else mx.bfloat16)
    assert not any(".module." in n for n in actual)


def test_dense_fixed_affine_policy_is_preserved(source, monkeypatch, tmp_path):
    from scripts.deepseek_v41_rd_build import output_entries

    c = b.recipe({"group_size": 128})
    entries = output_entries(
        "layers.0.attn.wq_a.weight",
        (32, 64),
        ("layers.0.attn.wq_a.weight",),
        "attention_other_8bit",
        8,
    )
    monkeypatch.setattr(b, "plan", lambda *a: (entries, {}, {}))
    planned, _ = b.output_plan(source, c)
    assert planned == entries
    block = b.bounded_block(source, 0, c)
    assert block.attn.wq_a.bits == 8 and block.attn.wq_a.group_size == 64
    with b.MemoryGuard(c) as guard:
        _, packed, _ = b.prepare_attention(source, 0, c, block, [], tmp_path, guard)
    assert "attn.wq_a" not in packed
    assert not isinstance(block.attn.wq_a, b.Probe)


def test_candidate_mask_estimate_matches_serialized_boolean_payload(source, tmp_path):
    from copy import copy

    from rapid_mlx.models.deepseek_v41_native.indexer import select_candidate_blocks
    from scripts.deepseek_v41_gptq_io import read_header, save_state, state_snapshot

    args = source.args
    c = b.recipe({"prefill_chunk": 3})
    lengths = [7]
    layer = args.candidate_source_layer
    ratio = args.compress_ratio(layer)
    without = copy(args)
    without.candidate_source_layer = -1
    estimate = b.boundary_estimate(args, lengths, layer, c) - b.boundary_estimate(
        without, lengths, layer, c
    )
    masks = []
    for start in range(0, lengths[0], c["prefill_chunk"]):
        end = min(lengths[0], start + c["prefill_chunk"])
        scores = mx.zeros((1, end - start, end // ratio))
        lens = mx.arange(start + 1, end + 1)[:, None] // ratio
        mask = select_candidate_blocks(
            scores, lens, args.candidate_topk_blocks, args.candidate_block_size
        )
        assert mask.dtype == mx.bool_
        masks.append(mask)
    path = tmp_path / "candidate-state"
    save_state(path, {"candidates": masks})
    header = json.loads(read_header(state_snapshot(path)[1])[8:])
    payload = sum(
        v["data_offsets"][1] - v["data_offsets"][0]
        for k, v in header.items()
        if k != "__metadata__"
    )
    assert payload == estimate
    assert all(v["dtype"] == "BOOL" for k, v in header.items() if k != "__metadata__")


def test_ffn_restart_inputs_are_durable_before_checkpoint(
    source, tmp_path, monkeypatch
):
    import os

    parts, scratch = tmp_path / "parts", tmp_path / "scratch"
    c = b.recipe({"prefill_chunk": 2, "ffn_chunk": 2})
    synced = set()
    real_sync, real_atomic = os.fsync, b.atomic_json

    def sync(fd):
        synced.add(os.fstat(fd).st_ino)
        real_sync(fd)

    def atomic(path, value):
        if path.name == "checkpoint.json" and value.get("phase") == "ffn":
            work = scratch / "working-0"
            for name in (
                "attention-quantized.safetensors",
                "x.bin",
                "ids.bin",
                "gates.bin",
            ):
                assert (work / name).stat().st_ino in synced
            assert work.stat().st_ino in synced
        return real_atomic(path, value)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(b, "atomic_json", atomic)
    with b.MemoryGuard(c) as guard:
        b.initialize(
            source,
            c,
            [np.array([1, 2], np.int32)],
            scratch,
            parts,
            "cal",
            guard,
            token_map=list(range(32)),
        )
        b.run_layer(source, c, parts, scratch, 0, guard)
    assert (parts / "layer-0-complete.json").exists()


def test_copied_metadata_is_bound_to_inventory(source, tmp_path):
    from scripts.deepseek_v41_gptq_io import copy_bound_metadata

    (source.source / "tokenizer.json").write_text('{"version":1}')
    (source.source / "LICENSE").write_text("fixture license")
    bound = PartialSource(source.source, inventory(source.source))
    destination = tmp_path / "out"
    destination.mkdir()
    copy_bound_metadata(bound, destination)
    assert (destination / "tokenizer.json").read_bytes() == (
        source.source / "tokenizer.json"
    ).read_bytes()
    (source.source / "tokenizer.json").write_text('{"version":2}')
    with pytest.raises(ValueError, match="metadata identity"):
        copy_bound_metadata(bound, destination)


def test_estimate_rejects_empty_calibration_with_actionable_error(source):
    with pytest.raises(ValueError, match="calibration manifest contains no windows"):
        b.estimates(source, b.recipe(), [])


@pytest.mark.parametrize(
    "field", ["memory_limit_gb", "cache_limit_gb", "resident_budget_gb"]
)
@pytest.mark.parametrize("value", [True, False, "1", None])
def test_recipe_memory_caps_require_real_numbers(field, value):
    with pytest.raises(ValueError, match=f"invalid {field}"):
        b.recipe({field: value})


@pytest.mark.parametrize("mismatch", ["missing", "run_identity", "layer", "window"])
def test_resume_rejects_unbound_or_stale_output_window(tmp_path, mismatch):
    target = tmp_path / "boundary-1"
    target.mkdir()
    binding = dict(run_identity="run", layer=1, window=0)
    state = {"h": mx.zeros((1, 2, 64))}
    if mismatch != "missing":
        binding[mismatch] = "stale"
        state["gptq_boundary"] = binding
    b.save_state(target / "window-0", state)
    with pytest.raises(ValueError, match="boundary window identity mismatch"):
        b.finalize_windows(
            None,
            [tmp_path / "missing-input"],
            {},
            target,
            2,
            None,
            b.recipe(),
            None,
            run_identity="run",
            layer=0,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_tokens", float("nan")),
        ("min_tokens", float("inf")),
        ("min_tokens", 1.5),
        ("min_tokens", True),
        ("group_size", 64.0),
        ("block_size", 128.0),
        ("block_size", True),
        ("damping", True),
        ("damping", "0.1"),
        ("low_token_damping", True),
        ("low_token_damping", None),
        ("gate_up_bits", 2.0),
        ("down_bits", 3.0),
        ("dense_bits", 4.0),
        ("engram_bits", 8.0),
        ("engram_group_size", 64.0),
    ],
)
def test_recipe_quantization_parameters_reject_invalid_numeric_types(field, value):
    with pytest.raises(ValueError):
        b.recipe({field: value})
