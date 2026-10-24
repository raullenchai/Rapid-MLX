"""Cache integrity and evaluator CLI behavior on deterministic CPU fixtures."""

import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_deepseek_v41_rd as evaluator
from scripts.deepseek_v41_rd_common import checkpoint_hashes, file_hash
from tests.test_deepseek_v41_evaluation import checkpoint_fixture, write_cache


@pytest.mark.parametrize(
    "bad", ["mode", "unsafe", "reference", "count", "shape", "ruler"]
)
def test_compare_rejects_incompatible_cache_metadata(tmp_path, bad):
    r, c = tmp_path / "r", tmp_path / "c"
    for root in (r, c):
        write_cache(root, [[1.0, 2.0], [3.0, 4.0]], "unpruned_reference")
    rm = json.loads((r / "metadata.json").read_text())
    cm = json.loads((c / "metadata.json").read_text())
    if bad == "mode":
        rm["mode"] = "packed_candidate"
    elif bad == "unsafe":
        rm["slices"]["../bad"] = rm["slices"].pop("domain")
        cm["slices"]["../bad"] = cm["slices"].pop("domain")
    elif bad == "reference":
        cm["reference_metadata_sha256"] = "bad"
    elif bad == "count":
        cm["slices"]["domain"]["tokens"] = 3
    elif bad == "ruler":
        cm["ruler"] = "wrong"
    elif bad == "shape":
        np.save(c / "domain-logits.npy", np.ones((1, 2)))
        cm["slices"]["domain"]["logits_sha256"] = file_hash(c / "domain-logits.npy")
    (r / "metadata.json").write_text(json.dumps(rm))
    (c / "metadata.json").write_text(json.dumps(cm))
    with pytest.raises(ValueError):
        evaluator.compare(r, c)


def test_online_rejects_invalid_score_arrays(tmp_path):
    r, c = tmp_path / "r", tmp_path / "c"
    for root in (r, c):
        write_cache(root, [[1.0, 2.0], [3.0, 4.0]], "unpruned_reference")
    rm = json.loads((r / "metadata.json").read_text())
    rm["compact_k"] = 1
    (r / "metadata.json").write_text(json.dumps(rm))
    cm = json.loads((c / "metadata.json").read_text())
    cm["reference_metadata_sha256"] = file_hash(r / "metadata.json")
    np.save(c / "domain-scores.npy", np.array([[0, np.nan], [0, 0]]))
    cm["slices"]["domain"].pop("logits_sha256")
    cm["slices"]["domain"]["scores_sha256"] = file_hash(c / "domain-scores.npy")
    (c / "metadata.json").write_text(json.dumps(cm))
    with pytest.raises(ValueError, match="invalid online"):
        evaluator.compare(r, c)


def test_mask_validation_and_knee_cli(tmp_path, monkeypatch, capsys):
    for layers in [
        {},
        {"0": {"kept_expert_ids": [0]}},
        {"0": {"kept_expert_ids": [0, 1], "bit_widths": [2, 0]}},
        {"0": {"kept_expert_ids": [0, 1]}},
    ]:
        with pytest.raises(ValueError):
            evaluator.validate_allocation({"layers": layers}, 1, 2, 2)
    points = [
        {"budget_gb": b, "worst_domain_kl_nats": k}
        for b, k in [(3, 0.001), (2, 0.002), (1, 0.1)]
    ]
    for invalid in [points[:2], [dict(p, worst_domain_kl_nats=0) for p in points]]:
        with pytest.raises(ValueError):
            evaluator.select_knee(invalid)
    for threshold in [float("nan"), float("inf"), -1]:
        with pytest.raises(ValueError):
            evaluator.select_knee(points, threshold)
    for key in ("budget_gb", "worst_domain_kl_nats"):
        with pytest.raises(ValueError):
            evaluator.select_knee([dict(points[0], **{key: float("nan")})] + points[1:])
    p = tmp_path / "points.json"
    p.write_text(json.dumps(points))
    monkeypatch.setattr(sys, "argv", ["eval", "select", "--points", str(p)])
    evaluator.main()
    assert json.loads(capsys.readouterr().out)["budget_gb"] == 2


@pytest.fixture
def collector(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    from tokenizers import Tokenizer, models

    source = tmp_path / "source"
    source.mkdir()
    tok = Tokenizer(models.WordLevel({"a": 0, "b": 1}, unk_token="a"))
    tok.save(str(source / "tokenizer.json"))
    (source / "config.json").write_text("{}")
    (source / "model.safetensors.index.json").write_text(checkpoint_fixture()[1])
    (source / "weights.safetensors").write_bytes(b"fixture weights")
    np.save(tmp_path / "a.npy", np.zeros(17, np.int32))
    manifest = tmp_path / "slices.json"
    manifest.write_text(json.dumps({"slices": {"a": ["a.npy"]}}))
    config = SimpleNamespace(
        n_layers=1, vocab_size=2, n_routed_experts=2, n_activated_experts=1
    )

    class Model:
        def make_cache(self, **kw):
            return None

        def __call__(self, ids, cache):
            return mx.zeros((*ids.shape, 2))

    model = Model()
    load = importlib.import_module("rapid_mlx.models.deepseek_v41_native.load")
    monkeypatch.setattr(load, "load", lambda *a, **kw: (model, config))
    a = SimpleNamespace(
        model=source,
        output=tmp_path / "cache",
        tokens=17,
        slices=manifest,
        model_revision="test",
        allocation=None,
        prefill_chunk=1,
        reference=None,
        serving_qmv=False,
    )
    try:
        yield a
    finally:
        mx.set_default_device(old)


@pytest.mark.requires_mlx
def test_collect_full_cache_mask_and_serving_hooks(collector, tmp_path, monkeypatch):
    a = collector
    evaluator.collect(a)
    assert np.load(a.output / "a-logits.npy").shape == (17, 2)
    assert (
        json.loads((a.output / "metadata.json").read_text())["mode"]
        == "unpruned_reference"
    )
    a.output = tmp_path / "masked"
    a.allocation = tmp_path / "allocation.json"
    a.allocation.write_text('{"layers":{}}')
    a.serving_qmv = True
    calls = []
    serving = importlib.import_module("rapid_mlx.models.deepseek_v41_native.serving")
    monkeypatch.setattr(serving, "install_target_qmv", lambda _: calls.append("qmv"))
    # Two slices must each contain exactly their own observations.
    a.slices.write_text(json.dumps({"slices": {"a": ["a.npy"], "b": ["a.npy"]}}))

    def install(model, allocation, masses):
        call = type(model).__call__

        def observe(self, ids, cache):
            masses[0].extend([0.5] * ids.shape[1])
            return call(self, ids, cache)

        monkeypatch.setattr(type(model), "__call__", observe)

    monkeypatch.setattr(evaluator, "install_native_masks", install)
    evaluator.collect(a)
    assert calls == ["qmv"]
    assert np.load(a.output / "a-mass.npy").shape == (17, 1)
    assert np.load(a.output / "b-mass.npy").shape == (17, 1)
    assert (
        json.loads((a.output / "metadata.json").read_text())["mode"]
        == "pruning_mask_only"
    )


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    "bad", ["exists", "tokens", "name", "tokenizer", "reference", "vocab", "corpus"]
)
def test_collect_guards(collector, tmp_path, bad):
    a = collector
    if bad == "exists":
        a.output.mkdir()
    elif bad == "tokens":
        a.tokens = 0
    elif bad == "name":
        a.slices.write_text(json.dumps({"slices": {"../a": ["a.npy"]}}))
    elif bad == "tokenizer":
        a.slices.write_text(
            json.dumps({"slices": {"a": ["a.npy"]}, "tokenizer_sha256": "bad"})
        )
    else:
        a.reference = tmp_path / "ref"
        a.reference.mkdir()
        rm = {
            "compact_k": 1,
            "mode": "unpruned_reference",
            "tokenizer_sha256": file_hash(a.model / "tokenizer.json"),
            "model_revision": "test",
            "slices": {"a": {"compact_sha256": {}, "corpus_sha256": []}},
            "vocab_size": 2,
            "source_checkpoint_sha256": checkpoint_hashes(a.model),
            "source_index_json": (a.model / "model.safetensors.index.json").read_text(),
        }
        if bad == "reference":
            rm["mode"] = "candidate"
        if bad == "vocab":
            rm["vocab_size"] = 3
        if bad == "corpus":
            from scripts.deepseek_v41_public_ruler import create_compact

            arrays = create_compact(a.reference, "a", 17, 2, 1)
            for v in arrays.values():
                v.flush()
        (a.reference / "metadata.json").write_text(json.dumps(rm))
    with pytest.raises((ValueError, FileExistsError)):
        evaluator.collect(a)


@pytest.mark.parametrize(
    "bad",
    ["modified", "truncated", "missing_hash", "wrong_layers", "missing_layers", "nan"],
)
def test_routing_mass_integrity(tmp_path, bad):
    r, c = tmp_path / "r", tmp_path / "c"
    for root in (r, c):
        write_cache(root, [[1.0, 2.0], [3.0, 4.0]], "unpruned_reference")
    path = c / "metadata.json"
    metadata = json.loads(path.read_text())
    mass = c / "domain-mass.npy"
    if bad == "modified":
        np.save(mass, np.full((2, 2), 0.5))
    elif bad in ("truncated", "nan"):
        np.save(
            mass, np.ones((2, 1)) if bad == "truncated" else np.full((2, 2), np.nan)
        )
        metadata["slices"]["domain"]["mass_sha256"] = file_hash(mass)
    elif bad == "missing_hash":
        del metadata["slices"]["domain"]["mass_sha256"]
    elif bad == "wrong_layers":
        metadata["num_layers"] = 3
    else:
        del metadata["num_layers"]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mass|layer counts"):
        evaluator.compare(r, c)


@pytest.mark.parametrize("bad", ["missing", "changed", "unbound_conversion"])
def test_comparison_binds_immutable_source_identity(tmp_path, bad):
    r, c = tmp_path / "r", tmp_path / "c"
    for root in (r, c):
        write_cache(root, [[1.0, 2.0], [3.0, 4.0]], "unpruned_reference")
    path = c / "metadata.json"
    metadata = json.loads(path.read_text())
    if bad == "missing":
        del metadata["source_checkpoint_sha256"]
    elif bad == "changed":
        metadata["source_checkpoint_sha256"]["config.json"] = "c" * 64
        metadata["checkpoint_sha256"] = metadata["source_checkpoint_sha256"]
    else:
        metadata["checkpoint_sha256"]["config.json"] = "c" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identity|provenance"):
        evaluator.compare(r, c)


def test_conversion_record_binds_exact_candidate_payloads(tmp_path):
    source, index = checkpoint_fixture()
    candidate = dict(source, **{"weights.safetensors": "d" * 64})
    path = tmp_path / "conversion.json"
    path.write_text(
        json.dumps(
            dict(
                source_checkpoint_sha256=source,
                checkpoint_sha256=candidate,
                source_index_json=index,
                checkpoint_index_json=index,
            )
        )
    )
    assert evaluator.source_identity_for_checkpoint(candidate, path, index) == (
        source,
        file_hash(path),
        index,
    )
    with pytest.raises(ValueError, match="candidate checkpoint mismatch"):
        evaluator.source_identity_for_checkpoint(source, path, index)
    for invalid in [None, {}, dict(source, **{"weights.safetensors": "not-a-hash"})]:
        with pytest.raises(ValueError, match="immutable checkpoint identity"):
            evaluator.validate_checkpoint_identity(invalid)
    r, c = tmp_path / "r", tmp_path / "c"
    for root in (r, c):
        write_cache(root, [[1.0, 2.0], [3.0, 4.0]], "unpruned_reference")
    metadata = json.loads((c / "metadata.json").read_text())
    metadata.update(
        checkpoint_sha256=candidate, conversion_provenance_sha256=file_hash(path)
    )
    (c / "metadata.json").write_text(json.dumps(metadata))
    record = c / "conversion-provenance.json"
    with pytest.raises(ValueError, match="record missing"):
        evaluator.compare(r, c)
    record.write_bytes(path.read_bytes())
    assert evaluator.compare(r, c)["kl_nats"] == 0
    record.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="record mismatch"):
        evaluator.compare(r, c)
    record.write_text(
        json.dumps(
            dict(
                source_checkpoint_sha256=candidate,
                checkpoint_sha256=candidate,
                source_index_json=index,
                checkpoint_index_json=index,
            )
        )
    )
    metadata["conversion_provenance_sha256"] = file_hash(record)
    (c / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="record mismatch"):
        evaluator.compare(r, c)
    record.write_text(
        json.dumps(
            dict(
                source_checkpoint_sha256=source,
                checkpoint_sha256=source,
                source_index_json=index,
                checkpoint_index_json=index,
            )
        )
    )
    with pytest.raises(ValueError, match="candidate checkpoint mismatch"):
        evaluator.compare(r, c)


@pytest.mark.requires_mlx
def test_collect_rejects_different_source_reference(collector, tmp_path):
    a = collector
    a.reference = tmp_path / "ref"
    a.reference.mkdir()
    rm = dict(
        compact_k=1,
        mode="unpruned_reference",
        tokenizer_sha256=file_hash(a.model / "tokenizer.json"),
        model_revision="test",
        slices={"a": {}},
        source_checkpoint_sha256=checkpoint_fixture()[0],
        source_index_json=checkpoint_fixture()[1],
    )
    (a.reference / "metadata.json").write_text(json.dumps(rm))
    with pytest.raises(ValueError, match="reference source checkpoint"):
        evaluator.collect(a)


@pytest.mark.requires_mlx
def test_collect_rejects_empty_manifest_without_output(collector):
    collector.slices.write_text('{"slices": {}}')
    with pytest.raises(ValueError, match="each slice needs"):
        evaluator.collect(collector)
    assert not collector.output.exists()


@pytest.mark.parametrize("roots", [("r",), ("c",), ("r", "c")])
def test_token_files_are_independently_authenticated(tmp_path, roots):
    r, c = tmp_path / "r", tmp_path / "c"
    for root in (r, c):
        write_cache(root, [[1.0, 2.0], [3.0, 4.0]], "unpruned_reference")
    for name in roots:
        np.save(tmp_path / name / "domain-tokens.npy", np.array([9, 9]))
    with pytest.raises(ValueError, match="token IDs hash mismatch"):
        evaluator.compare(r, c)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("changed", [False, True])
def test_collect_retains_verified_conversion_record(
    collector, tmp_path, monkeypatch, changed
):
    a = collector
    checkpoint = checkpoint_hashes(a.model)
    source = dict(checkpoint, **{"weights.safetensors": "d" * 64})
    index = (a.model / "model.safetensors.index.json").read_text()
    record = tmp_path / "conversion.json"
    record.write_text(
        json.dumps(
            dict(
                checkpoint_sha256=checkpoint,
                source_checkpoint_sha256=source,
                source_index_json=index,
                checkpoint_index_json=index,
            )
        )
    )
    a.conversion_provenance = record
    if changed:
        load = importlib.import_module("rapid_mlx.models.deepseek_v41_native.load")
        original = load.load

        def mutate(*args, **kwargs):
            record.write_text("{}")
            return original(*args, **kwargs)

        monkeypatch.setattr(load, "load", mutate)
        with pytest.raises(ValueError, match="changed during collection"):
            evaluator.collect(a)
        assert not (a.output / "metadata.json").exists()
    else:
        evaluator.collect(a)
        assert (
            a.output / "conversion-provenance.json"
        ).read_bytes() == record.read_bytes()
        metadata = json.loads((a.output / "metadata.json").read_text())
        assert metadata["source_checkpoint_sha256"] == source
        assert metadata["conversion_provenance_sha256"] == file_hash(record)
        assert evaluator.compare(a.output, a.output)["kl_nats"] == 0


@pytest.mark.requires_mlx
def test_collect_rechecks_checkpoint_before_metadata(collector, monkeypatch):
    a = collector
    load = importlib.import_module("rapid_mlx.models.deepseek_v41_native.load")
    original = load.load

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        config = a.model / "config.json"
        config.write_bytes(config.read_bytes() + b" ")
        return result

    monkeypatch.setattr(load, "load", mutate)
    with pytest.raises(ValueError, match="checkpoint changed during collection"):
        evaluator.collect(a)
    assert not (a.output / "metadata.json").exists()


@pytest.mark.requires_mlx
def test_allocation_digest_and_mask_share_retained_bytes(
    collector, tmp_path, monkeypatch
):
    a = collector
    a.allocation = tmp_path / "allocation.json"
    raw = b'{"layers": {"0": {"kept_expert_ids": [0]}}}'
    a.allocation.write_bytes(raw)

    def install(model, allocation, masses):
        assert allocation == json.loads(raw)
        a.allocation.write_text('{"changed": true}')
        call = type(model).__call__

        def observed(self, ids, cache):
            masses[0].extend([1.0] * ids.shape[1])
            return call(self, ids, cache)

        monkeypatch.setattr(type(model), "__call__", observed)

    monkeypatch.setattr(evaluator, "install_native_masks", install)
    evaluator.collect(a)
    assert (a.output / "allocation.json").read_bytes() == raw
    metadata = json.loads((a.output / "metadata.json").read_text())
    assert metadata["allocation_sha256"] == file_hash(a.output / "allocation.json")
    assert metadata["allocation_sha256"] != file_hash(a.allocation)


@pytest.mark.parametrize(
    "bad", ["missing", "modified", "layers", "experts", "topk", "dimensions"]
)
def test_pruning_allocation_is_authenticated_and_validated(tmp_path, bad):
    r, c = tmp_path / "r", tmp_path / "c"
    write_cache(r, [[1.0, 2.0], [3.0, 4.0]], "unpruned_reference")
    write_cache(c, [[1.0, 2.0], [3.0, 4.0]], "pruning_mask_only")
    assert evaluator.compare(r, c)["kl_nats"] == 0
    allocation = c / "allocation.json"
    metadata_path = c / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if bad == "missing":
        allocation.unlink()
    elif bad == "modified":
        allocation.write_text("{}")
    elif bad in ("layers", "experts", "topk"):
        value = json.loads(allocation.read_text())
        if bad == "layers":
            del value["layers"]["1"]
        elif bad == "experts":
            value["layers"]["0"]["kept_expert_ids"] = [3]
        else:
            value["layers"]["0"]["kept_expert_ids"] = []
        allocation.write_text(json.dumps(value))
        metadata["allocation_sha256"] = file_hash(allocation)
    else:
        metadata["routing_topk"] = 0
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        evaluator.compare(r, c)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("target", ["manifest", "corpus", "tokenizer"])
def test_collect_rechecks_tokenization_inputs(collector, tmp_path, monkeypatch, target):
    a = collector
    load = importlib.import_module("rapid_mlx.models.deepseek_v41_native.load")
    original = load.load

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = (
            a.slices
            if target == "manifest"
            else (
                a.model / "tokenizer.json"
                if target == "tokenizer"
                else tmp_path / "a.npy"
            )
        )
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(load, "load", mutate)
    with pytest.raises(ValueError, match="slice inputs changed"):
        evaluator.collect(a)
    assert not (a.output / "metadata.json").exists()


@pytest.mark.parametrize(
    "bad", ["missing_shard", "extra_shard", "index_hash", "empty_index", "unsafe_path"]
)
def test_checkpoint_identity_requires_complete_indexed_payloads(bad):
    import hashlib

    identity, index = checkpoint_fixture()
    assert evaluator.validate_checkpoint_identity(identity, index) == identity
    if bad == "missing_shard":
        del identity["weights.safetensors"]
    elif bad == "extra_shard":
        identity["unindexed.safetensors"] = "d" * 64
    elif bad == "index_hash":
        index += " "
    else:
        index = json.dumps(
            {
                "weight_map": {}
                if bad == "empty_index"
                else {"weight": "../outside.safetensors"}
            }
        )
        identity["model.safetensors.index.json"] = hashlib.sha256(
            index.encode()
        ).hexdigest()
    with pytest.raises(ValueError, match="index|weight map"):
        evaluator.validate_checkpoint_identity(identity, index)


@pytest.mark.requires_mlx
def test_collection_retains_exact_index_newlines(collector):
    a = collector
    index_path = a.model / "model.safetensors.index.json"
    raw = (
        json.dumps(json.loads(index_path.read_text()), indent=2)
        .replace("\n", "\r\n")
        .encode()
    )
    index_path.write_bytes(raw)
    evaluator.collect(a)
    metadata = json.loads((a.output / "metadata.json").read_text())
    assert metadata["checkpoint_index_json"].encode() == raw
    assert evaluator.compare(a.output, a.output)["kl_nats"] == 0
