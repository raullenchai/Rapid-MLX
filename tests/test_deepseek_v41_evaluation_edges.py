"""Cache integrity and evaluator CLI behavior on deterministic CPU fixtures."""

import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_deepseek_v41_rd as evaluator
from scripts.deepseek_v41_rd_common import file_hash
from tests.test_deepseek_v41_evaluation import write_cache


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
    (source / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    np.save(tmp_path / "a.npy", np.zeros(17, np.int32))
    manifest = tmp_path / "slices.json"
    manifest.write_text(json.dumps({"slices": {"a": ["a.npy"]}}))
    config = SimpleNamespace(n_layers=1, vocab_size=2)

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
    monkeypatch.setattr(
        evaluator,
        "install_native_masks",
        lambda m, a, masses: masses[0].extend([0.5] * 17),
    )
    # collect clears routing observations before each slice; fake model emits them.
    original = evaluator.install_native_masks

    def install(model, allocation, masses):
        original(model, allocation, masses)
        call = type(model).__call__

        def observe(self, ids, cache):
            masses[0].extend([0.5] * ids.shape[1])
            return call(self, ids, cache)

        monkeypatch.setattr(type(model), "__call__", observe)

    monkeypatch.setattr(evaluator, "install_native_masks", install)
    evaluator.collect(a)
    assert calls == ["qmv"]
    assert np.load(a.output / "a-mass.npy").shape == (17, 1)
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
