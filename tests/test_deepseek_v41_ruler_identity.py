"""Public labels bind retained manifests, pinned sources and authoritative tokens."""

import json
import shutil

import numpy as np
import pytest

from scripts import deepseek_v41_public_ruler as ruler
from scripts import evaluate_deepseek_v41_rd as evaluator
from tests.test_deepseek_v41_evaluation import checkpoint_fixture
from tests.test_deepseek_v41_ruler_cli import build_inputs


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    a = build_inputs(tmp_path, monkeypatch)
    ruler.build(a)
    return a.output


def refresh_provenance(root, manifest, value):
    (root / "provenance.json").write_text(json.dumps(value))
    data = json.loads(manifest.read_text())
    data["provenance_sha256"] = ruler.file_hash(root / "provenance.json")
    manifest.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "bad", ["provenance", "tokenizer", "coverage", "sources", "tokens", "shape", "path"]
)
def test_rejects_unverified_public_manifest(bundle, bad):
    manifest = bundle / "long17k.json"
    assert ruler.validate_ruler_manifest(manifest) == ruler.file_hash(manifest)
    data = json.loads(manifest.read_text())
    provenance = json.loads((bundle / "provenance.json").read_text())
    if bad == "provenance":
        data["provenance_sha256"] = "bad"
    elif bad == "tokenizer":
        data["tokenizer_sha256"] = "bad"
    elif bad == "coverage":
        del data["slices"]["long17k_en"]
    elif bad == "sources":
        provenance["sources"][0]["sha256"] = "bad"
        refresh_provenance(bundle, manifest, provenance)
        data = json.loads(manifest.read_text())
    elif bad == "tokens":
        np.save(bundle / "long17k_en.npy", np.zeros(17000, np.int32))
    elif bad == "path":
        data["slices"]["long17k_en"] = ["../other.npy"]
    else:
        np.save(bundle / "long17k_en.npy", np.zeros(2, np.int32))
        provenance["slices"]["long17k_en"]["sha256"] = ruler.file_hash(
            bundle / "long17k_en.npy"
        )
        refresh_provenance(bundle, manifest, provenance)
        data = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="public ruler"):
        ruler.validate_ruler_manifest(manifest)


def test_retention_detects_manifest_and_copy_changes(bundle, tmp_path, monkeypatch):
    path = bundle / "long17k.json"
    digest = ruler.file_hash(path)
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="manifest changed"):
        ruler.retain_ruler_manifest(path, cache, "wrong")
    original = shutil.copyfile

    def mutate(source, destination):
        result = original(source, destination)
        if destination.name == "manifest.json":
            destination.write_bytes(destination.read_bytes() + b" ")
        return result

    monkeypatch.setattr(ruler.shutil, "copyfile", mutate)
    with pytest.raises(ValueError, match="snapshot mismatch"):
        ruler.retain_ruler_manifest(path, cache, digest)


def public_cache(root, bundle):
    root.mkdir()
    manifest = bundle / "long17k.json"
    digest = ruler.file_hash(manifest)
    ruler.retain_ruler_manifest(manifest, root, digest)
    identity, index = checkpoint_fixture()
    metadata = dict(
        mode="unpruned_reference",
        model_revision="test",
        tokenizer_sha256=ruler.TOKENIZER_HASH,
        num_layers=2,
        checkpoint_sha256=identity,
        source_checkpoint_sha256=identity,
        source_index_json=index,
        checkpoint_index_json=index,
        ruler="long17k",
        ruler_manifest_sha256=digest,
        slices={},
    )
    for name, count in ruler.PUBLIC_RULERS["long17k"].items():
        shutil.copyfile(bundle / (name + ".npy"), root / (name + "-tokens.npy"))
        np.save(root / (name + "-logits.npy"), np.zeros((count, 2), np.float32))
        np.save(root / (name + "-mass.npy"), np.ones((count, 2), np.float32))
        metadata["slices"][name] = dict(
            tokens=count,
            corpus_sha256=[ruler.file_hash(bundle / (name + ".npy"))],
            tokens_sha256=ruler.file_hash(root / (name + "-tokens.npy")),
            logits_sha256=ruler.file_hash(root / (name + "-logits.npy")),
            mass_sha256=ruler.file_hash(root / (name + "-mass.npy")),
        )
    (root / "metadata.json").write_text(json.dumps(metadata))
    return metadata


@pytest.mark.parametrize(
    "bad", [None, "missing_digest", "label", "different_manifest", "tokens", "coverage"]
)
def test_public_comparison_requires_verified_matching_records(bundle, tmp_path, bad):
    r, c = tmp_path / "reference", tmp_path / "candidate"
    rm = public_cache(r, bundle)
    cm = public_cache(c, bundle)
    if bad is None:
        result = evaluator.compare(r, c)
        assert result["rulers"]["long17k"]["tokens"] == 34000 and result["kl_nats"] == 0
        return
    if bad == "missing_digest":
        del cm["ruler_manifest_sha256"]
    elif bad == "label":
        cm["ruler"] = "wiki1100"
    elif bad == "different_manifest":
        path = c / "ruler/manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        cm["ruler_manifest_sha256"] = ruler.file_hash(path)
    elif bad == "tokens":
        for root, metadata in ((r, rm), (c, cm)):
            p = root / "long17k_en-tokens.npy"
            ids = np.load(p)
            ids[0] += 1
            np.save(p, ids)
            metadata["slices"]["long17k_en"]["tokens_sha256"] = ruler.file_hash(p)
    else:
        del rm["slices"]["long17k_en"]
        del cm["slices"]["long17k_en"]
    (r / "metadata.json").write_text(json.dumps(rm))
    (c / "metadata.json").write_text(json.dumps(cm))
    with pytest.raises(ValueError, match="ruler"):
        evaluator.compare(r, c)


@pytest.mark.parametrize("bad", ["hash", "nonzero", "calibration"])
def test_public_ruler_requires_bound_zero_overlap_audit(bundle, bad):
    manifest = bundle / "long17k.json"
    audit_path = bundle / "overlap.json"
    audit = json.loads(audit_path.read_text())
    if bad == "hash":
        audit_path.write_text("{}")
    else:
        if bad == "nonzero":
            audit["shared_windows_total"] = 1
        else:
            audit["calibration_manifest_sha256"] = "invalid"
        audit_path.write_text(json.dumps(audit))
        provenance = json.loads((bundle / "provenance.json").read_text())
        provenance.update(
            overlap_sha256=ruler.file_hash(audit_path),
            calibration_manifest_sha256=audit["calibration_manifest_sha256"],
        )
        refresh_provenance(bundle, manifest, provenance)
        data = json.loads(manifest.read_text())
        data.update(
            overlap_sha256=ruler.file_hash(audit_path),
            calibration_manifest_sha256=audit["calibration_manifest_sha256"],
        )
        manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="overlap qualification"):
        ruler.validate_ruler_manifest(manifest)


def test_recomputed_hashes_cannot_substitute_arbitrary_public_tokens(bundle):
    manifest = bundle / "long17k.json"
    path = bundle / "long17k_en.npy"
    ids = np.load(path)
    ids[0] += 7
    np.save(path, ids)
    provenance = json.loads((bundle / "provenance.json").read_text())
    provenance["slices"]["long17k_en"]["sha256"] = ruler.file_hash(path)
    refresh_provenance(bundle, manifest, provenance)
    with pytest.raises(ValueError, match="pinned source recipe"):
        ruler.validate_ruler_manifest(manifest)


def test_pinned_tokenizer_and_book_markers_are_required(bundle):
    (bundle / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="tokenizer file mismatch"):
        ruler.validate_ruler_manifest(bundle / "long17k.json")
    (bundle / "raw/pg1342.txt").write_text("missing markers")
    with pytest.raises(ValueError, match="body markers"):
        ruler.reconstruct_public_slices(bundle, "long17k")
