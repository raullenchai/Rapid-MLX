"""Storage-bit and immutable-model admission contracts for real-model evidence."""

import hashlib

import numpy as np
import pytest

from scripts.qualify_qwen35_fused_gdn import compare_bits, snapshot_identity


def test_storage_comparison_catches_signed_zero_and_one_ulp():
    stock = np.array([0.0, 1.0, -2.0], dtype=np.float32).view(np.uint32)
    candidate = np.array(
        [-0.0, np.nextafter(np.float32(1), np.float32(2)), -2], dtype=np.float32
    ).view(np.uint32)
    metrics = compare_bits(stock, candidate)
    assert metrics["differing_elements"] == 2
    assert metrics["first_indices"] == [0, 1]
    assert metrics["stock_sha256"] != metrics["fused_sha256"]
    assert (
        metrics["indices_sha256"]
        == hashlib.sha256(np.array([0, 1], dtype="<i8").tobytes()).hexdigest()
    )


def test_exact_storage_comparison_and_metadata_validation():
    bits = np.arange(20, dtype=np.uint16).reshape(4, 5)
    metrics = compare_bits(bits, bits.copy())
    assert metrics["differing_elements"] == 0
    assert metrics["stock_sha256"] == metrics["fused_sha256"]
    assert metrics["first_indices"] == []
    with pytest.raises(ValueError, match="shape/dtype"):
        compare_bits(bits, bits.reshape(-1))
    with pytest.raises(ValueError, match="shape/dtype"):
        compare_bits(bits, bits.astype(np.uint32))


def test_snapshot_rejects_mutable_refs_and_missing_weights(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        snapshot_identity(tmp_path / "main")
    path = tmp_path / "models--owner--model" / "snapshots" / ("a" * 40)
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="complete local weights"):
        snapshot_identity(path)
    (path / "model.safetensors").write_bytes(b"local-weight-fixture")
    identity = snapshot_identity(path)
    assert identity["revision"] == "a" * 40
    assert identity["repository"] == "owner/model"
    assert identity["weights"] == [{"name": "model.safetensors", "size": 20}]


@pytest.fixture
def tiny_real_model(monkeypatch, tmp_path):
    """One production GDN layer with native prefill/decode and small vocabulary."""
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    import mlx_lm
    from mlx_lm.models.cache import ArraysCache
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

    from rapid_mlx import qwen35_fused_gdn_decode as fused

    if not mx.metal.is_available() or fused.probe_qwen35_fused_gdn_decode() is None:
        pytest.skip("runtime does not admit bit-exact Metal GDN")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 2048)
            self.gdn = GatedDeltaNet(
                TextModelArgs(
                    hidden_size=2048,
                    linear_num_key_heads=16,
                    linear_num_value_heads=32,
                    linear_key_head_dim=128,
                    linear_value_head_dim=128,
                )
            )
            self.head = nn.Linear(2048, 64, bias=False)

        def __call__(self, tokens, cache):
            return self.head(self.gdn(self.embed(tokens), cache=cache[0]))

        def make_cache(self):
            return [ArraysCache(2)]

    class Tokenizer:
        def encode(self, *args, **kwargs):
            return [1, 2, 3]

        def decode(self, tokens):
            return ",".join(map(str, tokens))

    mx.random.seed(4448)
    model = Model()
    model.set_dtype(mx.bfloat16)
    model.eval()
    monkeypatch.setattr(mlx_lm, "load", lambda *a, **kw: (model, Tokenizer()))
    snapshot = tmp_path / "models--test--gdn" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"local-weight-fixture")
    out = tmp_path / "evidence"
    out.mkdir()
    return model, snapshot, out


def test_one_token_prefill_tail_is_not_decode(tiny_real_model):
    from scripts.qualify_qwen35_fused_gdn import qualify

    _, snapshot, out = tiny_real_model
    report = qualify(snapshot, [257], 2, out)
    assert report["status"] == "exact"
    assert report["rows"] == 4
    assert [t["steps_completed"] for t in report["trajectories"]] == [2, 2]


def test_production_fallback_cannot_false_green(tiny_real_model, monkeypatch):
    from rapid_mlx import qwen35_fused_gdn_decode as fused
    from scripts.qualify_qwen35_fused_gdn import qualify

    _, snapshot, out = tiny_real_model
    monkeypatch.setattr(fused, "_projected_eligible", lambda *a, **kw: False)
    report = qualify(snapshot, [8], 2, out)
    assert report["status"] == "mismatch"
    assert report["rows"] == 0
    assert all("silently fell back" in t["error"] for t in report["trajectories"])
    assert all(t["steps_completed"] == 0 for t in report["trajectories"])


def test_prefill_failure_preserves_both_order_receipts(tiny_real_model, monkeypatch):
    from scripts.qualify_qwen35_fused_gdn import qualify

    model, snapshot, out = tiny_real_model

    def fail(*args, **kwargs):
        raise RuntimeError("injected prefill failure")

    monkeypatch.setattr(type(model), "__call__", fail)
    report = qualify(snapshot, [8], 2, out)
    assert report["status"] == "mismatch"
    assert len(report["trajectories"]) == 2
    assert all(t["error"] == "injected prefill failure" for t in report["trajectories"])
    assert all(t["steps_completed"] == 0 for t in report["trajectories"])


def test_state_only_corruption_is_red_and_retains_indices(tiny_real_model, monkeypatch):
    import gzip
    import json

    from rapid_mlx import qwen35_fused_gdn_decode as fused
    from scripts.qualify_qwen35_fused_gdn import qualify

    _, snapshot, out = tiny_real_model
    kernel = fused.fused_gdn_decode

    def corrupt(*a, **kw):
        output, conv, state = kernel(*a, **kw)
        state = state.at[0, 0, 0, 0].add(1.0)
        return output, conv, state

    monkeypatch.setattr(fused, "fused_gdn_decode", corrupt)
    report = qualify(snapshot, [8], 2, out)
    assert report["status"] == "mismatch"
    assert report["rows"] == 2
    assert all(t["steps_completed"] == 0 for t in report["trajectories"])
    rows = [json.loads(line) for line in gzip.open(out / "rows.jsonl.gz", "rt")]
    for row in rows:
        assert row["metrics"]["state"]["differing_elements"] == 1
        assert row["metrics"]["normalized_output"]["differing_elements"] == 0
    assert np.load(out / "mismatch-0-state-indices.npy").tolist() == [0]
    assert (out / "mismatch-0.safetensors").is_file()
