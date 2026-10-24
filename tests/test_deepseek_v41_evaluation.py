from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.deepseek_v41_rd_common import (
    bank_plan,
)
from scripts.evaluate_deepseek_v41_rd import (
    compare,
    select_knee,
    token_metrics,
    validate_allocation,
)


def test_kl_direction_shift_invariance_and_flip_rate():
    r = np.array([[1000.0, 1001.0], [-1000.0, 1000.0]])
    c = np.array([[1001.0, 1000.0], [-999.0, 999.0]])
    kl, flip = token_metrics(r, c)
    assert kl[0] == pytest.approx(np.tanh(0.5))
    assert flip.tolist() == [True, False]
    np.testing.assert_allclose(token_metrics(r, r + 777)[0], 0, atol=1e-12)
    with pytest.raises(ValueError):
        token_metrics(r, c[:, :1])


def checkpoint_fixture():
    index = json.dumps({"weight_map": {"weight": "weights.safetensors"}})
    return {
        "config.json": "a" * 64,
        "model.safetensors.index.json": hashlib.sha256(index.encode()).hexdigest(),
        "weights.safetensors": "c" * 64,
    }, index


def write_cache(root, logits, mode, ids=(1, 2), corpus="hash"):
    from scripts.deepseek_v41_rd_common import file_hash

    root.mkdir()
    np.save(root / "domain-logits.npy", np.array(logits, np.float32))
    np.save(root / "domain-tokens.npy", np.array(ids))
    np.save(root / "domain-mass.npy", np.ones((len(ids), 2)) * 0.75)
    allocation = {
        "layers": {
            str(i): {"kept_expert_ids": [0, 1], "bit_widths": [2, 2]} for i in range(2)
        }
    }
    (root / "allocation.json").write_text(json.dumps(allocation))
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "num_layers": 2,
                "n_routed_experts": 2,
                "routing_topk": 1,
                "allocation_sha256": file_hash(root / "allocation.json")
                if mode == "pruning_mask_only"
                else None,
                "checkpoint_sha256": checkpoint_fixture()[0],
                "source_checkpoint_sha256": checkpoint_fixture()[0],
                "checkpoint_index_json": checkpoint_fixture()[1],
                "source_index_json": checkpoint_fixture()[1],
                "model_revision": "revision",
                "tokenizer_sha256": "tokenizer",
                "slices": {
                    "domain": {
                        "corpus_sha256": [corpus],
                        "mass_sha256": file_hash(root / "domain-mass.npy"),
                        "tokens_sha256": file_hash(root / "domain-tokens.npy"),
                        "logits_sha256": file_hash(root / "domain-logits.npy"),
                    }
                },
            }
        )
    )


def test_cache_comparison_validates_identity_and_reports_domains(tmp_path):
    r, c = tmp_path / "reference", tmp_path / "candidate"
    logits = [[1.0, 2.0], [3.0, 0.0]]
    write_cache(r, logits, "unpruned_reference")
    write_cache(c, logits, "pruning_mask_only")
    report = compare(r, c)
    assert report["slices"]["domain"]["kl_nats"] == 0
    assert report["slices"]["domain"]["mean_retained_routing_mass"] == 0.75
    np.save(c / "domain-tokens.npy", np.array([1, 3]))
    metadata = json.loads((c / "metadata.json").read_text())
    from scripts.deepseek_v41_rd_common import file_hash

    metadata["slices"]["domain"]["tokens_sha256"] = file_hash(c / "domain-tokens.npy")
    (c / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="token IDs"):
        compare(r, c)


def test_cache_comparison_rejects_corrupted_logits(tmp_path):
    r, c = tmp_path / "reference", tmp_path / "candidate"
    write_cache(r, [[1.0, 2.0], [3.0, 0.0]], "unpruned_reference")
    write_cache(c, [[1.0, 2.0], [3.0, 0.0]], "pruning_mask_only")
    np.save(c / "domain-logits.npy", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="hash"):
        compare(r, c)


def test_knee_uses_safe_larger_neighbor_and_fixed_worst_domain_gate():
    points = [
        {"budget_gb": b, "worst_domain_kl_nats": k}
        for b, k in [(260, 0.001), (240, 0.002), (220, 0.004), (200, 0.04), (185, 0.1)]
    ]
    # Largest rise per GB is 200 ->185; safe-side 200 fails .02 ->220.
    assert select_knee(points, 0.02)["budget_gb"] == 220
    with pytest.raises(ValueError, match="no safe"):
        select_knee(points, 0)


@pytest.mark.requires_mlx
def test_native_mask_probe_preserves_shared_expert_and_topk():
    mx = pytest.importorskip("mlx.core")
    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from rapid_mlx.models.deepseek_v41_native.moe import MoE
    from scripts.evaluate_deepseek_v41_rd import install_native_masks

    with mx.stream(mx.cpu):
        args = ModelArgs(
            dim=64, moe_inter_dim=64, n_routed_experts=4, n_activated_experts=2
        )
        moe = MoE(args)
        model = SimpleNamespace(layers=[SimpleNamespace(ffn=moe)], args=args)
        allocation = {
            "layers": {"0": {**bank_plan([2, 0, 3, 0], 2), "bit_widths": [2, 0, 3, 0]}}
        }
        masses = {0: []}
        install_native_masks(model, allocation, masses)
        bias = np.array(moe.gate.bias)
        weights, ids = moe.gate(mx.ones((1, 64)))
        assert set(ids.tolist()[0]) == {0, 2}
        assert weights.sum().item() == pytest.approx(args.route_scale)
        np.testing.assert_array_equal(np.array(moe.gate.bias), bias)
        assert 0 <= masses[0][0] <= 1
        bad = json.loads(json.dumps(allocation))
        bad["layers"]["0"]["kept_expert_ids"] = [0, 1]
        with pytest.raises(ValueError):
            validate_allocation(bad, 1, 4, 2)
