"""CPU math, provenance and scoring contracts for public rulers."""

import json

import numpy as np
import pytest

from scripts.deepseek_v41_public_ruler import (
    COMPACT_FIELDS,
    compact_hashes,
    compact_logits,
    compact_metrics,
    slice_tokens,
    windows,
)
from scripts.deepseek_v41_rd_common import file_hash
from scripts.evaluate_deepseek_v41_rd import compare, token_metrics
from tests.test_deepseek_v41_evaluation import checkpoint_fixture


@pytest.mark.parametrize("k", [1, 8, 31, 32])
def test_compact_is_kl_lower_bound_with_exact_top1(k):
    rng = np.random.default_rng(1017)
    r = rng.normal(size=(16, 32)).astype(np.float32)
    c = r + rng.normal(0, 0.3, size=r.shape)
    full, flip = token_metrics(r, c)
    small, small_flip = compact_metrics(compact_logits(r, k), c)
    assert np.all(small <= full + 3e-7)
    assert np.array_equal(flip, small_flip)
    if k == 32:
        np.testing.assert_allclose(small, full, atol=3e-7)


def test_compact_tail_extremes_and_argmax_ties():
    r = np.array([[100.0, -1000.0, -1000.0, -1000.0], [0.0, 0.0, 0.0, 0.0]])
    c = np.array([[-1000.0, 100.0, -1000.0, -1000.0], [1.0, 0.0, 0.0, 0.0]])
    compact = compact_logits(r, 2)
    kl, flip = compact_metrics(compact, c)
    assert np.isfinite(kl).all()
    assert compact["argmax"].tolist() == [0, 0]
    assert flip.tolist() == [True, False]


def test_token_slices_are_exact_without_decode_roundtrip(tmp_path):
    from tokenizers import Tokenizer, models

    tok = Tokenizer(models.WordLevel({"a": 0, "b": 1}, unk_token="a"))
    p = tmp_path / "tokens.npy"
    np.save(p, np.array([1, 0, 1], np.int32))
    assert slice_tokens([p], tok, 3).tolist() == [1, 0, 1]
    with pytest.raises(ValueError, match="requested"):
        slice_tokens([p], tok, 2)
    assert windows(range(100)) & windows(range(36, 100))
    assert not windows(range(63))


@pytest.mark.parametrize("online", [False, True])
def test_compare_compact_and_online_fields_and_integrity(tmp_path, online):
    ref, cand = tmp_path / "ref", tmp_path / "cand"
    ref.mkdir()
    cand.mkdir()
    for root in (ref, cand):
        np.save(root / "a-tokens.npy", np.arange(11, dtype=np.int32))
    rng = np.random.default_rng(7)
    r = rng.normal(size=(11, 32)).astype(np.float32)
    c = r + rng.normal(0, 0.2, size=r.shape)
    small = compact_logits(r, 8)
    for f in COMPACT_FIELDS:
        np.save(ref / f"a-{f}.npy", small[f])
    base = dict(
        mode="unpruned_reference",
        model_revision="test",
        tokenizer_sha256="tok",
        ruler="tiny",
        vocab_size=32,
        num_layers=2,
        checkpoint_sha256=checkpoint_fixture()[0],
        source_checkpoint_sha256=checkpoint_fixture()[0],
        checkpoint_index_json=checkpoint_fixture()[1],
        source_index_json=checkpoint_fixture()[1],
    )
    rm = dict(
        base,
        compact_k=8,
        slices=dict(
            a=dict(
                tokens=11,
                tokens_sha256=file_hash(ref / "a-tokens.npy"),
                corpus_sha256=["text"],
                compact_sha256=compact_hashes(ref, "a"),
            )
        ),
    )
    (ref / "metadata.json").write_text(json.dumps(rm))
    kl, flip = compact_metrics(small, c)
    if online:
        np.save(cand / "a-scores.npy", np.column_stack([kl, flip]))
        entry = dict(scores_sha256=file_hash(cand / "a-scores.npy"))
    else:
        np.save(cand / "a-logits.npy", c)
        entry = dict(logits_sha256=file_hash(cand / "a-logits.npy"))
    np.save(cand / "a-mass.npy", np.ones((11, 2)))
    entry["mass_sha256"] = file_hash(cand / "a-mass.npy")
    entry["tokens_sha256"] = file_hash(cand / "a-tokens.npy")
    cm = dict(base, slices=dict(a=dict(tokens=11, corpus_sha256=["text"], **entry)))
    if online:
        cm["reference_metadata_sha256"] = file_hash(ref / "metadata.json")
    (cand / "metadata.json").write_text(json.dumps(cm))
    for root in (ref, cand):
        np.save(root / "a-tokens.npy", np.arange(11, dtype=np.int32))
    np.save(cand / "a-mass.npy", np.ones((11, 2)))
    got = compare(ref, cand)
    assert got["rulers"]["tiny"]["tokens"] == 11
    assert got["kl_nats"] == pytest.approx(kl.mean())
    assert got["p99_kl_nats"] == pytest.approx(np.quantile(kl, 0.99))
    assert got["top1_agreement"] == pytest.approx(1 - flip.mean())
    np.save(ref / "a-tail_mass.npy", small["tail_mass"] * 0.5)
    with pytest.raises(ValueError, match="hash"):
        compare(ref, cand)
