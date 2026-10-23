from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from scripts.evaluate_deepseek_v41_rd import compare
from tests.test_deepseek_v41_evaluation import write_cache


def test_subset_compare_default_unchanged_and_cli(caches, tmp_path, monkeypatch):
    from scripts.evaluate_deepseek_v41_rd import main

    r, c = caches
    with pytest.raises(ValueError, match="provenance/slice"):
        compare(r, c)
    result = compare(r, c, subset=True)
    assert result["compared_slices"] == ["domain"]
    assert result["mean_domain_kl_nats"] == 0
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "compare",
            "--reference",
            str(r),
            "--candidate",
            str(c),
            "--subset",
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize(
    "bad",
    ["empty", "unknown", "revision", "tokenizer", "corpus", "logits", "tokens", "mass"],
)
def test_subset_preserves_all_guards(caches, bad):
    r, c = caches
    path = c / "metadata.json"
    metadata = json.loads(path.read_text())
    if bad == "empty":
        metadata["slices"] = {}
    elif bad == "unknown":
        metadata["slices"]["unknown"] = metadata["slices"].pop("domain")
    elif bad in ("revision", "tokenizer"):
        metadata["model_revision" if bad == "revision" else "tokenizer_sha256"] = (
            "changed"
        )
    elif bad == "corpus":
        metadata["slices"]["domain"]["corpus_sha256"] = "changed"
    else:
        np.save(c / f"domain-{bad}.npy", np.zeros((2, 2)))
        if bad == "mass":
            np.save(c / "domain-mass.npy", np.full((2, 2), np.nan))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        compare(r, c, subset=True)


@pytest.fixture
def caches(tmp_path):
    reference, candidate = tmp_path / "r", tmp_path / "c"
    logits = [[1, 2], [3, 4]]
    write_cache(reference, logits, "unpruned_reference")
    write_cache(candidate, logits, "packed_quantized")
    path = reference / "metadata.json"
    metadata = json.loads(path.read_text())
    # A slice absent from candidate need not have payload files read at all.
    metadata["slices"]["extra"] = metadata["slices"]["domain"]
    path.write_text(json.dumps(metadata))
    return reference, candidate
