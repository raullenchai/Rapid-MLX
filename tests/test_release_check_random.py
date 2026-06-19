# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/release_check_m3_random.py — G12 release-gauntlet
random-coverage gate. The orchestrator script lives outside the
``vllm_mlx`` package so we import it via importlib."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "release_check_m3_random.py"
)


@pytest.fixture(scope="module")
def g12():
    """Load the orchestrator script as a module so its helpers can be
    unit-tested without spawning subprocesses."""
    spec = importlib.util.spec_from_file_location("g12", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fake_aliases() -> dict[str, dict]:
    """A shrunken aliases.json fixture covering every eligibility
    branch (size in/out of band, vision marker, kimi marker, gemma-4
    marker, 8bit vs 4bit, missing hf_path, fail-closed name-without-
    size token)."""
    return {
        # In-band (4-12 B, 4-bit, no special excludes)
        "qwen3.5-9b-4bit": {"hf_path": "mlx-community/Qwen3.5-9B-4bit"},
        "qwen3-8b-4bit": {"hf_path": "mlx-community/Qwen3-8B-4bit"},
        "hermes3-8b-4bit": {"hf_path": "mlx-community/Hermes-3-Llama-3.1-8B-4bit"},
        # Fail-closed: the repo name carries no parameter-count token
        # (``Air`` is a variant name, ``4.5`` is the version). The old
        # parser regex extracted ``4`` from ``-4bit`` and admitted this
        # entry as a 4 B model — codex PR #693 review caught it. The
        # post-fix parser refuses to guess and skips the entry.
        "glm4.5-air-4bit": {"hf_path": "mlx-community/GLM-4.5-Air-4bit"},
        # Out-of-band: too small (< 4 B) — harnesses would false-fail.
        "qwen3-0.6b-4bit": {"hf_path": "mlx-community/Qwen3-0.6B-4bit"},
        "llama3-1b-4bit": {"hf_path": "mlx-community/Llama-3.2-1B-Instruct-4bit"},
        "smollm3-3b-4bit": {"hf_path": "mlx-community/SmolLM3-3B-4bit"},
        # Out-of-band: too large (> 12 B).
        "qwen3.5-27b-4bit": {"hf_path": "mlx-community/Qwen3.5-27B-4bit"},
        # Excluded: vision variant.
        "qwen3-vl-8b-4bit": {"hf_path": "mlx-community/Qwen3-VL-8B-Instruct-4bit"},
        # Excluded: kimi family (heavy + user-flagged).
        "kimi-k2-9b-4bit": {"hf_path": "fake/Kimi-K2-9B-4bit"},
        # Excluded: gemma-4 family (issue #686 thought-loop hang).
        "gemma-4-12b-4bit": {"hf_path": "mlx-community/gemma-4-12B-it-4bit"},
        "gemma-4-e4b-4bit": {"hf_path": "mlx-community/gemma-4-e4b-it-4bit"},
        # Excluded: 8-bit quant (we sample 4-bit only).
        "qwen3.5-9b-8bit": {"hf_path": "mlx-community/Qwen3.5-9B-8bit"},
        # Skipped silently: missing hf_path field.
        "broken-9b-4bit": {"tool_call_parser": "hermes"},
    }


def test_eligible_aliases_filters_correctly(g12, tmp_path):
    """The 3 in-band entries survive; everything else is filtered.

    Note ``glm4.5-air-4bit`` is **expected** to be excluded — its repo
    name (``GLM-4.5-Air-4bit``) carries no parameter-count token, so
    the post-codex-#693 parser fails closed rather than mis-attributing
    a 4 B size from the ``-4bit`` quantization suffix.
    """
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps(_fake_aliases()))
    eligible = g12._eligible_aliases(p)
    names = {name for name, _ in eligible}
    assert names == {
        "qwen3.5-9b-4bit",
        "qwen3-8b-4bit",
        "hermes3-8b-4bit",
    }


def test_eligible_aliases_does_not_parse_quant_suffix_as_size(g12, tmp_path):
    """Regression: the size parser MUST NOT match ``-4bit`` / ``-8bit``
    as a fake 4 B / 8 B model size.

    Round-1 codex review of PR #693 caught this — the original regex
    ``(\\d+(?:\\.\\d+)?)b`` greedily matched the ``4b`` inside ``4bit``,
    so any 4-bit alias without a real size token in its repo name
    (e.g. ``GLM-4.5-Air-4bit``) slipped past the 4-12 B disk filter as
    a phantom 4 B model. The post-fix parser requires the ``b`` token
    to be bounded by name separators so the quant suffix can't match.
    """
    aliases = {
        # Repo name has NO parameter-count token. Must fail closed.
        "phantom-air-4bit": {"hf_path": "fake/Phantom-Air-4bit"},
        # Same idea, 8-bit suffix.
        "phantom-air-8bit": {"hf_path": "fake/Phantom-Air-8bit"},
        # Real size token IS present — must survive (control case).
        "good-9b-4bit": {"hf_path": "fake/Good-9B-4bit"},
    }
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps(aliases))
    eligible = dict(g12._eligible_aliases(p))
    assert "phantom-air-4bit" not in eligible, (
        "size parser must not extract 4 from -4bit quant suffix"
    )
    assert "phantom-air-8bit" not in eligible, (
        "the 8-bit alias is filtered by quant-only rule anyway, but its "
        "repo name also has no real size token — both filters apply"
    )
    assert eligible.get("good-9b-4bit") == "fake/Good-9B-4bit"


def test_eligible_aliases_sorted_for_reproducible_sampling(g12, tmp_path):
    """``random.sample`` is order-sensitive — eligibility must return
    in deterministic order so the same seed picks the same models
    across machines / future aliases.json additions to unrelated
    entries.

    The function sorts by ``(size_B, name)``. We verify determinism by
    invoking twice and checking order also doesn't depend on dict
    insertion order in the source JSON.
    """
    p = tmp_path / "aliases.json"
    aliases = _fake_aliases()
    p.write_text(json.dumps(aliases))
    eligible_a = g12._eligible_aliases(p)
    eligible_b = g12._eligible_aliases(p)
    assert eligible_a == eligible_b
    # Write the same aliases in REVERSED insertion order — sort must
    # produce the same output regardless of dict-iteration order.
    p_reversed = tmp_path / "aliases_reversed.json"
    p_reversed.write_text(json.dumps(dict(reversed(list(aliases.items())))))
    eligible_c = g12._eligible_aliases(p_reversed)
    assert eligible_a == eligible_c, (
        "eligibility must be order-stable across source-file dict orderings"
    )


def test_real_aliases_json_yields_nonzero_pool(g12):
    """Sanity check against the in-tree aliases.json: at least 5
    eligible models must exist or the gauntlet has nothing to sample.
    If this trips after a future aliases prune, raise the floor or
    adjust the filter to admit a wider band."""
    real = Path(__file__).resolve().parent.parent / "vllm_mlx" / "aliases.json"
    eligible = g12._eligible_aliases(real)
    assert len(eligible) >= 5, (
        f"need ≥5 sample-eligible aliases for meaningful G12 random "
        f"coverage; only {len(eligible)} pass the filter — "
        f"check the size band / new exclude rules in "
        f"release_check_m3_random.py"
    )


def test_hf_cache_dir_shape(g12, monkeypatch):
    """Cleanup path must match HuggingFace's actual snapshot layout
    (``models--<owner>--<repo>``) so the rm -rf at end-of-model
    actually deletes the right tree, not a sibling.

    Pin the env to the default branch — other tests in the module may
    leave ``HF_HOME`` / ``HF_HUB_CACHE`` set in this process.
    """
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(env, raising=False)
    p = g12._hf_cache_dir("mlx-community/Qwen3.5-9B-4bit")
    assert p.name == "models--mlx-community--Qwen3.5-9B-4bit"
    assert p.parent.name == "hub"
    assert p.parent.parent.name == "huggingface"


def test_free_disk_gb_walks_to_existing_ancestor(g12, tmp_path):
    """``_free_disk_gb`` must tolerate a non-existent leaf — the cache
    root may be on a custom mount whose ``models--owner--repo`` leaf
    hasn't been created until the first download. Without the ancestor
    walk, ``shutil.disk_usage`` raises ``FileNotFoundError`` on a
    brand-new ``HF_HUB_CACHE=/data/hf-cache`` rig where ``/data/``
    exists but ``hf-cache`` doesn't. Codex round-2 PR #693 review.
    """
    missing_leaf = tmp_path / "nonexistent" / "deeper" / "still-missing"
    assert not missing_leaf.exists()
    # Should return a real positive number, not raise.
    free_gb = g12._free_disk_gb(missing_leaf)
    assert free_gb > 0


def test_hf_cache_root_honors_env_vars(g12, tmp_path, monkeypatch):
    """``_hf_cache_root`` must respect ``HF_HUB_CACHE``,
    ``HUGGINGFACE_HUB_CACHE`` and ``HF_HOME`` in the same precedence
    order as ``huggingface_hub.constants.HF_HUB_CACHE`` — otherwise
    G12 downloads into one place and tries to ``rm -rf`` another,
    leaving the actual snapshots on disk to balloon across releases.
    Codex round-1 review of PR #693 caught this.
    """
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(env, raising=False)

    # Modern override
    target = tmp_path / "modern"
    monkeypatch.setenv("HF_HUB_CACHE", str(target))
    assert g12._hf_cache_root() == target
    monkeypatch.delenv("HF_HUB_CACHE")

    # Legacy override
    target = tmp_path / "legacy"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(target))
    assert g12._hf_cache_root() == target
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE")

    # HF_HOME — cache root is ``$HF_HOME/hub``
    target = tmp_path / "home"
    monkeypatch.setenv("HF_HOME", str(target))
    assert g12._hf_cache_root() == target / "hub"
    monkeypatch.delenv("HF_HOME")

    # Default
    assert g12._hf_cache_root() == Path.home() / ".cache" / "huggingface" / "hub"
