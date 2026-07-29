# SPDX-License-Identifier: Apache-2.0
"""Tests for ``vllm_mlx.model_sizes`` and the checked-in size manifest.

Pins the download-size feature behind ``rapid-mlx models`` / ``rapid-mlx info``
(issue #1286):

* The manifest (``model_sizes.json``) parses and every value is a positive int
  or ``null`` — no floats / bools / strings / negatives.
* ``size_bytes`` / ``format_size`` map known repos to bytes and unknown or
  ``null`` repos to "unknown".
* COVERAGE: every alias ``hf_path`` (text) and audio ``hf_id`` has a manifest
  entry — the forcing function that fails CI when someone adds an alias without
  running ``scripts/gen_model_sizes.py``.

No test here hits the network — everything reads the committed manifest.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx import model_sizes
from vllm_mlx.model_aliases import list_profiles


def _raw_manifest() -> dict:
    """The manifest exactly as committed (keeps ``null`` values)."""
    return json.loads(model_sizes._MANIFEST_PATH.read_text())


def _deepseek_v4_flash_4bit_hf() -> str:
    return list_profiles()["deepseek-v4-flash-4bit"].hf_path


def test_manifest_parses_and_value_types():
    raw = _raw_manifest()
    assert isinstance(raw.get("sizes"), dict)
    for repo, val in raw["sizes"].items():
        assert isinstance(repo, str) and repo
        # int (positive) or None — a bool is NOT an acceptable size.
        assert val is None or (
            isinstance(val, int) and not isinstance(val, bool) and val > 0
        ), f"{repo} has invalid size {val!r}"


def test_size_bytes_known_and_unknown():
    known = _deepseek_v4_flash_4bit_hf()
    n = model_sizes.size_bytes(known)
    assert isinstance(n, int) and n > 0
    assert model_sizes.size_bytes("definitely/not-a-real-repo-xyz") is None


def test_null_and_missing_both_read_as_unknown():
    raw = _raw_manifest()
    null_keys = [k for k, v in raw["sizes"].items() if v is None]
    # A committed ``null`` (unresolvable repo) must surface as None, exactly
    # like a key that was never listed at all.
    for k in null_keys:
        assert model_sizes.size_bytes(k) is None
    assert model_sizes.size_bytes("some/never-listed-repo") is None


def test_is_listed_distinguishes_null_from_absent():
    raw = _raw_manifest()
    null_keys = [k for k, v in raw["sizes"].items() if v is None]
    # A listed-but-null repo is still "listed" (so info() skips a live probe);
    # a repo the registry never carried is not.
    for k in null_keys:
        assert model_sizes.is_listed(k) is True
        assert model_sizes.size_bytes(k) is None
    assert model_sizes.is_listed(_deepseek_v4_flash_4bit_hf()) is True
    assert model_sizes.is_listed("some/never-listed-repo") is False


def test_format_size_known_and_unknown():
    known = _deepseek_v4_flash_4bit_hf()
    s = model_sizes.format_size(known)
    assert s.endswith("iB")  # KiB/MiB/GiB/TiB — base-1024 like the gate
    assert model_sizes.format_size("no/such-repo") == "—"
    assert model_sizes.format_size("no/such-repo", unknown="n/a") == "n/a"


def test_issue_example_deepseek_v4_flash_is_large():
    # The report that motivated #1286: deepseek-v4-flash-4bit surprised a user
    # at ~141 GB. Anchor that it reads as a large multi-hundred-GiB footprint
    # (guards a unit/formatter regression turning GiB into MiB, etc.).
    n = model_sizes.size_bytes(_deepseek_v4_flash_4bit_hf())
    assert n is not None and n > 100 * 1024**3
    assert model_sizes.format_size(_deepseek_v4_flash_4bit_hf()).endswith("GiB")


def test_every_text_alias_has_a_manifest_entry():
    keys = set(_raw_manifest()["sizes"])
    missing = sorted(
        p.hf_path for p in list_profiles().values() if p.hf_path not in keys
    )
    assert not missing, (
        "text aliases missing from model_sizes.json — run "
        f"`python3.12 scripts/gen_model_sizes.py`: {missing}"
    )


def test_every_audio_alias_has_a_manifest_entry():
    try:
        from vllm_mlx.audio.registry import list_audio_aliases

        entries = list_audio_aliases()
    except Exception:
        pytest.skip("audio registry unavailable in this environment")
    keys = set(_raw_manifest()["sizes"])
    missing = sorted(e.hf_id for e in entries if e.hf_id not in keys)
    assert not missing, (
        "audio aliases missing from model_sizes.json — run "
        f"`python3.12 scripts/gen_model_sizes.py`: {missing}"
    )


@pytest.mark.parametrize(
    "body",
    [
        "{ this is not json",  # invalid JSON
        "[]",  # valid JSON, wrong top-level shape
        '{"sizes": null}',  # sizes present but not a dict
        '{"sizes": [1, 2, 3]}',  # sizes is a list
        "{}",  # no sizes key at all
    ],
)
def test_loader_survives_corrupt_or_wrong_shape_manifest(tmp_path, monkeypatch, body):
    # Any unreadable / wrong-shape manifest → empty map (the listing degrades
    # to "—", never crashes with AttributeError/TypeError).
    bogus = tmp_path / "model_sizes.json"
    bogus.write_text(body)
    monkeypatch.setattr(model_sizes, "_MANIFEST_PATH", bogus)
    model_sizes._raw.cache_clear()
    try:
        assert model_sizes._raw() == {}
        assert model_sizes.size_bytes(_deepseek_v4_flash_4bit_hf()) is None
        assert model_sizes.is_listed(_deepseek_v4_flash_4bit_hf()) is False
        assert model_sizes.format_size("anything") == "—"
    finally:
        model_sizes._raw.cache_clear()  # don't poison other tests


def test_loader_normalises_bad_values_to_none(tmp_path, monkeypatch):
    # A hand-edited/malformed size (negative, zero, bool, string, float) must
    # never render as a bogus footprint — it degrades to "unknown".
    bogus = tmp_path / "model_sizes.json"
    bogus.write_text(
        '{"sizes": {"a/neg": -5, "a/zero": 0, "a/bool": true, '
        '"a/str": "12", "a/float": 1.5, "a/ok": 1073741824}}'
    )
    monkeypatch.setattr(model_sizes, "_MANIFEST_PATH", bogus)
    model_sizes._raw.cache_clear()
    try:
        for bad in ("a/neg", "a/zero", "a/bool", "a/str", "a/float"):
            assert model_sizes.size_bytes(bad) is None, bad
            assert model_sizes.is_listed(bad) is True  # still listed, just unknown
        assert model_sizes.size_bytes("a/ok") == 1073741824
    finally:
        model_sizes._raw.cache_clear()
