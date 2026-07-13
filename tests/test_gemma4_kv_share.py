# SPDX-License-Identifier: Apache-2.0
"""Verify + lock Gemma 4 cross-layer KV-sharing.

Cross-layer KV-sharing (Gemma-3n / Gemma-4) already ships in the vendored
text stack (``vllm_mlx/models/gemma4_vendored/language.py``, since 0.10.1):
the last ``num_kv_shared_layers`` decoder layers are "borrowers" that compute
no K/V and reuse the last same-type producer layer's K/V. ``make_cache()``
therefore returns a *producer-only* cache list (borrowers get no cache
object) — which reduces the resident KV cache (measured ~2.3x smaller
footprint on gemma-4-e2b-4bit; the prefill/TTFT wall-time delta was ~1.0x on
that size, so the demonstrated benefit is memory, not decode speed).

Nothing in the tree asserted this held, so a future refactor that
"normalizes" ``make_cache()`` back to one-cache-per-layer would silently
break the borrow with zero coverage — and a checkpoint whose ``config.json``
sets ``num_kv_shared_layers=0`` would silently lose the KV-memory reduction.

These tests lock both:

* the load-time guard ``_check_kv_share_config`` (INFO on inactive, RAISE on
  malformed, DEBUG on active), and
* the producer→borrower map + ``make_cache()`` length for all 5 Gemma 4
  size shapes (E2B / E4B / 12B / 26B-A4B / 31B), using their real
  ``num_hidden_layers`` / ``num_kv_shared_layers`` values (no weights loaded).

Real config values (from the mlx-community ``config.json`` ``text_config``
blocks, confirmed 2026-07-13):

    size       num_hidden_layers  num_kv_shared_layers
    E2B        35                 20
    E4B        42                 18
    12B        48                  0   (dense — no sharing, by design)
    26B-A4B    30                  0   (dense/MoE — no sharing, by design)
    31B        60                  0   (dense — no sharing, by design)

Only the Gemma-3n-lineage E-series (E2B / E4B) shares K/V; the dense large
sizes ship ``num_kv_shared_layers=0`` and never borrow. The guard logs the
``0`` case at INFO (not WARNING, not raise) precisely because it is
legitimate — and the common case — for those dense sizes.
"""

from __future__ import annotations

import json
import logging

import pytest

from vllm_mlx.models.gemma4_text import _check_kv_share_config
from vllm_mlx.models.gemma4_vendored.config import TextConfig
from vllm_mlx.models.gemma4_vendored.language import LanguageModel

# (size label, num_hidden_layers, num_kv_shared_layers)
GEMMA4_SIZES = [
    ("E2B", 35, 20),
    ("E4B", 42, 18),
    ("12B", 48, 0),
    ("26B-A4B", 30, 0),
    ("31B", 60, 0),
]


def _build_text_config(num_hidden_layers: int, num_kv_shared_layers: int) -> TextConfig:
    """Construct a vendored ``TextConfig`` for a given size shape.

    Only the layer count + share count + ``layer_types`` topology drive the
    producer/borrower split under test, so we shrink hidden/intermediate/vocab/
    head dims to tiny consistent values. This keeps ``LanguageModel(tc)`` builds
    (constructed by the make_cache / producer-map tests up to 60 layers) cheap
    — the default E2B dims (hidden 1536, vocab 262144) would allocate a
    262144x1536 embedding per build. ``layer_types`` is still derived by
    ``__post_init__`` from ``sliding_window_pattern`` (4 sliding + 1 full),
    matching how the real checkpoints interleave attention types.
    """
    return TextConfig.from_dict(
        {
            "num_hidden_layers": num_hidden_layers,
            "num_kv_shared_layers": num_kv_shared_layers,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": False,
        }
    )


def _expected_previous_kvs(tc: TextConfig) -> list[int]:
    """Ground-truth borrower→producer map, computed independently of the
    code under test.

    Producers are layers ``[0, M)`` where ``M = num_hidden_layers -
    num_kv_shared_layers``. Each borrower ``j >= M`` reuses the K/V of the
    LAST producer of the SAME ``layer_type`` (full vs sliding). Producer
    layers map to themselves (identity).
    """
    n = tc.num_hidden_layers
    m = n - (tc.num_kv_shared_layers or 0)
    prev = list(range(n))  # identity by default (producers)
    if tc.num_kv_shared_layers:
        last_of_type: dict[str, int] = {}
        for i in range(m):
            last_of_type[tc.layer_types[i]] = i
        for j in range(m, n):
            prev[j] = last_of_type[tc.layer_types[j]]
    return prev


# --------------------------------------------------------------------------
# 1. Load-time guard ``_check_kv_share_config``
# --------------------------------------------------------------------------


def test_guard_active_logs_debug(caplog):
    """0 < num_kv_shared_layers < num_hidden_layers → sharing active, DEBUG."""
    tc = _build_text_config(35, 20)
    with caplog.at_level(logging.DEBUG, logger="vllm_mlx.models.gemma4_text"):
        _check_kv_share_config(
            {"num_hidden_layers": 35, "num_kv_shared_layers": 20}, tc, "test/e2b"
        )
    text = caplog.text
    assert "KV-sharing ACTIVE" in text
    assert "15 producer" in text  # 35 - 20
    assert "20 borrower" in text
    # No WARNING, no raise.
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_guard_explicit_zero_logs_info_not_warning(caplog):
    """num_kv_shared_layers=0 (real 12B/26B/31B case) → INFO, not WARNING, no
    raise. The dense sizes legitimately ship 0 and are the common case, so a
    WARNING on every such load would be a false-positive alert."""
    tc = _build_text_config(48, 0)
    with caplog.at_level(logging.INFO, logger="vllm_mlx.models.gemma4_text"):
        _check_kv_share_config(
            {"num_hidden_layers": 48, "num_kv_shared_layers": 0}, tc, "test/12b"
        )
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    msg = infos[0].getMessage()
    assert "KV-sharing INACTIVE" in msg
    assert "test/12b" in msg
    assert "num_kv_shared_layers=0" in msg
    # Explicitly NOT a warning — no false-positive production alert.
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_guard_absent_key_defaults_to_active(caplog):
    """An absent key is filled by the dataclass default (20) → the model IS
    built with 20 borrowers, so the guard truthfully reports ACTIVE and notes
    that the default was applied. (``from_dict`` masks absence — the guard
    keys severity off the value the model is actually built from.)"""
    tc = _build_text_config_absent(35)
    assert tc.num_kv_shared_layers == 20  # dataclass default masks absence
    with caplog.at_level(logging.DEBUG, logger="vllm_mlx.models.gemma4_text"):
        _check_kv_share_config({"num_hidden_layers": 35}, tc, "test/absent")
    text = caplog.text
    assert "KV-sharing ACTIVE" in text
    assert "checkpoint omitted the key" in text


def _build_text_config_absent(num_hidden_layers: int) -> TextConfig:
    """TextConfig from a dict that OMITS num_kv_shared_layers entirely."""
    return TextConfig.from_dict({"num_hidden_layers": num_hidden_layers})


@pytest.mark.parametrize("bad", [-1, 35, 100])
def test_guard_invalid_raises(bad):
    """num_kv_shared_layers < 0 or >= num_hidden_layers → ValueError."""
    # Build the config bypassing __post_init__'s reliance on the value being
    # sane; the guard reads tc.num_kv_shared_layers directly.
    tc = _build_text_config(35, max(bad, 0))
    tc.num_kv_shared_layers = bad
    with pytest.raises(ValueError, match="KV-sharing config INVALID"):
        _check_kv_share_config(
            {"num_hidden_layers": 35, "num_kv_shared_layers": bad}, tc, "test/bad"
        )


@pytest.mark.parametrize("bad", ["20", 3.5, [], True])
def test_guard_non_int_shared_raises(bad):
    """A non-int (or bool) num_kv_shared_layers is a malformed config → clear
    ValueError, not an incidental TypeError from the range comparison."""
    tc = _build_text_config(35, 20)
    tc.num_kv_shared_layers = bad
    with pytest.raises(ValueError, match="KV-sharing config INVALID"):
        _check_kv_share_config(
            {"num_hidden_layers": 35, "num_kv_shared_layers": bad}, tc, "test/badtype"
        )


def test_text_config_default_helper():
    """_text_config_default_num_kv_shared reads the dataclass field DEFAULT.
    An int default → that int; a None default (or non-dataclass) → 0 (fail-safe
    inactive)."""
    import dataclasses

    from vllm_mlx.models.gemma4_text import _text_config_default_num_kv_shared

    @dataclasses.dataclass
    class _IntDefault:
        num_kv_shared_layers: int = 7

    @dataclasses.dataclass
    class _NoneDefault:
        num_kv_shared_layers: int | None = None

    assert _text_config_default_num_kv_shared(_IntDefault()) == 7
    assert _text_config_default_num_kv_shared(_NoneDefault()) == 0  # None → 0
    assert _text_config_default_num_kv_shared(object()) == 0  # non-dataclass → 0
    # And the real vendored TextConfig default is 20 (E2B shape).
    assert _text_config_default_num_kv_shared(TextConfig()) == 20


def test_guard_explicit_null_raises():
    """An EXPLICIT ``num_kv_shared_layers: null`` in the config dict is
    malformed → ValueError. We must not guess a size-specific value (forcing
    the default would give e.g. E4B 20 borrowers instead of 18)."""
    tc = _build_text_config(35, 20)
    tc.num_kv_shared_layers = None
    with pytest.raises(ValueError, match="explicitly null"):
        _check_kv_share_config(
            {"num_hidden_layers": 35, "num_kv_shared_layers": None}, tc, "test/null"
        )


def test_guard_absent_key_with_none_field_uses_default(caplog):
    """When the key is ABSENT from the config dict but the field on the tc
    instance is ``None``, fall back to the dataclass DEFAULT (20 for the
    vendored TextConfig) — the legitimate "checkpoint didn't override the
    default" path, distinct from the explicit-null case (which raises). The
    helper's own None-default → 0 fallback is covered by
    test_text_config_default_helper."""
    tc = _build_text_config(35, 20)
    tc.num_kv_shared_layers = None
    with caplog.at_level(logging.DEBUG, logger="vllm_mlx.models.gemma4_text"):
        # dict OMITS the key → absent, not explicit null.
        _check_kv_share_config({"num_hidden_layers": 35}, tc, "test/absentnone")
    assert tc.num_kv_shared_layers == 20  # written back to the dataclass default
    assert "KV-sharing ACTIVE" in caplog.text
    assert "checkpoint omitted the key" in caplog.text
    # Model must build without a TypeError on the (now int) value.
    lm = LanguageModel(tc)
    assert lm.model.first_kv_shared_layer_idx == 15  # 35 - 20


def test_guard_active_requires_usable_layer_types():
    """Sharing on but layer_types missing/wrong-length → ValueError (cannot
    establish the producer→borrower map), NOT a bare ACTIVE log."""
    tc = _build_text_config(35, 20)
    tc.layer_types = None
    with pytest.raises(ValueError, match="layer_types is missing or"):
        _check_kv_share_config(
            {"num_hidden_layers": 35, "num_kv_shared_layers": 20}, tc, "test/nolt"
        )
    tc2 = _build_text_config(35, 20)
    tc2.layer_types = ["full_attention"] * 10  # wrong length
    with pytest.raises(ValueError, match="wrong length"):
        _check_kv_share_config(
            {"num_hidden_layers": 35, "num_kv_shared_layers": 20}, tc2, "test/shortlt"
        )
    # Non-string / unhashable entry → clear ValueError, not an incidental
    # TypeError from the set() construction.
    tc3 = _build_text_config(4, 2)
    tc3.layer_types = [
        "sliding_attention",
        "full_attention",
        ["oops"],
        "sliding_attention",
    ]
    with pytest.raises(ValueError, match="must be a list of attention-type strings"):
        _check_kv_share_config(
            {"num_hidden_layers": 4, "num_kv_shared_layers": 2}, tc3, "test/badlt"
        )


def test_guard_orphan_borrower_type_raises():
    """A borrower attention type with no producer of that type below the split
    is a malformed layer_types layout → ValueError (borrower has nothing to
    borrow)."""
    tc = _build_text_config(4, 2)
    # 2 producers, 2 borrowers. Make the only full_attention layer a borrower
    # so the full-attention borrower has no full-attention producer.
    tc.layer_types = [
        "sliding_attention",  # producer 0
        "sliding_attention",  # producer 1
        "full_attention",  # borrower 2 — no full producer below split!
        "sliding_attention",  # borrower 3
    ]
    with pytest.raises(ValueError, match="have no producer layer of that type"):
        _check_kv_share_config(
            {"num_hidden_layers": 4, "num_kv_shared_layers": 2}, tc, "test/orphan"
        )


@pytest.mark.parametrize("bad_hidden", [0, -1, None, "35", True])
def test_guard_invalid_num_hidden_raises(bad_hidden):
    """A malformed num_hidden_layers (0, negative, None, string, bool) is a
    broken config → clear ValueError, not a silent skip that lets the model
    build fail cryptically or produce an empty stack."""

    class _Stub:
        num_hidden_layers = bad_hidden
        num_kv_shared_layers = 0

    with pytest.raises(ValueError, match="num_hidden_layers"):
        _check_kv_share_config({}, _Stub(), "test/badhidden")


# --------------------------------------------------------------------------
# 2. Producer map + make_cache() length across all 5 size shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label,n_hidden,n_shared", GEMMA4_SIZES)
def test_make_cache_producer_only_length(label, n_hidden, n_shared):
    """``make_cache()`` returns exactly ``num_hidden_layers -
    num_kv_shared_layers`` cache objects — producer-only, borrowers get none.

    For dense sizes (n_shared == 0) that means one cache per layer (no
    borrowing) — which is correct: those checkpoints do not share.
    """
    tc = _build_text_config(n_hidden, n_shared)
    lm = LanguageModel(tc)
    caches = lm.make_cache()

    expected_producers = n_hidden - n_shared
    assert lm.model.first_kv_shared_layer_idx == expected_producers, label
    assert len(caches) == expected_producers, (
        f"{label}: make_cache() returned {len(caches)} cache objs, "
        f"expected {expected_producers} (= {n_hidden} - {n_shared})"
    )
    if n_shared > 0:
        # Sharing sizes: strictly fewer caches than layers.
        assert len(caches) < len(lm.model.layers), label
    else:
        # Dense sizes: one cache per layer, no borrowing.
        assert len(caches) == len(lm.model.layers), label


@pytest.mark.parametrize("label,n_hidden,n_shared", GEMMA4_SIZES)
def test_borrower_maps_to_last_same_type_producer(label, n_hidden, n_shared):
    """Each borrower reuses the last same-type (full vs sliding) producer's
    K/V; producers map to themselves. Compares the model's built
    ``previous_kvs`` against an independently computed ground truth."""
    tc = _build_text_config(n_hidden, n_shared)
    lm = LanguageModel(tc)

    expected = _expected_previous_kvs(tc)
    assert lm.model.previous_kvs == expected, (
        f"{label}: previous_kvs mismatch\n"
        f"  got     ={lm.model.previous_kvs}\n"
        f"  expected={expected}"
    )

    # Cross-check the semantic property directly for every borrower.
    m = n_hidden - n_shared
    for j in range(m, n_hidden):
        producer = lm.model.previous_kvs[j]
        assert producer < m, f"{label}: borrower {j} maps to non-producer {producer}"
        assert lm.model.layers[producer].layer_type == lm.model.layers[j].layer_type, (
            f"{label}: borrower {j} type mismatch with producer {producer}"
        )
        # It is the LAST same-type producer below the split.
        later_same_type = [
            i
            for i in range(producer + 1, m)
            if lm.model.layers[i].layer_type == lm.model.layers[j].layer_type
        ]
        assert not later_same_type, (
            f"{label}: borrower {j} should map to the LAST same-type producer, "
            f"but producers {later_same_type} come after {producer}"
        )


def test_e2b_borrow_is_active_smoke():
    """E2B shape (35 layers / 20 shared): borrow is structurally active (the
    shipped behaviour). Guards the mlx-lm ``make_cache`` deferral contract — if
    a future edit made make_cache() return one-cache-per-layer, this fails.
    Uses the tiny-dim builder so it doesn't allocate the full 262144x1536
    embedding."""
    tc = _build_text_config(35, 20)  # E2B topology, tiny dims
    lm = LanguageModel(tc)
    caches = lm.make_cache()
    assert len(caches) == 15 < len(lm.model.layers) == 35
    # Both a full-attention and a sliding-attention producer feed the top block.
    borrower_producers = {lm.model.previous_kvs[j] for j in range(15, 35)}
    producer_types = {lm.model.layers[p].layer_type for p in borrower_producers}
    assert producer_types == {"full_attention", "sliding_attention"}


def _assert_e2b_active_sharing(cfg_cls, model_cls):
    """Build an E2B-shape (35/20) model with the given class pair and assert
    make_cache() is producer-only (borrow active). Works for both the upstream
    mlx-vlm and the vendored class implementations. Uses tiny hidden/vocab/head
    dims (only the 35/20 topology matters) to keep the build cheap."""
    tc = cfg_cls.from_dict(
        {
            "num_hidden_layers": 35,
            "num_kv_shared_layers": 20,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": False,
        }
    )
    lm = model_cls(tc)
    caches = lm.make_cache()
    assert len(caches) == 15, (
        f"{model_cls.__module__}: make_cache() returned {len(caches)} caches, "
        "expected 15 (producer-only) — borrow is NOT active"
    )
    assert lm.model.first_kv_shared_layer_idx == 15
    assert len(caches) < len(lm.model.layers) == 35

    # Also validate the borrower→producer mapping on THIS class (not just the
    # cache count): every borrower must reuse the last same-type producer below
    # the split. A resolved upstream impl with a wrong previous_kvs mapping
    # would pass the count check but produce wrong attention — this catches it.
    expected = _expected_previous_kvs(tc)
    assert list(lm.model.previous_kvs) == expected, (
        f"{model_cls.__module__}: previous_kvs mismatch\n"
        f"  got     ={list(lm.model.previous_kvs)}\n"
        f"  expected={expected}"
    )
    return tc


def test_resolved_load_path_borrow_active():
    """Exercise the SAME class resolution production uses
    (``_resolve_gemma4_text_classes`` — upstream mlx-vlm when installed, else
    vendored). If the class production actually loads ever stops sharing, this
    fails — the vendored-only tests above would not catch that. Also runs the
    load-time guard on the resolved active config so the guard cannot silently
    reject the upstream TextConfig shape production resolves."""
    from vllm_mlx.models.gemma4_text import _resolve_gemma4_text_classes

    cfg_cls, model_cls = _resolve_gemma4_text_classes()
    tc = _assert_e2b_active_sharing(cfg_cls, model_cls)
    # Guard must ACCEPT the resolved active config (no raise) and log ACTIVE.
    _check_kv_share_config(
        {"num_hidden_layers": 35, "num_kv_shared_layers": 20},
        tc,
        "test/resolved-e2b",
    )


def test_vendored_fallback_borrow_active():
    """The fresh-install path (no mlx-vlm ``[vision]`` extra) must also keep
    borrow active. Force the vendored branch regardless of what is installed."""
    _assert_e2b_active_sharing(TextConfig, LanguageModel)


# --------------------------------------------------------------------------
# 3. Guard fires on the real load path
# --------------------------------------------------------------------------


def _write_gemma4_config(tmp_path, num_kv_shared_layers, num_hidden_layers=2):
    layer_types = (["sliding_attention"] * (num_hidden_layers - 1)) + ["full_attention"]
    cfg = {
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 16,
            "num_hidden_layers": num_hidden_layers,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": num_kv_shared_layers,
            "sliding_window_pattern": 2,
            "layer_types": layer_types,
            "use_double_wide_mlp": False,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_loader_logs_inactive_sharing(tmp_path, caplog):
    """Through the real ``load_gemma4_text`` path, a config with
    ``num_kv_shared_layers=0`` emits the INACTIVE INFO line before the loader
    reaches the (absent) weight files."""
    from vllm_mlx.models.gemma4_text import load_gemma4_text

    model_dir = _write_gemma4_config(tmp_path, num_kv_shared_layers=0)
    # No safetensors present → loader raises FileNotFoundError AFTER the guard
    # has already run and logged.
    with (
        caplog.at_level(logging.INFO, logger="vllm_mlx.models.gemma4_text"),
        pytest.raises(FileNotFoundError),
    ):
        load_gemma4_text(model_dir, None)
    assert any(
        "KV-sharing INACTIVE" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO
    ), "loader did not emit the INACTIVE info line for num_kv_shared_layers=0"


def test_loader_raises_on_malformed_split(tmp_path):
    """Through the real load path, a config with num_kv_shared_layers >=
    num_hidden_layers raises the malformed-config ValueError (before the
    weight-file check)."""
    from vllm_mlx.models.gemma4_text import load_gemma4_text

    # 2 layers, 2 shared → invalid (no producers).
    model_dir = _write_gemma4_config(
        tmp_path, num_kv_shared_layers=2, num_hidden_layers=2
    )
    with pytest.raises(ValueError, match="KV-sharing config INVALID"):
        load_gemma4_text(model_dir, None)
