"""--pflash-threshold-basis uncached: pure threshold arithmetic (scheduler/cache contract: test_pflash_threshold_basis_scheduler.py)."""

from types import SimpleNamespace

import pytest

from rapid_mlx.pflash import (
    PFlashConfig,
    compress_request_tokens,
    compress_tokens,
    config_from_args,
)


def _toks(n):
    return [(i * 7919) % 50021 for i in range(n)]


CFG = dict(
    mode="auto",
    threshold=1000,
    keep_ratio=0.5,
    min_keep_tokens=0,
    sink_tokens=16,
    tail_tokens=64,
    block_size=16,
    query_window=32,
    stride_blocks=8,
    skip_when_tools=False,
)


def test_total_basis_ignores_cache_and_compresses():
    cfg = PFlashConfig(**CFG).validate()
    r = compress_tokens(_toks(1500), cfg, cached_prefix_tokens=1400)
    assert r.compressed and r.reason == "compressed"


def test_uncached_basis_skips_when_remainder_small():
    cfg = PFlashConfig(**CFG, threshold_basis="uncached").validate()
    r = compress_tokens(_toks(1500), cfg, cached_prefix_tokens=1400)
    assert not r.compressed and r.reason == "cached_prefix"
    assert r.tokens == _toks(1500)


def test_uncached_basis_compresses_when_remainder_large():
    cfg = PFlashConfig(**CFG, threshold_basis="uncached").validate()
    r = compress_tokens(
        _toks(3000), cfg, cached_prefix_tokens=1000
    )  # 2000 uncached >= 1000
    assert r.compressed


def test_uncached_basis_cold_prompt_behaves_like_total():
    cfg = PFlashConfig(**CFG, threshold_basis="uncached").validate()
    assert compress_tokens(_toks(1500), cfg).compressed
    assert compress_tokens(_toks(900), cfg).reason == "threshold"


def test_always_mode_unaffected_by_basis():
    cfg = PFlashConfig(
        **{**CFG, "mode": "always"}, threshold_basis="uncached"
    ).validate()
    assert compress_tokens(_toks(1500), cfg, cached_prefix_tokens=1499).compressed


def test_invalid_basis_rejected():
    with pytest.raises(ValueError):
        PFlashConfig(**CFG, threshold_basis="bogus").validate()


def test_metadata_and_args_wiring():
    cfg = PFlashConfig(**CFG, threshold_basis="uncached").validate()
    _, md = compress_request_tokens(_toks(1500), cfg, cached_prefix_tokens=1400)
    assert md["cached_prefix_tokens"] == 1400 and md["reason"] == "cached_prefix"
    args = SimpleNamespace(
        pflash="auto",
        pflash_threshold=131072,
        pflash_keep_ratio=0.5,
        pflash_min_keep_tokens=2048,
        pflash_sink_tokens=256,
        pflash_tail_tokens=2048,
        pflash_block_size=128,
        pflash_query_window=512,
        pflash_stride_blocks=8,
        pflash_include_tools=True,
        pflash_threshold_basis="uncached",
    )
    assert config_from_args(args).threshold_basis == "uncached"
    del args.pflash_threshold_basis
    assert config_from_args(args).threshold_basis == "total"


@pytest.mark.parametrize("reported, expected", [(-4, 0), (2000, 1500)])
def test_metadata_clamps_cached_prefix_to_prompt(reported, expected):
    cfg = PFlashConfig(**CFG, threshold_basis="uncached").validate()
    _, metadata = compress_request_tokens(
        _toks(1500), cfg, cached_prefix_tokens=reported
    )
    assert metadata["cached_prefix_tokens"] == expected
