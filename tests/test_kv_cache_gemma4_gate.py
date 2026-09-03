# SPDX-License-Identifier: Apache-2.0
"""Capability gate for quantized KV on Gemma-4-class families (#78).

Bug #78: ``--kv-cache-dtype int8/int4`` on a Gemma-4 alias slipped the
sliding-window safelist (the multimodal config nests ``sliding_window``
under ``text_config``), the server reported ready, and then either the
text lane 503'd every request (quantized tuples reached ``gemma4_text``
attention) or the MLLM lane silently ignored the flag.

Pinned contract: nested ``text_config`` is detected structurally; an
EXPLICIT int8/int4 request that cannot be honored raises the single
typed ``KVCacheQuantizationUnsupportedError`` before ready on every
lane (text CLI, MLLM engine start, scheduler backstop, residency);
auto/profile selection may downgrade to bf16; supported full-attention
families are unchanged.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vllm_mlx.kv_cache_dtype import (
    KVCacheQuantizationUnsupportedError,
    quantized_kv_unsupported_reason,
    resolve_kv_cache_dtype,
)

# Mirrors mlx-community/gemma-4-26b-a4b-it-4bit / gemma-4-e2b-it-8bit:
# no top-level sliding_window, backbone fields nested under text_config.
GEMMA4_MLLM_CONFIG = {
    "model_type": "gemma4",
    "architectures": ["Gemma4ForConditionalGeneration"],
    "text_config": {"model_type": "gemma4_text", "sliding_window": 1024},
    "vision_config": {"model_type": "siglip_vision_model"},
}

# Mirrors mlx-community/gemma-4-12B-it-4bit (unified audio variant).
GEMMA4_UNIFIED_CONFIG = {
    "model_type": "gemma4_unified",
    "text_config": {"model_type": "gemma4_unified_text", "sliding_window": 1024},
}

# Full-attention control family (Qwen3.5-style): no sliding window, no MLA.
SUPPORTED_CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
}


# ---------------------------------------------------------------------------
# Nested text_config detection (structural, no model-name patterns)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [GEMMA4_MLLM_CONFIG, GEMMA4_UNIFIED_CONFIG])
def test_gemma4_nested_config_is_unsupported_for_quantized_kv(cfg):
    reason = quantized_kv_unsupported_reason(model_name="some-alias", hf_config=cfg)
    assert reason is not None
    assert "sliding-window" in reason.lower()


@pytest.mark.parametrize("dtype", ["int8", "int4"])
def test_gemma4_nested_auto_request_downgrades_to_bf16(dtype):
    """Auto/profile-selected quantization may safely fall back to bf16."""
    decision = resolve_kv_cache_dtype(
        dtype,
        model_name="gemma-4-26b-alias",
        hf_config=GEMMA4_MLLM_CONFIG,
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True
    assert "sliding-window" in decision.reason.lower()


@pytest.mark.parametrize("dtype", ["int8", "int4"])
def test_gemma4_nested_explicit_request_is_rejected(dtype):
    """Explicit --kv-cache-dtype int8/int4 must fail, never silently serve."""
    with pytest.raises(KVCacheQuantizationUnsupportedError) as exc_info:
        resolve_kv_cache_dtype(
            dtype,
            explicit=True,
            model_name="mlx-community/gemma-4-e2b-it-8bit",
            hf_config=GEMMA4_MLLM_CONFIG,
        )
    message = str(exc_info.value)
    assert dtype in message
    assert "gemma-4-e2b" in message
    # Actionable: tells the operator what to do instead.
    assert "bf16" in message


def test_gemma4_unified_explicit_request_is_rejected():
    with pytest.raises(KVCacheQuantizationUnsupportedError):
        resolve_kv_cache_dtype(
            "int8",
            explicit=True,
            model_name="gemma-4-12b-alias",
            hf_config=GEMMA4_UNIFIED_CONFIG,
        )


@pytest.mark.parametrize("dtype", ["int8", "int4"])
def test_supported_family_explicit_request_is_unchanged(dtype):
    """Regression control: full-attention families keep quantized KV."""
    decision = resolve_kv_cache_dtype(
        dtype,
        explicit=True,
        model_name="qwen3.5-9b-4bit",
        hf_config=SUPPORTED_CONFIG,
    )
    assert decision.dtype == dtype
    assert decision.downgraded is False


def test_explicit_bf16_never_rejects():
    """bf16 is safe everywhere — the gate only guards sub-bf16 requests."""
    decision = resolve_kv_cache_dtype(
        "bf16",
        explicit=True,
        model_name="gemma-4-26b-alias",
        hf_config=GEMMA4_MLLM_CONFIG,
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is False


# ---------------------------------------------------------------------------
# CLI: pre-load exit before any weights load (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.requires_mlx  # boots `serve`, which imports mlx at CLI load
@pytest.mark.parametrize("dtype", ["int8", "int4"])
def test_serve_exits_before_load_for_explicit_gemma4_dtype(tmp_path, dtype):
    """``serve <local-gemma4-dir> --kv-cache-dtype int8/int4`` must exit
    non-zero at the resolver gate — before any weight load or ready
    report. A config-only local directory keeps this fast and
    cache/network-independent (the gate reads ``<dir>/config.json``);
    if the gate did NOT fire, the run would instead die later with a
    weights-loading error that lacks the actionable dtype message.
    """
    # The unified (no vision_config) shape keeps the run on the text lane
    # so no vision-runtime preflight fires before the dtype gate; the
    # nested-text_config detection is identical for both shapes.
    model_dir = tmp_path / "gemma4-config-only"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(GEMMA4_UNIFIED_CONFIG))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_mlx.cli",
            "serve",
            str(model_dir),
            "--kv-cache-dtype",
            dtype,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode != 0
    assert f"--kv-cache-dtype {dtype}" in combined
    assert "cannot be honored" in combined
    assert "bf16" in combined


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    "flags",
    [
        ["--kv-cache-dtype", "int8"],
        ["--kv-cache-quantization", "--kv-cache-quantization-bits", "8"],
    ],
)
def test_serve_command_maps_both_explicit_flag_shapes_to_exit_two(
    tmp_path, flags, capsys
):
    """Exercise the in-process CLI exception mapping as well as subprocess E2E."""
    from vllm_mlx import cli

    model_dir = tmp_path / "gemma4-config-only"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(GEMMA4_UNIFIED_CONFIG))
    args = cli.build_parser().parse_args(["serve", str(model_dir), "--no-mllm", *flags])

    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)

    assert exc_info.value.code == 2
    assert "cannot be honored" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# MLLM-lane admission (pre-ready)
# ---------------------------------------------------------------------------


def _scheduler_config(**overrides):
    defaults = {
        "kv_cache_quantization": True,
        "kv_cache_quantization_bits": 8,
        "kv_cache_dtype": "int8",
        "kv_cache_dtype_explicit": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_mllm_lane_rejects_explicit_quantized_kv_before_ready():
    """The MLLM lane has no quantized-KV wiring; explicit requests must
    fail during engine start (pre-ready) with the shared typed error."""
    from vllm_mlx.engine.batched import _check_mllm_kv_quantization

    with pytest.raises(KVCacheQuantizationUnsupportedError) as exc_info:
        _check_mllm_kv_quantization(
            _scheduler_config(), "mlx-community/gemma-4-e2b-it-8bit"
        )
    message = str(exc_info.value)
    assert "int8" in message
    assert "MLLM" in message
    assert "bf16" in message


def test_mllm_lane_explicit_int4_rejected_with_dtype_in_message():
    from vllm_mlx.engine.batched import _check_mllm_kv_quantization

    with pytest.raises(KVCacheQuantizationUnsupportedError, match="int4"):
        _check_mllm_kv_quantization(
            _scheduler_config(kv_cache_dtype="int4", kv_cache_quantization_bits=4),
            "gemma-4-26b-alias",
        )


def test_mllm_lane_auto_quantization_warns_and_serves(caplog):
    """Auto/profile-selected quantization (e.g. --reasoning pin) keeps the
    lane serving bf16 but says so instead of staying silent."""
    from vllm_mlx.engine.batched import _check_mllm_kv_quantization

    with caplog.at_level(logging.WARNING):
        _check_mllm_kv_quantization(
            _scheduler_config(kv_cache_dtype_explicit=False), "qwen3-vl-alias"
        )
    assert any("bf16" in rec.message for rec in caplog.records)


def test_mllm_lane_without_quantization_is_untouched(caplog):
    from vllm_mlx.engine.batched import _check_mllm_kv_quantization

    with caplog.at_level(logging.WARNING):
        _check_mllm_kv_quantization(
            _scheduler_config(kv_cache_quantization=False), "qwen3-vl-alias"
        )
    assert not caplog.records


# ---------------------------------------------------------------------------
# Text-lane structural backstop (config-free capability probe)
# ---------------------------------------------------------------------------


def _scheduler_stub(explicit: bool):
    from vllm_mlx.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.config = _scheduler_config(
        kv_cache_dtype_explicit=explicit,
        kv_cache_turboquant=None,
        kv_cache_quantization_group_size=64,
        model_name="probe-model",
    )
    return sched


@pytest.mark.requires_mlx  # imports mlx_lm.models.cache + vllm_mlx.scheduler (mlx)
def test_live_cache_probe_flags_rotating_cache():
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from vllm_mlx.scheduler import Scheduler

    class _FakeModel:
        def make_cache(self):
            return [KVCache(), RotatingKVCache(max_size=8, keep=0)]

    assert (
        Scheduler._quantized_live_cache_incompatibility(_FakeModel())
        == "RotatingKVCache"
    )


@pytest.mark.requires_mlx  # imports mlx_lm.models.cache + vllm_mlx.scheduler (mlx)
def test_live_cache_probe_accepts_plain_kvcache():
    from mlx_lm.models.cache import KVCache

    from vllm_mlx.scheduler import Scheduler

    class _FakeModel:
        def make_cache(self):
            return [KVCache(), KVCache()]

    assert Scheduler._quantized_live_cache_incompatibility(_FakeModel()) is None


@pytest.mark.requires_mlx
def test_live_cache_probe_treats_empty_cache_list_as_unprobeable():
    from vllm_mlx.scheduler import Scheduler

    class _EmptyModel:
        def make_cache(self):
            return []

    assert (
        Scheduler._quantized_live_cache_incompatibility(_EmptyModel())
        == Scheduler._KV_CACHE_UNPROBEABLE
    )


@pytest.mark.requires_mlx  # imports mlx_lm.models.cache + vllm_mlx.scheduler (mlx)
def test_explicit_request_fails_closed_on_incompatible_cache():
    """Rotating cache + explicit request: engine start raises pre-ready."""
    from mlx_lm.models.cache import RotatingKVCache

    class _FakeModel:
        def make_cache(self):
            return [RotatingKVCache(max_size=8, keep=0)]

    with pytest.raises(KVCacheQuantizationUnsupportedError, match="RotatingKVCache"):
        _scheduler_stub(explicit=True)._init_kv_quantization(_FakeModel())


@pytest.mark.requires_mlx  # _scheduler_stub imports vllm_mlx.scheduler (mlx)
def test_explicit_request_fails_closed_on_unprobeable_cache():
    """A cache layout that cannot be verified must not report ready and
    gamble on the first request (fail closed for explicit requests)."""

    class _Broken:
        def make_cache(self):
            raise ValueError("stub")

    with pytest.raises(KVCacheQuantizationUnsupportedError, match="probed"):
        _scheduler_stub(explicit=True)._init_kv_quantization(_Broken())


@pytest.mark.requires_mlx  # imports mlx_lm.models.cache + vllm_mlx.scheduler (mlx)
def test_auto_request_disables_quantization_on_incompatible_cache(caplog):
    from mlx_lm.models.cache import RotatingKVCache

    class _FakeModel:
        def make_cache(self):
            return [RotatingKVCache(max_size=8, keep=0)]

    sched = _scheduler_stub(explicit=False)
    with caplog.at_level(logging.WARNING):
        sched._init_kv_quantization(_FakeModel())
    assert sched._kv_quant_live_disabled is True


@pytest.mark.requires_mlx
def test_serve_command_maps_runtime_backstop_to_exit_two(tmp_path, monkeypatch, capsys):
    """A late structural rejection keeps the same actionable CLI contract."""
    from vllm_mlx import cli, server

    model_dir = tmp_path / "supported-config"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(SUPPORTED_CONFIG))
    args = cli.build_parser().parse_args(
        ["serve", str(model_dir), "--no-mllm", "--kv-cache-dtype", "int8"]
    )
    error = KVCacheQuantizationUnsupportedError(
        requested="int8",
        model_name="probe-model",
        family_reason="the live cache layout could not be probed",
    )
    monkeypatch.setattr(
        cli,
        "_gather_kv_cache_dtype_inputs",
        lambda model: (SUPPORTED_CONFIG, None),
    )
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_check_alias_min_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        server, "configure_model_residency", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        server, "load_model", lambda *args, **kwargs: (_ for _ in ()).throw(error)
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)

    assert exc_info.value.code == 2
    output = capsys.readouterr().out
    assert "--kv-cache-dtype int8 cannot be honored" in output
    assert "live cache layout could not be probed" in output


# ---------------------------------------------------------------------------
# Residency (runtime) path shares the same SSOT
# ---------------------------------------------------------------------------


def test_resident_performance_explicit_int8_gemma4_rejected(monkeypatch):
    import vllm_mlx.cli as cli
    from vllm_mlx.runtime.resident_models import (
        ResidentPerformanceConfig,
        resolve_resident_performance,
    )

    monkeypatch.setattr(
        cli,
        "_gather_kv_cache_dtype_inputs",
        lambda name: (GEMMA4_MLLM_CONFIG, None),
    )
    with pytest.raises(KVCacheQuantizationUnsupportedError):
        resolve_resident_performance(
            ResidentPerformanceConfig(kv_cache_dtype="int8"),
            model_name="gemma-4-26b-alias",
            model_path=None,
        )


def test_resident_performance_supported_family_passes(monkeypatch):
    import vllm_mlx.cli as cli
    from vllm_mlx.runtime.resident_models import (
        ResidentPerformanceConfig,
        resolve_resident_performance,
    )

    monkeypatch.setattr(
        cli,
        "_gather_kv_cache_dtype_inputs",
        lambda name: (SUPPORTED_CONFIG, None),
    )
    performance = resolve_resident_performance(
        ResidentPerformanceConfig(kv_cache_dtype="int8"),
        model_name="qwen3.5-9b-4bit",
        model_path=None,
    )
    assert performance is not None
    assert performance.kv_cache_dtype == "int8"


# ---------------------------------------------------------------------------
# Legacy --kv-cache-quantization shape shares the same reject-before-ready
# contract (both bit widths), not just the new --kv-cache-dtype flag.
# ---------------------------------------------------------------------------


@pytest.mark.requires_mlx  # boots `serve`, which imports mlx at CLI load
@pytest.mark.parametrize("bits", [4, 8])
def test_serve_exits_before_load_for_explicit_gemma4_legacy_flag(tmp_path, bits):
    """The deprecated ``--kv-cache-quantization [--kv-cache-quantization-bits
    N]`` shape is equally operator-explicit and must reject an unsupported
    family BEFORE the server reports ready, with the same exit code (2) and
    actionable message as ``--kv-cache-dtype`` (#78). A config-only local
    directory keeps the run fast and cache/network-independent so the gate is
    reachable without downloading Gemma-4 weights."""
    model_dir = tmp_path / "gemma4-config-only"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(GEMMA4_UNIFIED_CONFIG))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_mlx.cli",
            "serve",
            str(model_dir),
            "--kv-cache-quantization",
            "--kv-cache-quantization-bits",
            str(bits),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 2, combined
    dtype = "int8" if bits == 8 else "int4"
    assert f"--kv-cache-dtype {dtype}" in combined
    assert "cannot be honored" in combined
    assert "bf16" in combined
