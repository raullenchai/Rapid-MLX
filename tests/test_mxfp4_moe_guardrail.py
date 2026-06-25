# SPDX-License-Identifier: Apache-2.0
"""R15 task #297 — load-time guardrail for the MoE+MXFP4+multi-device cliff.

Covers the detection matrix for the two upstream MLX issues exposed by
``vllm_mlx/_mxfp4_moe_guardrail.py``:

* mlx#3402 — MoE + MXFP4 + multi-device throughput cliff
  (3-of-3 → fire; any 2-of-3 → silent).
* mlx#2962 — MoE + NVFP4 dynamic-range loss
  (fires regardless of device count, MoE required).

The tests are pure: every signal is constructed by hand via
:class:`GuardrailSignal`, so we don't pull any MoE weights from disk
just to exercise the guardrail (per the task disk-hygiene constraint).
The mlx.distributed probe is exercised through ``check_from_profile``
with monkeypatched detectors so we don't need a real MPI run either.
"""

from __future__ import annotations

import logging

import pytest

from vllm_mlx import _mxfp4_moe_guardrail as g


@pytest.fixture(autouse=True)
def _reset_guardrail_state():
    """Zero the module counters around every test for isolation."""
    g.reset_for_tests()
    yield
    g.reset_for_tests()


# ---------------------------------------------------------------------------
# _detect_quant_format — path-name heuristic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hf_path,expected",
    [
        ("mlx-community/MiniMax-M2.7-4bit-mxfp4", "mxfp4"),
        ("nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx", "mxfp4"),
        ("mlx-community/gpt-oss-20b-MXFP4-Q8", "mxfp4"),  # case-insensitive
        ("vendor/Some-Model-NVFP4", "nvfp4"),
        ("vendor/some-nvfp4-moe", "nvfp4"),
        ("mlx-community/Qwen3-7B-4bit", None),
        ("mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit", None),
        ("", None),
        (None, None),
    ],
)
def test_detect_quant_format(hf_path, expected):
    """Path-name heuristic detects mxfp4/nvfp4 case-insensitively."""
    assert g._detect_quant_format(hf_path) == expected


def test_detect_quant_format_mxfp4_wins_over_nvfp4():
    """When both tokens are present, the louder mxfp4 warning wins."""
    # Synthetic — no real model carries both, but the priority matters.
    assert g._detect_quant_format("vendor/weird-mxfp4-nvfp4") == "mxfp4"


# ---------------------------------------------------------------------------
# check_load_time_guardrails — the 3-flag matrix
# ---------------------------------------------------------------------------


def _signal(is_moe, quant, world):
    return g.GuardrailSignal(
        is_moe=is_moe,
        quant_format=quant,
        distributed_world_size=world,
    )


def test_three_tuple_fires_mxfp4_moe_distributed_warning(caplog):
    """The full 3-of-3 match fires the cliff warning and bumps the counter."""
    caplog.set_level(logging.WARNING, logger=g.logger.name)
    fired = g.check_load_time_guardrails(
        _signal(is_moe=True, quant="mxfp4", world=2),
        hf_path="vendor/some-moe-mxfp4",
        alias="some-alias",
    )
    assert fired == ["mxfp4_moe_distributed"]
    assert g.snapshot()["mxfp4_moe_distributed_warnings_total"] == 1
    assert g.snapshot()["nvfp4_moe_warnings_total"] == 0
    # Warning carries the upstream issue link so operators can self-route.
    joined = " ".join(r.message for r in caplog.records)
    assert g.MLX_3402_URL in joined
    assert "mxfp4" in joined.lower() or "MXFP4" in joined


@pytest.mark.parametrize(
    "is_moe,quant,world,label",
    [
        # 2-of-3 misses: MoE off
        (False, "mxfp4", 4, "no_moe"),
        # 2-of-3 misses: quant off
        (True, None, 4, "no_mxfp4"),
        (True, "int4", 4, "wrong_quant"),
        # 2-of-3 misses: single-device
        (True, "mxfp4", 1, "single_device"),
        # 1-of-3 / 0-of-3
        (False, None, 1, "none"),
        (False, "mxfp4", 1, "mxfp4_only"),
        (True, None, 1, "moe_only"),
        (False, None, 4, "distributed_only"),
    ],
)
def test_two_of_three_does_not_fire_cliff(is_moe, quant, world, label):
    """Any 2-of-3 combo stays silent for the cliff guardrail."""
    fired = g.check_load_time_guardrails(
        _signal(is_moe=is_moe, quant=quant, world=world),
        hf_path=f"vendor/{label}",
        alias=label,
    )
    # The MoE+NVFP4 guardrail is independent — it must not fire for any
    # of these cases either (none of them carry nvfp4).
    assert "mxfp4_moe_distributed" not in fired
    assert "nvfp4_moe" not in fired
    assert g.snapshot()["mxfp4_moe_distributed_warnings_total"] == 0
    assert g.snapshot()["nvfp4_moe_warnings_total"] == 0


def test_nvfp4_moe_fires_regardless_of_device_count(caplog):
    """mlx#2962 dynamic-range loss bites even single-device."""
    caplog.set_level(logging.WARNING, logger=g.logger.name)
    for world in (1, 2, 8):
        g.reset_for_tests()
        caplog.clear()
        fired = g.check_load_time_guardrails(
            _signal(is_moe=True, quant="nvfp4", world=world),
            hf_path="vendor/nvfp4-moe",
            alias="alias-x",
        )
        assert fired == ["nvfp4_moe"], f"world={world}"
        assert g.snapshot()["nvfp4_moe_warnings_total"] == 1
        joined = " ".join(r.message for r in caplog.records)
        assert g.MLX_2962_URL in joined


def test_nvfp4_without_moe_silent():
    """NVFP4 + dense (non-MoE) does not trip the guardrail."""
    fired = g.check_load_time_guardrails(
        _signal(is_moe=False, quant="nvfp4", world=4),
        hf_path="vendor/dense-nvfp4",
        alias=None,
    )
    assert fired == []
    assert g.snapshot()["nvfp4_moe_warnings_total"] == 0


def test_counters_monotonic_across_repeated_fires():
    """Counters must accumulate across repeated three-tuple loads."""
    for _ in range(3):
        g.check_load_time_guardrails(
            _signal(is_moe=True, quant="mxfp4", world=2),
            hf_path="vendor/x",
            alias="x",
        )
    assert g.snapshot()["mxfp4_moe_distributed_warnings_total"] == 3


# ---------------------------------------------------------------------------
# check_from_profile — adapter from server.load_model() call site
# ---------------------------------------------------------------------------


class _FakeProfile:
    """Minimal stand-in for AliasProfile — just is_moe + hf_path."""

    def __init__(self, *, hf_path, is_moe):
        self.hf_path = hf_path
        self.is_moe = is_moe


def test_check_from_profile_three_tuple(monkeypatch, caplog):
    """Adapter path: AliasProfile + mocked distributed size triggers the cliff."""
    caplog.set_level(logging.WARNING, logger=g.logger.name)
    monkeypatch.setattr(g, "_detect_distributed_world_size", lambda: 4)
    profile = _FakeProfile(
        hf_path="nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx",
        is_moe=True,
    )
    fired = g.check_from_profile(
        model_name="qwen3.5-122b-mxfp4",
        profile=profile,
        alias="qwen3.5-122b-mxfp4",
    )
    assert fired == ["mxfp4_moe_distributed"]
    assert g.snapshot()["mxfp4_moe_distributed_warnings_total"] == 1


def test_check_from_profile_single_device_silent(monkeypatch):
    """Adapter path: single-device default (world=1) stays silent on MoE+MXFP4."""
    monkeypatch.setattr(g, "_detect_distributed_world_size", lambda: 1)
    profile = _FakeProfile(
        hf_path="nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx",
        is_moe=True,
    )
    fired = g.check_from_profile(
        model_name="qwen3.5-122b-mxfp4",
        profile=profile,
    )
    assert fired == []
    assert g.snapshot()["mxfp4_moe_distributed_warnings_total"] == 0


def test_check_from_profile_no_profile_treated_as_non_moe(monkeypatch):
    """A bare HF path with no alias is conservatively treated as non-MoE."""
    monkeypatch.setattr(g, "_detect_distributed_world_size", lambda: 4)
    # Path carries mxfp4 + we're distributed, but profile=None → no is_moe
    # signal, so the guardrail stays silent. The conservative bias is
    # documented in the module docstring.
    fired = g.check_from_profile(
        model_name="some/unknown-mxfp4-model",
        profile=None,
    )
    assert fired == []


def test_check_from_profile_nvfp4_moe(monkeypatch, caplog):
    """Adapter path: NVFP4 + MoE fires regardless of mocked device count."""
    caplog.set_level(logging.WARNING, logger=g.logger.name)
    monkeypatch.setattr(g, "_detect_distributed_world_size", lambda: 1)
    profile = _FakeProfile(
        hf_path="vendor/some-nvfp4-moe",
        is_moe=True,
    )
    fired = g.check_from_profile(
        model_name="vendor/some-nvfp4-moe",
        profile=profile,
    )
    assert fired == ["nvfp4_moe"]


# ---------------------------------------------------------------------------
# /metrics rendering — counters surface in Prometheus output
# ---------------------------------------------------------------------------


def test_metrics_endpoint_exposes_guardrail_counters():
    """The Prometheus exposition format must carry both new counters."""
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import metrics as metrics_module

    # Trip both counters once each so the rendered body shows non-zero
    # values rather than just the HELP/TYPE lines.
    g.check_load_time_guardrails(
        _signal(is_moe=True, quant="mxfp4", world=2),
        hf_path="vendor/a",
        alias="a",
    )
    g.check_load_time_guardrails(
        _signal(is_moe=True, quant="nvfp4", world=1),
        hf_path="vendor/b",
        alias="b",
    )

    cfg = get_config()
    body = metrics_module._render_prometheus(cfg)

    assert "rapid_mlx_mxfp4_moe_distributed_warnings_total" in body
    assert "rapid_mlx_nvfp4_moe_warnings_total" in body
    # Counter type discoverable via TYPE line — required by Prometheus
    # exposition format spec.
    assert (
        "# TYPE rapid_mlx_mxfp4_moe_distributed_warnings_total counter" in body
    )
    assert "# TYPE rapid_mlx_nvfp4_moe_warnings_total counter" in body
    # Values were both bumped to 1.
    assert "rapid_mlx_mxfp4_moe_distributed_warnings_total 1" in body
    assert "rapid_mlx_nvfp4_moe_warnings_total 1" in body
