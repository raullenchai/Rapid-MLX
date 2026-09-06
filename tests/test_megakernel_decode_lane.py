# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the opt-in plain-decode megakernel lane gate.

The lane's kernel lives in the mlx-lm build, not here, so these tests are
pure: they exercise Rapid's *decision surface* (config parsing, capability
probe, per-model arming, and the width-1-plain routing gate) with hand-built
fakes, never loading a model or touching the GPU. The design contract under
test is fail-closed — every path that cannot use the lane must leave the
ordinary ``generate_step`` decode in place.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_mlx import _megakernel_decode_lane as mk


@pytest.fixture(autouse=True)
def _reset_counters():
    mk.reset_for_tests()
    yield
    mk.reset_for_tests()


# ---------------------------------------------------------------------------
# MegakernelLaneConfig.from_env
# ---------------------------------------------------------------------------
def test_config_defaults_off():
    cfg = mk.MegakernelLaneConfig.from_env({})
    assert cfg.enabled is False
    assert cfg.geometry == "auto"
    assert cfg.max_context == 0


def test_config_reads_env():
    cfg = mk.MegakernelLaneConfig.from_env(
        {
            mk.ENV_ENABLE: "1",
            mk.ENV_GEOMETRY: "Qwen4_Exp",
            mk.ENV_MAX_CONTEXT: "65536",
        }
    )
    assert cfg.enabled is True
    assert cfg.geometry == "qwen4_exp"  # normalised to lower-case
    assert cfg.max_context == 65536


def test_config_bad_max_context_is_zero():
    cfg = mk.MegakernelLaneConfig.from_env(
        {mk.ENV_ENABLE: "1", mk.ENV_MAX_CONTEXT: "not-a-number"}
    )
    assert cfg.max_context == 0


@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True),
                                          ("on", True), ("yes", True),
                                          ("0", False), ("off", False),
                                          ("", False), (None, False)])
def test_truthy(raw, expected):
    assert mk._truthy(raw) is expected


# ---------------------------------------------------------------------------
# configure_process_env — only enables, never disables
# ---------------------------------------------------------------------------
def test_configure_process_env_noop_when_disabled(monkeypatch):
    monkeypatch.delenv(mk._MLX_MASTER_ENV, raising=False)
    mk.configure_process_env(mk.MegakernelLaneConfig(enabled=False))
    import os

    assert mk._MLX_MASTER_ENV not in os.environ


def test_configure_process_env_sets_switches(monkeypatch):
    import os

    for name in (mk._MLX_MASTER_ENV, mk._MLX_LANE_ENV, mk._MLX_MAX_WIDTH_ENV):
        monkeypatch.delenv(name, raising=False)
    mk.configure_process_env(mk.MegakernelLaneConfig(enabled=True))
    assert os.environ[mk._MLX_MASTER_ENV] == "1"
    assert os.environ[mk._MLX_LANE_ENV] == "1"
    assert os.environ[mk._MLX_MAX_WIDTH_ENV] == "1"


def test_configure_process_env_does_not_clobber_operator_value(monkeypatch):
    import os

    monkeypatch.setenv(mk._MLX_MASTER_ENV, "0")  # operator turned it off
    mk.configure_process_env(mk.MegakernelLaneConfig(enabled=True))
    assert os.environ[mk._MLX_MASTER_ENV] == "0"  # setdefault leaves it


# ---------------------------------------------------------------------------
# enable_for_model — fail-closed arming
# ---------------------------------------------------------------------------
def _fake_model(model_type):
    return SimpleNamespace(
        language_model=SimpleNamespace(args=SimpleNamespace(model_type=model_type))
    )


def test_enable_disabled_config():
    d = mk.enable_for_model(mk.MegakernelLaneConfig(enabled=False), _fake_model("x"))
    assert d.route is False


def test_enable_unavailable_build(monkeypatch):
    monkeypatch.setattr(mk, "lane_available", lambda: False)
    d = mk.enable_for_model(mk.MegakernelLaneConfig(enabled=True), _fake_model("x"))
    assert d.route is False
    assert "stock wheel" in d.reason
    assert mk.snapshot_counters()["rapid_mlx_megakernel_lane_unavailable_total"] == 1


def test_enable_no_geometry(monkeypatch):
    monkeypatch.setattr(mk, "lane_available", lambda: True)
    monkeypatch.setattr(mk, "geometry_name_for_model", lambda m: None)
    d = mk.enable_for_model(mk.MegakernelLaneConfig(enabled=True), _fake_model("llama"))
    assert d.route is False
    assert "no registered megakernel geometry" in d.reason


def test_enable_geometry_mismatch(monkeypatch):
    monkeypatch.setattr(mk, "lane_available", lambda: True)
    monkeypatch.setattr(mk, "geometry_name_for_model", lambda m: "qwen4_exp")
    cfg = mk.MegakernelLaneConfig(enabled=True, geometry="qwen36_35b_a3b")
    d = mk.enable_for_model(cfg, _fake_model("qwen4_exp_text"))
    assert d.route is False
    assert "does not match" in d.reason


def test_enable_auto_geometry_arms(monkeypatch):
    monkeypatch.setattr(mk, "lane_available", lambda: True)
    monkeypatch.setattr(mk, "geometry_name_for_model", lambda m: "qwen4_exp")
    d = mk.enable_for_model(mk.MegakernelLaneConfig(enabled=True), _fake_model("q"))
    assert d.route is True
    assert "qwen4_exp" in d.reason


def test_enable_pinned_geometry_match(monkeypatch):
    monkeypatch.setattr(mk, "lane_available", lambda: True)
    monkeypatch.setattr(mk, "geometry_name_for_model", lambda m: "qwen36_35b_a3b")
    cfg = mk.MegakernelLaneConfig(enabled=True, geometry="qwen36_35b_a3b")
    d = mk.enable_for_model(cfg, _fake_model("qwen3_5_moe_text"))
    assert d.route is True


# ---------------------------------------------------------------------------
# route_decision — width-1 plain gate
# ---------------------------------------------------------------------------
def _plain_params():
    return SimpleNamespace(temperature=0.7, top_p=0.9)


def _route(**over):
    base = dict(
        config=mk.MegakernelLaneConfig(enabled=True),
        armed=True,
        context_len=1000,
        max_tokens=128,
        batch_size=1,
        is_speculative=False,
        sampling_params=_plain_params(),
        capacity=262144,
    )
    base.update(over)
    cfg = base.pop("config")
    return mk.route_decision(cfg, **base)


def test_route_plain_within_profile():
    assert _route().route is True


def test_route_not_armed():
    assert _route(armed=False).route is False


def test_route_batch_declined():
    d = _route(batch_size=4)
    assert d.route is False and "batch" in d.reason
    assert mk.snapshot_counters()["rapid_mlx_megakernel_lane_declined_total"] == 1


def test_route_speculative_declined():
    d = _route(is_speculative=True)
    assert d.route is False and "speculative" in d.reason


@pytest.mark.parametrize("attr", ["guided_json", "guided_grammar", "guided_regex",
                                  "guided_choice", "grammar", "logits_processors",
                                  "response_format"])
def test_route_tool_constrained_declined(attr):
    params = SimpleNamespace(temperature=0.7, top_p=0.9)
    setattr(params, attr, object())
    d = _route(sampling_params=params)
    assert d.route is False and (
        "tools" in d.reason or "guided" in d.reason
    )


def test_route_over_capacity_declined():
    d = _route(context_len=262100, max_tokens=100, capacity=262144)
    assert d.route is False and "capacity" in d.reason


def test_route_over_operator_cap_declined():
    cfg = mk.MegakernelLaneConfig(enabled=True, max_context=8192)
    d = _route(config=cfg, context_len=9000)
    assert d.route is False and "operator cap" in d.reason


def test_route_unbounded_max_tokens_ok():
    # max_tokens<=0 (unbounded) must not blow the capacity projection.
    assert _route(max_tokens=-1).route is True


# ---------------------------------------------------------------------------
# note_engagement — counter bookkeeping from generate_step's status dict
# ---------------------------------------------------------------------------
def test_note_engagement_used():
    assert mk.note_engagement({"used": True, "geometry": "qwen4_exp"}) is True
    assert mk.snapshot_counters()["rapid_mlx_megakernel_lane_engaged_total"] == 1


def test_note_engagement_declined():
    assert mk.note_engagement({"decline_reason": "another lane active"}) is False
    assert mk.snapshot_counters()["rapid_mlx_megakernel_lane_declined_total"] == 1


def test_note_engagement_empty():
    assert mk.note_engagement(None) is False
    assert mk.note_engagement({}) is False
    assert mk.snapshot_counters()["rapid_mlx_megakernel_lane_declined_total"] == 2


# ---------------------------------------------------------------------------
# lane_available — probe is honest about the installed build
# ---------------------------------------------------------------------------
def test_lane_available_is_bool():
    assert isinstance(mk.lane_available(), bool)
