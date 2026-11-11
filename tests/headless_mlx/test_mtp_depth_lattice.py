"""Depth policy and serving wiring without Metal or model weights."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice.mtp import resolve_max_depth

ENV = "RAPID_MLX_MTP_MAX_DEPTH"


def context(**changes):
    return replace(
        lattice.runtime_context({}, mode="mtp", has_mtp_head=True), **changes
    )


def model(**changes):
    return SimpleNamespace(
        mtp=object(), mtp_forward=lambda: None, make_mtp_cache=lambda: [], **changes
    )


@pytest.mark.parametrize("raw,expected", [(str(k), k) for k in range(1, 5)])
def test_integer_override(raw, expected):
    decision = lattice.resolve("mtp_max_depth", context(), environ={ENV: raw})
    assert type(decision.value) is int
    assert (decision.value, decision.source) == (expected, "env")


@pytest.mark.parametrize("raw", ["", "yes", "false", "0", "-1", "5", "2.5", "x"])
def test_invalid_override_defers(raw):
    assert lattice.resolve("mtp_max_depth", context(), environ={ENV: raw}) == (
        lattice.Decision(1, "default")
    )


@pytest.mark.parametrize(
    "change", [{"has_mtp_head": False}, {"mode": "plain"}, {"batch": 2}]
)
def test_override_cannot_bypass_capability_or_scope(change):
    decision = lattice.resolve("mtp_max_depth", context(**change), environ={ENV: "4"})
    assert decision.value == 1 and decision.source == "capability"


@pytest.mark.parametrize("chip", ["unknown", "Apple M3 Ultra", "Apple M5 Max"])
def test_unmeasured_default(chip):
    ctx = lattice.runtime_context({"device_name": chip}, mode="mtp", has_mtp_head=True)
    assert lattice.resolve("mtp_max_depth", ctx, environ={}).value == 1


def test_typed_local_and_table_precedence(monkeypatch):
    from rapid_mlx.lattice.defaults import Profile

    ctx = context(hardware=lattice.HardwareProfile(gpu_generation=15, tier="Ultra"))
    cell = dict(
        id="test-depth",
        hardware=[dict(gpu_generation=15, tier="Ultra")],
        knobs={"mtp_max_depth": 3},
        provenance={"evidence": "test only"},
    )
    monkeypatch.setattr(lattice, "builtin_profile", lambda: Profile("test", (cell,)))
    assert lattice.resolve("mtp_max_depth", ctx, environ={}).value == 3
    for invalid in (True, 0, 5, "2"):
        monkeypatch.setattr(
            lattice,
            "local_calibration",
            lambda c, value=invalid: {"mtp_max_depth": value},
        )
        assert (
            lattice.resolve("mtp_max_depth", ctx, environ={}).source
            == "builtin:test-depth"
        )
    monkeypatch.setattr(lattice, "local_calibration", lambda c: {"mtp_max_depth": 2})
    assert lattice.resolve("mtp_max_depth", ctx, environ={}).source == "local"
    assert lattice.resolve("mtp_max_depth", ctx, environ={ENV: "4"}).value == 4


def test_adapter_preserves_explicit_and_requires_real_head(monkeypatch):
    import mlx.core as mx

    probe = Mock(side_effect=AssertionError("no hardware probe"))
    monkeypatch.setattr(mx, "device_info", probe, raising=False)
    monkeypatch.setenv(ENV, "4")
    assert resolve_max_depth(model(), 2, explicit=True) == 2
    assert resolve_max_depth(object(), 1, explicit=False) == 1
    probe.assert_not_called()


def test_adapter_defaults_and_override(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(mx, "device_info", lambda: {}, raising=False)
    monkeypatch.delenv(ENV, raising=False)
    assert resolve_max_depth(model(), 1, explicit=False) == 1
    assert resolve_max_depth(model(), 3, explicit=False) == 3
    monkeypatch.setenv(ENV, "4")
    assert (
        resolve_max_depth(
            model(args=SimpleNamespace(model_type="other")), 1, explicit=False
        )
        == 4
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_eligibility_carries_explicit_depth(explicit):
    from rapid_mlx.cli import _apply_mtp_cli_model_type_reconciliation
    from rapid_mlx.scheduler import SchedulerConfig

    config = SchedulerConfig(spec_decode="mtp", mtp_max_k=1)
    _apply_mtp_cli_model_type_reconciliation(
        scheduler_config=config,
        hf_cfg_eligibility={"model_type": "qwen3_5_moe"},
        requested_depth=1,
        explicit_depth=explicit,
    )
    assert config.mtp_depth_explicit is explicit


@pytest.mark.parametrize("explicit", [False, True])
def test_scheduler_creation_carries_explicit_depth(monkeypatch, explicit):
    from rapid_mlx import scheduler as module
    from rapid_mlx.request import SamplingParams

    bg = SimpleNamespace()
    install = Mock(return_value=True)
    monkeypatch.setattr(module, "BatchGenerator", lambda **kwargs: bg)
    monkeypatch.setattr(module, "make_sampler", lambda **kwargs: object())
    monkeypatch.setattr(module, "_install_mtp_vendored", install)
    monkeypatch.setattr(module, "_install_dense_sampler_fastpath", lambda bg: None)
    monkeypatch.setattr(
        "rapid_mlx.singleton_cache_fastpath.install_singleton_cache_fastpath",
        lambda: None,
    )
    scheduler = module.Scheduler.__new__(module.Scheduler)
    scheduler.model = model()
    scheduler.model_config = None
    scheduler.requests = {}
    scheduler.uid_to_request_id = {}
    scheduler.uid_to_request_processors = {}
    scheduler._get_stop_tokens = lambda: set()
    scheduler.config = module.SchedulerConfig(
        spec_decode="mtp", mtp_depth_explicit=explicit
    )
    assert scheduler._create_batch_generator(SamplingParams(max_tokens=4)) is bg
    assert install.call_args.kwargs["explicit_depth"] is explicit


def test_installed_gate_resolves_before_checkpoint_limit(monkeypatch, caplog):
    import mlx.core as mx

    from rapid_mlx import scheduler as module
    from rapid_mlx.spec_decode.mtp import draft_k_controller_v2 as controller

    monkeypatch.setattr(mx, "device_info", lambda: {}, raising=False)
    monkeypatch.setenv(ENV, "4")

    class BoundaryReachedError(Exception):
        pass

    def stop_at_key(m):
        raise BoundaryReachedError

    monkeypatch.setattr(controller, "derive_controller_key", stop_at_key)
    bg = SimpleNamespace(_generation_batch=SimpleNamespace())
    with pytest.raises(BoundaryReachedError):
        module._install_mtp_vendored(bg, model(mtp_max_speculative_tokens=2), max_k=1)
    assert "caps speculative depth at K=2 (requested K=4)" in caplog.text


def test_calibration_uses_every_resolved_width(monkeypatch):
    import contextlib
    import sys
    from types import ModuleType

    from rapid_mlx.spec_decode.mtp import reproducible_depth as rd

    # Keep the mutable import stubs intact for later lifecycle tests.
    mx = ModuleType("mlx.core")
    mx.device_info = lambda: {}
    mx.zeros = lambda shape, **kwargs: shape
    mx.eval = lambda *args: None
    mx.stream = lambda stream: contextlib.nullcontext()
    mx.default_stream = lambda device: None
    mx.default_device = lambda: None
    mx.uint32 = object()
    mlx = ModuleType("mlx")
    mlx.core = mx
    cache = ModuleType("mlx_lm.models.cache")
    cache.make_prompt_cache = lambda model: []
    models = ModuleType("mlx_lm.models")
    models.cache = cache
    for name, module in (
        ("mlx", mlx),
        ("mlx.core", mx),
        ("mlx_lm.models", models),
        ("mlx_lm.models.cache", cache),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv(ENV, "4")
    cap = resolve_max_depth(model(), 1, explicit=False)
    seen = []

    class Hidden:
        def __getitem__(self, index):
            return self

    class Target:
        def __call__(self, tokens, **kwargs):
            seen.append(tokens[1])
            return None, Hidden()

        def make_mtp_cache(self):
            return []

        def mtp_forward(self, *args):
            return None

    result = rd.time_target_forwards(Target(), cap)
    assert len(result[1]) == cap
    assert seen == list(range(1, cap + 2)) * (
        rd.TIMING_ROUNDS + rd.TIMING_WARMUP_ROUNDS
    )


@pytest.mark.parametrize("raw,value", [("0", False), ("1", True)])
def test_existing_boolean_override_keeps_its_type(raw, value):
    ctx = lattice.runtime_context({}, mode="plain")
    decision = lattice.resolve("lane_simd_probe", ctx, environ={})
    assert type(decision.value) is bool
    ctx = lattice.runtime_context(
        {"device_name": "Apple M3 Ultra"},
        model=lattice.ModelProfile("qwen35_moe", *lattice.QWEN35_AFFINE4),
    )
    decision = lattice.resolve(
        "qwen35_affine4_bank_views",
        ctx,
        environ={"RAPID_MLX_QWEN35_AFFINE4_BANK_VIEWS": raw},
    )
    assert decision == lattice.Decision(value, "env")
