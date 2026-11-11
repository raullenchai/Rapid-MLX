"""Effective eager scheduling equivalence and lattice integration without Metal."""

import itertools
import json
import subprocess
import sys
import textwrap
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice import defaults
from tests.test_lattice_gate_characterization import eager  # noqa: F401

NAME = "qwen35_eager_layer_dispatch"


@pytest.fixture(autouse=True)
def clean_policy(monkeypatch):
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)


def old_effective(layer, shape, enabled=True, compiled=False):
    norm_shape = tuple(
        getattr(
            getattr(getattr(layer, "input_layernorm", None), "weight", None),
            "shape",
            (),
        )
    )
    mlp = getattr(layer, "mlp", None)
    return (
        enabled
        and not compiled
        and len(shape) >= 2
        and int(shape[0]) * int(shape[1]) <= 64
        and norm_shape == (2048,)
        and getattr(mlp, "num_experts", None) == 256
        and getattr(mlp, "top_k", None) == 8
    )


def test_effective_cartesian_grid(eager, monkeypatch):
    module, Layer, submitted = eager
    for hidden, experts, top_k, shape, enabled, compiled in itertools.product(
        (None, 2047, 2048, 2049),
        (None, 255, 256, 257),
        (None, 7, 8, 9),
        (
            (1, 0, 8),
            (1, 1, 8),
            (1, 64, 8),
            (1, 65, 8),
            (8, 8, 8),
            (5, 13, 8),
            (0, 64, 8),
            (8,),
            (),
        ),
        (False, True),
        (False, True),
    ):
        layer = Layer()
        layer.input_layernorm = SimpleNamespace(weight=SimpleNamespace(shape=(hidden,)))
        layer.mlp = SimpleNamespace(num_experts=experts, top_k=top_k)
        monkeypatch.setattr(module, "_ENABLED", enabled)
        monkeypatch.setattr(
            module, "_in_compiled_decode", lambda compiled=compiled: compiled
        )
        submitted.clear()
        output = SimpleNamespace(shape=shape)
        assert layer(output) is output
        assert bool(submitted) == old_effective(layer, shape, enabled, compiled)


def test_unknown_hardware_and_requirement_blocks(monkeypatch):
    ctx = lattice.LatticeContext(
        lattice.HardwareProfile(),
        "unknown",
        "unknown",
        lattice.ModelProfile(
            "qwen3_5_decoder_layer", hidden_size=2048, num_experts=256, top_k=8
        ),
        flattened_rows=64,
    )
    assert lattice.KNOBS[NAME].validator is lattice.boolean
    assert lattice.KNOBS[NAME].default is False
    for hardware in (
        lattice.HardwareProfile(),
        lattice.HardwareProfile(gpu_generation=99, architecture="future"),
    ):
        assert (
            lattice.resolve(NAME, replace(ctx, hardware=hardware), environ={}).value
            is True
        )
    force = {"RAPID_MLX_BENCHMARK_KNOBS": json.dumps({NAME: True})}
    for blocked in (
        replace(ctx, mode="compiled"),
        replace(ctx, flattened_rows=None),
        replace(ctx, model=replace(ctx.model, hidden_size=None)),
        replace(ctx, model=replace(ctx.model, num_experts=0)),
        replace(ctx, model=replace(ctx.model, top_k=0)),
        replace(ctx, model=replace(ctx.model, family="unknown")),
    ):
        assert lattice.resolve(NAME, blocked, environ=force).source == "capability"


@pytest.mark.parametrize("override", [None, False, True])
@pytest.mark.parametrize("legacy_enabled", [False, True])
def test_overlay_and_candidate_override(monkeypatch, override, legacy_enabled):
    # The new overlay/benchmark controls intentionally select policy; the old
    # import-time master switch still wins over all of them.
    monkeypatch.setattr(
        lattice,
        "overlay_profile",
        lambda: defaults.Profile(
            "test",
            (
                {
                    "id": "disable",
                    "knobs": {NAME: False},
                    "workload": {"flattened_rows": [0, 32]},
                },
            ),
        ),
    )
    if override is not None:
        monkeypatch.setenv("RAPID_MLX_BENCHMARK_KNOBS", json.dumps({NAME: override}))
    module, Layer, submitted = eager.__wrapped__(monkeypatch)
    monkeypatch.setattr(module, "_ENABLED", legacy_enabled)
    for rows in (0, 1, 32, 33, 64, 65):
        submitted.clear()
        output = SimpleNamespace(shape=(1, rows, 8))
        assert Layer()(output) is output
        policy = override if override is not None else 32 < rows <= 64
        assert bool(submitted) == (legacy_enabled and policy)


def test_cache_hits_skip_resolution_and_refresh_actual_facts(eager, monkeypatch):
    module, Layer, submitted = eager
    monkeypatch.setattr(module, "_ENABLED", True)
    actual = module.resolve
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1])
        return actual(*args, **kwargs)

    monkeypatch.setattr(module, "resolve", counted)
    layer = Layer()
    output = SimpleNamespace(shape=(8, 8, 8))
    for _ in range(100):
        layer(output)
    assert len(submitted) == 100 and len(calls) == 1
    assert calls[0].flattened_rows == 64
    layer.mlp = SimpleNamespace(num_experts=255, top_k=8)
    layer(output)
    assert len(submitted) == 100 and len(calls) == 2


def test_rank_and_each_missing_fact(eager, monkeypatch):
    module, Layer, submitted = eager
    monkeypatch.setattr(module, "_ENABLED", True)
    for norm_shape in ((), (2048, 1), (1, 2048)):
        layer = Layer()
        layer.input_layernorm = SimpleNamespace(
            weight=SimpleNamespace(shape=norm_shape)
        )
        output = SimpleNamespace(shape=(1, 1, 8))
        layer(output)
        assert not old_effective(layer, output.shape) and not submitted
    for mlp in (SimpleNamespace(top_k=8), SimpleNamespace(num_experts=256)):
        layer = Layer()
        layer.mlp = mlp
        layer(output)
        assert not old_effective(layer, output.shape) and not submitted


def test_install_fires_on_real_serve_import_path():
    script = textwrap.dedent("""
        import sys
        from types import ModuleType
        # Inert tensor imports only; production tokenizer runs the installer.
        sys.modules['mlx'] = None
        sys.modules['mlx_lm'] = None
        from tests.headless_mlx._headless_mlx import install_headless_mlx_import_stubs
        install_headless_mlx_import_stubs()
        q = ModuleType('mlx_lm.models.qwen3_5')
        class DecoderLayer:
            def __call__(self, value): return value
        class TextModel:
            def sanitize(self, weights): return weights
        q.DecoderLayer, q.TextModel = DecoderLayer, TextModel
        sys.modules[q.__name__] = q
        sys.modules['mlx_lm.models'].qwen3_5 = q
        import rapid_mlx.utils.tokenizer
        from rapid_mlx.patches.qwen3_5_eager_dispatch import is_installed
        assert is_installed()
        assert q._RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED is True
        assert q.DecoderLayer.__call__ is not q._RAPID_MLX_ORIG_DECODER_LAYER_CALL
        print('OK')
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.parametrize(
    "device,applies",
    [
        ({"device_name": "Apple M3 Ultra", "architecture": "applegpu_g15d"}, True),
        ({"device_name": "Apple M4 Max", "architecture": "applegpu_g16s"}, False),
        ({}, False),
    ],
)
def test_chip_specific_cells_see_the_real_device(monkeypatch, device, applies):
    """The resolver uses the installed device, so chip-specific cells apply."""
    monkeypatch.setattr(
        lattice,
        "overlay_profile",
        lambda: defaults.Profile(
            "test",
            (
                {
                    "id": "m3-ultra-off",
                    "hardware": [{"gpu_generation": 15, "tier": "Ultra"}],
                    "knobs": {NAME: False},
                },
            ),
        ),
    )
    from tests import test_lattice_gate_characterization as characterization

    class DeviceModule(characterization.ModuleType):
        def device_info(self):
            return device

    monkeypatch.setattr(characterization, "ModuleType", DeviceModule)
    module, Layer, submitted = eager.__wrapped__(monkeypatch)
    monkeypatch.setattr(module, "_ENABLED", True)
    output = SimpleNamespace(shape=(1, 1, 8))
    assert Layer()(output) is output
    assert bool(submitted) == (not applies)
