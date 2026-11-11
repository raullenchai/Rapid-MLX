"""Effective attention dispatch equality; no numerical or Metal qualification."""

import importlib
import itertools
from unittest.mock import Mock

import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice import attention, defaults, profiles
from tests.headless_mlx.test_lattice_gate_characterization import (
    ReachedDispatchError,
    Tensor,
    affine_call,
    fake_mx,
)

NAMES = (
    "affine8_attention_min_kv",
    "qsa_stage1_min_query",
    "qsa_stage1_min_kv",
    "qsa_indexed_splitk_min_kv",
    "qsa_block_sparse_min_query",
    "qsa_block_sparse_min_kv",
)
DEVICES: tuple[dict[str, str], ...] = (
    {},
    {"device_name": "Apple M3 Ultra", "architecture": "applegpu_g15d"},
    {"device_name": "Apple M9 Max", "architecture": "applegpu_g21x"},
)
FUNCTIONS = {
    "qsa_stage1": "qsa_stage1_decline_reason",
    "qsa_indexed_splitk": "indexed_splitk_decline_reason",
    "qsa_block_sparse": "block_sparse_decline_reason",
}


@pytest.fixture(autouse=True)
def clean_policy(monkeypatch):
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)


def setup(monkeypatch, name, device, *, version="0.32.2", metal=True):
    module = importlib.import_module("rapid_mlx.kernels." + name)
    mx = fake_mx()
    mx.device_info.return_value = device
    mx.metal.is_available.return_value = metal
    mx.__version__ = version
    monkeypatch.setattr(module, "mx", mx)
    monkeypatch.setattr(
        module, "_thresholds", attention.AttentionThresholds(lambda: mx)
    )
    monkeypatch.setattr(profiles, "mlx_version", lambda: version)
    if name == "affine8_attention":
        monkeypatch.setattr(module, "_DISABLED", False)
        monkeypatch.setattr(module, "_qualified_runtime", lambda: True)
        monkeypatch.setattr(
            module, "_kernel", lambda: Mock(side_effect=ReachedDispatchError)
        )
        monkeypatch.delenv("RAPID_MLX_AFFINE8_ATTENTION", raising=False)
    else:
        monkeypatch.setenv(module.ENABLE_ENV, "1")
        if name != "qsa_block_sparse":
            monkeypatch.setattr(module, "_mlx_version", lambda: version)
            monkeypatch.setattr(
                module,
                "_metal_architecture",
                lambda: device.get("architecture", "unknown"),
            )
    return module


def affine_dispatched(module, width, kv):
    try:
        assert affine_call(module, width, kv) is None
    except ReachedDispatchError:
        return True
    return False


@pytest.mark.parametrize("device", DEVICES)
def test_affine_effective_boundaries(monkeypatch, device):
    # Qualification is independently characterized unchanged. Force admission
    # here to prove universal crossover behavior even on unknown/future devices.
    module = setup(monkeypatch, "affine8_attention", device)
    for width, delta in itertools.product((*range(1, 10), 63, 64, 65), (-1, 0, 1)):
        threshold = 16384 if width == 1 else 8192 if width < 4 else 2048
        kv = threshold + delta
        old = 1 <= width <= 8 and kv >= threshold
        assert affine_dispatched(module, width, kv) == old, (device, width, kv)


def old_qsa_reason(name, query, kv, *, enabled, training, batch, version, arch, metal):
    """Frozen pre-migration predicates, including refusal order."""
    if not enabled:
        return "disabled"
    if training:
        return "training"
    if name != "qsa_block_sparse" and batch != 1:
        return "batch size is not qualified"
    if name == "qsa_indexed_splitk":
        if query not in (1, 3):
            return "query length is not qualified"
        minimum = 65536 if query == 1 else 16384
    else:
        if query < 64:
            return "query below crossover"
        minimum = 65024 if name == "qsa_stage1" else 16384
    if kv < minimum:
        return "physical KV below crossover"
    if not metal:
        return "Metal runtime unavailable"
    if name != "qsa_block_sparse":
        if version != "0.32.2":
            return f"unqualified MLX version {version}"
        allowed = (
            ("applegpu_g15d", "applegpu_g14s")
            if name == "qsa_stage1"
            else ("applegpu_g15d",)
        )
        if arch not in allowed:
            return f"unqualified Metal architecture {arch}"
    return None


@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("device", (*DEVICES, {"architecture": "applegpu_g14s"}))
@pytest.mark.parametrize("version", ("0.32.2", "0.32.3", "unknown"))
def test_qsa_effective_grid(monkeypatch, name, device, version):
    module = setup(monkeypatch, name, device, version=version)
    fn = getattr(module, FUNCTIONS[name])
    for query, delta, enabled, training, batch, metal in itertools.product(
        (*range(1, 10), 63, 64, 65),
        (-1, 0, 1),
        (False, True),
        (False, True),
        (1, 2),
        (False, True),
    ):
        minimum = (
            (65536 if query == 1 else 16384)
            if name == "qsa_indexed_splitk"
            else 65024
            if name == "qsa_stage1"
            else 16384
        )
        kv = minimum + delta
        monkeypatch.setenv(module.ENABLE_ENV, "1" if enabled else "0")
        module.mx.metal.is_available.return_value = metal
        kwargs = dict(training=training)
        if name != "qsa_block_sparse":
            kwargs["batch_size"] = batch
        actual = fn(query, kv, **kwargs)
        expected = old_qsa_reason(
            name,
            query,
            kv,
            enabled=enabled,
            training=training,
            batch=batch,
            version=version,
            arch=device.get("architecture", "unknown"),
            metal=metal,
        )
        assert actual == expected, (name, device, version, query, kv, kwargs)


def overlay(monkeypatch, tmp_path, knobs, **selectors):
    import json

    path = tmp_path / "attention.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version=1,
                profile_version="attention.1",
                generated_at="2026-11-11T00:00:00Z",
                cells=[
                    dict(
                        id="attention-test",
                        knobs=knobs,
                        provenance={"evidence": "test"},
                        **selectors,
                    )
                ],
            )
        )
    )
    profile = defaults.load_profile(path, lattice.KNOBS)
    assert len(profile.cells) == 1
    monkeypatch.setattr(lattice, "overlay_profile", lambda: profile)


@pytest.mark.parametrize(
    "name,query,knob,minimum",
    [
        ("affine8_attention", 4, NAMES[0], 2048),
        ("qsa_stage1", 64, NAMES[1], 64),
        ("qsa_stage1", 64, NAMES[2], 65024),
        ("qsa_indexed_splitk", 1, NAMES[3], 65536),
        ("qsa_indexed_splitk", 3, NAMES[3], 16384),
        ("qsa_block_sparse", 64, NAMES[4], 64),
        ("qsa_block_sparse", 64, NAMES[5], 16384),
    ],
)
def test_overlay_changes_effective_dispatch(
    monkeypatch, tmp_path, name, query, knob, minimum
):
    overlay(monkeypatch, tmp_path, {knob: minimum + 1})
    module = setup(monkeypatch, name, DEVICES[1])
    if name == "affine8_attention":
        assert not affine_dispatched(module, query, minimum)
        assert affine_dispatched(module, query, minimum + 1)
        monkeypatch.setenv("RAPID_MLX_AFFINE8_ATTENTION", "0")
        assert not affine_dispatched(module, query, minimum + 1)
        monkeypatch.setenv("RAPID_MLX_AFFINE8_ATTENTION", "1")
        monkeypatch.setattr(module, "_qualified_runtime", lambda: False)
        assert not affine_dispatched(module, query, minimum + 1)
    else:
        fn = getattr(module, FUNCTIONS[name])
        kwargs = {} if name == "qsa_block_sparse" else {"batch_size": 1}
        kv = 65536 if knob.endswith("query") else minimum
        assert fn(query, kv, **kwargs) is not None
        query += int(knob.endswith("query"))
        kv += int(knob.endswith("kv"))
        assert fn(query, kv, **kwargs) is None
        monkeypatch.delenv(module.ENABLE_ENV)
        assert fn(query, kv, **kwargs) == "disabled"
        monkeypatch.setenv(module.ENABLE_ENV, "1")
        assert fn(query, kv, training=True, **kwargs) == "training"


@pytest.mark.parametrize("device", DEVICES)
def test_chip_specific_threshold_uses_actual_facts(monkeypatch, tmp_path, device):
    overlay(
        monkeypatch,
        tmp_path,
        {NAMES[5]: 16385},
        hardware=[dict(gpu_generation=15, tier="Ultra")],
        software={"mlx": "===0.32.2"},
    )
    module = setup(monkeypatch, "qsa_block_sparse", device)
    expected = "physical KV below crossover" if device == DEVICES[1] else None
    assert module.block_sparse_decline_reason(64, 16384) == expected
    module.mx.device_info.assert_called_once()


def test_cache_hits_and_bounds(monkeypatch):
    module = setup(monkeypatch, "qsa_block_sparse", DEVICES[1])
    spy = Mock(wraps=attention.resolve)
    monkeypatch.setattr(attention, "resolve", spy)
    assert module.block_sparse_decline_reason(64, 16384) is None
    assert spy.call_count == 2
    for _ in range(10):
        assert module.block_sparse_decline_reason(64, 16384) is None
    assert spy.call_count == 2
    # Candidate settings freeze on first threshold use; legacy switch is dynamic.
    monkeypatch.setenv(
        "RAPID_MLX_BENCHMARK_KNOBS", '{"qsa_block_sparse_min_kv": 999999}'
    )
    assert module.block_sparse_decline_reason(65, 16384) is None
    for query in range(66, 400):
        module.block_sparse_decline_reason(query, 16384)
    assert module._thresholds.minimum.cache_info().currsize == 256
    module.mx.device_info.assert_called_once()
    monkeypatch.setenv(module.ENABLE_ENV, "0")
    assert module.block_sparse_decline_reason(64, 16384) == "disabled"


def test_threshold_cells_are_universal_and_integer_domains():
    cells = [
        c
        for c in defaults.builtin_profile().cells
        if set(NAMES).intersection(c["knobs"])
    ]
    assert len(cells) == 12
    for cell in cells:
        assert "hardware" not in cell and "software" not in cell
    for name in NAMES:
        assert lattice.KNOBS[name].valid(0) and lattice.KNOBS[name].valid(123456)
        for bad in (-1, True, False, 1.5, "64", None):
            assert not lattice.KNOBS[name].valid(bad)


def test_affine_local_constraints_survive_threshold_override(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_BENCHMARK_KNOBS", '{"affine8_attention_min_kv": 0}')
    module = setup(monkeypatch, "affine8_attention", DEVICES[1])
    packed = (
        Tensor((1, 2, 2048, 64), "u32"),
        Tensor((1, 2, 2048, 4)),
        Tensor((1, 2, 2048, 4)),
    )
    cases = [
        (Tensor((1, 16, 8, 256)), {}, packed),  # padded rows > 56
        (Tensor((1, 2, 4, 256), "u32"), {}, packed),
        (Tensor((1, 2, 4, 256)), {"mask": "additive"}, packed),
        (Tensor((1, 2, 4, 256)), {"bits": 4}, packed),
        (Tensor((1, 2, 4, 256)), {"group_size": 32}, packed),
        (Tensor((1, 2, 4, 256)), {}, (Tensor((1, 2, 2048, 63), "u32"), *packed[1:])),
    ]
    for q, kwargs, k in cases:
        assert (
            module.try_affine8_attention(
                q, k, packed, scale=1.0, **{"mask": "causal", **kwargs}
            )
            is None
        )


@pytest.mark.parametrize("name", ("qsa_stage1", "qsa_indexed_splitk"))
@pytest.mark.parametrize("version", ("0.32.1", "0.32.2", "0.32.3", "unknown"))
@pytest.mark.parametrize(
    "arch", ("applegpu_g14s", "applegpu_g15d", "applegpu_g21x", "unknown")
)
def test_zero_thresholds_cannot_bypass_runtime_qualification(
    monkeypatch, name, version, arch
):
    import json

    module = setup(monkeypatch, name, {"architecture": arch}, version=version)
    monkeypatch.setenv(
        "RAPID_MLX_BENCHMARK_KNOBS",
        json.dumps(
            {
                **dict.fromkeys(NAMES, 0),
                name: True,
            }
        ),
    )
    query = 64 if name == "qsa_stage1" else 3
    kwargs = dict(batch_size=1)
    reason = getattr(module, FUNCTIONS[name])(query, 1, **kwargs)
    allowed = arch == "applegpu_g15d" or (
        name == "qsa_stage1" and arch == "applegpu_g14s"
    )
    assert (reason is None) == (version == "0.32.2" and allowed)


@pytest.mark.parametrize(
    "name,query,kv",
    (
        ("affine8_attention", 3, 8191),
        ("qsa_stage1", 64, 65023),
        ("qsa_indexed_splitk", 3, 16383),
        ("qsa_block_sparse", 64, 16383),
    ),
)
def test_each_route_warm_thresholds_skip_resolution(monkeypatch, name, query, kv):
    module = setup(monkeypatch, name, DEVICES[1])

    def call():
        if name == "affine8_attention":
            return affine_dispatched(module, query, kv)
        kwargs = {} if name == "qsa_block_sparse" else dict(batch_size=1)
        return getattr(module, FUNCTIONS[name])(query, kv, **kwargs)

    expected = call()

    def unexpected(*args, **kwargs):
        pytest.fail("warm threshold cache entered resolution")

    monkeypatch.setattr(attention, "resolve", unexpected)
    for _ in range(10):
        assert call() == expected
    module.mx.device_info.assert_called_once()


def test_missing_device_info_preserves_universal_thresholds(monkeypatch):
    module = setup(monkeypatch, "qsa_block_sparse", {})
    module.mx.device_info.side_effect = RuntimeError("unavailable")
    assert (
        module.block_sparse_decline_reason(64, 16383) == "physical KV below crossover"
    )
    assert module.block_sparse_decline_reason(64, 16384) is None
    module.mx.device_info.assert_called_once()


def test_block_sparse_activation_log(monkeypatch):
    module = setup(monkeypatch, "qsa_block_sparse", {})
    logger = Mock()
    monkeypatch.setattr(module, "logger", logger)
    module._log_activation.cache_clear()
    try:
        module._log_activation()
        logger.info.assert_called_once()
    finally:
        module._log_activation.cache_clear()
