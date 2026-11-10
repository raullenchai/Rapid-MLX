"""Performance policy is portable, scoped, and subordinate to explicit switches."""

import os
import subprocess
import sys

import pytest

from rapid_mlx import tuning


@pytest.mark.parametrize(
    "name,key",
    [
        ("Apple M3 Ultra", "M3 Ultra"),
        ("Apple M4 Pro", "M4 Pro"),
        ("Apple M5 Max", "M5 Max"),
        ("Apple M1", "M1"),
        (" Apple M12 Ultra ", "M12 Ultra"),
        ("Apple M5 Extreme", "unknown"),
        ("Intel", "unknown"),
        (None, "unknown"),
    ],
)
def test_device_key(name, key):
    assert tuning.device_key({"device_name": name}) == key
    assert tuning.device_key({}) == "unknown"


def resolve(name="qwen35_affine4_bank_views", chip="Apple M3 Ultra", **kw):
    args = dict(
        family="qwen35_moe",
        geometry=tuning.QWEN35_AFFINE4,
        mode="plain",
        width=1,
        library_version="0.32.4",
    )
    args.update(kw)
    return tuning.enabled(name, {"device_name": chip}, **args)


@pytest.mark.parametrize("name", tuning.KNOBS)
def test_precedence_and_fallback(monkeypatch, name):
    switch = tuning.KNOBS[name].env
    monkeypatch.delenv(switch, raising=False)
    assert resolve(name)
    assert not resolve(name, chip="Apple M99 Max")
    assert not resolve(name, library_version="unmeasured")
    monkeypatch.setenv(switch, "yes")
    assert resolve(name, chip="unknown", library_version="unmeasured")
    monkeypatch.setenv(switch, "0")
    assert not resolve(name)
    monkeypatch.setenv(switch, "garbage")
    assert resolve(name)
    assert not resolve(name, chip="unknown")


@pytest.mark.parametrize("name", tuning.KNOBS)
@pytest.mark.parametrize(
    "change", [{"family": "other"}, {"geometry": (1,)}, {"width": 2}, {"mode": "other"}]
)
def test_override_cannot_expand_scope(monkeypatch, name, change):
    monkeypatch.setenv(tuning.KNOBS[name].env, "1")
    assert not resolve(name, **change)


def test_mtp_scope(monkeypatch):
    for knob in tuning.KNOBS.values():
        monkeypatch.setenv(knob.env, "1")
    assert resolve(mode="mtp")
    assert not resolve("qwen35_residual_norm", mode="mtp")


def test_import_does_not_load_mlx():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rapid_mlx.tuning; assert 'mlx.core' not in sys.modules",
        ],
        env=env,
        check=True,
    )


def test_both_gates_consult_registry(monkeypatch):
    pytest.importorskip("mlx.core")
    from rapid_mlx import qwen35_residual_norm as residual
    from rapid_mlx.patches import qwen35_affine4 as affine

    calls = []

    def decline(name, info, **scope):
        calls.append((name, scope))
        return False

    monkeypatch.delenv("RAPID_MLX_QWEN35_RESIDUAL_NORM", raising=False)
    monkeypatch.setattr(affine, "enabled", decline)
    monkeypatch.setattr(residual, "enabled", decline)
    assert not affine._singleton_bank_views_enabled()
    assert residual.install_qwen35_residual_norm(object()) == 0
    assert [name for name, _ in calls] == list(tuning.KNOBS)
    assert all(scope["width"] == 1 for _, scope in calls)
