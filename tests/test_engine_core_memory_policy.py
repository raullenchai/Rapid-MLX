"""Exercise production memory setup through the existing no-MLX import seam."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("override", [False, True])
def test_engine_memory_setup(override):
    # Isolate inert MLX modules from the rest of the test process.
    script = """
        import asyncio
        import json
        import os
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from tests.headless_mlx._headless_mlx import install_headless_mlx_import_stubs
        with patch("importlib.util.find_spec", return_value=None):
            install_headless_mlx_import_stubs()

        from rapid_mlx import engine_core, lattice
        from rapid_mlx.engine.batched import _compute_metal_cache_limit
        from rapid_mlx.lattice import defaults, memory

        override = OVERRIDE
        knobs = dict(metal_cache_floor_bytes=1, metal_cache_cap_bytes=100,
                     metal_cache_divisor=2, memory_pressure_margin=0.1,
                     memory_pressure_ceiling=0.8, memory_pressure_fallback_bytes=1000,
                     memory_check_interval=3)
        if override:
            os.environ['RAPID_MLX_BENCHMARK_KNOBS'] = json.dumps(knobs)
        resolve = Mock(wraps=memory.resolve)
        probe = Mock(return_value={'memory_size': 100})
        with patch.object(lattice, 'overlay_profile', return_value=defaults.Profile()), \
             patch.object(memory, 'resolve', resolve), \
             patch.object(engine_core.mx, 'device_info', probe):
            policy = memory.memory_policy()
            thresholds = []
            intervals = []
            class ObservedPolicy:
                @property
                def memory_check_interval(self):
                    intervals.append(policy.memory_check_interval)
                    return policy.memory_check_interval
                def pressure_threshold(self, *args):
                    value = policy.pressure_threshold(*args)
                    thresholds.append(value)
                    return value

            with patch.object(engine_core, 'memory_policy', return_value=ObservedPolicy()):
                for _ in range(20):
                    engine = SimpleNamespace(
                        _mlx_executor=None, _idle_event=None, _running=False,
                        config=SimpleNamespace(step_interval=0.01, stream_interval=1,
                                               gpu_memory_utilization=0.5))
                    asyncio.run(engine_core.EngineCore._engine_loop(engine))
                    expected = 49 if override else 99
                    assert _compute_metal_cache_limit(99) == expected
            assert thresholds == [60 if override else 55] * 20
            assert intervals == [3 if override else 16] * 20
            assert probe.call_count == 20  # One read per engine setup.
            assert resolve.call_count == 7
    """
    env = dict(os.environ)
    env.pop("RAPID_MLX_BENCHMARK_KNOBS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(script).replace("OVERRIDE", repr(override)),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_device_info_uses_fallback():
    # An MLX build without device_info must keep the legacy 200 GiB fallback
    # instead of aborting engine-loop setup.
    script = """
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch

        from tests.headless_mlx._headless_mlx import install_headless_mlx_import_stubs
        with patch("importlib.util.find_spec", return_value=None):
            install_headless_mlx_import_stubs()

        from rapid_mlx import engine_core, lattice
        from rapid_mlx.lattice import defaults, memory

        thresholds = []
        policy = memory.MemoryPolicy()

        class ObservedPolicy:
            memory_check_interval = policy.memory_check_interval

            def pressure_threshold(self, *args):
                value = policy.pressure_threshold(*args)
                thresholds.append(value)
                return value

        if hasattr(engine_core.mx, "device_info"):
            delattr(engine_core.mx, "device_info")
        assert not hasattr(engine_core.mx, "device_info")
        with patch.object(lattice, "overlay_profile", return_value=defaults.Profile()), \
             patch.object(engine_core, "memory_policy", return_value=ObservedPolicy()):
            engine = SimpleNamespace(
                _mlx_executor=None, _idle_event=None, _running=False,
                config=SimpleNamespace(step_interval=0.01, stream_interval=1,
                                       gpu_memory_utilization=0.5))
            asyncio.run(engine_core.EngineCore._engine_loop(engine))
        assert thresholds == [200 * 1024**3], thresholds
    """
    env = dict(os.environ)
    env.pop("RAPID_MLX_BENCHMARK_KNOBS", None)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
