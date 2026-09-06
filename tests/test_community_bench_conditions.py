# SPDX-License-Identifier: Apache-2.0
"""``hardware.run_conditions()`` — volatile run conditions.

Kept MLX-free (no ``requires_mlx`` marker) so the hosted Linux lane runs the
parsing/degradation tests, and listed in the Apple lane so the in-process
``NSProcessInfo`` path is exercised on real hardware.
"""

from __future__ import annotations

import sys

import pytest

# ---------------------------------------------------------------------------
# hardware.run_conditions() — volatile run conditions
# ---------------------------------------------------------------------------


def _fake_probe(outputs: dict[str, str]):
    """Return a ``_run`` stand-in keyed on the joined argv.

    Missing keys raise ``RuntimeError`` exactly like a failed probe so the
    per-field degradation path is exercised, not just the happy path.
    """

    def run(cmd, timeout):
        key = " ".join(cmd)
        if key not in outputs:
            raise RuntimeError(f"probe {cmd!r} failed")
        return outputs[key]

    return run


_BATT_AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged"
_BATT_BATTERY = (
    "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t61%; discharging"
)


def test_run_conditions_maps_every_probe_onto_the_schema_enums(monkeypatch) -> None:
    from vllm_mlx.community_bench import hardware

    monkeypatch.setattr(
        hardware,
        "_run",
        _fake_probe(
            {
                "/usr/bin/pmset -g batt": _BATT_BATTERY,
                "/usr/sbin/sysctl -n kern.memorystatus_vm_pressure_level": "2",
                "/usr/sbin/sysctl -n vm.page_free_count vm.page_speculative_count "
                "vm.page_purgeable_count hw.pagesize": "1024\n1024\n2048\n16384",
            }
        ),
    )
    monkeypatch.setattr(hardware, "_thermal_state", lambda: "fair")
    monkeypatch.setattr(hardware, "_low_power_mode", lambda: True)
    conditions = hardware.run_conditions()
    assert conditions == {
        "power_source": "battery",
        "low_power_mode": True,
        "thermal_state": "fair",
        "memory_pressure": "warning",
        # (1024 + 1024 + 2048) pages * 16 KiB = 64 MiB
        "available_memory_mib": 64,
    }


def test_run_conditions_reads_ac_and_normal_pressure(monkeypatch) -> None:
    from vllm_mlx.community_bench import hardware

    monkeypatch.setattr(
        hardware,
        "_run",
        _fake_probe(
            {
                "/usr/bin/pmset -g batt": _BATT_AC,
                "/usr/sbin/sysctl -n kern.memorystatus_vm_pressure_level": "1",
            }
        ),
    )
    monkeypatch.setattr(hardware, "_thermal_state", lambda: "nominal")
    monkeypatch.setattr(hardware, "_low_power_mode", lambda: False)
    conditions = hardware.run_conditions()
    assert conditions["power_source"] == "ac"
    assert conditions["low_power_mode"] is False
    assert conditions["memory_pressure"] == "normal"
    # The memory probe was not answered: that one field degrades, nothing else.
    assert conditions["available_memory_mib"] is None


def test_run_conditions_degrades_each_field_independently(monkeypatch) -> None:
    """Every probe failing must still yield a schema-valid object."""
    from vllm_mlx.catalog.validation import ContractValidator
    from vllm_mlx.community_bench import hardware

    monkeypatch.setattr(hardware, "_run", _fake_probe({}))
    monkeypatch.setattr(hardware, "_thermal_state", lambda: "unknown")
    monkeypatch.setattr(hardware, "_low_power_mode", lambda: None)
    conditions = hardware.run_conditions()
    assert conditions == {
        "power_source": "unknown",
        "low_power_mode": None,
        "thermal_state": "unknown",
        "memory_pressure": "unknown",
        "available_memory_mib": None,
    }
    # Unrecognised raw values map to "unknown" too, never to a wrong bucket.
    monkeypatch.setattr(
        hardware,
        "_run",
        _fake_probe(
            {
                "/usr/bin/pmset -g batt": "Now drawing from 'UPS Power'",
                "/usr/sbin/sysctl -n kern.memorystatus_vm_pressure_level": "3",
            }
        ),
    )
    assert hardware.run_conditions()["power_source"] == "unknown"
    assert hardware.run_conditions()["memory_pressure"] == "unknown"
    observation = {
        "schema_version": 1,
        "profile_completeness": "partial",
        "profile": {
            "chip": "Apple M3 Pro",
            "memory_gib": 18,
            "cpu_cores": 12,
            "gpu_cores": 18,
        },
        "profile_digest": "sha256:" + "0" * 64,
        "os": {"name": "macOS", "version": "15.6.1", "architecture": "arm64"},
        "conditions_before": conditions,
        "conditions_after": hardware.run_conditions(),
    }
    # profile_digest is checked by the run validator, not the atomic one.
    ContractValidator().validate("machine_observation", observation)


@pytest.mark.skipif(
    sys.platform != "darwin", reason="Objective-C runtime is macOS-only"
)
def test_process_info_probes_read_real_values_on_macos() -> None:
    from vllm_mlx.community_bench import hardware

    assert hardware._thermal_state() in {"nominal", "fair", "serious", "critical"}
    assert isinstance(hardware._low_power_mode(), bool)


def test_process_info_probes_degrade_when_the_runtime_is_unavailable(
    monkeypatch,
) -> None:
    from vllm_mlx.community_bench import hardware

    monkeypatch.setattr(hardware, "_process_info", lambda selector, restype: None)
    assert hardware._thermal_state() == "unknown"
    assert hardware._low_power_mode() is None
    # An out-of-range raw thermal value is "unknown", never a wrong bucket.
    monkeypatch.setattr(hardware, "_process_info", lambda selector, restype: 7)
    assert hardware._thermal_state() == "unknown"


def test_run_normalises_process_creation_failures(monkeypatch) -> None:
    """An ``OSError`` from spawning must surface as the documented RuntimeError.

    Otherwise the optional post-measurement probe in ``run_local`` would
    abort after the benchmark completed and the result would be lost.
    """
    from vllm_mlx.community_bench import hardware

    def cannot_spawn(*args, **kwargs):
        raise OSError(35, "Resource temporarily unavailable")

    monkeypatch.setattr(hardware.subprocess, "run", cannot_spawn)
    with pytest.raises(RuntimeError, match="probe .* failed"):
        hardware._run(["/usr/sbin/sysctl", "-n", "hw.ncpu"], timeout=1.0)
    # And the composed snapshot still comes back schema-valid.
    monkeypatch.setattr(hardware, "_process_info", lambda selector, restype: None)
    assert hardware.run_conditions()["power_source"] == "unknown"


def test_pmset_is_on_the_allowlist_and_nothing_else_was_added() -> None:
    """The privacy contract enumerates every binary; pin the expansion."""
    from vllm_mlx.community_bench import hardware

    expected = frozenset(
        {
            "/usr/sbin/sysctl",
            "/usr/bin/sw_vers",
            "/usr/sbin/system_profiler",
            "/usr/bin/pmset",
        }
    )
    assert expected == hardware._PERMITTED_BINARIES


@pytest.mark.skipif(
    sys.platform != "darwin", reason="Objective-C runtime is macOS-only"
)
def test_process_info_degrades_when_the_objc_runtime_cannot_load(monkeypatch) -> None:
    import ctypes

    from vllm_mlx.community_bench import hardware

    def cannot_load(*args, **kwargs):
        raise OSError("dlopen failed")

    monkeypatch.setattr(ctypes, "CDLL", cannot_load)
    assert hardware._process_info(b"thermalState", ctypes.c_long) is None
    assert hardware._thermal_state() == "unknown"
    assert hardware._low_power_mode() is None


@pytest.mark.skipif(
    sys.platform != "darwin", reason="Objective-C runtime is macOS-only"
)
def test_process_info_probes_work_in_a_fresh_interpreter() -> None:
    """A clean process (no Foundation imported by anything else) must still
    resolve NSProcessInfo — pr_validate codex on #3146."""
    import json
    import subprocess

    code = (
        "import json, sys; "
        "from vllm_mlx.community_bench import hardware; "
        "print(json.dumps([hardware._thermal_state(), hardware._low_power_mode()]))"
    )
    out = subprocess.run(
        [sys.executable, "-S", "-c", code], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    thermal, low_power = json.loads(out.stdout.strip().splitlines()[-1])
    assert thermal in {"nominal", "fair", "serious", "critical"}
    assert isinstance(low_power, bool)


def test_process_info_is_unavailable_off_darwin(monkeypatch) -> None:
    from vllm_mlx.community_bench import hardware

    monkeypatch.setattr(hardware.sys, "platform", "linux")
    assert hardware._process_info(b"thermalState", int) is None


@pytest.mark.skipif(
    sys.platform != "darwin", reason="Objective-C runtime is macOS-only"
)
def test_process_info_degrades_when_the_class_cannot_be_resolved(monkeypatch) -> None:
    """A null NSProcessInfo (class not registered) degrades to None, not a crash."""
    import ctypes

    from vllm_mlx.community_bench import hardware

    monkeypatch.setattr(ctypes, "cast", lambda *args, **kwargs: lambda *call: 0)
    assert hardware._process_info(b"thermalState", ctypes.c_long) is None
    assert hardware._thermal_state() == "unknown"


def test_process_info_declines_selectors_the_runtime_does_not_recognise() -> None:
    """An unknown selector degrades to ``None`` instead of an ObjC exception."""
    import ctypes

    from vllm_mlx.community_bench import hardware

    assert hardware._process_info(b"rapidMlxNoSuchSelector", ctypes.c_long) is None
    if sys.platform == "darwin":
        # The guard must not reject selectors NSProcessInfo really has.
        assert hardware._process_info(b"thermalState", ctypes.c_long) is not None
