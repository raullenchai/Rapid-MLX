# SPDX-License-Identifier: Apache-2.0
"""Tests for the dev-only LTX benchmark summarization helpers."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path

PATH = Path(__file__).parents[1] / "bench" / "bench_ltx_video.py"
SPEC = spec_from_file_location("bench_ltx_video", PATH)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_swap_parser_handles_megabytes(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.subprocess,
        "check_output",
        lambda *args, **kwargs: "total = 2048.00M  used = 12.50M  free = 2035.50M",
    )
    assert MODULE._swap_used_bytes() == round(12.5 * 1024**2)


def test_duration_rounds_to_nearest_ltx_frame_shape() -> None:
    assert MODULE._frames_for_seconds(5, 24) == 121
    assert MODULE._frames_for_seconds(10, 24) == 241


def test_ltx25_worker_does_not_require_controller_python(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_ltx_video.py",
            "--worker",
            "--runtime",
            "ltx25",
            "--model",
            "/model",
            "--frames",
            "9",
        ],
    )
    sentinel = object()
    monkeypatch.setattr(MODULE, "_worker", lambda args: sentinel)
    assert MODULE.main() is sentinel


def test_transformer_phase_end_marks_weights_ready() -> None:
    events = []
    lines = []
    MODULE._read_stream(
        StringIO("[Loading transformer (weights.safetensors)] done in 0.4s\n"),
        "stderr",
        MODULE.time.monotonic_ns(),
        events,
        lines,
    )
    assert [event["kind"] for event in events] == ["phase_end", "weights_ready"]


def test_markdown_excludes_failed_and_swap_limited_runs_from_medians() -> None:
    document = {
        "machine": {
            "hostname": "mzr",
            "hardware_model": "Mac16,11",
            "architecture": "arm64",
            "memory_bytes": 48 * 1024**3,
            "macos_version": "26.5.1",
            "macos_build": "25F80",
        },
        "config": {
            "model": "example/ltx",
            "model_revision": "abc123",
            "runtime": "mlx23",
            "runtime_revision": "0.1.36",
            "prompt": "fixed prompt",
            "width": 768,
            "height": 512,
            "frames": 121,
            "fps": 24,
            "seed": 42,
        },
        "results": [
            {
                "status": "completed",
                "cold_start_to_first_step_s": 10.0,
                "cold_start_to_weights_ready_s": 9.5,
                "step_median_s": 3.0,
                "stage_step_median_s": {"1": 2.5, "2": 7.5},
                "total_s": 20.0,
                "peak_tree_rss_bytes": 2 * 1024**3,
                "mlx_peak_bytes": 1024**3,
                "swap_growth_bytes": 0,
                "swap_limited": False,
            },
            {
                "status": "failed",
                "cold_start_to_first_step_s": 999.0,
                "cold_start_to_weights_ready_s": 998.0,
                "step_median_s": 999.0,
                "stage_step_median_s": {},
                "total_s": 999.0,
                "peak_tree_rss_bytes": 3 * 1024**3,
                "mlx_peak_bytes": None,
                "swap_growth_bytes": 0,
                "swap_limited": False,
            },
            {
                "status": "completed",
                "cold_start_to_first_step_s": 777.0,
                "cold_start_to_weights_ready_s": 776.0,
                "step_median_s": 777.0,
                "stage_step_median_s": {"1": 777.0, "2": 777.0},
                "total_s": 777.0,
                "peak_tree_rss_bytes": 4 * 1024**3,
                "mlx_peak_bytes": 2 * 1024**3,
                "swap_growth_bytes": 512 * 1024**2,
                "swap_limited": True,
            },
        ],
    }
    rendered = MODULE._markdown(document)
    assert "10.00 s" in rendered
    assert "9.50 s" in rendered
    assert "abc123" in rendered
    assert "48.00 GiB" in rendered
    assert "3.00 s" in rendered
    assert "20.00 s" in rendered
    assert "Median stage 1 diffusion step: 2.50 s" in rendered
    assert "Median stage 2 diffusion step: 7.50 s" in rendered
    assert "999.00 s" not in rendered
    assert "777.00 s" not in rendered
    assert "4.00 GiB" in rendered
    assert "0.50 GiB" in rendered
