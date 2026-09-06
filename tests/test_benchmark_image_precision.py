from __future__ import annotations

import argparse
import base64
import io
import socket
from pathlib import Path

import pytest
from PIL import Image

from scripts import benchmark_image_precision as bench


def _png(size=(512, 512), *, uniform=False) -> bytes:
    image = Image.new("RGB", size, "red")
    if not uniform:
        image.putpixel((0, 0), (0, 0, 255))
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("generate:512x512:4", ("generate", 512, 512, 4)),
        ("edit:1024x768:20", ("edit", 1024, 768, 20)),
    ],
)
def test_parse_workload(value, expected):
    parsed = bench.parse_workload(value)
    assert (
        parsed["operation"],
        parsed["width"],
        parsed["height"],
        parsed["steps"],
    ) == expected


@pytest.mark.parametrize(
    "value",
    ["chat:512x512:4", "generate:500x512:4", "edit:512x512:0", "broken"],
)
def test_parse_workload_rejects_invalid_contract(value):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_workload(value)


def test_parse_sequence_requires_balanced_precisions():
    assert bench.parse_sequence("q4,bf16,bf16,q4") == ("q4", "bf16", "bf16", "q4")
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_sequence("q4,bf16,bf16")


def test_low_power_mode_reads_ac_power_only(monkeypatch):
    monkeypatch.setattr(
        bench,
        "command",
        lambda *args, **kwargs: (
            "Battery Power:\n lowpowermode 1\nAC Power:\n lowpowermode 0"
        ),
    )
    assert bench.low_power_mode() == 0


def test_port_probe_rejects_listener_but_not_released_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="already in use"):
            bench.ensure_port_free(port)
    bench.ensure_port_free(port)


def test_running_rapid_servers_ignores_unrelated_processes(monkeypatch):
    monkeypatch.setattr(bench.os, "getpid", lambda: 20)
    monkeypatch.setattr(
        bench,
        "command",
        lambda *args, **kwargs: (
            "10 /usr/bin/node portal_server.py\n"
            "20 python benchmark_image_precision.py\n"
            "30 /usr/bin/rapid-mlx --no-banner serve model --port 8000\n"
        ),
    )
    assert bench.running_rapid_servers() == [30]


def test_server_environment_forces_pinned_cache_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    environment = bench.server_environment(Path("/tmp/qualification-cache"))
    assert environment["HF_HUB_CACHE"] == "/tmp/qualification-cache"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_validate_png_checks_format_dimensions_and_nonuniformity():
    raw = _png()
    response = {"data": [{"b64_json": base64.b64encode(raw).decode()}]}
    digest, byte_count = bench.validate_png(response, 512, 512)
    assert len(digest) == 64
    assert byte_count == len(raw)

    with pytest.raises(RuntimeError, match="format/size"):
        bench.validate_png(response, 1024, 1024)

    uniform = _png(uniform=True)
    with pytest.raises(RuntimeError, match="uniform"):
        bench.validate_png(
            {"data": [{"b64_json": base64.b64encode(uniform).decode()}]}, 512, 512
        )


def test_summarize_requires_stable_hashes_and_reports_median():
    sessions = [
        {
            "precision": "q4",
            "samples": [
                {
                    "workload": "generate-512x512-4",
                    "wall_s": 2.0,
                    "sha256": "a",
                    "peak_footprint_gib": 4.0,
                },
                {
                    "workload": "generate-512x512-4",
                    "wall_s": 1.0,
                    "sha256": "a",
                    "peak_footprint_gib": 5.0,
                },
            ],
        }
    ]
    row = bench.summarize(sessions)[0]
    assert row["median_wall_s"] == 1.5
    assert row["peak_footprint_gib"] == 5.0

    sessions[0]["samples"][1]["sha256"] = "b"
    with pytest.raises(RuntimeError, match="non-deterministic"):
        bench.summarize(sessions)
