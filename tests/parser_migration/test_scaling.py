"""Deterministic scaling check for the complete parser path.

The default unit run must not depend on CPU or wall-clock timing. Instead, the
test counts the characters handed to the Hermes tool-call scanner over a whole
stream, including finalization. Linear streaming work keeps that count per
chunk roughly constant as the argument body grows; re-scanning the accumulated
text on every chunk makes it grow with the body size. Manual CPU timing remains
available through ``python -m tests.parser_migration.benchmark``.
"""

import pytest

from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

from .benchmark import run_stream


class ScalingRegressionError(Exception):
    pass


def _scanned_chars_per_chunk(monkeypatch, size):
    scanned = 0
    original = HermesToolParser.__dict__["_scan_tool_call_shapes"].__func__

    def counting(cls, text, request=None):
        nonlocal scanned
        scanned += len(text)
        return original(cls, text, request)

    with monkeypatch.context() as patch:
        patch.setattr(HermesToolParser, "_scan_tool_call_shapes", classmethod(counting))
        chunk_count = run_stream(size)
    assert scanned > 0, "Hermes scanner was not reached; update this probe"
    return scanned / chunk_count


@pytest.mark.xfail(
    strict=True,
    raises=ScalingRegressionError,
    reason="pre-existing in v0.16.9 (baseline parity); fix via upstream-aligned parser port in 0.17.0 — #5273 / R4",
)
def test_full_path_scan_work_scaling(monkeypatch):
    # A 4x larger body may at most double the scanned characters per chunk.
    # Linear work stays near 1x; whole-text rescans per chunk grow about 4x.
    previous = _scanned_chars_per_chunk(monkeypatch, 1024)
    for size in (4096, 16384):
        current = _scanned_chars_per_chunk(monkeypatch, size)
        if current > previous * 2:
            raise ScalingRegressionError((size, previous, current))
        previous = current
