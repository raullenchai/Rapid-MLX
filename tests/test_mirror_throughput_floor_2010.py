# SPDX-License-Identifier: Apache-2.0
"""Issue #2010: the R2 mirror serves a fast prefix then throttles a large
weight file to ~0.2 MB/s. A silent stall trips ``_FILE_TIMEOUT``; a slow
trickle does not, so the download crawls for minutes. ``_ThroughputFloor``
bails to the HuggingFace fallback once the windowed rate collapses below a
floor. These tests drive the guard with an injected clock (no real sleeps) and
exercise the wiring in ``_do_r2_download`` against a slow fake response.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import vllm_mlx._mirror as _mirror
from vllm_mlx._mirror import _mirror_floor_bytes_per_sec, _ThroughputFloor


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _run(floor: _ThroughputFloor, samples):
    """Feed (dt, bytes_this_step) samples; return the time it aborted, or None."""
    read = 0
    for dt, delta in samples:
        floor._clock.advance(dt)  # type: ignore[attr-defined]
        read += delta
        if floor.record(read):
            return floor._clock.now  # type: ignore[attr-defined]
    return None


def _floor(clock, bps=1_000_000, window=8.0, grace=6.0, consecutive=2):
    return _ThroughputFloor(
        bps, window_s=window, grace_s=grace, consecutive=consecutive, clock=clock
    )


def test_healthy_transfer_never_aborts():
    c = _Clock()
    f = _floor(c)
    # 10 MB/s for 20 s
    assert _run(f, [(1.0, 10_000_000)] * 20) is None


def test_stall_after_fast_prefix_aborts():
    c = _Clock()
    f = _floor(c)
    # 50 MB in the first second, then 180 KB/s
    samples = [(1.0, 50_000_000)] + [(1.0, 180_000)] * 60
    t = _run(f, samples)
    assert t is not None
    # grace 6 s, then the two consecutive sub-floor windows needed to bail.
    assert t <= 6.0 + 3 * 8.0 + 1


def test_slow_from_start_aborts():
    c = _Clock()
    f = _floor(c)
    t = _run(f, [(1.0, 200_000)] * 60)  # 200 KB/s throughout
    assert t is not None
    assert t <= 6.0 + 3 * 8.0 + 1


def test_slow_startup_then_healthy_does_not_abort():
    # Codex round 1: a 6 s cold-connection pause followed by a steady rate just
    # above the floor must NOT abort — the grace + post-grace anchoring keeps
    # the slow opening out of any window's rate.
    c = _Clock()
    f = _floor(c)
    samples = [(6.0, 0)] + [(1.0, 1_100_000)] * 30  # pause, then 1.1 MB/s
    assert _run(f, samples) is None


def test_single_bursty_pause_is_tolerated():
    # Codex round 1: 8 MiB chunks with one stretched gap. A single sub-floor
    # window must not bail while the aggregate stays healthy.
    c = _Clock()
    f = _floor(c)
    eight_mib = 8 * 1024 * 1024
    # Fast chunks every second (well above floor), then one 12 s gap, then fast
    # again — the stretched window is a single sub-floor sample, reset by the
    # healthy windows around it.
    samples = [(1.0, eight_mib)] * 12 + [(12.0, eight_mib)] + [(1.0, eight_mib)] * 12
    assert _run(f, samples) is None


def test_two_consecutive_slow_windows_are_required():
    c = _Clock()
    f = _floor(c)
    eight_mib = 8 * 1024 * 1024
    # Past grace, one slow window then a healthy one then another slow window —
    # never two in a row, so it must not abort.
    samples = (
        [(1.0, eight_mib)] * 8  # healthy anchor window
        + [(8.0, 100_000)]  # slow window 1
        + [(8.0, eight_mib)]  # healthy window (resets the counter)
        + [(8.0, 100_000)]  # slow window 1 again
        + [(1.0, eight_mib)] * 8  # healthy
    )
    assert _run(f, samples) is None


def test_grace_period_tolerates_a_slow_opening():
    c = _Clock()
    f = _floor(c)
    # Nothing arrives for the first 5 s (< grace) — must not abort yet.
    assert not f.record(0)
    c.advance(5.0)
    assert not f.record(0)


def test_floor_zero_disables_the_guard():
    c = _Clock()
    f = _floor(c, bps=0)
    assert _run(f, [(100.0, 1)] * 50) is None


def test_env_override_parsing(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_MIRROR_MIN_MBPS", raising=False)
    assert _mirror_floor_bytes_per_sec() == pytest.approx(1_000_000)
    monkeypatch.setenv("RAPID_MLX_MIRROR_MIN_MBPS", "2.5")
    assert _mirror_floor_bytes_per_sec() == pytest.approx(2_500_000)
    monkeypatch.setenv("RAPID_MLX_MIRROR_MIN_MBPS", "0")
    assert _mirror_floor_bytes_per_sec() == 0.0
    monkeypatch.setenv("RAPID_MLX_MIRROR_MIN_MBPS", "-3")
    assert _mirror_floor_bytes_per_sec() == 0.0
    # Unparseable / non-finite fall back to the default (never silently off, and
    # never an infinite floor that would abort everything).
    for bad in ("not-a-number", "inf", "-inf", "nan"):
        monkeypatch.setenv("RAPID_MLX_MIRROR_MIN_MBPS", bad)
        assert _mirror_floor_bytes_per_sec() == pytest.approx(1_000_000), bad
    # A finite-but-enormous value overflows to inf when scaled to bytes/s — the
    # scaled result is re-checked, so it falls back to the default too.
    monkeypatch.setenv("RAPID_MLX_MIRROR_MIN_MBPS", "1e308")
    assert _mirror_floor_bytes_per_sec() == pytest.approx(1_000_000)


class _SlowResponse:
    """A 200 response that trickles bytes while a shared clock advances, so the
    windowed rate stays far below the floor. ``read1`` returns after a single
    (small) socket read; ``read(n)`` would BLOCK until all ``n`` bytes arrived,
    which under a throttle is minutes — so it advances the clock by a whole
    window-sized gulp per call. The production loop uses ``read1``; if a
    regression switched it back to ``read`` the floor could never sample and
    these tests would time out / not abort in the same bound."""

    def __init__(
        self,
        clock: _Clock,
        *,
        total: int,
        per_read: int,
        dt: float,
        status: int = 200,
        content_range: str | None = None,
    ):
        self.status = status
        self.headers = {"Content-Length": str(total)}
        if content_range is not None:
            self.headers["Content-Range"] = content_range
        self._clock = clock
        self._total = total
        self._per_read = per_read
        self._dt = dt
        self._served = 0
        self.read1_calls = 0
        self.read_calls = 0

    def read1(self, n: int = -1) -> bytes:
        self.read1_calls += 1
        self._clock.advance(self._dt)
        if self._served >= self._total:
            return b""
        take = min(
            self._per_read if n < 0 else min(self._per_read, n),
            self._total - self._served,
        )
        self._served += take
        return b"x" * take

    def read(self, n: int = -1) -> bytes:
        # A real ``read(n)`` BLOCKS until all n bytes arrive; under a throttle
        # that is minutes for one 8 MiB chunk, and the whole file lands in a
        # single call — so the floor would never sample. Model that: return the
        # ENTIRE remaining body at once. A loop that (wrongly) used ``read``
        # would then complete without ever calling the floor, so the
        # bail-to-HF assertion below would fail — pinning that production uses
        # ``read1``.
        self.read_calls += 1
        self._clock.advance(self._dt)
        rest = self._total - self._served
        self._served = self._total
        return b"x" * rest

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_do_r2_download_bails_to_hf_on_slow_mirror(tmp_path, monkeypatch):
    target = tmp_path / "snap" / "model.safetensors"
    sidecar = tmp_path / "sidecar"
    part = sidecar / "model.part"
    target.parent.mkdir(parents=True)
    sidecar.mkdir(parents=True)

    clock = _Clock()
    monkeypatch.setattr(_mirror.time, "monotonic", clock)
    monkeypatch.delenv("RAPID_MLX_MIRROR_MIN_MBPS", raising=False)

    # 100 MB advertised, delivered at 200 KB per 5 s ≈ 40 KB/s — far below the
    # 1 MB/s floor, so the guard fires long before the body is complete.
    resp = _SlowResponse(clock, total=100_000_000, per_read=200_000, dt=5.0)
    with patch("urllib.request.urlopen", return_value=resp):
        ok, reason = _mirror._do_r2_download(
            "https://models.example/model.safetensors",
            target,
            part,
            100_000_000,
        )

    assert not ok
    assert reason == "slow-mirror"
    # The partial prefix is discarded so HF's full-file download can't be
    # concatenated onto a stale R2 prefix.
    assert not part.exists()
    assert not target.exists()
    # The loop must stream via read1 (prompt) — a blocking read would starve
    # the floor and this would never abort.
    assert resp.read1_calls > 0
    assert resp.read_calls == 0


def test_do_r2_download_keeps_a_complete_file_whose_last_chunk_is_slow(
    tmp_path, monkeypatch
):
    # Codex round 3: the final chunk can complete Content-Length AND close the
    # second sub-floor window. That is not a stall — the file is done — so the
    # guard must not delete a complete, valid file and refetch it from HF.
    target = tmp_path / "snap" / "model.safetensors"
    sidecar = tmp_path / "sidecar"
    part = sidecar / "model.part"
    target.parent.mkdir(parents=True)
    sidecar.mkdir(parents=True)

    clock = _Clock()
    monkeypatch.setattr(_mirror.time, "monotonic", clock)
    monkeypatch.delenv("RAPID_MLX_MIRROR_MIN_MBPS", raising=False)

    # 3 reads of 200 KB every 8 s: anchor at t=8, slow window at t=16 (count 1),
    # and at t=24 the third read both completes the 600 KB file and would close
    # the second slow window. Total rate is sub-floor throughout.
    resp = _SlowResponse(clock, total=600_000, per_read=200_000, dt=8.0)
    with patch("urllib.request.urlopen", return_value=resp):
        ok, reason = _mirror._do_r2_download(
            "https://models.example/model.safetensors",
            target,
            part,
            600_000,
        )

    assert ok, reason
    assert target.read_bytes() == b"x" * 600_000


def test_do_r2_download_guards_a_resumed_download(tmp_path, monkeypatch):
    # Codex round 6 (pr_validate): a resumed 206 whose Content-Length is only
    # the REMAINING body must still be guarded against the absolute final size,
    # not the suffix — otherwise monitoring stops early on a resume.
    target = tmp_path / "snap" / "model.safetensors"
    sidecar = tmp_path / "sidecar"
    part = sidecar / "model.part"
    target.parent.mkdir(parents=True)
    sidecar.mkdir(parents=True)
    existing = 400_000
    part.write_bytes(b"x" * existing)  # a prior aborted run left this prefix

    clock = _Clock()
    monkeypatch.setattr(_mirror.time, "monotonic", clock)
    monkeypatch.delenv("RAPID_MLX_MIRROR_MIN_MBPS", raising=False)

    total = 100_000_000
    remaining = total - existing
    resp = _SlowResponse(
        clock,
        total=remaining,
        per_read=200_000,
        dt=8.0,
        status=206,
        content_range=f"bytes {existing}-{total - 1}/{total}",
    )
    with patch("urllib.request.urlopen", return_value=resp):
        ok, reason = _mirror._do_r2_download(
            "https://models.example/model.safetensors",
            target,
            part,
            total,
        )
    assert not ok
    assert reason == "slow-mirror"


def test_do_r2_download_guards_when_content_length_absent_but_size_known(
    tmp_path, monkeypatch
):
    # Codex round 6 (pr_validate): if the response omits Content-Length but HF
    # gave us the size, the guard must still fire — a chunked multi-GB shard
    # must not bypass the mitigation.
    target = tmp_path / "snap" / "model.safetensors"
    sidecar = tmp_path / "sidecar"
    part = sidecar / "model.part"
    target.parent.mkdir(parents=True)
    sidecar.mkdir(parents=True)

    clock = _Clock()
    monkeypatch.setattr(_mirror.time, "monotonic", clock)
    monkeypatch.delenv("RAPID_MLX_MIRROR_MIN_MBPS", raising=False)

    resp = _SlowResponse(clock, total=100_000_000, per_read=200_000, dt=8.0)
    resp.headers = {}  # no Content-Length on the wire
    with patch("urllib.request.urlopen", return_value=resp):
        ok, reason = _mirror._do_r2_download(
            "https://models.example/model.safetensors",
            target,
            part,
            100_000_000,  # but HF told us the size
        )
    assert not ok
    assert reason == "slow-mirror"


def test_do_r2_download_does_not_abort_an_unknown_length_response(
    tmp_path, monkeypatch
):
    # Codex round 5: with no Content-Length the loop can't tell "final chunk"
    # from "mid-stream", so the floor is skipped entirely — a slow unknown-length
    # response (only tiny config assets in practice) must complete, not be
    # discarded. Model it with a response that omits Content-Length.
    target = tmp_path / "snap" / "config.json"
    sidecar = tmp_path / "sidecar"
    part = sidecar / "config.part"
    target.parent.mkdir(parents=True)
    sidecar.mkdir(parents=True)

    clock = _Clock()
    monkeypatch.setattr(_mirror.time, "monotonic", clock)
    monkeypatch.delenv("RAPID_MLX_MIRROR_MIN_MBPS", raising=False)

    resp = _SlowResponse(clock, total=90_000, per_read=10_000, dt=8.0)
    resp.headers = {}  # no Content-Length -> length == 0
    with patch("urllib.request.urlopen", return_value=resp):
        ok, reason = _mirror._do_r2_download(
            "https://models.example/config.json",
            target,
            part,
            None,  # expected_size unknown
        )

    assert ok, reason
    assert target.read_bytes() == b"x" * 90_000


def test_do_r2_download_completes_a_fast_transfer(tmp_path, monkeypatch):
    target = tmp_path / "snap" / "small.bin"
    sidecar = tmp_path / "sidecar"
    part = sidecar / "small.part"
    target.parent.mkdir(parents=True)
    sidecar.mkdir(parents=True)

    clock = _Clock()
    monkeypatch.setattr(_mirror.time, "monotonic", clock)
    monkeypatch.delenv("RAPID_MLX_MIRROR_MIN_MBPS", raising=False)

    # 4 MB delivered at 4 MB per 0.1 s = 40 MB/s — well above the floor.
    resp = _SlowResponse(clock, total=4_000_000, per_read=4_000_000, dt=0.1)
    with patch("urllib.request.urlopen", return_value=resp):
        ok, reason = _mirror._do_r2_download(
            "https://models.example/small.bin",
            target,
            part,
            4_000_000,
        )

    assert ok, reason
    assert target.read_bytes() == b"x" * 4_000_000
