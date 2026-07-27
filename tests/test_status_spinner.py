"""Tests for ``vllm_mlx.cli._StatusSpinner``.

The spinner covers the silent "Resolving…" window before a cold model
download starts (disk-space probe + mirror metadata + catalog fetch) so a
first-run user sees activity instead of an apparent hang. It must:

* be fully inert on non-TTY / ``NO_COLOR`` streams (clean CI logs + pipes),
* animate on a TTY and clear its line on stop,
* have an idempotent, thread-safe ``stop()`` (it doubles as a download
  ``on_pull_start`` hook AND the ``__exit__`` clear).

All timing-sensitive assertions poll a thread-safe fake stream with a
generous timeout rather than sleeping a fixed amount, so they stay
deterministic under load.
"""

from __future__ import annotations

import io
import threading
import time

import pytest

from vllm_mlx.cli import _StatusSpinner


class _FakeTTY:
    """Thread-safe writable stream that reports ``isatty() == True``."""

    def __init__(self, is_tty: bool = True):
        self._is_tty = is_tty
        self._lock = threading.Lock()
        self._parts: list[str] = []

    def write(self, s: str) -> int:
        with self._lock:
            self._parts.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return self._is_tty

    def value(self) -> str:
        with self._lock:
            return "".join(self._parts)

    def write_count(self) -> int:
        with self._lock:
            return len(self._parts)


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_inert_on_non_tty_stringio():
    """A plain (non-TTY) stream → no thread, no output, no-op stop."""
    buf = io.StringIO()  # StringIO.isatty() is False
    sp = _StatusSpinner("Resolving foo …", stream=buf)
    assert sp._enabled is False
    with sp:
        pass
    sp.stop()
    sp.stop()  # idempotent
    assert buf.getvalue() == ""
    assert sp._thread is None


def test_inert_when_no_color(monkeypatch: pytest.MonkeyPatch):
    """NO_COLOR mutes the spinner even on a real TTY stream."""
    monkeypatch.setenv("NO_COLOR", "1")
    stream = _FakeTTY(is_tty=True)
    sp = _StatusSpinner("Resolving foo …", stream=stream)
    assert sp._enabled is False
    with sp:
        pass
    assert stream.value() == ""


def test_animates_and_clears_on_tty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = _FakeTTY(is_tty=True)
    sp = _StatusSpinner("Resolving qwen …", stream=stream)
    assert sp._enabled is True
    with sp:
        # The first draw happens before the first 0.1s wait — it lands fast.
        assert _wait_until(lambda: stream.write_count() >= 1), "spinner never drew"
        mid = stream.value()
        assert "Resolving qwen …" in mid
        # A braille frame char from the animation set is present.
        assert any(ch in mid for ch in _StatusSpinner._FRAMES)
    # __exit__ → stop(): the final write clears the line with a CR.
    final = stream.value()
    assert final.rstrip().endswith("\r") or final.endswith("\r")
    assert sp._done is True


def test_stop_halts_the_thread(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = _FakeTTY(is_tty=True)
    sp = _StatusSpinner("Resolving x …", stream=stream)
    with sp:
        assert _wait_until(lambda: stream.write_count() >= 1)
    # After stop, the worker thread must be gone and quiet.
    assert sp._thread is not None
    assert not sp._thread.is_alive()
    count_after_stop = stream.write_count()
    time.sleep(0.25)  # longer than the 0.1s draw interval
    assert stream.write_count() == count_after_stop, "thread still writing after stop"


def test_stop_is_idempotent_and_thread_safe(monkeypatch: pytest.MonkeyPatch):
    """``stop`` doubles as the download ``on_pull_start`` hook (called once)
    AND the ``__exit__`` clear (called again) — both must be safe.
    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = _FakeTTY(is_tty=True)
    sp = _StatusSpinner("Resolving y …", stream=stream)
    with sp:
        assert _wait_until(lambda: stream.write_count() >= 1)
        sp.stop()  # simulate the on_pull_start hook firing mid-with
        first_done = sp._done
    # No exception on the second (exit) stop; state stays done.
    assert first_done is True
    assert sp._done is True


def test_stop_before_enter_is_safe():
    """Calling stop() without ever entering must not raise."""
    sp = _StatusSpinner("Resolving z …", stream=_FakeTTY(is_tty=True))
    sp.stop()
    assert sp._done is True
