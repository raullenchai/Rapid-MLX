# SPDX-License-Identifier: Apache-2.0
"""Chat readiness rendering with HTTP responses and an advancing fake clock."""

import io
import os
from types import SimpleNamespace

import pytest
import requests
from rich.cells import cell_len

from rapid_mlx import cli


class Terminal(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


@pytest.fixture
def wait(monkeypatch):
    def run(payloads, *, tty=True, width=100, timeout=10, exited=False):
        clock = [0.0]
        calls = []
        output = Terminal(tty)
        responses = iter(payloads)

        def get(url, **kwargs):
            calls.append((clock[0], url, kwargs))
            item = next(responses, (200, {}))
            if isinstance(item, Exception):
                raise item
            code, payload = item

            def json():
                if isinstance(payload, Exception):
                    raise payload
                return payload

            return SimpleNamespace(status_code=code, json=json)

        monkeypatch.setattr(requests, "get", get)
        monkeypatch.setattr("time.monotonic", lambda: clock[0])
        monkeypatch.setattr(
            "time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        )
        monkeypatch.setattr(
            "shutil.get_terminal_size", lambda fallback: os.terminal_size((width, 24))
        )
        monkeypatch.setattr(cli.sys, "stderr", output)
        proc = SimpleNamespace(poll=lambda: 1 if exited else None, returncode=1)
        error = None
        try:
            cli._wait_for_chat_server("http://localhost:1234", proc, timeout_s=timeout)
        except (RuntimeError, TimeoutError) as exc:
            error = exc
        return output.getvalue(), calls, error

    return run


def weights(done=1_600_000_000, total=2_500_000_000, model="Qwen-4B"):
    return {
        "phase": "loading_weights",
        "model": model,
        "progress": {"done": done, "total": total, "unit": "bytes"},
    }


def frames(output):
    return [part for part in output.split("\r\x1b[2K") if part]


def test_determinate_bar_and_http_poll_rate(wait):
    output, calls, error = wait([(503, weights()), (200, {})])
    assert error is None
    rendered = frames(output)
    assert len(rendered) == 10
    assert all("Loading Qwen-4B" in frame for frame in rendered)
    assert all("64%" in frame and "1.6/2.5 GB" in frame for frame in rendered)
    assert "━" in rendered[0] and "─" in rendered[0]
    assert len(calls) == 2
    assert calls[1][0] - calls[0][0] == pytest.approx(1.0)
    assert calls[0][1].endswith("/health/ready")
    assert output.endswith("\r\x1b[2K")


def test_starting_and_phase_only_spinner(wait):
    output, _, error = wait(
        [
            requests.ConnectionError(),
            (503, {"phase": "warming_up"}),
            (200, {}),
        ]
    )
    assert error is None
    rendered = frames(output)
    assert "Starting…" in rendered[0]
    assert "Warming up…" in rendered[10]
    assert rendered[0][0] != rendered[1][0]
    assert "%" not in output


def test_narrow_terminal_only_percentage(wait):
    output, _, error = wait([(503, weights())], width=59)
    assert error is None
    assert set(frames(output)) == {"64%"}


def test_non_tty_quiet(wait):
    output, calls, error = wait([(503, weights())], tty=False)
    assert error is None
    assert len(calls) == 2
    assert output == ""


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        ValueError("not JSON"),
        {"detail": "model loading"},
        {"phase": []},
        {"progress": "invalid"},
        {"progress": {"done": -1, "total": 10}},
        {"progress": {"done": True, "total": 10}},
        {"progress": {"done": 5, "total": 0}},
        {"progress": {"done": 11, "total": 10}},
        {"progress": {"done": 1, "total": 10**1000}},
    ],
)
def test_old_or_malformed_readiness_falls_back(wait, payload):
    output, _, error = wait([(503, payload)])
    assert error is None
    assert "Loading model…" in output
    assert "%" not in output


def test_long_unicode_model_cannot_wrap_or_inject_controls(wait):
    output, _, error = wait(
        [(503, weights(model="模型" * 100 + "\n\x1b[31m\u202e"))], width=60
    )
    assert error is None
    assert all(cell_len(frame) <= 59 for frame in frames(output))
    assert "\n" not in output and "\u202e" not in output
    assert "\x1b[31m" not in output


def test_shorter_phase_frame_erases_previous_bar(wait):
    output, _, error = wait([(503, weights()), (503, {"phase": "initializing"})])
    assert error is None
    assert "\r\x1b[2K⠋ Initializing…" in output
    assert output.endswith("\r\x1b[2K")


@pytest.mark.parametrize("exited", [False, True])
def test_error_paths_erase_line(wait, exited):
    output, _, error = wait([(503, weights())], timeout=0.5, exited=exited)
    assert isinstance(error, RuntimeError if exited else TimeoutError)
    assert output.endswith("\r\x1b[2K")
