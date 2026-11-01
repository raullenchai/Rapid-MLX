"""Offline cancellation coverage with real executor workers and signals."""

import os
import selectors
import signal
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from rapid_mlx import _download_cancel as cancel
from rapid_mlx._download_gate import confirm_or_abort


@pytest.mark.parametrize(
    ("mode", "signum", "code", "hint"),
    [
        ("pull", signal.SIGINT, 130, "Retry: rapid-mlx pull o/m"),
        ("serve", signal.SIGTERM, 143, "Run the same command again to continue."),
        # A background job starts with SIGINT ignored; SIGTERM must still cancel.
        ("serve-sigint-ignored", signal.SIGTERM, 143, "Run the same command again"),
    ],
)
def test_transfer_signal_exits_promptly(tmp_path, mode, signum, code, hint):
    blobs = tmp_path / "models--o--m" / "blobs"
    blobs.mkdir(parents=True)
    owned = blobs / ("a" * 64 + ".1234abcd.incomplete")
    foreign = blobs / ("b" * 64 + ".abcd1234.incomplete")
    legacy = blobs / ("c" * 64 + ".incomplete")
    part = tmp_path / ".rapid-mlx-mirror" / "x.part"
    part.parent.mkdir()
    for path in (foreign, legacy, part):
        path.write_bytes(b"keep")
    script = textwrap.dedent(
        """
        import os
        import time
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        from types import SimpleNamespace
        import huggingface_hub
        from rapid_mlx import cli
        import rapid_mlx._download_gate as gate

        def transfer(*args, **kwargs):
            def blocked():
                path = Path(os.environ["OWNED"])
                with path.open("wb") as stream:
                    stream.write(b"scratch")
                    stream.flush()
                    print("READY", flush=True)
                    time.sleep(30)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(blocked).result()

        huggingface_hub.snapshot_download = transfer
        huggingface_hub.model_info = lambda *a, **kw: SimpleNamespace(sha="0"*40, siblings=[])
        cli._try_mirror_prefetch = lambda *a, **kw: False
        cli._resolve_variant_allow_patterns = lambda *a, **kw: None
        cli._cache_runnability = lambda *a: False
        cli._check_disk_space = lambda *a, **kw: None
        cli._offline_hub_mode_active = lambda: False
        gate.reap_orphan_incomplete_blobs = lambda *a, **kw: (0, 0)
        if os.environ["MODE"] == "pull":
            cli._pull_repository(SimpleNamespace(model="o/m"))
        else:
            cli._ensure_model_downloaded("o/m")
        """
    )
    env = {
        **os.environ,
        "HF_HUB_CACHE": str(tmp_path),
        "OWNED": str(owned),
        "MODE": mode,
        "RAPID_MLX_TELEMETRY": "0",
        "RAPID_MLX_DISABLE_VERSION_CHECK": "1",
        "RAPID_MLX_AUTO_PULL": "1",
    }
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        # Pin the child's starting SIGINT disposition so the test does not
        # depend on whether the runner itself ignores SIGINT.
        preexec_fn=(
            (lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
            if mode == "serve-sigint-ignored"
            else (lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        ),
    )
    try:
        # Read bytes so buffered readline cannot hide the READY marker.
        output = b""
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 5
            while b"READY\n" not in output:
                remaining = deadline - time.monotonic()
                assert remaining > 0, "transfer did not become ready"
                assert selector.select(remaining), "transfer did not become ready"
                chunk = os.read(process.stdout.fileno(), 4096)
                assert chunk, "subprocess exited before READY"
                output += chunk
        started = time.monotonic()
        process.send_signal(signum)
        _, stderr = process.communicate(timeout=5)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    stderr = stderr.decode()
    assert elapsed < 5
    assert process.returncode == code, stderr
    assert "\n  Cancelling download…" in stderr
    assert "  Download cancelled. Completed files stay cached." in stderr
    assert hint in stderr
    assert "Traceback" not in stderr
    assert "KeyboardInterrupt" not in stderr
    if mode != "pull":
        assert "rapid-mlx pull" not in stderr
    assert not owned.exists()
    for path in (foreign, legacy, part):
        assert path.read_bytes() == b"keep"


@pytest.fixture
def default_signals():
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    signal.signal(signal.SIGINT, signal.default_int_handler)
    yield previous
    for sig, handler in previous.items():
        signal.signal(sig, handler)


@pytest.mark.parametrize(
    "error", [None, ValueError("bad"), SystemExit(1), KeyboardInterrupt()]
)
def test_result_and_exception_identity(default_signals, error):
    previous = {sig: signal.getsignal(sig) for sig in default_signals}
    value = object()

    def operation():
        assert threading.current_thread().name == "rapid-mlx-download"
        assert threading.current_thread().daemon
        if error is not None:
            raise error
        return value

    if error is None:
        assert cancel.run_cancellable_download(operation, retry_hint="retry") is value
    else:
        with pytest.raises(type(error)) as raised:
            cancel.run_cancellable_download(operation, retry_hint="retry")
        assert raised.value is error
    for sig, handler in previous.items():
        assert signal.getsignal(sig) is handler


def test_non_main_thread_calls_directly():
    with ThreadPoolExecutor(max_workers=1) as pool:

        def operation():
            caller = threading.current_thread()
            return (
                cancel.run_cancellable_download(
                    threading.current_thread, retry_hint="retry"
                )
                is caller
            )

        assert pool.submit(operation).result(timeout=2)


def test_custom_handler_calls_directly(default_signals):
    signal.signal(signal.SIGINT, lambda *_: None)
    assert (
        cancel.run_cancellable_download(threading.current_thread, retry_hint="retry")
        is threading.main_thread()
    )


@pytest.mark.parametrize("second_signal", [signal.SIGINT, signal.SIGTERM])
def test_second_signal_hard_exits(monkeypatch, default_signals, second_signal):
    release = threading.Event()
    exits = []

    def first_signal(_self):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

    def cleanup():
        signal.getsignal(second_signal)(second_signal, None)

    monkeypatch.setattr(cancel.threading.Thread, "start", first_signal)
    monkeypatch.setattr(cancel, "_remove_owned_incomplete_files", cleanup)

    def hard_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(cancel.os, "_exit", hard_exit)
    monkeypatch.setattr(cancel, "_hard_exit_cancelled", lambda code: None)
    try:
        with pytest.raises(SystemExit) as raised:
            cancel.run_cancellable_download(lambda: release.wait(), retry_hint="retry")
        assert raised.value.code == 130
        assert exits == [130]
    finally:
        release.set()


@pytest.mark.parametrize("suggest_pull", [True, False])
def test_abort_prompt(monkeypatch, capsys, suggest_pull):
    monkeypatch.delenv("TEST_AUTO_YES", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(SystemExit) as raised:
        confirm_or_abort(
            "o/m",
            100,
            threshold_bytes=1,
            auto_yes_env="TEST_AUTO_YES",
            suggest_pull=suggest_pull,
        )
    assert raised.value.code == 1
    output = capsys.readouterr().out
    if suggest_pull:
        assert "Use 'rapid-mlx pull o/m'" in output
    else:
        assert "  Aborted. Set TEST_AUTO_YES=1 to skip this prompt." in output
        assert "rapid-mlx pull" not in output
