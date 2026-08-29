#!/usr/bin/env python3
"""Generate one real image through a freshly assembled Desktop sidecar."""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

_SIZE = re.compile(r"([1-9][0-9]*)x([1-9][0-9]*)")


class _RequestDeadlineExceededError(RuntimeError):
    """The complete HTTP exchange exceeded its wall-clock budget."""


@contextlib.contextmanager
def _wall_clock_deadline(timeout: float):
    """Interrupt socket reads that make progress without ever completing."""

    def _expire(_signum, _frame) -> None:
        raise _RequestDeadlineExceededError(
            f"HTTP request exceeded {timeout:g}s wall-clock deadline"
        )

    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, _expire)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        previous_remaining, previous_interval = previous_timer
        if previous_remaining > 0:
            previous_remaining = max(
                1e-6, previous_remaining - (time.monotonic() - started)
            )
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, previous_remaining, previous_interval)


def _bound_local_listener() -> socket.socket:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    return listener


def _request_json(url: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    # urllib's timeout applies per socket operation. A peer that keeps sending
    # partial bytes can therefore keep json.load() alive forever; the release
    # gate needs a deadline for the entire render response.
    with (
        _wall_clock_deadline(timeout),
        urllib.request.urlopen(request, timeout=timeout) as response,
    ):
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        body = json.load(response)
    if not isinstance(body, dict):
        raise RuntimeError(f"{url} returned a non-object JSON response")
    return body


def _wait_until_ready(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sidecar server exited early with {process.returncode}")
        try:
            health = _request_json(f"{base_url}/health", None, 2)
            if health.get("ready") is True and health.get("engine_type") == "image":
                return
            last_error = RuntimeError(f"unexpected health payload: {health!r}")
        except (OSError, ValueError, RuntimeError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"image sidecar was not ready after {timeout}s: {last_error}")


def _resolve_model(model: str, revision: str | None) -> Path:
    local_path = Path(model)
    if local_path.exists():
        return local_path
    if not revision:
        raise SystemExit(
            "image smoke: a repository model requires --revision so the "
            "release proof is content-addressed"
        )
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=True,
        )
    )


def _serve_model(model: str, snapshot: Path, revision: str | None) -> str:
    if revision is None:
        return str(snapshot)

    # Image-family routing is registry-backed. Passing snapshots/<sha> to the
    # CLI discards that modality metadata and makes a component-layout FLUX
    # checkpoint look like a root-sharded text model. Keep the repository ID
    # on the wire, but first prove the production resolver will hand mflux the
    # same content-addressed snapshot that the release manifest selected.
    from vllm_mlx._download_gate import mflux_local_snapshot

    runtime_snapshot = mflux_local_snapshot(model)
    if runtime_snapshot is None:
        raise RuntimeError(
            f"image runtime cannot resolve a complete offline snapshot for {model}"
        )
    if Path(runtime_snapshot).resolve() != snapshot.resolve():
        raise RuntimeError(
            "image runtime snapshot does not match the release pin: "
            f"runtime={runtime_snapshot}, pinned={snapshot}"
        )
    return model


def _parse_size(value: str) -> tuple[int, int]:
    match = _SIZE.fullmatch(value)
    if not match:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    return int(match.group(1)), int(match.group(2))


def _validate_generated_png(body: dict, expected_size: tuple[int, int]) -> int:
    if body.get("cancelled") is True:
        raise RuntimeError("image request reported cancellation")
    data = body.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise RuntimeError("image response must contain exactly one data item")
    encoded = data[0].get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError("image response has no non-empty b64_json")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("image response b64_json is invalid base64") from exc

    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.format != "PNG":
                raise RuntimeError(
                    f"image response format is {image.format}, expected PNG"
                )
            if image.size != expected_size:
                raise RuntimeError(
                    f"image response is {image.size}, expected {expected_size}"
                )
            extrema = image.convert("RGB").getextrema()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("image response is not a decodable PNG") from exc
    if not any(low < high for low, high in extrema):
        raise RuntimeError(
            "image response is uniform; no generated detail was produced"
        )
    return len(payload)


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--size", type=_parse_size, default=_parse_size("512x512"))
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--request-timeout", type=float, default=300)
    args = parser.parse_args()

    executable = args.sidecar_root / "bin" / "rapid-mlx"
    if not executable.exists():
        raise SystemExit(f"image smoke: sidecar executable not found: {executable}")
    snapshot = _resolve_model(args.model, args.revision)
    serve_model = _serve_model(args.model, snapshot, args.revision)
    size_text = f"{args.size[0]}x{args.size[1]}"

    listener = _bound_local_listener()
    base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    process: subprocess.Popen | None = None
    with tempfile.NamedTemporaryFile(
        prefix="rapid-sidecar-image-", suffix=".log", delete=False
    ) as log:
        log_path = Path(log.name)

    try:
        with log_path.open("ab") as log:
            try:
                process = subprocess.Popen(
                    [
                        str(executable),
                        "serve",
                        serve_model,
                        "--listen-fd",
                        str(listener.fileno()),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(listener.fileno(),),
                    env={
                        **os.environ,
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONNOUSERSITE": "1",
                    },
                )
            finally:
                listener.close()

        _wait_until_ready(base_url, process, args.startup_timeout)
        started = time.monotonic()
        body = _request_json(
            f"{base_url}/v1/images/generations",
            {
                "model": serve_model,
                "prompt": (
                    "A red sailboat on a calm alpine lake at sunrise, "
                    "clean product illustration"
                ),
                "n": 1,
                "size": size_text,
                "response_format": "b64_json",
                "seed": 42,
            },
            args.request_timeout,
        )
        png_bytes = _validate_generated_png(body, args.size)
        elapsed = time.monotonic() - started
        print(
            f"image smoke: HTTP 200; PNG={size_text}; bytes={png_bytes}; "
            f"elapsed={elapsed:.2f}s"
        )
        return 0
    except Exception:
        if process is not None:
            # Flush the most relevant final failure output before surfacing it.
            _stop_process(process)
        print(f"image smoke server log: {log_path}")
        print(log_path.read_text(errors="replace"))
        raise
    finally:
        listener.close()
        if process is not None:
            _stop_process(process)
        log_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
