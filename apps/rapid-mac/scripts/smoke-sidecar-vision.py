#!/usr/bin/env python3
"""Run one real image request through a freshly built Desktop sidecar."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


def _bound_local_listener() -> socket.socket:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    return listener


def _completion_matches(text: object, expected: str) -> bool:
    """Accept only the deterministic class requested for a fixture."""
    if not isinstance(text, str):
        return False
    return text.strip().rstrip(".!?").casefold() == expected.casefold()


def _request_json(url: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return json.load(response)


def _wait_until_ready(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sidecar server exited early with {process.returncode}")
        try:
            _request_json(f"{base_url}/v1/models", None, 2)
            return
        except (OSError, ValueError, RuntimeError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"sidecar server was not ready after {timeout}s: {last_error}")


def _resolve_model(model: str, revision: str | None) -> Path:
    local_path = Path(model)
    if local_path.exists():
        return local_path
    if not revision:
        raise SystemExit(
            "vision smoke: a repository model requires --revision so the "
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
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--negative-image", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--request-timeout", type=float, default=240)
    args = parser.parse_args()

    executable = args.sidecar_root / "bin" / "rapid-mlx"
    for path, label in (
        (executable, "sidecar executable"),
        (args.image, "image"),
        (args.negative_image, "negative image"),
    ):
        if not path.exists():
            raise SystemExit(f"vision smoke: {label} not found: {path}")
    model = _resolve_model(args.model, args.revision)

    listener = _bound_local_listener()
    base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    process: subprocess.Popen | None = None
    with tempfile.NamedTemporaryFile(
        prefix="rapid-sidecar-vision-", suffix=".log", delete=False
    ) as log:
        log_path = Path(log.name)

    try:
        with log_path.open("ab") as log:
            try:
                process = subprocess.Popen(
                    [
                        str(executable),
                        "serve",
                        str(model),
                        "--mllm",
                        "--listen-fd",
                        str(listener.fileno()),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(listener.fileno(),),
                    env={**os.environ, "PYTHONNOUSERSITE": "1"},
                )
            finally:
                # Popen duplicates pass_fds into the child before it returns.
                # Close the parent's copy only after that atomic handoff.
                listener.close()

        _wait_until_ready(base_url, process, args.startup_timeout)
        for image_path, expected in (
            (args.image, "SPOTTED_CAT"),
            (args.negative_image, "OTHER"),
        ):
            image = base64.b64encode(image_path.read_bytes()).decode("ascii")
            payload = {
                "model": str(model),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Classify the main subject in this image. Reply "
                                    "with exactly SPOTTED_CAT for a spotted wild cat "
                                    "such as a cheetah or leopard, or OTHER for "
                                    "anything else."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + image},
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
            }
            body = _request_json(
                f"{base_url}/v1/chat/completions", payload, args.request_timeout
            )
            content = body.get("choices", [{}])[0].get("message", {}).get("content")
            if not _completion_matches(content, expected):
                raise RuntimeError(
                    f"vision smoke expected {expected} for {image_path.name}, "
                    f"got: {content!r}"
                )
        print("vision smoke: HTTP 200; fixture verdicts=SPOTTED_CAT,OTHER")
        return 0
    except Exception:
        print(f"vision smoke server log: {log_path}")
        print(log_path.read_text(errors="replace"))
        raise
    finally:
        listener.close()
        if process is not None:
            _stop_process(process)
        log_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
