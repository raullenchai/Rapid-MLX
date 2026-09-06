#!/usr/bin/env python3
"""Qualify FLUX.2 Klein q4 and BF16 on real Apple M1/M2 hardware.

The harness runs the real server and OpenAI-compatible image routes. It is
fail-closed: both immutable checkpoints must already be cached, the host must
identify as Apple M1/M2 with at least 32 GiB, every response must be a
non-uniform PNG of the requested size, and output hashes must be stable within
each precision/workload pair. It never downloads weights or changes Rapid's
default precision policy.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PIL import Image, ImageOps

GIB = 1 << 30
ALIASES = {
    "q4": "flux2-klein-4b",
    "bf16": "flux2-klein-4b-bf16",
}
DEFAULT_SEQUENCE = ("q4", "bf16", "bf16", "q4")
DEFAULT_WORKLOADS = ("generate:512x512:4", "generate:1024x1024:4", "edit:1024x1024:4")


def command(*args: str, timeout: float = 30, check: bool = True) -> str:
    return subprocess.run(
        args, check=check, capture_output=True, text=True, timeout=timeout
    ).stdout.strip()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def parse_workload(value: str) -> dict[str, object]:
    try:
        operation, size, steps_text = value.split(":", 2)
        width_text, height_text = size.lower().split("x", 1)
        width, height, steps = int(width_text), int(height_text), int(steps_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "workload must be OPERATION:WIDTHxHEIGHT:STEPS"
        ) from exc
    if operation not in {"generate", "edit"}:
        raise argparse.ArgumentTypeError("operation must be generate or edit")
    if width < 256 or height < 256 or width % 64 or height % 64:
        raise argparse.ArgumentTypeError("dimensions must be >=256 and divisible by 64")
    if not 1 <= steps <= 100:
        raise argparse.ArgumentTypeError("steps must be between 1 and 100")
    return {
        "id": f"{operation}-{width}x{height}-{steps}",
        "operation": operation,
        "width": width,
        "height": height,
        "steps": steps,
    }


def parse_sequence(value: str) -> tuple[str, ...]:
    sequence = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not sequence or any(item not in ALIASES for item in sequence):
        raise argparse.ArgumentTypeError("sequence may contain only q4 and bf16")
    if sequence.count("q4") != sequence.count("bf16"):
        raise argparse.ArgumentTypeError("sequence must balance q4 and bf16 sessions")
    return sequence


def json_request(url: str, payload: dict | None = None, timeout: float = 900) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def multipart_request(
    url: str, fields: dict[str, str], image_bytes: bytes, timeout: float = 900
) -> dict:
    boundary = f"rapid-image-precision-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        b'Content-Disposition: form-data; name="image"; filename="source.png"\r\n'
        b"Content-Type: image/png\r\n\r\n"
    )
    body.extend(image_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    request = urllib.request.Request(url, data=bytes(body))
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_png(response: dict, width: int, height: int) -> tuple[str, int]:
    try:
        raw = base64.b64decode(response["data"][0]["b64_json"], validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError("response did not contain a decodable PNG") from exc
    if image.format != "PNG" or image.size != (width, height):
        raise RuntimeError(
            f"unexpected image format/size: {image.format} {image.size}; "
            f"expected PNG {(width, height)}"
        )
    extrema = image.convert("RGB").getextrema()
    if all(low == high for low, high in extrema):
        raise RuntimeError("response PNG is uniform")
    return hashlib.sha256(raw).hexdigest(), len(raw)


def edit_source(width: int, height: int) -> bytes:
    gradient = Image.linear_gradient("L").resize((width, height))
    image = ImageOps.colorize(gradient, black="#173c70", white="#f4a261")
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def footprint(pid: int) -> tuple[float, float]:
    text = command("footprint", "-p", str(pid), timeout=30)
    current = re.search(r"Footprint:\s+([0-9.]+)\s+(MB|GB)", text)
    peak = re.search(r"phys_footprint_peak:\s+([0-9.]+)\s+(MB|GB)", text)
    if not current or not peak:
        raise RuntimeError("could not parse macOS footprint output")

    def gib(match: re.Match[str]) -> float:
        value = float(match.group(1))
        return value if match.group(2) == "GB" else value / 1024

    return round(gib(current), 3), round(gib(peak), 3)


def low_power_mode() -> int:
    text = command("pmset", "-g", "custom")
    ac_power = text.split("AC Power:", 1)[-1].split("Battery Power:", 1)[0]
    match = re.search(r"^\s*lowpowermode\s+(\d+)\s*$", ac_power, re.MULTILINE)
    if not match:
        raise RuntimeError("could not parse AC lowpowermode")
    return int(match.group(1))


def running_rapid_servers() -> list[int]:
    rows = command("ps", "-axo", "pid=,command=")
    matches = []
    for row in rows.splitlines():
        pid_text, _, process_command = row.strip().partition(" ")
        if not pid_text.isdigit() or int(pid_text) == os.getpid():
            continue
        if re.search(r"(?:rapid-mlx|vllm_mlx\.cli).*\bserve\b", process_command):
            matches.append(int(pid_text))
    return matches


def swap_used_mb() -> float:
    text = command("sysctl", "-n", "vm.swapusage")
    match = re.search(r"used = ([0-9.]+)([MG])", text)
    if not match:
        raise RuntimeError("could not parse vm.swapusage")
    value = float(match.group(1))
    return value * 1024 if match.group(2) == "G" else value


def wait_ready(proc: subprocess.Popen[str], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with status {proc.returncode}")
        try:
            json_request(f"http://127.0.0.1:{port}/healthz", timeout=2)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(1)
    raise TimeoutError(f"server was not ready after {timeout:.0f}s")


def ensure_port_free(port: int) -> None:
    # Binding can fail for a recently stopped server whose connections remain
    # in TIME_WAIT even though no process is listening. A real connect probes
    # the condition that would make the next server unsafe to start.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"benchmark port {port} is already in use")


def stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait(timeout=10)
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def request_image(
    port: int, alias: str, workload: dict[str, object], seed: int
) -> dict:
    width, height = int(workload["width"]), int(workload["height"])
    started = time.monotonic()
    if workload["operation"] == "generate":
        response = json_request(
            f"http://127.0.0.1:{port}/v1/images/generations",
            {
                "model": alias,
                "prompt": "A folded paper crane on a walnut desk, soft window light",
                "size": f"{width}x{height}",
                "steps": workload["steps"],
                "seed": seed,
            },
        )
    else:
        response = multipart_request(
            f"http://127.0.0.1:{port}/v1/images/edits",
            {
                "model": alias,
                "prompt": "Turn the scene into a warm sunset while preserving the composition",
                "steps": str(workload["steps"]),
                "seed": str(seed),
            },
            edit_source(width, height),
        )
    elapsed = time.monotonic() - started
    digest, byte_count = validate_png(response, width, height)
    return {"wall_s": round(elapsed, 3), "sha256": digest, "png_bytes": byte_count}


def cache_identity(hf_cache: Path, alias: str) -> dict[str, object]:
    from vllm_mlx._download_gate import IMAGE_MODEL_REVISIONS, mflux_missing_weights
    from vllm_mlx.model_aliases import resolve_model

    repo = resolve_model(alias)
    revision = IMAGE_MODEL_REVISIONS[repo]
    snapshot = (
        hf_cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    )
    if not snapshot.is_dir():
        raise RuntimeError(f"pinned snapshot is not cached: {repo}@{revision}")
    missing = mflux_missing_weights(repo)
    if missing != []:
        raise RuntimeError(f"checkpoint is incomplete: {repo}@{revision}: {missing!r}")
    return {
        "alias": alias,
        "repo": repo,
        "revision": revision,
        "snapshot_present": True,
    }


def server_environment(hf_cache: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["HF_HUB_CACHE"] = str(hf_cache.expanduser())
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def run_session(
    precision: str,
    session_index: int,
    workloads: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    alias = ALIASES[precision]
    log_path = Path(args.log_dir) / f"{session_index:02d}-{precision}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = server_environment(Path(args.hf_cache))
    swap_before = swap_used_mb()
    ensure_port_free(args.port)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-P",
                "-u",
                "-s",
                "-m",
                "vllm_mlx.cli",
                "--no-telemetry",
                "--no-banner",
                "serve",
                alias,
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        try:
            wait_ready(proc, args.port, args.load_timeout)
            samples: list[dict[str, object]] = []
            warmups: list[dict[str, object]] = []

            def session_record(status: str, swap_after: float) -> dict[str, object]:
                return {
                    "session": session_index,
                    "precision": precision,
                    "alias": alias,
                    "status": status,
                    "swap_before_mb": round(swap_before, 2),
                    "swap_after_mb": round(swap_after, 2),
                    "swap_delta_mb": round(max(0.0, swap_after - swap_before), 2),
                    "warmups": warmups,
                    "samples": samples,
                    "log": str(log_path),
                }

            for workload in workloads:
                warmup = request_image(args.port, alias, workload, args.seed)
                current_gib, peak_gib = footprint(proc.pid)
                warmups.append(
                    {
                        "workload": workload["id"],
                        **warmup,
                        "current_footprint_gib": current_gib,
                        "peak_footprint_gib": peak_gib,
                    }
                )
                swap_now = swap_used_mb()
                if swap_now - swap_before > args.abort_swap_mb:
                    return session_record("aborted_swap", swap_now)
                for repeat in range(args.repeats):
                    sample = request_image(args.port, alias, workload, args.seed)
                    current_gib, peak_gib = footprint(proc.pid)
                    samples.append(
                        {
                            "workload": workload["id"],
                            "repeat": repeat + 1,
                            **sample,
                            "current_footprint_gib": current_gib,
                            "peak_footprint_gib": peak_gib,
                        }
                    )
                    swap_now = swap_used_mb()
                    if swap_now - swap_before > args.abort_swap_mb:
                        return session_record("aborted_swap", swap_now)
                print(
                    f"session {session_index} {precision} {workload['id']}: "
                    f"warmup={warmup['wall_s']}s",
                    flush=True,
                )
            swap_after = swap_used_mb()
            return session_record("ok", swap_after)
        finally:
            stop(proc)
            time.sleep(args.cooldown)


def summarize(sessions: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for session in sessions:
        for sample in session["samples"]:  # type: ignore[index]
            key = (str(session["precision"]), str(sample["workload"]))
            grouped.setdefault(key, []).append(sample)
    rows = []
    for (precision, workload), samples in sorted(grouped.items()):
        hashes = {str(sample["sha256"]) for sample in samples}
        if len(hashes) != 1:
            raise RuntimeError(f"non-deterministic output for {precision}/{workload}")
        times = [float(sample["wall_s"]) for sample in samples]
        rows.append(
            {
                "precision": precision,
                "workload": workload,
                "samples": len(times),
                "median_wall_s": round(statistics.median(times), 3),
                "p95_wall_s_nearest_rank": sorted(times)[
                    max(0, int(0.95 * len(times) + 0.999) - 1)
                ],
                "peak_footprint_gib": max(
                    float(sample["peak_footprint_gib"]) for sample in samples
                ),
                "sha256": hashes.pop(),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-git-sha", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument(
        "--hf-cache", required=True, help="concrete HF Hub cache directory"
    )
    parser.add_argument("--workload", action="append", type=parse_workload)
    parser.add_argument("--sequence", type=parse_sequence, default=DEFAULT_SEQUENCE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--port", type=int, default=18792)
    parser.add_argument("--load-timeout", type=float, default=300)
    parser.add_argument("--cooldown", type=float, default=15)
    parser.add_argument("--max-start-swap-mb", type=float, default=256)
    parser.add_argument("--abort-swap-mb", type=float, default=256)
    parser.add_argument("--log-dir", default="/tmp/rapid-image-precision-logs")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("this qualification harness requires macOS")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    chip = command("sysctl", "-n", "machdep.cpu.brand_string")
    ram_gib = int(command("sysctl", "-n", "hw.memsize")) / GIB
    if not re.fullmatch(r"Apple M[12](?: Pro| Max| Ultra)?", chip):
        parser.error(f"host is not an Apple M1/M2 family system: {chip}")
    if ram_gib < 32:
        parser.error(
            f"BF16 qualification requires at least 32 GiB; host has {ram_gib:.1f}"
        )
    power_mode = low_power_mode()
    if power_mode != 0:
        parser.error("qualification requires AC low-power mode to be off")
    active_servers = running_rapid_servers()
    if active_servers:
        parser.error(
            "qualification requires an otherwise-idle host; active Rapid server PID: "
            f"{active_servers[0]}"
        )
    start_swap_mb = swap_used_mb()
    if start_swap_mb > args.max_start_swap_mb:
        parser.error(
            f"qualification requires a clean swap baseline; host already uses "
            f"{start_swap_mb:.1f} MiB (limit {args.max_start_swap_mb:.1f} MiB)"
        )

    hf_cache = Path(args.hf_cache).expanduser()
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    identities = {
        precision: cache_identity(hf_cache, alias)
        for precision, alias in ALIASES.items()
    }
    workloads = args.workload or [parse_workload(value) for value in DEFAULT_WORKLOADS]
    result: dict[str, object] = {
        "schema_version": 1,
        "environment": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chip": chip,
            "physical_ram_gib": round(ram_gib, 2),
            "macos": platform.mac_ver()[0],
            "low_power_mode": power_mode,
            "start_swap_mb": round(start_swap_mb, 2),
            "python": platform.python_version(),
            "rapid_mlx": package_version("rapid-mlx"),
            "mflux": package_version("mflux"),
            "mlx": package_version("mlx"),
            "pillow": package_version("Pillow"),
            "candidate_git_sha": args.candidate_git_sha,
            "wheel_sha256": args.wheel_sha256,
        },
        "method": {
            "sequence": list(args.sequence),
            "repeats_per_session": args.repeats,
            "warmup_per_workload_per_session": 1,
            "seed": args.seed,
            "workloads": workloads,
            "models_are_sequential": True,
            "downloads_allowed": False,
            "max_start_swap_mb": args.max_start_swap_mb,
            "abort_swap_mb": args.abort_swap_mb,
        },
        "artifacts": identities,
        "sessions": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for session_index, precision in enumerate(args.sequence, start=1):
            session = run_session(precision, session_index, workloads, args)
            result["sessions"].append(session)  # type: ignore[union-attr]
            output.write_text(json.dumps(result, indent=2) + "\n")
            if session["status"] != "ok":
                raise RuntimeError(
                    f"session {session_index} {precision} aborted after "
                    f"{session['swap_delta_mb']} MiB new swap"
                )
        result["summary"] = summarize(result["sessions"])  # type: ignore[arg-type]
        result["status"] = "pass"
        output.write_text(json.dumps(result, indent=2) + "\n")
        return 0
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        output.write_text(json.dumps(result, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
