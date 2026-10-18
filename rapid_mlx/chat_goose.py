"""Run the pinned Goose CLI against the same gated provider as Desktop Chat."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from rapid_mlx import goose_runtime

GOOSE_VERSION = "1.54.0"
GOOSE_URL = (
    "https://github.com/aaif-goose/goose/releases/download/v1.54.0/"
    "goose-aarch64-apple-darwin.tar.gz"
)
GOOSE_SHA256 = "feda29744cd4e13f9a5211a968ed47c67fb8fa79647f4d271d92d564998a393c"
GOOSE_BINARY_SHA256 = "4b8258c24adad9e07957d1dddfc7d965b97883dce2eff64af7e3d3c005655d81"
BUILTINS = "developer,computercontroller,memory,autovisualiser,tutorial"
ASSETS = Path(goose_runtime.__file__).parent


def _cache_root() -> Path:
    return Path.home() / ".cache" / "rapid-mlx" / "goose" / GOOSE_VERSION


def _goose_binary() -> Path | None:
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        return None
    root = _cache_root()
    binary = root / "goose"
    stamp = root / "archive.sha256"
    if binary.is_file() and os.access(binary, os.X_OK) and stamp.is_file():
        if (
            stamp.read_text().strip() == GOOSE_SHA256
            and _sha256(binary) == GOOSE_BINARY_SHA256
        ):
            return binary
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    archive = root / "goose.tar.gz"
    if not archive.is_file() or _sha256(archive) != GOOSE_SHA256:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as target:
            pending = Path(target.name)
            try:
                with urllib.request.urlopen(GOOSE_URL, timeout=120) as source:
                    shutil.copyfileobj(source, target)
            except BaseException:
                pending.unlink(missing_ok=True)
                raise
        if _sha256(pending) != GOOSE_SHA256:
            pending.unlink(missing_ok=True)
            raise ValueError("Goose archive checksum mismatch")
        pending.replace(archive)
    with tarfile.open(archive, "r:gz") as package:
        entries = [m for m in package if m.isfile() and Path(m.name).name == "goose"]
        if len(entries) != 1:
            raise ValueError("Goose archive has no unique executable")
        source = package.extractfile(entries[0])
        if source is None:
            raise ValueError("Goose executable could not be read")
        with source, tempfile.NamedTemporaryFile(dir=root, delete=False) as target:
            pending = Path(target.name)
            shutil.copyfileobj(source, target)
    pending.chmod(0o755)
    if _sha256(pending) != GOOSE_BINARY_SHA256:
        pending.unlink(missing_ok=True)
        raise ValueError("Goose executable checksum mismatch")
    pending.replace(binary)
    stamp.write_text(GOOSE_SHA256 + "\n")
    return binary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ready(port: int, process: subprocess.Popen, timeout: float = 10) -> None:
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError("Goose inference helper exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("Goose inference helper did not become ready")


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def _run_goose(command: list[str], *, cwd: str, env: dict[str, str]) -> int:
    """Keep Goose's terminal streams while the watchdog owns its tool tree."""
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            str(ASSETS / "helper_watchdog.py"),
            str(os.getpid()),
            *command,
        ],
        cwd=cwd,
        env=env,
    )
    try:
        while True:
            try:
                return process.wait()
            except KeyboardInterrupt:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
    finally:
        _stop(process)


def _local_port(base_url: str) -> int | None:
    url = urlsplit(base_url)
    if (
        url.scheme != "http"
        or url.hostname not in {"localhost", "127.0.0.1"}
        or url.username is not None
        or url.password is not None
        or url.path.rstrip("/") not in {"", "/v1"}
    ):
        return None
    return url.port or 80


def _served_context_limit(base_url: str, model: str, api_key: str) -> int:
    """Use the server's advertised model window, as Desktop Chat does."""
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/models", headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            models = json.load(response).get("data", [])
        for entry in models:
            if entry.get("id") == model:
                limit = entry.get("context_window")
                if isinstance(limit, int) and limit > 0:
                    return limit
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return 32768


def _metric_sequence(port: int, key: str) -> int:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/metrics", headers={"X-Metrics-Key": key}
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return int(json.load(response)["sequence"])


def _private_directory(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError("Goose workspace contains a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("Goose workspace is not a directory")


def _write_owned(path: Path, content: str) -> None:
    if path.is_symlink():
        raise RuntimeError("Goose workspace contains a symlink")
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
        stream.write(content)
    pending.replace(path)


def _fallback(reason: str) -> bool:
    print(f"  Goose unavailable ({reason}); using built-in chat.", file=sys.stderr)
    return False


def run(args, base_url: str, model: str, *, max_tokens_explicit: bool) -> bool:
    """Return false only when the old REPL should handle this session."""
    script_mode = not sys.stdin.isatty()
    unsupported = []
    if getattr(args, "mcp_config", None):
        unsupported.append("--mcp-config")
    if getattr(args, "mcp_max_rounds", 8) != 8:
        unsupported.append("--mcp-max-rounds")
    if getattr(args, "response_timeout", 600) != 600:
        unsupported.append("--response-timeout")
    if unsupported:
        return _fallback(f"{', '.join(unsupported)} is not supported by Goose chat")
    rapid_port = _local_port(base_url)
    if rapid_port is None:
        return _fallback("Goose chat requires a local HTTP Rapid server")
    try:
        goose = _goose_binary()
    except (OSError, ValueError, tarfile.TarError) as exc:
        return _fallback(str(exc))
    if goose is None:
        return _fallback("Goose 1.54.0 requires macOS arm64")

    root = (
        _cache_root() / "workspaces" / hashlib.sha256(os.getcwd().encode()).hexdigest()
    )
    launch_id = secrets.token_hex(8)
    run_root = root / "runs" / launch_id
    provider_name = "custom_rapid_local_" + launch_id
    provider_path = root / "config" / "custom_providers" / (provider_name + ".json")
    contract_path = run_root / "chat-contract.json"
    for directory in (
        root,
        root / "config",
        root / "config" / "custom_providers",
        root / "runs",
        run_root,
    ):
        _private_directory(directory)
    scoped_key, proxy_key = secrets.token_hex(32), secrets.token_hex(32)
    gate_port, bridge_port = _port(), _port()
    while bridge_port == gate_port:
        bridge_port = _port()
    upstream_key = os.environ.get("RAPID_MLX_API_KEY", "")
    provider = {
        "name": provider_name,
        "engine": "openai",
        "display_name": "Rapid Local",
        "api_key_env": "RAPID_LOCAL_WORK_KEY",
        "base_url": f"http://127.0.0.1:{bridge_port}/v1/chat/completions",
        "models": [
            {
                "name": model,
                "context_limit": args.context_length
                or _served_context_limit(base_url, model, upstream_key),
            }
        ],
        "supports_streaming": True,
        "requires_auth": True,
        "dynamic_models": False,
    }
    _write_owned(provider_path, json.dumps(provider))
    contract = {
        "system_prompt": args.system or "",
        "use_goose_sampling_defaults": not (
            getattr(args, "_temperature_was_explicit", args.temperature != 0.7)
            or getattr(args, "_think_was_explicit", args.think)
            or max_tokens_explicit
        ),
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "repetition_penalty": 1.0,
        "enable_thinking": args.think,
    }
    _write_owned(contract_path, json.dumps(contract))
    shell = run_root / "bash"
    if shell.is_symlink():
        raise RuntimeError("Goose workspace contains a symlink")
    if not shell.exists() or _sha256(shell) != _sha256(ASSETS / "helper_watchdog.py"):
        shutil.copyfile(ASSETS / "helper_watchdog.py", shell)
        shell.chmod(0o700)
    common = {
        "HOME": str(root),
        "GOOSE_PATH_ROOT": str(root),
        "PATH": str(Path(sys.executable).parent)
        + os.pathsep
        + os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        "LANG": "en_US.UTF-8",
        "NO_PROXY": "127.0.0.1,localhost",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    gate_env = {
        **common,
        "WORK_PROXY_BEARER": proxy_key,
        "WORK_PROXY_UPSTREAM_BEARER": upstream_key or proxy_key,
        "WORK_PROXY_METRICS_KEY": secrets.token_hex(32),
    }
    bridge_env = {
        **common,
        "WORK_BRIDGE_SCOPED_KEY": scoped_key,
        "WORK_BRIDGE_UPSTREAM_KEY": proxy_key,
        "WORK_BRIDGE_CONTRACT_FILE": str(contract_path),
    }
    goose_env = {
        **common,
        "RAPID_LOCAL_WORK_KEY": scoped_key,
        "GOOSE_PROVIDER": provider_name,
        "GOOSE_MODEL": model,
        "GOOSE_MODE": "approve",
        "GOOSE_SHELL": str(shell),
        "GOOSE_TELEMETRY_OFF": "1",
        "GOOSE_TELEMETRY_ENABLED": "false",
        "GOOSE_DISABLE_KEYRING": "1",
        "WORK_HELPER_OWNERSHIP_TOKEN": secrets.token_hex(16),
    }
    processes: list[subprocess.Popen] = []
    try:
        gate = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(ASSETS / "inference_proxy.py"),
                str(gate_port),
                str(rapid_port),
                model,
                "--full-access",
                "--goose-chat",
            ],
            cwd=root,
            env=gate_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(gate)
        _ready(gate_port, gate)
        bridge = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(ASSETS / "helper_watchdog.py"),
                "--credential-bridge",
                str(bridge_port),
                str(gate_port),
            ],
            cwd=root,
            env=bridge_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(bridge)
        _ready(bridge_port, bridge)
        if not args.disable_prefix_cache and not args.system:
            body = json.loads((ASSETS / "prefix.json").read_text())
            body.update(model=model, max_tokens=1, temperature=0, stream=True)
            request = urllib.request.Request(
                f"http://127.0.0.1:{gate_port}/v1/chat/completions",
                json.dumps(body).encode(),
                {
                    "Authorization": "Bearer " + proxy_key,
                    "Content-Type": "application/json",
                },
            )
            print(
                "  Warming Goose Chat prefix …",
                file=sys.stderr if script_mode else sys.stdout,
                flush=True,
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    response.read()
            except OSError as exc:
                print(f"  Prefix warmup skipped: {exc}", file=sys.stderr)
        if script_mode:
            first_prompt = sys.stdin.read().strip()
            if not first_prompt:
                return _fallback("piped chat input is empty")
        else:
            print(
                "  Goose Chat ready. Use Goose's approval prompts for tool calls.",
                flush=True,
            )
            while True:
                try:
                    first_prompt = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return True
                if first_prompt.lower() in {"/exit", "exit", "quit"}:
                    return True
                if first_prompt:
                    break
        metric_before = _metric_sequence(gate_port, gate_env["WORK_PROXY_METRICS_KEY"])
        command = [
            str(goose),
            "run",
            "--provider",
            provider_name,
            "--model",
            model,
            "--with-builtin",
            BUILTINS,
            "--output-format",
            args.output_format,
            "--text",
            first_prompt,
        ]
        if not script_mode:
            command.insert(command.index("--with-builtin"), "--interactive")
        if args.resume:
            command.append("--resume")
        args._goose_pending_prompt = first_prompt
        goose_status = _run_goose(command, cwd=os.getcwd(), env=goose_env)
        if goose_status:
            if (
                _metric_sequence(gate_port, gate_env["WORK_PROXY_METRICS_KEY"])
                == metric_before
            ):
                return _fallback(
                    f"Goose exited before starting (status {goose_status})"
                )
            print(f"  Goose exited with status {goose_status}.", file=sys.stderr)
        args._goose_pending_prompt = None
        return True
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"  Goose startup failed ({exc}); using built-in chat.", file=sys.stderr)
        return False
    finally:
        for process in reversed(processes):
            _stop(process)
        for path in (shell, contract_path, provider_path):
            path.unlink(missing_ok=True)
        try:
            run_root.rmdir()
        except OSError:
            pass
