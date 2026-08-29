#!/usr/bin/env python3
"""Serialize large model commands and re-check memory while holding the lock."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
ALIASES = (
    SCRIPT_DIR / "aliases.json"
    if (SCRIPT_DIR / "aliases.json").is_file()
    else ROOT / "vllm_mlx/aliases.json"
)
SIZES = (
    SCRIPT_DIR / "model_sizes.json"
    if (SCRIPT_DIR / "model_sizes.json").is_file()
    else ROOT / "vllm_mlx/model_sizes.json"
)
DEFAULT_LOCK = Path("/var/tmp/rapid-mlx-large-model.lock")
GIB = 1024**3


def estimate_gib(model: str) -> float:
    aliases = json.loads(ALIASES.read_text())
    repository = aliases.get(model, {}).get("hf_path", model)
    sizes = json.loads(SIZES.read_text())["sizes"]
    size = sizes.get(repository)
    if not isinstance(size, int) or size <= 0:
        raise ValueError(
            f"model size is unknown for {model!r}; pass --working-set-gb explicitly"
        )
    return size * 1.5 / GIB


def available_gib() -> float:
    override = os.environ.get("RAPID_LARGE_MODEL_TEST_AVAILABLE_GB")
    if override is not None:
        if os.environ.get("RAPID_HOST_SAFETY_TESTING") != "1":
            raise ValueError(
                "test memory override requires RAPID_HOST_SAFETY_TESTING=1"
            )
        return float(override)
    try:
        import psutil

        return psutil.virtual_memory().available / GIB
    except ImportError:
        page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"]))
        vm = subprocess.check_output(["vm_stat"], text=True)
        pages = 0
        for line in vm.splitlines():
            if line.startswith(("Pages free:", "Pages inactive:", "Pages purgeable:")):
                pages += int(line.rsplit(None, 1)[-1].rstrip("."))
        return pages * page_size / GIB


def run(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model")
    choice.add_argument("--working-set-gb", type=float)
    parser.add_argument("--threshold-gb", type=float, default=20.0)
    parser.add_argument("--reserve-gb", type=float, default=4.0)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--lock-timeout", type=int, default=7200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(raw_argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    required = (
        args.working_set_gb
        if args.working_set_gb is not None
        else estimate_gib(args.model)
    )
    if required <= 0:
        parser.error("working-set size must be positive")
    if required <= args.threshold_gb:
        os.execvp(command[0], command)

    if os.environ.get("RAPID_LARGE_MODEL_LOCK_HELD") != "1":
        lockf = shutil.which("lockf")
        if lockf is None:
            print(
                "large-model-run: lockf is required for a large model load",
                file=sys.stderr,
            )
            return 2
        print(
            f"large-model-run: waiting for host lock ({required:.1f} GiB working set)",
            file=sys.stderr,
            flush=True,
        )
        inner = [sys.executable, str(Path(__file__).resolve()), *raw_argv]
        env = dict(os.environ, RAPID_LARGE_MODEL_LOCK_HELD="1")
        child = subprocess.Popen(
            [lockf, "-k", "-t", str(args.lock_timeout), str(args.lock_file), *inner],
            env=env,
            start_new_session=True,
        )
        previous: dict[signal.Signals, object] = {}

        def forward(signum: int, _frame: object) -> None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

        for name in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[name] = signal.signal(name, forward)
        try:
            return child.wait()
        finally:
            for name, handler in previous.items():
                signal.signal(name, handler)

    available = available_gib()
    minimum = required + args.reserve_gb
    if available < minimum:
        print(
            f"large-model-run: insufficient memory under lock: {available:.1f} GiB "
            f"available, {minimum:.1f} GiB required ({required:.1f} + "
            f"{args.reserve_gb:.1f} reserve)",
            file=sys.stderr,
        )
        return 1
    print(
        "large-model-run: host lock acquired; memory precheck passed", file=sys.stderr
    )
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except ValueError as exc:
        print(f"large-model-run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
