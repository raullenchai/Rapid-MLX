#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G12 release-gauntlet random-coverage gate.

The fixed gauntlet only exercises ``qwen3.5-9b-4bit`` — every release
ships without ever booting the other ~28 registered small/medium
aliases. PR #687 (gemma-4 ``<|tool_call>`` wire-marker leak) is a
class of bug that only surfaces when you actually run the model.

This script bolts a randomized sweep onto the existing gauntlet:

    for each of N seeded-random models (from the eligible alias set):
        boot rapid-mlx serve <model> --port $PORT --no-thinking
        for each of M seeded-random harnesses (from the 5 first-class):
            for r in 1..K rounds:
                run `bench --tier harness` with the env-filter scoped
                to just that one harness, against the booted server
        stop the server (clean shutdown, wait for port to free)
        rm -rf ~/.cache/huggingface/hub/models--<repo> so the disk
        doesn't balloon across release cycles

Defaults: N=2, M=2, K=3 → 12 sweeps × ~30s avg ≈ 6-12 min wall-clock
(plus model download + boot time, which dominates for cold caches).

The seed is today's UTC date (``YYYYMMDD``) — same calendar day cuts
of the release reproduce the same model × harness picks, so a failure
is repro-able by another contributor running the script on the same
day with the same alias inventory.

Failure handling: any harness round that fails surfaces a non-zero
exit code. The shell gauntlet ``set -e``'s out on the first bad gate.

Disk safety: the script REFUSES to start if free space < 30 GB
(typical 4-bit small models are 2-6 GB on disk; the largest 12B 4-bit
land at ~7 GB; a worst-case 2-model sweep with no cleanup would
allocate ~14 GB, but cleanup runs after each model so peak working set
is one model at a time + 5 GB headroom).

Eligibility filter (sample pool):
  * 4 ≤ size_B ≤ 12  (smaller models can't actually solve harness tasks
                       → false-fail spam; larger models bust the disk
                       budget on M3 16/32 GB systems)
  * 4-bit quant only  (8-bit is 2x download cost for the same coverage)
  * no kimi-*         (deliberately heavy class, explicit user exclude)
  * no -vl- variants  (vision; harness tasks are text-only)
  * no gemma-4-*      (known model-side hang on tool-use prompts; see
                       issue #686 + huggingface/google/gemma-4-12B-it
                       discussion #41 — would burn 156s+ per round on
                       a known-bad model and add zero signal)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirror of ``vllm_mlx.bench.tier_runner.HARNESS_PROFILES`` — hardcoded
# here so this script doesn't need to import the package (which would
# pull mlx_lm at module-load and fail in a clean-venv sanity run).
HARNESS_PROFILES = ("codex", "opencode", "hermes", "aider", "langchain")

# Disk safety floor — refuse to start if the cache disk has less than
# this. Sized for one 12B-4bit model + 5 GB headroom.
MIN_FREE_DISK_GB = 30

# Per-server-boot deadline. First-time downloads can be slow; this is
# tight enough to flag a hung boot but loose enough to tolerate a slow
# HF connection on a 7-GB shard.
SERVE_READY_TIMEOUT_S = 600  # 10 minutes

# Per-harness-round timeout. The harness runner's own per-profile cap
# is 300s (HARNESS_PROFILE_TIMEOUT_S); we add headroom for the
# bench-CLI startup cost.
ROUND_TIMEOUT_S = 360


# Match a parameter-count token bounded by name separators (``-``,
# ``_``, ``.``, start, or end) followed by ``b``/``B`` and another
# separator/end. Rejects the quantization suffix ``-4bit`` (the ``b``
# is followed by ``it``) and the version-number-only names like
# ``glm4.5-air`` (no ``b`` after the digits at all). Parsing the
# parameter count from the **hf_path's repo segment** is more reliable
# than from the alias slug — repo names by upstream convention spell
# the size as ``\d+(\.\d+)?B`` (Qwen3.5-9B, Llama-3.2-1B, gemma-4-12B)
# whereas alias slugs sometimes encode model version instead of size.
# Fail-closed: if no size token is found, the alias is skipped rather
# than guessed at — guessing landed us with ``glm4.5-air-4bit`` parsing
# as a 4 B model and slipping past the disk-budget filter.
_SIZE_TOKEN_RE = re.compile(r"(?:^|[-_.])(\d+(?:\.\d+)?)[bB](?=[-_.]|$)")


def _eligible_aliases(aliases_path: Path) -> list[tuple[str, str]]:
    """Return ``[(alias_name, hf_repo_path), ...]`` after applying the
    eligibility filter documented in the module docstring.

    Sorted by size then name so the seeded random.sample is stable
    across machines that read the same aliases.json — list order
    matters for ``random.sample`` reproducibility.
    """
    data = json.loads(aliases_path.read_text())
    out: list[tuple[float, str, str]] = []
    for name, entry in data.items():
        if not name.endswith("-4bit"):
            continue
        if "kimi" in name.lower():
            continue
        if "-vl-" in name.lower():
            continue
        if name.lower().startswith("gemma-4-"):
            # Known-bad: model-side ``thought\n…`` loop on agent prompts.
            # See issue #686 + HF discussion google/gemma-4-12B-it#41.
            continue
        # Use .get() — a future schema change that omits ``hf_path``
        # should silently skip the entry, not crash the gauntlet.
        hf_path = entry.get("hf_path") if isinstance(entry, dict) else None
        if not hf_path:
            continue
        repo_name = hf_path.split("/")[-1]
        match = _SIZE_TOKEN_RE.search(repo_name)
        if not match:
            # Fail closed: cannot parse a real parameter count from the
            # repo name — skip rather than guess and risk admitting an
            # oversized model into the sweep.
            continue
        size_b = float(match.group(1))
        if not (4.0 <= size_b <= 12.0):
            continue
        out.append((size_b, name, hf_path))
    out.sort()
    return [(name, hf) for _, name, hf in out]


def _free_disk_gb(path: Path) -> float:
    """Free space in GB on the filesystem holding ``path``."""
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def _hf_cache_root() -> Path:
    """Resolve the HuggingFace Hub cache root, mirroring
    ``huggingface_hub.constants.HF_HUB_CACHE`` lookup order:

      1. ``HF_HUB_CACHE`` (modern)
      2. ``HUGGINGFACE_HUB_CACHE`` (legacy)
      3. ``$HF_HOME/hub`` (when HF_HOME is set)
      4. ``~/.cache/huggingface/hub`` (default)

    Hard-coding ``~/.cache/huggingface/hub`` meant installs that point
    HF elsewhere (CI runners, multi-disk dev rigs) would download into
    one place and have G12 try to clean another, ballooning disk usage
    across release cycles. The cleanup must target the actual snapshot
    tree this run's download landed in.
    """
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(env)
        if v:
            return Path(v).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_cache_dir(hf_repo_path: str) -> Path:
    """Path of the HuggingFace cache entry for ``hf_repo_path``."""
    return _hf_cache_root() / f"models--{hf_repo_path.replace('/', '--')}"


def _wait_for_server(
    proc: subprocess.Popen, port: int, deadline_s: float, log_path: Path
) -> bool:
    """Poll ``/v1/models`` until the server responds 200, the child
    exits, or the deadline expires. Returns True on success, False
    otherwise.

    Watching ``proc.poll()`` matters: if ``rapid-mlx serve`` aborts at
    import time (missing alias, port collision raced past the
    pre-flight, mlx-lm import error on a clean venv), there is no port
    that will ever come up. Without this check we'd burn the full 600 s
    deadline polling a dead child.
    """
    url = f"http://127.0.0.1:{port}/v1/models"
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        rc = proc.poll()
        if rc is not None:
            print(
                f"  serve process exited early (rc={rc}) before reaching ready state",
                file=sys.stderr,
            )
            break
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(2)
    # Dump the last 30 lines of the server log so the operator sees
    # why we gave up — same shape the shell gauntlet uses.
    if log_path.exists():
        print("  server log (last 30 lines):", file=sys.stderr)
        for line in log_path.read_text(errors="replace").splitlines()[-30:]:
            print(f"    {line}", file=sys.stderr)
    return False


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _stop_server(proc: subprocess.Popen, port: int, deadline_s: float = 30) -> None:
    """Gracefully terminate the server and wait for the port to free.

    The server's SIGTERM handler flushes the prefix cache (post-PR #667
    deadline-aware shutdown), so we give it real time to land.
    """
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=deadline_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    # Belt-and-braces — confirm the port released before we move on.
    start = time.monotonic()
    while time.monotonic() - start < 5 and not _port_free(port):
        time.sleep(0.5)


def _run_harness_round(
    *,
    alias: str,
    harness: str,
    base_url: str,
    log_path: Path,
) -> tuple[bool, float, str]:
    """Run one ``bench --tier harness`` invocation scoped to one
    harness. Returns ``(ok, wall_clock_s, error_excerpt)``."""
    env = {**os.environ, "RAPID_MLX_HARNESS_PROFILES_FILTER": harness}
    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "bench",
        alias,
        "--tier",
        "harness",
        "--base-url",
        base_url,
    ]
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            timeout=ROUND_TIMEOUT_S,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        dur = time.monotonic() - t0
        return False, dur, f"round timed out after {ROUND_TIMEOUT_S}s"
    dur = time.monotonic() - t0
    # Append the subprocess output to our log so a failure has
    # a debuggable trail. ``"a"`` mode is single-write-atomic enough for
    # our single-threaded sweep loop.
    with log_path.open("a") as fh:
        fh.write(
            f"\n=== {alias}/{harness} (exit={result.returncode}, {dur:.1f}s) ===\n"
        )
        fh.write(result.stdout or "")
        if result.stderr:
            fh.write("\n--- stderr ---\n")
            fh.write(result.stderr)
    if result.returncode != 0:
        # Pull the FAIL line from the bench output as the excerpt.
        excerpt = ""
        for line in (result.stdout or "").splitlines():
            if "FAIL" in line:
                excerpt = line.strip()[:200]
                break
        if not excerpt:
            excerpt = f"exit {result.returncode}; see {log_path}"
        return False, dur, excerpt
    return True, dur, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        default=time.strftime("%Y%m%d", time.gmtime()),
        help="Deterministic seed (default: today's UTC date YYYYMMDD).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to boot rapid-mlx serve on (default: 8000).",
    )
    parser.add_argument(
        "--models",
        type=int,
        default=2,
        help="Number of models to sample (default: 2).",
    )
    parser.add_argument(
        "--harnesses",
        type=int,
        default=2,
        help="Number of harnesses to sample per model (default: 2).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Rounds per (model × harness) pair (default: 3).",
    )
    parser.add_argument(
        "--report",
        default="/tmp/release-check-m3-random.log",
        help="Path to write the human-readable summary report to.",
    )
    parser.add_argument(
        "--aliases-json",
        default=str(REPO_ROOT / "vllm_mlx" / "aliases.json"),
        help="Path to aliases.json (default: in-tree copy).",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Skip the per-model HF cache cleanup (debug aid).",
    )
    args = parser.parse_args()

    # ===== Argument bounds =====
    # ``random.sample(population, k)`` raises ``ValueError`` for k > len.
    # We want an actionable release-gate error instead of a Python
    # traceback when someone passes ``G12_HARNESSES=6`` or
    # ``G12_MODELS=0`` from the shell wrapper.
    if args.models < 1:
        print(
            f"  Error: --models must be ≥1 (got {args.models}).",
            file=sys.stderr,
        )
        return 2
    if not (1 <= args.harnesses <= len(HARNESS_PROFILES)):
        print(
            f"  Error: --harnesses must be 1..{len(HARNESS_PROFILES)} "
            f"(got {args.harnesses}); the registry has "
            f"{len(HARNESS_PROFILES)} harness profile(s).",
            file=sys.stderr,
        )
        return 2
    if args.rounds < 1:
        print(
            f"  Error: --rounds must be ≥1 (got {args.rounds}).",
            file=sys.stderr,
        )
        return 2

    # ===== Pre-flight =====
    if not _port_free(args.port):
        print(
            f"  Error: port {args.port} already in use — kill the existing "
            f"server before running G12.",
            file=sys.stderr,
        )
        return 2

    free_gb = _free_disk_gb(Path.home() / ".cache")
    if free_gb < MIN_FREE_DISK_GB:
        print(
            f"  Error: only {free_gb:.1f} GB free on the HF cache disk; "
            f"refusing to start (need {MIN_FREE_DISK_GB} GB). Clear caches "
            f"and retry.",
            file=sys.stderr,
        )
        return 2

    # ===== Sample =====
    eligible = _eligible_aliases(Path(args.aliases_json))
    if len(eligible) < args.models:
        print(
            f"  Error: only {len(eligible)} eligible aliases; need {args.models}.",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    sampled_models = rng.sample(eligible, args.models)
    # Independent seed stream per model so the harness pick for model A
    # doesn't shift when we change ``--models``.
    sampled = []
    for alias, hf_path in sampled_models:
        per_model_rng = random.Random(f"{args.seed}::{alias}")
        hs = per_model_rng.sample(list(HARNESS_PROFILES), args.harnesses)
        sampled.append((alias, hf_path, hs))

    print("=" * 60)
    print("  G12 — random-coverage release gate")
    print(f"  seed:     {args.seed}")
    print(f"  models:   {args.models} (of {len(eligible)} eligible)")
    print(f"  harnesses:{args.harnesses} (of {len(HARNESS_PROFILES)})")
    print(f"  rounds:   {args.rounds}")
    print(f"  report:   {args.report}")
    print(f"  free GB:  {free_gb:.1f}")
    print("=" * 60)
    print("  Sampled matrix:")
    for alias, _, hs in sampled:
        print(f"    {alias:<28} × harnesses={hs}")
    print("=" * 60)

    # Reset the report log.
    report_path = Path(args.report)
    report_path.write_text(
        f"G12 random-coverage report (seed={args.seed})\n" + "=" * 60 + "\n"
    )

    # ===== Sweep =====
    failures: list[str] = []
    for alias, hf_path, harnesses in sampled:
        print()
        print(f"  >> Booting {alias} on port {args.port}…")
        log_path = Path(f"/tmp/release-check-m3-random-{alias}.log")
        log_path.write_text("")
        with log_path.open("w") as logfh:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                    alias,
                    "--port",
                    str(args.port),
                    "--no-thinking",
                ],
                stdout=logfh,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
            )
        try:
            if not _wait_for_server(proc, args.port, SERVE_READY_TIMEOUT_S, log_path):
                msg = f"{alias}: server did not respond within {SERVE_READY_TIMEOUT_S}s"
                print(f"  FAIL  {msg}", file=sys.stderr)
                with report_path.open("a") as fh:
                    fh.write(f"FAIL  {msg}\n")
                failures.append(msg)
                continue
            print(f"     server up ({alias}); harnesses={harnesses}")
            base_url = f"http://127.0.0.1:{args.port}"
            for harness in harnesses:
                for r in range(1, args.rounds + 1):
                    ok, dur, excerpt = _run_harness_round(
                        alias=alias,
                        harness=harness,
                        base_url=base_url,
                        log_path=log_path,
                    )
                    marker = "PASS" if ok else "FAIL"
                    line = (
                        f"     {marker} {alias}/{harness} "
                        f"round {r}/{args.rounds} ({dur:.1f}s)"
                    )
                    if excerpt:
                        line += f"  — {excerpt}"
                    print(line)
                    with report_path.open("a") as fh:
                        fh.write(line + "\n")
                    if not ok:
                        failures.append(f"{alias}/{harness} round {r}: {excerpt}")
        finally:
            print(f"  << Stopping {alias}…")
            _stop_server(proc, args.port)
            if not args.keep_cache:
                cache_dir = _hf_cache_dir(hf_path)
                if cache_dir.exists():
                    print(f"     rm -rf {cache_dir}")
                    shutil.rmtree(cache_dir, ignore_errors=True)

    # ===== Verdict =====
    print()
    print("=" * 60)
    if failures:
        print(f"  G12: {len(failures)} failure(s)")
        for f in failures:
            print(f"    - {f}")
        print(f"  Full log: {args.report}")
        print("=" * 60)
        return 1
    print("  G12: ALL rounds passed")
    print(f"  Full log: {args.report}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
