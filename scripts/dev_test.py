#!/usr/bin/env python3
"""Unified dev testing entry point for Rapid-MLX.

NOT shipped with pip — this is for local development only.
Orchestrates all test levels from quick lint to overnight benchmarks.

Usage:
    python scripts/dev_test.py lint          # ruff + import check (~10s)
    python scripts/dev_test.py unit          # pytest unit suite (~30s)
    python scripts/dev_test.py smoke         # lint + unit (~1 min)
    python scripts/dev_test.py stress        # 8-test stress suite (needs server)
    python scripts/dev_test.py soak          # 10-min agent soak test (needs server)
    python scripts/dev_test.py cross-model   # multi-model stress (auto starts servers)
    python scripts/dev_test.py all           # everything except soak + cross-model
    python scripts/dev_test.py full          # everything including soak
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
PY = sys.executable


def run(cmd, label, timeout=600):
    """Run a command, print result, return success."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, timeout=timeout)
        elapsed = time.perf_counter() - t0
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"  [{status}] {label} ({elapsed:.1f}s)")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"  [FAIL] {label} (timeout after {elapsed:.0f}s)")
        return False


def run_lint():
    import shutil

    # Try python -m ruff first, fall back to standalone binary
    result = subprocess.run(
        [PY, "-m", "ruff", "--version"], capture_output=True, cwd=REPO_ROOT
    )
    if result.returncode == 0:
        return run([PY, "-m", "ruff", "check", "vllm_mlx/", "tests/"], "Lint (ruff)")
    ruff_bin = shutil.which("ruff")
    if ruff_bin:
        return run([ruff_bin, "check", "vllm_mlx/", "tests/"], "Lint (ruff)")
    print("  ruff not installed — pip install ruff")
    return False


def run_unit():
    return run(
        [PY, "-m", "pytest", "tests/", "-q",
         "--ignore=tests/integrations",
         "--deselect", "tests/test_event_loop.py",
         "--deselect", "tests/test_batching_deterministic.py",
         "--deselect", "tests/test_reasoning_parsers.py"],
        "Unit tests (pytest)",
        timeout=120,
    )


def run_stress(port=8000):
    return run(
        [PY, os.path.join(SCRIPTS_DIR, "stress_test.py"), "--port", str(port)],
        "Stress test (8 scenarios)",
    )


def run_soak(port=8000, duration=600):
    return run(
        [PY, os.path.join(SCRIPTS_DIR, "agent_soak_test.py"),
         "--url", f"http://localhost:{port}/v1",
         "--duration", str(duration)],
        f"Agent soak test ({duration}s)",
        timeout=duration + 1200,  # generous buffer for slow models
    )


def run_cross_model():
    return run(
        [PY, os.path.join(SCRIPTS_DIR, "cross_model_stress.py")],
        "Cross-model stress test",
        timeout=3600,
    )


def check_server(port=8000):
    """Check if a server is running."""
    import urllib.request
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Rapid-MLX dev testing suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tier",
        choices=["lint", "unit", "smoke", "stress", "soak", "cross-model", "all", "full"],
        help="Test tier to run",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port for stress/soak")
    parser.add_argument("--duration", type=int, default=600, help="Soak test duration (seconds)")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Rapid-MLX Dev Test Suite — {args.tier}")
    print(f"{'=' * 60}")

    results = {}

    if args.tier in ("lint", "smoke", "all", "full"):
        results["lint"] = run_lint()

    if args.tier in ("unit", "smoke", "all", "full"):
        results["unit"] = run_unit()

    if args.tier in ("stress", "all", "full"):
        if not check_server(args.port):
            print(f"\n  ⚠ No server on port {args.port}. Start one first:")
            print(f"    rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit --port {args.port}")
            results["stress"] = False
        else:
            results["stress"] = run_stress(args.port)

    if args.tier in ("soak", "full"):
        if not check_server(args.port):
            print(f"\n  ⚠ No server on port {args.port}.")
            results["soak"] = False
        else:
            results["soak"] = run_soak(args.port, args.duration)

    if args.tier == "cross-model":
        results["cross-model"] = run_cross_model()

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {passed}/{total} passed")
    print(f"{'=' * 60}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
