# SPDX-License-Identifier: Apache-2.0
"""Step 5 — stress + e2e + multi-model bench.

The heaviest gate: only runs when blast radius is high (scheduler /
engine / memory_cache / cli / server). Boots a real server, runs
``scripts/stress_test.py``, runs each agent integration in
``golden_models.yaml`` against each selected model, and records bench
numbers.

Model selection: ``golden_models.yaml`` defines families; for each
family we pick the highest-quality candidate that fits machine RAM.
Bench regression threshold is 5% on cold TTFT and decode TPS vs the
last saved baseline at ``harness/baselines/<model>.json`` (when
present) — missing baselines mean "first time, record it" and we don't
fail.

Implementation notes:
* Server boot uses an unusual port (``8451``) to avoid colliding with
  any locally-running rapid-mlx; we ALSO check for stale processes on
  that port at startup and refuse if one is found rather than killing
  it (don't accidentally murder someone's debug session).
* The MVP uses two models — smoke + small — to keep total runtime
  under ~5 minutes. The full m×n matrix is the eventual goal; expanding
  it is just adding entries to the YAML.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import Step, StepResult
from ..context import Context, env_truthy

REGISTRY_PATH = Path(__file__).parent.parent / "golden_models.yaml"
BASELINE_DIR = Path("harness/baselines")
BENCH_PORT = 8451
BENCH_THRESHOLD_PCT = 5.0
SERVER_BOOT_TIMEOUT_S = 180
# Per-request timeout. Sized for the worst case in the matrix: cold-start
# of a 27B-class model (~17 GB weights to mmap) immediately after the
# previous server's lifespan finished writing its prefix cache to disk.
# 60s was tuned for a 4B small-model matrix; raised to 180s when the
# golden registry moved to real-capacity 27-35B models (PR #208).
SERVER_REQUEST_TIMEOUT_S = 180
# How long to wait for the previous server's port to free between back-
# to-back model boots. SIGINT triggers a lifespan shutdown that writes
# the prefix cache to disk — for large models this can run past the old
# 10s budget. 30s covers a 35B model's cache flush comfortably.
PORT_FREE_TIMEOUT_S = 30
RAM_HEADROOM_GB = 8.0  # leave this much free for the OS + model load spike


@dataclass
class ModelChoice:
    family: str
    model_id: str
    ram_gb_required: float
    quality_tier: str
    extra_args: list[str]


class StressE2EBenchStep(Step):
    name = "stress_e2e_bench"
    description = "stress + integration matrix + bench (high blast only)"

    def should_run(self, ctx: Context) -> bool:
        if env_truthy("PR_VALIDATE_NO_STRESS"):
            return False
        return ctx.blast_radius == "high"

    def run(self, ctx: Context) -> StepResult:
        try:
            registry = _load_registry()
        except Exception as e:  # noqa: BLE001
            return StepResult(
                name=self.name,
                status="error",
                summary=f"could not load model registry: {e}",
            )

        try:
            available_gb = _available_ram_gb()
        except Exception as e:  # noqa: BLE001
            # Don't silently fall back — that has caused us to either
            # under-pick (bench skips on a beefy Mac) or over-pick (OOM
            # on a small machine). Surface the error so the operator
            # sets PR_VALIDATE_RAM_GB explicitly.
            return StepResult(
                name=self.name,
                status="error",
                summary=(
                    f"RAM probe failed ({type(e).__name__}: {e}) — "
                    "set PR_VALIDATE_RAM_GB=<int> and retry"
                ),
            )
        usable_gb = max(0, available_gb - RAM_HEADROOM_GB)

        choices = _select_models(registry, usable_gb)
        if not choices:
            return StepResult(
                name=self.name,
                status="error",
                summary=(
                    f"no model in registry fits {usable_gb:.1f}GB usable RAM "
                    f"(machine reports {available_gb:.1f}GB total)"
                ),
            )

        ctx.run_log(
            f"selected {len(choices)} model(s): "
            + ", ".join(f"{c.family}:{c.model_id}" for c in choices)
        )

        all_findings: list[str] = []
        all_artifacts: list[str] = []
        any_fail = False

        for choice in choices:
            ctx.run_log(f"--- {choice.family} ({choice.model_id}) ---")
            try:
                with _server(choice, ctx) as server_log:
                    all_artifacts.append(server_log)
                    # Stress.
                    stress_result = _run_stress(ctx, choice)
                    if stress_result["status"] != "pass":
                        any_fail = True
                        all_findings.append(
                            f"[BLOCKING] stress on {choice.model_id}: "
                            + stress_result["summary"]
                        )
                    if stress_result.get("artifact"):
                        all_artifacts.append(stress_result["artifact"])

                    # Integration matrix.
                    for agent in registry["agents"]:
                        if choice.quality_tier == "smoke" and agent.get(
                            "skip_for_smoke"
                        ):
                            continue
                        ag_result = _run_agent(ctx, choice, agent)
                        if ag_result["status"] != "pass":
                            any_fail = True
                            all_findings.append(
                                f"[BLOCKING] {agent['name']} on "
                                f"{choice.model_id}: " + ag_result["summary"]
                            )
                        if ag_result.get("artifact"):
                            all_artifacts.append(ag_result["artifact"])

                    # Bench.
                    bench_result = _run_bench(ctx, choice)
                    if bench_result["status"] != "pass":
                        any_fail = True
                        all_findings.append(
                            f"[BLOCKING] bench on {choice.model_id}: "
                            + bench_result["summary"]
                        )
                    if bench_result.get("artifact"):
                        all_artifacts.append(bench_result["artifact"])

            except _ServerStartError as e:
                any_fail = True
                all_findings.append(
                    f"[BLOCKING] could not boot server with {choice.model_id}: {e}"
                )
                continue

        status = "fail" if any_fail else "pass"
        summary = (
            f"matrix {len(choices)}×{len([a for a in registry['agents']])}, "
            f"{'PASSED' if not any_fail else 'see findings'}"
        )
        return StepResult(
            name=self.name,
            status=status,
            summary=summary,
            findings=all_findings,
            artifacts=all_artifacts,
        )


# ---------------------------------------------------------------------------
# Registry + selection
# ---------------------------------------------------------------------------


def _load_registry() -> dict[str, Any]:
    """Parse golden_models.yaml. We avoid PyYAML to keep the script
    dependency-free; the file format is hand-restricted to a subset
    we can parse trivially."""
    text = REGISTRY_PATH.read_text()
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        # PyYAML is in test deps but may be absent from a slim install.
        raise RuntimeError(
            "golden_models.yaml needs PyYAML — `pip install pyyaml`"
        ) from e
    return yaml.safe_load(text)


def _select_models(registry: dict[str, Any], usable_gb: float) -> list[ModelChoice]:
    """For each family, pick the highest-quality candidate that fits."""
    out: list[ModelChoice] = []
    overrides = registry.get("overrides", {}) or {}
    for family in registry.get("families", []):
        for cand in family["candidates"]:
            if cand["ram_gb_required"] <= usable_gb:
                out.append(
                    ModelChoice(
                        family=family["family"],
                        model_id=cand["id"],
                        ram_gb_required=float(cand["ram_gb_required"]),
                        quality_tier=cand.get("quality_tier", "unknown"),
                        extra_args=list(
                            (overrides.get(cand["id"], {}) or {}).get("args", [])
                        ),
                    )
                )
                break  # stop at first fit per family
    return out


def _available_ram_gb() -> float:
    """Probe how much RAM is *currently free*, not total.

    macOS: ``vm_stat`` page counts × page size. Linux: ``MemAvailable``
    line in ``/proc/meminfo``. The previous version returned total RAM,
    which on a 256GB Mac with the model already loaded would over-pick
    a 200GB candidate and OOM.

    Override with ``PR_VALIDATE_RAM_GB=<int>`` if the probe is wrong
    for your environment.
    """
    override = os.environ.get("PR_VALIDATE_RAM_GB")
    if override:
        return float(override)

    # Linux first — cheap and unambiguous.
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / (1024**2)

    # macOS — vm_stat reports pages of free / inactive / speculative.
    # Free + inactive is what we can realistically reclaim for a model.
    if shutil.which("vm_stat"):
        proc = subprocess.run(  # noqa: S603
            ["vm_stat"], capture_output=True, text=True, check=True
        )
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (ValueError, AttributeError, OSError):
            page_size = 16384  # Apple Silicon default; vm_stat fallback
        free_pages = inactive_pages = 0
        for line in proc.stdout.splitlines():
            if line.startswith("Mach Virtual Memory Statistics"):
                # Header includes the page size: "(page size of N bytes)"
                m = re.search(r"page size of (\d+) bytes", line)
                if m:
                    page_size = int(m.group(1))
            elif "Pages free:" in line:
                free_pages = int(line.rsplit(":", 1)[1].strip().rstrip("."))
            elif "Pages inactive:" in line:
                inactive_pages = int(line.rsplit(":", 1)[1].strip().rstrip("."))
        bytes_free = (free_pages + inactive_pages) * page_size
        return bytes_free / (1024**3)

    raise RuntimeError(
        "no RAM probe available (need /proc/meminfo or `vm_stat`); "
        "set PR_VALIDATE_RAM_GB=<int> to override"
    )


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class _ServerStartError(RuntimeError):
    pass


@contextmanager
def _server(choice: ModelChoice, ctx: Context):
    """Start a rapid-mlx server with `choice.model_id` on BENCH_PORT.
    Yield the path to its log file. Stop on context exit. Refuses to
    proceed if BENCH_PORT is already bound."""
    if _port_in_use(BENCH_PORT):
        raise _ServerStartError(
            f"port {BENCH_PORT} already in use — refusing to clobber"
        )

    log_path = ctx.artifact_path(f"server-{_safe_name(choice.model_id)}.log")
    cmd = [
        "python3.12",
        "-m",
        "vllm_mlx.cli",
        "serve",
        choice.model_id,
        "--port",
        str(BENCH_PORT),
        *choice.extra_args,
    ]
    # Open the log inside the try so a Popen failure (e.g. fork EAGAIN,
    # disk full) doesn't leak the file handle.
    log_f = None
    proc = None
    try:
        log_f = open(log_path, "w")  # noqa: SIM115 — explicit close in finally
        proc = subprocess.Popen(  # noqa: S603
            cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(ctx.repo_root)
        )
        if not _wait_for_server(BENCH_PORT, SERVER_BOOT_TIMEOUT_S):
            raise _ServerStartError(
                f"server did not respond on :{BENCH_PORT} within "
                f"{SERVER_BOOT_TIMEOUT_S}s — see {log_path}"
            )
        yield str(log_path)
    finally:
        # Nested try/finally so log_f.close() runs even if the proc
        # cleanup or _wait_for_port_free raises (e.g., _port_in_use's
        # socket call can raise OSError under fd exhaustion).
        try:
            # Graceful first (lifespan saves prefix cache); SIGKILL the
            # direct child if SIGINT timed out; lsof-based fallback for
            # stragglers (zombies / port still held).  Keep BOTH paths:
            # proc.kill() handles a process that survived SIGINT but isn't
            # holding the port any more (would otherwise leak), and
            # _force_kill_port handles a process tree where a child of the
            # spawned proc is what's actually bound.
            if proc is not None:
                proc.send_signal(2)  # SIGINT
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass  # _force_kill_port handles port-bound orphans
                if not _wait_for_port_free(BENCH_PORT, timeout=PORT_FREE_TIMEOUT_S):
                    _force_kill_port(BENCH_PORT)
        finally:
            if log_f is not None:
                log_f.close()


def _port_in_use(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _wait_for_server(port: int, timeout_s: int) -> bool:
    """Poll /health/ready until 200 or timeout. Each attempt tolerates
    connection refused (still booting) and 503 (engine loading +
    warmup + prefix-cache load still in progress).

    /health/ready returns 200 only after lifespan has finished
    engine.start() + warmup + load_from_disk + MCP init. Polling it
    (instead of /v1/models) means the first stress/bench request
    isn't racing the cold-start work — what previously looked like a
    "request timeout" was actually warmup latency leaking past the
    moment the FastAPI app started accepting connections.
    """
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health/ready"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def _wait_for_port_free(port: int, timeout: int = 10) -> bool:
    """Poll until ``port`` is no longer bound or ``timeout`` expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def _force_kill_port(port: int) -> None:
    """Find and SIGKILL any process holding ``port``."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return
    if not result.stdout:
        return
    for pid_str in result.stdout.strip().split():
        try:
            pid = int(pid_str)
            os.kill(pid, 9)
        except (ValueError, OSError):
            pass
    # One more wait for the port to drain.
    _wait_for_port_free(port, timeout=5)


# ---------------------------------------------------------------------------
# Sub-runners (stress / agent / bench)
# ---------------------------------------------------------------------------


def _run_stress(ctx: Context, choice: ModelChoice) -> dict[str, Any]:
    log = ctx.artifact_path(f"stress-{_safe_name(choice.model_id)}.log")
    proc = subprocess.run(  # noqa: S603
        ["python3.12", "scripts/stress_test.py", "--port", str(BENCH_PORT)],
        capture_output=True,
        text=True,
        cwd=str(ctx.repo_root),
        timeout=900,
    )
    log.write_text((proc.stdout or "") + (proc.stderr or ""))
    summary = _grep_last(proc.stdout, "passed")
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "summary": summary or f"exit {proc.returncode}",
        "artifact": str(log),
    }


def _run_agent(
    ctx: Context, choice: ModelChoice, agent: dict[str, Any]
) -> dict[str, Any]:
    log = ctx.artifact_path(f"agent-{agent['name']}-{_safe_name(choice.model_id)}.log")
    script = ctx.repo_root / agent["script"]
    if not script.exists():
        return {"status": "skip", "summary": f"script missing: {agent['script']}"}

    env = {
        **os.environ,
        "RAPID_MLX_BASE_URL": f"http://127.0.0.1:{BENCH_PORT}/v1",
    }
    # 1200s (20 min) per agent script. Sized for the slowest model in the
    # matrix — Gemma 4 26B routes through MLLMModel (mlx-vlm) which has
    # higher per-token decode overhead than the qwen3.5/qwen3.6 path. 600s
    # was enough for 27B-class qwen models but truncated gemma4 langchain
    # mid-suite (observed in pr_validate against PR #208).
    proc = subprocess.run(  # noqa: S603
        ["python3.12", str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ctx.repo_root),
        timeout=1200,
    )
    log.write_text((proc.stdout or "") + (proc.stderr or ""))
    summary = _grep_last(proc.stdout, "passed") or _grep_last(proc.stdout, "FAIL")
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "summary": summary or f"exit {proc.returncode}",
        "artifact": str(log),
    }


def _run_bench(ctx: Context, choice: ModelChoice) -> dict[str, Any]:
    """Inline bench — measure cold TTFT + decode TPS, compare to
    baseline. Implementation borrowed from the inline benchmark we ran
    against PR #200; keeps this step dependency-free."""
    import statistics
    import urllib.error
    import urllib.request

    sys = (
        "You are a helpful assistant. Provide thoughtful and clear answers. "
        "Be concise but informative."
    )

    def call(prompt: str, max_tok: int = 80) -> tuple[float, int]:
        body = json.dumps(
            {
                "model": choice.model_id,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tok,
                "stream": False,
                "temperature": 0.0,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{BENCH_PORT}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(  # noqa: S310
                req, timeout=SERVER_REQUEST_TIMEOUT_S
            ) as resp:
                payload = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(f"bench request failed: {e}") from e
        dt_ms = (time.time() - t0) * 1000
        toks = payload.get("usage", {}).get("completion_tokens", 0)
        return dt_ms, toks

    # Cold: 5 different prompts (no cache hits).
    cold_times = []
    for i in range(5):
        dt, _ = call(f"Cold prompt #{i} — say something brief")
        cold_times.append(dt)

    # Warm: 5 repeats of same prompt (full cache hit after warmup).
    call("warmup", 30)
    call("warmup", 30)
    warm_times = []
    for _ in range(5):
        dt, _ = call("Warm prompt — say something brief")
        warm_times.append(dt)

    cold = statistics.median(cold_times)
    warm = statistics.median(warm_times)
    speedup = cold / warm if warm else 0
    metrics = {
        "model": choice.model_id,
        "cold_request_ms_median": cold,
        "warm_request_ms_median": warm,
        "speedup_x": speedup,
    }

    bench_path = ctx.artifact_path(f"bench-{_safe_name(choice.model_id)}.json")
    bench_path.write_text(json.dumps(metrics, indent=2))

    # Compare to baseline if present.
    baseline_path = (
        ctx.repo_root / BASELINE_DIR / f"bench-{_safe_name(choice.model_id)}.json"
    )
    if not baseline_path.exists():
        return {
            "status": "pass",
            "summary": (
                f"cold={cold:.0f}ms warm={warm:.0f}ms ({speedup:.2f}x) "
                f"— no baseline, recorded for next run"
            ),
            "artifact": str(bench_path),
        }

    baseline = json.loads(baseline_path.read_text())
    # Accept both old (ttft) and new key names for backward compat. Treat
    # an explicit ``None`` (JSON ``null``) the same as a missing key so we
    # don't end up with ``base_cold=None`` flowing into the division below.
    # This is stricter than a plain ``or`` because we want a real ``0`` to
    # be honoured (it's a legit value, even if the downstream guard makes
    # it a no-op).
    base_cold = baseline.get("cold_request_ms_median")
    if base_cold is None:
        base_cold = baseline.get("cold_ttft_ms_median", cold)
    base_warm = baseline.get("warm_request_ms_median")
    if base_warm is None:
        base_warm = baseline.get("warm_ttft_ms_median", warm)

    # Slowdown = current / baseline. >5% slower on cold OR warm = fail.
    cold_slow = (cold / base_cold - 1) * 100 if base_cold else 0
    warm_slow = (warm / base_warm - 1) * 100 if base_warm else 0
    if cold_slow > BENCH_THRESHOLD_PCT or warm_slow > BENCH_THRESHOLD_PCT:
        return {
            "status": "fail",
            "summary": (
                f"perf regression: cold {cold_slow:+.1f}%, warm {warm_slow:+.1f}% "
                f"vs baseline (threshold {BENCH_THRESHOLD_PCT}%)"
            ),
            "artifact": str(bench_path),
        }
    return {
        "status": "pass",
        "summary": (
            f"cold {cold_slow:+.1f}%, warm {warm_slow:+.1f}% "
            f"vs baseline (within {BENCH_THRESHOLD_PCT}%)"
        ),
        "artifact": str(bench_path),
    }


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def _grep_last(text: str, needle: str) -> str:
    """Return the last line containing `needle`, stripped — handy for
    pulling pytest/stress summary lines out of subprocess output."""
    for line in reversed((text or "").splitlines()):
        if needle in line:
            return line.strip()
    return ""
