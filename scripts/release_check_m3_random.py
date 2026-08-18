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
  * 6 ≤ size_B ≤ 12  (smaller models can't actually solve harness tasks
                       → they burn the whole per-profile clock retrying
                       and the gauntlet goes red on a working engine, see
                       #1672; larger models bust the disk budget on M3
                       16/32 GB systems)
  * 4-bit quant only  (8-bit is 2x download cost for the same coverage)
  * no kimi-*         (deliberately heavy class, explicit user exclude)
  * no multimodal     (vision; harness tasks are text-only — matched with
                       the engine's own MLLM_PATTERNS, not a name guess:
                       UI-TARS and Gemma 3 carry no ``VL`` marker)
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
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirror of ``vllm_mlx.bench.tier_runner.HARNESS_PROFILES`` — hardcoded
# here so this script doesn't need to import the package (which would
# pull mlx_lm at module-load and fail in a clean-venv sanity run).
HARNESS_PROFILES = (
    "codex",
    "opencode",
    "hermes",
    "aider",
    "langchain",
    "deepseek-harness",
)

# Mirror of ``vllm_mlx.api.utils.MLLM_PATTERNS``, for the same reason as
# HARNESS_PROFILES: importing the package pulls mlx_lm at module load.
#
# Hand-rolling this was a bug. The filter used to exclude vision models by
# looking for ``-vl-`` in the alias name, which is precisely the test the engine
# documents as insufficient: UI-TARS is a Qwen2.5-VL-based GUI agent whose
# public name carries no ``VL``, and Gemma 3 is multimodal with no marker at
# all. Both were in the sample pool. Booting one through a text-only agentic
# harness is not thin coverage, it is a crash on an install without the vision
# extra — and something worse than a crash on one with it.
#
# ``tests/test_release_check_random.py`` parses the real list out of
# ``vllm_mlx/api/utils.py`` and fails if this mirror stops covering it, so a
# family added upstream cannot silently reappear here.
MLLM_NAME_PATTERNS = (
    "-vl-",
    "-vl/",
    "vl-",
    "llava",
    "idefics",
    "paligemma",
    "gemma-3",
    "gemma3",
    "medgemma",
    "pixtral",
    "molmo",
    "phi3-vision",
    "phi-3-vision",
    "cogvlm",
    "internvl",
    "deepseek-vl",
    "ui-tars",
    "ui_tars",
)


def _is_multimodal(*names: str) -> bool:
    """True when any of ``names`` looks like a multimodal model.

    Same case-folded substring rule as ``vllm_mlx.api.utils.is_mllm_model``.
    Both the alias and the HF path are checked: the alias is what a human
    recognises, the repo name is where the family marker usually survives.
    """
    return any(
        pattern in name.lower()
        for name in names
        if name
        for pattern in MLLM_NAME_PATTERNS
    )


# Disk safety floor — refuse to start if the cache disk has less than
# this. Sized for one 12B-4bit model + 5 GB headroom.
MIN_FREE_DISK_GB = 30

# Per-server-boot deadline. First-time downloads can be slow; this is
# tight enough to flag a hung boot but loose enough to tolerate a slow
# HF connection on a 7-GB shard.
SERVE_READY_TIMEOUT_S = 600  # 10 minutes

# Per-harness-round timeout. A scoped ``bench --tier harness`` round
# runs ONE harness profile end-to-end; the hermes profile alone is
# ~740-800s on a dense 9B once its deep agentic tests actually run
# (post #1326/#1330). The old 360s was sized against a stale 300s
# per-profile cap and killed every hermes round. Track the real profile
# length with headroom, and keep it below the inherited per-profile cap
# (HARNESS_PROFILE_TIMEOUT_S, 1200s in the gauntlet) so in the gauntlet
# the outer subprocess timeout — not the inner cap — bounds a hung round.
ROUND_TIMEOUT_S = 1080

# Sample-pool size window, in billions of parameters.
#
# Every profile in HARNESS_PROFILES drives a multi-step tool-calling loop, and
# a model that cannot hold one does not fail fast: it emits malformed calls and
# retries until the per-profile clock runs out. G12 can only report that as a
# red gauntlet, so the pool has to exclude models that cannot do the work —
# the same rule the hybrid, gemma-4 and low-active-param excludes below apply.
#
# The floor sits at 6 because that is the conservative cut between the two
# sizes actually measured on this hardware:
#
#   * 4B fails — ``qwen3-4b-instruct-2507-4bit`` × ``hermes`` burned the whole
#     1020 s cap while the engine answered without a single 5xx, and v0.12.7
#     does exactly the same, so it is the model's ceiling, not a regression
#     (#1672). The other three 4B aliases are excluded with it rather than
#     waiting to be drawn and measured one by one: a coverage sweep that cries
#     wolf on a random calendar day stops being read at all.
#   * 9B passes — a dense 9B completes hermes in ~740-800 s (see
#     ROUND_TIMEOUT_S above), which is where that budget came from.
#
# 7B and 8B are in between and unmeasured; they stay in the pool because that
# band is where most of the sweep's coverage lives (what is left of it: the
# multimodal filter above takes the three UI-TARS aliases as well, and the pool
# is down to five). If one of them turns out to
# share the 4B ceiling, this constant moves again — that is how the gemma-4 and
# hybrid excludes got here.
#
# What this costs, stated plainly rather than waved away: the 4B tier keeps
# only the coverage it already had, and that is thin. ``qwen3.5-4b-4bit`` is a
# release-fleet coherence model, so G0a boots it every gauntlet — but
# ``evals/coherence_gate.py`` sends short, single-turn, tool-free prompts at
# temperature 0. CI's ``l1-smoke`` adds one forced tool-call format check on the
# same alias, and is not a required check. Neither covers multi-turn contexts,
# streaming tool calls, parser stress under repeated agent traffic, or the other
# three 4B aliases at all. Buying that back needs a small-model tier that is
# sized for what small models can finish, not a sweep that hands them the same
# agentic profiles as a 12B — tracked separately.
_MIN_PARAMS_B = 6.0
# Ceiling: larger models bust the disk budget on M3 16/32 GB systems.
_MAX_PARAMS_B = 12.0


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

# Match an MoE ACTIVE-parameter token: ``A1B`` in ``LFM2.5-8B-A1B``,
# ``A10B`` in ``Qwen3.5-122B-A10B``. Bounded by name separators. Used to
# apply the capability floor to a MoE's active (not total) params — an
# 8B-total/1B-active model is far too weak for agentic harness tasks.
_ACTIVE_TOKEN_RE = re.compile(r"(?:^|[-_.])[aA](\d+(?:\.\d+)?)[bB](?=[-_.]|$)")


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
        if name.lower().startswith("gemma-4-"):
            # Known-bad: model-side ``thought\n…`` loop on agent prompts.
            # See issue #686 + HF discussion google/gemma-4-12B-it#41.
            continue
        if isinstance(entry, dict) and entry.get("is_hybrid"):
            # Hybrid models can't spec/suffix-decode and run agentic
            # harness rounds several times slower → they blow the
            # per-round timeout (esp. hermes) and add false-fail spam,
            # not coverage signal (same class as the gemma-4 exclude).
            # G0a/G0b already cover hybrids on non-agentic prompts.
            continue
        # Use .get() — a future schema change that omits ``hf_path``
        # should silently skip the entry, not crash the gauntlet.
        hf_path = entry.get("hf_path") if isinstance(entry, dict) else None
        if not hf_path:
            continue
        # Multimodal models, checked against the engine's own rule rather than
        # a hand-rolled name test — the harness profiles are text-only, so a
        # VLM here is a crash on a base install and a meaningless result on a
        # vision-enabled one. The HF path matters as much as the alias: the
        # family marker survives there when the alias has dropped it.
        if _is_multimodal(name, hf_path):
            continue
        repo_name = hf_path.split("/")[-1]
        match = _SIZE_TOKEN_RE.search(repo_name)
        if not match:
            # Fail closed: cannot parse a real parameter count from the
            # repo name — skip rather than guess and risk admitting an
            # oversized model into the sweep.
            continue
        size_b = float(match.group(1))
        if not (_MIN_PARAMS_B <= size_b <= _MAX_PARAMS_B):
            continue
        # Effective-capacity floor for MoEs: e.g. ``LFM2.5-8B-A1B`` is 8B
        # total but ~1B ACTIVE/token. The same floor, applied to the params
        # that actually do the work — a model's ability to hold an agentic
        # loop follows its active parameters, not its total.
        active_m = _ACTIVE_TOKEN_RE.search(repo_name)
        if active_m and float(active_m.group(1)) < _MIN_PARAMS_B:
            continue
        out.append((size_b, name, hf_path))
    out.sort()
    return [(name, hf) for _, name, hf in out]


def _free_disk_gb(path: Path) -> float:
    """Free space in GB on the filesystem holding ``path``.

    Walks up to the nearest existing ancestor when ``path`` itself
    doesn't exist yet — the cache root may be on a custom mount whose
    leaf hasn't been created until the first model is downloaded.
    ``shutil.disk_usage`` errors on missing paths, which would block
    G12 from starting on a brand-new ``HF_HUB_CACHE=/data/hf-cache``
    rig where ``/data/`` exists but the leaf doesn't.
    """
    p = path
    while not p.exists():
        parent = p.parent
        if parent == p:
            # Walked all the way to the root and still nothing exists.
            # Let shutil.disk_usage raise — something is very wrong.
            break
        p = parent
    usage = shutil.disk_usage(p)
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


def _run_model_rounds(
    *,
    proc: subprocess.Popen,
    port: int,
    alias: str,
    harnesses: list[str],
    rounds: int,
    bench_log: Path,
    report_path: Path,
) -> tuple[list[str], bool]:
    """Run every (harness × round) for one booted model.

    Returns ``(failures, drifted)``. ``drifted`` means the port stopped being
    ours partway through — an infrastructure failure rather than a result, and
    the caller's cue to stop the whole sweep rather than boot the next model
    into whatever took it.

    Extracted from ``main`` so the control flow can be tested: that ownership
    is asked before AND after each round, that a round whose ownership changed
    is not reported as a result, and that the remaining harnesses are skipped.
    Reading the source to check a call site is present does not establish any
    of those.
    """
    failures: list[str] = []
    base_url = f"http://127.0.0.1:{port}"

    def _record(msg: str) -> None:
        print(f"     FAIL  {msg}", file=sys.stderr)
        with report_path.open("a") as fh:
            fh.write(f"FAIL  {msg}\n")
        failures.append(msg)

    for harness in harnesses:
        for r in range(1, rounds + 1):
            # Ownership is not a one-time fact. Readiness proved the port was
            # ours when the sweep started; a takeover after that (#1618 again)
            # hands every later round to a stranger and the numbers come back
            # attributed to the sampled alias. Ask before AND after — before,
            # so we do not measure someone else; after, so a takeover mid-round
            # cannot be reported as a result.
            drift = _still_ours(proc, port)
            if drift:
                _record(f"{alias}/{harness} round {r}: {drift} before the round")
                return failures, True
            ok, dur, excerpt = _run_harness_round(
                alias=alias,
                harness=harness,
                base_url=base_url,
                bench_log=bench_log,
            )
            drift = _still_ours(proc, port)
            if drift:
                _record(
                    f"{alias}/{harness} round {r}: {drift} during the round"
                    " — its result cannot be attributed to this model"
                )
                return failures, True
            marker = "PASS" if ok else "FAIL"
            line = f"     {marker} {alias}/{harness} round {r}/{rounds} ({dur:.1f}s)"
            if excerpt:
                line += f"  — {excerpt}"
            print(line)
            with report_path.open("a") as fh:
                fh.write(line + "\n")
            if not ok:
                failures.append(f"{alias}/{harness} round {r}: {excerpt}")
    return failures, False


def _as_text(payload: str | bytes | None) -> str:
    """``TimeoutExpired`` carries bytes even when the run asked for text."""
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return payload


# One private directory per run, created 0700, holding every per-alias log.
#
# The paths used to be `/tmp/release-check-m3-random-<alias>.log`, which is
# predictable and sits in a world-writable directory: any other local process
# can pre-create that name as a symlink, and this script's `write_text("")`
# follows it and truncates whatever it points at. Owning the directory removes
# the name-guessing step entirely — nobody else can create entries in it.
_LOG_DIR: Path | None = None


def _log_dir() -> Path:
    """The per-run log directory, created on first use."""
    global _LOG_DIR
    if _LOG_DIR is None:
        _LOG_DIR = Path(tempfile.mkdtemp(prefix="release-check-m3-random-"))
        # mkdtemp is already 0700; make that a stated requirement rather than
        # an implementation detail we inherited.
        _LOG_DIR.chmod(0o700)
        print(f"  logs:     {_LOG_DIR}")
    return _LOG_DIR


def _serve_log_path(alias: str) -> Path:
    """Where the sampled server's own stdout goes."""
    return _log_dir() / f"{alias}.log"


def _bench_log_path(alias: str) -> Path:
    """Where each round's bench transcript goes.

    Deliberately NOT the serve log. The server writes through a descriptor it
    opened ``"w"``, tracking its own byte offset; appending to the same path
    from here moves the end of the file without moving that offset, and the
    server's next line lands on top of the transcript. Two writers, one of
    them offset-based, corrupts the artifact on exactly the runs someone
    needs to read.
    """
    return _log_dir() / f"{alias}.bench.log"


def _parent_pid(pid: int) -> int | None:
    """PPID of ``pid``, or None when it cannot be read."""
    try:
        out = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # Same rule as `_listening_pids`: a non-zero exit discards the output. A
    # `ps` that failed after printing something stale would otherwise hand back
    # a parent that makes a stranger look like our descendant.
    if out.returncode != 0:
        return None
    text = out.stdout.strip()
    return int(text) if text.isdigit() else None


def _listening_pids(port: int) -> list[int]:
    """PIDs with a LISTEN socket on ``port``; ``[]`` if that cannot be
    established (``lsof`` missing, permission denied, timeout, partial run).

    A non-zero exit discards the output rather than trusting it. ``lsof`` can
    print some rows and then fail on a socket it may not inspect, and half a
    listener list is worse than none here: the caller reads "all of these are
    ours" as ownership, so the one row that was not printed is exactly the
    stranger it was asked about.
    """
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [int(tok) for tok in out.stdout.split() if tok.isdigit()]


def _owns_port(proc: subprocess.Popen, port: int, max_depth: int = 8) -> bool:
    """True when every listener on ``port`` is ``proc`` or a descendant.

    A 200 on the port proves something is there, not that it is ours, and
    identity is not ownership either: asking ``/v1/models`` what it serves
    accepts a leftover sidecar or an older engine that happens to hold the
    same weights. This walks the process tree instead, so the only thing
    that satisfies it is the child we started.

    Not hypothetical. While a gauntlet was running, a desktop app on this
    machine swept port 8000, SIGTERM'd the gauntlet's server and bound its
    own sidecar (#1618); every later request went there. It went unnoticed
    because the replacement answered perfectly well.

    Fails CLOSED: an empty listener list while the port demonstrably answers
    means ``lsof`` could not tell us, and an unverifiable port is refused.
    """
    # Every process is a descendant of init, so "the listener descends from
    # pid 1" is true of every listener and proves nothing. Measured, not
    # theorised: with a real listener on a real port, this returned True for
    # pid 1 until the guard existed — and the unit tests could not see it,
    # because they stub the parent walk.
    if proc.pid <= 1:
        return False
    listeners = _listening_pids(port)
    if not listeners:
        return False
    for pid in listeners:
        cur: int | None = pid
        # `max_depth` is a number of ancestry EDGES, so the walk needs
        # max_depth + 1 identity checks: the listener itself, then one after
        # each hop. Checking only `max_depth` times tested every generation
        # except the last one it walked to — so a listener exactly max_depth
        # edges below our process was reported as a stranger and aborted the
        # release as ownership drift.
        for _ in range(max_depth + 1):
            if cur == proc.pid:
                break
            if cur is None or cur <= 1:
                return False
            cur = _parent_pid(cur)
        else:
            return False
    return True


def _still_ours(proc: subprocess.Popen, port: int) -> str:
    """``""`` while the server we started still owns the port, else why not.

    Readiness establishes ownership once, at the start of a sweep that then
    runs for tens of minutes. Everything measured after a takeover belongs to
    whoever took over, so this is asked around every round rather than trusted
    from the beginning.
    """
    rc = proc.poll()
    if rc is not None:
        return f"the server exited (rc={rc})"
    if not _owns_port(proc, port):
        return (
            f"port {port} is no longer served by our process (pid {proc.pid}); "
            f"listeners={_listening_pids(port) or '<could not determine>'}"
        )
    return ""


def _wait_for_server(
    proc: subprocess.Popen, port: int, deadline_s: float, log_path: Path
) -> tuple[bool, bool]:
    """Poll ``/v1/models`` until our own server responds 200, the child
    exits, or the deadline expires. Returns True on success, False
    otherwise.

    Watching ``proc.poll()`` matters: if ``rapid-mlx serve`` aborts at
    import time (missing alias, port collision raced past the
    pre-flight, mlx-lm import error on a clean venv), there is no port
    that will ever come up. Without this check we'd burn the full 600 s
    deadline polling a dead child.

    It is not sufficient on its own, though: while our child is still
    loading weights it is very much alive, so a stranger already on the
    port answers first and the sweep would report that stranger's numbers
    under the sampled alias's name. Hence ``_owns_port``.

    Returns ``(ready, saw_stranger)``. The second value is the difference
    between "this model would not boot" — one model's problem, move on — and
    "somebody else has the port", which is the environment's problem and makes
    every later model's numbers meaningless too. Collapsing both into ``False``
    is how the sweep used to carry on into a contaminated machine.
    """
    url = f"http://127.0.0.1:{port}/v1/models"
    start = time.monotonic()
    warned_stranger = False
    saw_stranger = False
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
                    if _owns_port(proc, port):
                        return True, saw_stranger
                    saw_stranger = True
                    if not warned_stranger:
                        warned_stranger = True
                        print(
                            f"  port {port} answers, but the listener is not our "
                            f"server (pid {proc.pid}); listeners="
                            f"{_listening_pids(port) or '<could not determine>'} "
                            f"— still waiting",
                            file=sys.stderr,
                        )
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(2)
    # Dump the last 30 lines of the server log so the operator sees
    # why we gave up — same shape the shell gauntlet uses.
    if log_path.exists():
        print("  server log (last 30 lines):", file=sys.stderr)
        for line in log_path.read_text(errors="replace").splitlines()[-30:]:
            print(f"    {line}", file=sys.stderr)
    return False, saw_stranger


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
    bench_log: Path,
) -> tuple[bool, float, str]:
    """Run one ``bench --tier harness`` invocation scoped to one
    harness. Returns ``(ok, wall_clock_s, error_excerpt)``.

    ``bench_log`` is a DIFFERENT file from the server log on purpose. The
    server holds its own descriptor on that file, opened ``"w"`` — no
    ``O_APPEND``, so it writes at a position it tracks itself. Appending
    bench output to the same path moves the end of the file without moving
    that position, and the server's next line lands on top of what we just
    wrote. Two writers, one of them offset-based, is a corrupted artifact on
    exactly the runs anyone needs to read.
    """
    env = {**os.environ, "RAPID_MLX_HARNESS_PROFILES_FILTER": harness}
    # Right-size the inner per-profile cap for standalone runs. Via
    # release_check_m3.sh the gauntlet exports HARNESS_PROFILE_TIMEOUT_S
    # (1200s) and this setdefault is a no-op; standalone, default it to
    # ROUND_TIMEOUT_S - 60 so the ~800s hermes profile isn't killed by the
    # 300s library default. A genuine hang still surfaces as a failure —
    # the inner cap trips a hair before the outer subprocess timeout.
    env.setdefault("HARNESS_PROFILE_TIMEOUT_S", str(ROUND_TIMEOUT_S - 60))
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
    except subprocess.TimeoutExpired as expired:
        dur = time.monotonic() - t0
        # Whatever bench managed to say before the clock stopped it is the only
        # evidence there will be about WHERE it got stuck. Returning without
        # writing it leaves the one failure class that most needs a transcript
        # with none at all.
        with bench_log.open("a") as fh:
            fh.write(f"\n=== {alias}/{harness} (TIMED OUT after {dur:.1f}s) ===\n")
            fh.write(_as_text(expired.stdout))
            stderr_text = _as_text(expired.stderr)
            if stderr_text:
                fh.write("\n--- stderr ---\n")
                fh.write(stderr_text)
        return False, dur, f"round timed out after {ROUND_TIMEOUT_S}s"
    dur = time.monotonic() - t0
    # Append the subprocess output to the bench log so a failure has
    # a debuggable trail. ``"a"`` mode is single-write-atomic enough for
    # our single-threaded sweep loop.
    with bench_log.open("a") as fh:
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
            excerpt = f"exit {result.returncode}; see {bench_log}"
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
        default=None,
        help=(
            "Path to write the human-readable summary report to. Defaults to "
            "a file inside this run's private log directory — the previous "
            "default was a fixed name in world-writable /tmp, which any other "
            "local process could pre-create as a symlink for this script to "
            "truncate."
        ),
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
    # Ownership is established from the listener process tree. Without lsof,
    # an answering server is indistinguishable from a stranger and the ready
    # loop would burn its full ten-minute deadline before saying so. The shell
    # wrapper checks this too, but this script is also a supported standalone
    # entry point.
    if shutil.which("lsof") is None:
        print(
            "  Error: lsof is required to verify G12 server ownership.",
            file=sys.stderr,
        )
        return 2
    if not _port_free(args.port):
        print(
            f"  Error: port {args.port} already in use — kill the existing "
            f"server before running G12.",
            file=sys.stderr,
        )
        return 2

    # Check free space on the disk that ACTUALLY holds the HF cache —
    # an install with ``HF_HUB_CACHE=/data/hf-cache`` may have plenty of
    # space on ``/data`` while ``~/.cache`` is tight (or vice-versa).
    # Codex round-2 PR #693 caught this — ``~/.cache`` is wrong for any
    # non-default HF install.
    cache_root = _hf_cache_root()
    free_gb = _free_disk_gb(cache_root)
    if free_gb < MIN_FREE_DISK_GB:
        print(
            f"  Error: only {free_gb:.1f} GB free on the HF cache disk "
            f"({cache_root}); refusing to start (need {MIN_FREE_DISK_GB} GB). "
            f"Clear caches and retry.",
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

    # Resolved before the banner because the banner prints it, and inside
    # `_log_dir()` so an unspecified report shares the run's private
    # directory rather than a fixed name in world-writable /tmp.
    report_path = Path(args.report) if args.report else _log_dir() / "report.log"

    print("=" * 60)
    print("  G12 — random-coverage release gate")
    print(f"  seed:     {args.seed}")
    print(f"  models:   {args.models} (of {len(eligible)} eligible)")
    print(f"  harnesses:{args.harnesses} (of {len(HARNESS_PROFILES)})")
    print(f"  rounds:   {args.rounds}")
    print(f"  report:   {report_path}")
    print(f"  free GB:  {free_gb:.1f}")
    print("=" * 60)
    print("  Sampled matrix:")
    for alias, _, hs in sampled:
        print(f"    {alias:<28} × harnesses={hs}")
    print("=" * 60)

    # Reset the report log.
    report_path.write_text(
        f"G12 random-coverage report (seed={args.seed})\n" + "=" * 60 + "\n"
    )

    # ===== Sweep =====
    failures: list[str] = []
    drifted = False
    for alias, hf_path, harnesses in sampled:
        print()
        print(f"  >> Booting {alias} on port {args.port}…")
        log_path = _serve_log_path(alias)
        bench_log = _bench_log_path(alias)
        log_path.write_text("")
        bench_log.write_text("")
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
            ready, saw_stranger = _wait_for_server(
                proc, args.port, SERVE_READY_TIMEOUT_S, log_path
            )
            if not ready:
                if saw_stranger:
                    # Not this model's failure. Something else holds the port,
                    # and it will still hold it for the next model.
                    msg = (
                        f"{alias}: port {args.port} was answered by another "
                        f"process while this model was booting"
                    )
                    drifted = True
                else:
                    msg = (
                        f"{alias}: server did not respond within "
                        f"{SERVE_READY_TIMEOUT_S}s"
                    )
                print(f"  FAIL  {msg}", file=sys.stderr)
                with report_path.open("a") as fh:
                    fh.write(f"FAIL  {msg}\n")
                failures.append(msg)
                # Deliberately NOT `continue`. A `continue` here runs the
                # `finally` and then starts the next model, stepping straight
                # over the `if drifted: break` below — so a stranger detected
                # during boot recorded the failure and then carried on
                # benchmarking into the same contaminated port, which is the
                # exact outcome the drift check exists to prevent.
            else:
                print(f"     server up ({alias}); harnesses={harnesses}")
                round_failures, drifted = _run_model_rounds(
                    proc=proc,
                    port=args.port,
                    alias=alias,
                    harnesses=harnesses,
                    rounds=args.rounds,
                    bench_log=bench_log,
                    report_path=report_path,
                )
                failures.extend(round_failures)
        finally:
            print(f"  << Stopping {alias}…")
            _stop_server(proc, args.port)
            if not args.keep_cache:
                cache_dir = _hf_cache_dir(hf_path)
                if cache_dir.exists():
                    print(f"     rm -rf {cache_dir}")
                    shutil.rmtree(cache_dir, ignore_errors=True)
        if drifted:
            # Ownership drift is an infrastructure failure, not this model's
            # result. The environment that took the port is still out there;
            # booting the next model into it produces numbers nobody can
            # attribute, and can overlap its GPU allocation with whatever is
            # tearing down. Stop, with our own server already reaped by the
            # `finally` above.
            print(
                "  !! aborting the sweep — the port was taken from us; "
                "remaining models were not run",
                file=sys.stderr,
            )
            with report_path.open("a") as fh:
                fh.write("ABORT ownership drift — remaining models not run\n")
            break

    # ===== Verdict =====
    print()
    print("=" * 60)
    if failures:
        print(f"  G12: {len(failures)} failure(s)")
        for f in failures:
            print(f"    - {f}")
        print(f"  Full log: {report_path}")
        print("=" * 60)
        return 1
    print("  G12: ALL rounds passed")
    print(f"  Full log: {report_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
