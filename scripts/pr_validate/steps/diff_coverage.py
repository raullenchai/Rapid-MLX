# SPDX-License-Identifier: Apache-2.0
"""Advisory step — patch (diff) coverage measurement. MEASURE-FIRST, NO GATE.

This step reports what fraction of the *production lines this PR adds or
changes* are exercised by the unit suite. It NEVER blocks a merge.

Why advisory-only (dev-flow gate proposal, 2026-07): a hard threshold
picked before we have any distribution is a number out of thin air. The
plan is to publish patch-coverage across ~20 PRs, look at where the
numbers actually land, and only *then* decide whether a gate is
justified. Until then this step exists purely to make the signal
visible in the scorecard (which is posted as a PR comment).

Mechanism:
  1. Run the same unit set ``full_unit`` runs, but under ``coverage``
     instrumentation (``pytest --cov=vllm_mlx --cov-report=xml``). This
     is a SEPARATE, self-contained run — we deliberately do NOT
     piggyback on ``full_unit``. ``full_unit`` is a merge-*gating*
     step; an advisory measurement feature must not be able to break
     the gate. The asymmetry is the whole argument: a false block on
     the gate costs a maintainer a blocked-PR investigation, while
     sharing the run would save only ~40 s. Isolation wins.
  2. Feed the coverage XML to ``diff-cover`` scoped to the diff vs the
     base branch → patch-coverage %.

Because it's advisory, EVERY failure path here returns ``pass`` or
``skip`` — never ``fail`` / ``error``. A crash in an advisory measurer
must not be the reason a good PR can't merge. The base ``execute``
wrapper would turn an *uncaught* exception into ``error`` (which the
scorecard treats as blocking), so ``run`` catches everything and
downgrades to ``skip``. The subprocess calls carry bounded timeouts so
a hung test or diff-cover can't wedge the (auto-deploy) pipeline, and
the diagnostic-log writes are best-effort (a disk-full / permission
error while logging must not escape as a blocking ``error`` either).
"""

from __future__ import annotations

import importlib.util
import os
import re
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from ..base import Step, StepResult
from ..context import Context

# The package we measure. Coverage is scoped to production code only —
# test files aren't the subject of "is this PR's new code tested".
_COV_PACKAGE = "vllm_mlx"

# Bounded so a hung test or a wedged diff-cover can't block the pipeline
# — the whole point of an advisory step is that it never blocks. On
# expiry ``subprocess.run`` SIGKILLs the child; we catch and skip. The
# suite timeout is deliberately generous (the instrumented full suite is
# ~3 min today) — a false timeout would just silently drop a
# measurement, so err long.
_PYTEST_TIMEOUT_S = 1800
_DIFF_COVER_TIMEOUT_S = 180

# pytest exit codes that still yield a trustworthy coverage.xml. 0 = all
# passed; 1 = tests ran but some FAILED — coverage is a complete record of
# the lines that executed either way (pytest-cov writes on session finish
# regardless of outcomes). 2 = interrupted, 3 = internal error, 4 = usage
# error, 5 = no tests collected — those leave no dependable coverage, so we
# skip on them. Accepting exit 1 is deliberate: an ADVISORY baseline
# collector must survive an unrelated flaky test (this repo has known
# GPU-contention flakes, e.g. ``test_batching_improves_throughput``) rather
# than discard the whole measurement and skip on most real PRs (codex
# #1220). ``full_unit`` still owns gating on a red suite.
_PYTEST_COVERAGE_VALID_EXITS = (0, 1)

# Bounded reap after a group SIGKILL. SIGKILL is uncatchable so the leader
# dies at once; this bound only guards the pathological case where a
# descendant that inherited the pipes is wedged in an uninterruptible
# syscall — we must never block the (auto-deploy) pipeline waiting on it.
_REAP_TIMEOUT_S = 10

# Keep the diagnostic log readable — we tail (not head) subprocess output so
# the most recent lines (pytest's failure summary) always survive truncation.
_LOG_TAIL_CHARS = 4000

# Cap on how much subprocess output we read back into memory from the temp
# files (the tail). 1 MiB dwarfs diff-cover's footer and pytest's ``-q``
# summary (both a few KB), so nothing real is ever lost, yet a runaway test
# that streams gigabytes to disk still can't bloat the validator (codex
# #1220 r8).
_STDOUT_CAP_BYTES = 1_048_576

# Disk quota: the largest file the child — or anything in its session — may
# create, enforced by the KERNEL via ``RLIMIT_FSIZE`` (set by an exec-based
# wrapper, ``_rlimit_wrap``). The output goes to disk (not RAM), but disk is finite on
# an auto-deploy host, so a runaway / hung test that streams without bound
# must not exhaust it (codex #1220 r10). A kernel limit closes the gap a
# userspace size-poll leaves — a child that dumps gigabytes and exits BETWEEN
# polls, or in the very first sub-poll window, can't overshoot because the
# write past the limit is refused (EFBIG) / SIGXFSZ-killed at the syscall,
# regardless of timing, and the cap is INHERITED so even a descendant that
# ``setsid()``s out of the group still can't exhaust disk (codex #1220 r11).
# 256 MiB is orders of magnitude above any real ``-q`` suite / diff-cover run
# or coverage artifact, so it only ever trips on genuinely pathological output.
_OUTPUT_QUOTA_BYTES = 256 * 1024 * 1024


class DiffCoverageStep(Step):
    name = "diff_coverage"
    description = "advisory patch (diff) coverage — measure-only, never gates"

    # An advisory measurer must never stop the pipeline. (Belt-and-braces:
    # ``run`` already catches every exception, so ``execute`` can't reach
    # its error path — but if it somehow did, don't halt later steps.)
    continue_on_error = True

    def should_run(self, ctx: Context) -> bool:
        # Nothing to measure on docs/example-only PRs.
        if ctx.blast_radius == "low":
            return False
        # Only meaningful when the PR touches production Python under the
        # measured package — a tests-only or config-only PR has no
        # ``vllm_mlx`` lines for diff-cover to score.
        return any(
            f.startswith(f"{_COV_PACKAGE}/") and f.endswith(".py")
            for f in ctx.files_changed
        )

    def run(self, ctx: Context) -> StepResult:
        # Advisory contract: swallow ALL failures. Never fail/error.
        # Resolve the artifact path INSIDE the protected block —
        # ``artifact_path()`` does a ``mkdir`` that can itself raise on
        # disk-full / permission errors, and even that must not escape
        # through ``execute()`` as a blocking ``error`` (codex #1220).
        # The fallback tolerates ``log_path`` never getting assigned.
        log_path: Path | None = None
        try:
            log_path = ctx.artifact_path("diff-coverage.log")
            return self._measure(ctx, log_path)
        except Exception as e:  # noqa: BLE001 — advisory must not block merge
            # ``_safe_write`` never raises; guard the path being unset too.
            if log_path is not None:
                _safe_write(log_path, traceback.format_exc())
            return self._skip(
                f"advisory coverage skipped (internal error: {type(e).__name__}: {e})",
                log_path,
            )

    # ------------------------------------------------------------------

    def _measure(self, ctx: Context, log_path: Path) -> StepResult:
        # 1. Tooling present? Both are declared in the ``[test]``/``[dev]``
        #    extras, but the operator may run validate from a leaner env.
        #    Missing tooling is a clean skip, not a failure.
        if importlib.util.find_spec("pytest_cov") is None:
            return self._skip(
                "pytest-cov not installed — advisory coverage unavailable"
            )
        if importlib.util.find_spec("diff_cover") is None:
            return self._skip(
                "diff-cover not installed — advisory coverage unavailable"
            )

        xml_path = ctx.artifact_path("coverage.xml")
        # Fresh-file guarantee: never let a stale coverage.xml from an
        # earlier run masquerade as this run's result. If pytest fails to
        # regenerate it below, the ``not xml_path.exists()`` path fires
        # instead of publishing yesterday's numbers.
        xml_path.unlink(missing_ok=True)

        # coverage.py writes its data file to ``$COVERAGE_FILE`` or, unset,
        # ``.coverage`` in the CWD — which here is the repo root. Point it at a
        # dedicated artifact path so an advisory run can NEVER erase/overwrite a
        # developer's own ``.coverage`` database sitting at the repo root (codex
        # #1220 r12). The XML report is still emitted to ``xml_path``; only the
        # intermediate data file moves.
        cov_env = {
            **os.environ,
            "COVERAGE_FILE": str(ctx.artifact_path("coverage.data")),
        }

        # 2. Instrumented suite — mirrors ``full_unit``'s selection so the
        #    coverage picture matches what we actually gate on.
        pytest_cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--ignore=tests/integrations",
            "--ignore=tests/test_event_loop.py",
            f"--cov={_COV_PACKAGE}",
            # Suppress the terminal cov table (noise); we only need XML.
            "--cov-report=",
            f"--cov-report=xml:{xml_path}",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ]
        ctx.run_log("diff_coverage: running instrumented suite (advisory)…")
        try:
            pytest_proc = _run_group_bounded(
                pytest_cmd, str(ctx.repo_root), _PYTEST_TIMEOUT_S, env=cov_env
            )
        except subprocess.TimeoutExpired as exc:
            _safe_write(log_path, _timeout_dump("instrumented pytest", pytest_cmd, exc))
            return self._skip(
                f"advisory coverage skipped (instrumented suite exceeded "
                f"{_PYTEST_TIMEOUT_S}s or its output quota — process group killed)",
                log_path,
            )

        # 3. Coverage trust model (see ``_PYTEST_COVERAGE_VALID_EXITS``). We
        #    accept exit 0 (all passed) and exit 1 (some tests failed but the
        #    run completed and coverage.xml is a valid record of executed
        #    lines) and skip only on 2-5 (interrupted / internal error /
        #    usage error / no tests) — those leave no dependable coverage. A
        #    failing test that WAS meant to exercise the changed lines simply
        #    leaves them uncovered, which is honest advisory signal, not
        #    contamination; the caveat below flags that the % may under-count.
        if pytest_proc.returncode not in _PYTEST_COVERAGE_VALID_EXITS:
            _safe_write(log_path, _pytest_dump(pytest_cmd, pytest_proc))
            return self._skip(
                f"advisory coverage skipped (instrumented suite exit "
                f"{pytest_proc.returncode} — interrupted/error, not merely "
                f"test failures; see full_unit)",
                log_path,
            )
        suite_had_failures = pytest_proc.returncode == 1

        if not xml_path.exists():
            _safe_write(log_path, _pytest_dump(pytest_cmd, pytest_proc))
            return self._skip(
                "advisory coverage skipped (no coverage.xml produced)", log_path
            )

        # 4. diff-cover: patch coverage of the changed lines vs the PR's
        #    base. The compare ref is the PR's ACTUAL base — ``ctx.base_sha``
        #    (its ``baseRefOid``, a concrete commit) — so a PR targeting a
        #    release/maintenance branch is scored against ITS base, never a
        #    hardcoded ``main`` (codex #1220 r5). A SHA also resolves in a
        #    detached CI checkout, where a *bare* local branch name may not
        #    exist. So the no-metadata fallback qualifies the target branch
        #    with the remote — ``origin/<base_branch>`` — which exists after
        #    fetch even detached, rather than a bare ``main`` that would fail
        #    to resolve and skip every fallback run (codex #1220 r6). Invoke
        #    via ``-m`` so it runs in the SAME interpreter the coverage was
        #    produced with (matches targeted_tests' policy).
        compare_ref = ctx.base_sha or f"origin/{ctx.base_branch}"
        dc_cmd = [
            sys.executable,
            "-m",
            "diff_cover.diff_cover_tool",
            str(xml_path),
            "--compare-branch",
            compare_ref,
        ]
        try:
            dc_proc = _run_group_bounded(
                dc_cmd, str(ctx.repo_root), _DIFF_COVER_TIMEOUT_S
            )
        except subprocess.TimeoutExpired as exc:
            _safe_write(log_path, _timeout_dump("diff-cover", dc_cmd, exc))
            return self._skip(
                f"advisory coverage skipped (diff-cover exceeded "
                f"{_DIFF_COVER_TIMEOUT_S}s or its output quota — process group killed)",
                log_path,
            )

        # Record pytest's own output too (tail-truncated) — on an accepted
        # exit-1 measurement the reader needs to see WHICH tests failed and
        # why, not just the bare exit code (codex #1220 r5 nit).
        _safe_write(
            log_path,
            "# diff_coverage advisory run\n\n"
            f"## pytest cmd\n`{' '.join(pytest_cmd)}`\n\n"
            f"## pytest exit: {pytest_proc.returncode}\n\n"
            "## pytest stdout (tail)\n"
            + _tail(pytest_proc.stdout)
            + "\n## pytest stderr (tail)\n"
            + _tail(pytest_proc.stderr)
            + f"\n## diff-cover cmd\n`{' '.join(dc_cmd)}`\n\n"
            f"## diff-cover exit: {dc_proc.returncode}\n\n"
            "## diff-cover stdout\n"
            + (dc_proc.stdout or "")
            + "\n## diff-cover stderr\n"
            + (dc_proc.stderr or ""),
        )

        # A nonzero diff-cover exit means it errored (bad XML, git
        # failure, interrupted) — even if it happened to print a footer
        # first, that number is not trustworthy. Skip BEFORE parsing so a
        # failed run can't publish a misleading success (codex #1220).
        # Verified exit codes: scored-lines=0, no-lines=0, bad-xml=1,
        # bad-branch=1 — so this never false-skips the happy path.
        if dc_proc.returncode != 0:
            return self._skip(
                f"advisory coverage skipped (diff-cover exit "
                f"{dc_proc.returncode} — not scored)",
                log_path,
            )

        parsed = _parse_diff_cover(dc_proc.stdout)
        if parsed is None:
            # diff-cover ran but found no changed lines it could score
            # (e.g. every changed line is a comment / blank / not in the
            # coverage source map). Nothing to report — clean skip.
            return self._skip(
                "advisory coverage skipped (no measurable production lines in diff)",
                log_path,
            )
        if parsed is _PARSE_FAILED:
            # diff-cover exited 0 but we recognized NEITHER its explicit
            # no-lines message NOR a Total/Missing footer — most likely a
            # diff-cover version whose output format drifted (the ``>=8.0.0``
            # floor permits future majors). Surface this as a DISTINCT
            # tooling-format skip rather than silently reporting "no lines",
            # which would let the baseline quietly die (codex #1220 r8).
            return self._skip(
                "advisory coverage skipped (diff-cover output format "
                "unrecognized — parser may need updating for this diff-cover "
                "version)",
                log_path,
            )

        pct, covered, total = parsed
        caveat = (
            " NOTE: some unit tests failed this run (see full_unit) — the % "
            "may under-count lines a failing test would otherwise have covered."
            if suite_had_failures
            else ""
        )
        # One decimal (not {:.0f}) so 99.5% doesn't read as a misleading
        # "100%" nor 0.5% as "0%" — matches the finding's precision (codex
        # #1220 r6 nit).
        summary = (
            f"patch coverage {pct:.1f}% ({covered}/{total} changed lines) · "
            "advisory — not gating"
        )
        finding = (
            f"[ADVISORY] patch (diff) coverage: {pct:.1f}% — "
            f"{covered}/{total} newly changed {_COV_PACKAGE} lines exercised "
            "by the unit suite. Measure-only; no threshold enforced yet "
            f"(collecting baseline across ~20 PRs before deciding a gate).{caveat}"
        )
        return StepResult(
            name=self.name,
            status="pass",  # ALWAYS pass — advisory
            summary=summary,
            findings=[finding],
            # Advertise only artifacts that actually made it to disk —
            # ``_safe_write`` may have swallowed a log write failure, and the
            # existence probe itself must not raise (codex #1220 r8), mirroring
            # the skip path's guard.
            artifacts=[str(p) for p in (log_path, xml_path) if _path_exists(p)],
        )

    # ------------------------------------------------------------------

    def _skip(self, summary: str, log_path: Path | None = None) -> StepResult:
        """Uniform advisory skip. Attaches the log artifact only when it
        actually made it to disk (``_safe_write`` may have swallowed a
        write error)."""
        return StepResult(
            name=self.name,
            status="skip",
            summary=summary,
            artifacts=[str(log_path)] if _path_exists(log_path) else [],
        )


def _run_group_bounded(
    cmd: list[str], cwd: str, timeout: int, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` in its OWN process group with a hard timeout.

    ``subprocess.run(timeout=...)`` SIGKILLs only the direct child on
    expiry — a timed-out pytest could leave spawned servers/workers alive
    to contaminate later validation steps (codex #1220). We start the
    child in a new session (``start_new_session=True`` → new process
    group) and, on timeout, SIGKILL the WHOLE group so no descendant
    survives. Re-raises ``subprocess.TimeoutExpired`` so callers keep
    their existing skip-on-timeout handling.

    Output is redirected to temp FILES, not PIPEs (codex #1220 r8). This
    bounds validator memory — a noisy / runaway test streams to disk, and
    we read back only a capped tail — and, because there are no pipes for a
    surviving descendant to hold open, it removes the reap-wedge class
    entirely. The tail keeps the bytes every caller needs — diff-cover's
    footer and pytest's ``-q`` failure summary both sit at the END of their
    streams. Disk is bounded too, but by the KERNEL not a userspace poll: the
    child is spawned under an ``RLIMIT_FSIZE`` of ``_OUTPUT_QUOTA_BYTES``, so a
    write past the cap is refused at the syscall the instant it happens — no
    matter how fast the child writes or whether it exits between two userspace
    polls, and even if a descendant ``setsid()``s out of the group (the limit
    is inherited). A child killed by ``SIGXFSZ`` (or exiting nonzero on
    ``EFBIG``) then fails the exit-code gate and the run is skipped (codex
    #1220 r11). The limit is applied by a tiny exec-based WRAPPER (see
    ``_rlimit_wrap``) rather than a ``preexec_fn`` callback: ``preexec_fn``
    runs between fork and exec in a copy of the (possibly multi-threaded)
    parent's address space, where grabbing an internal lock another thread
    held at fork can deadlock the child forever — before ``Popen`` even
    returns, so the timeout above could never fire and the pipeline would
    wedge. The wrapper instead sets the limit from a FRESH, single-threaded
    process (post-exec) and then ``exec``s the real command, sidestepping
    that class entirely (codex #1220 r12).

    On timeout the raised ``TimeoutExpired`` carries the captured (capped)
    stdout/stderr so the caller can log WHAT hung before the temp files are
    deleted (codex #1220 r10).

    The isolated group is swept on EVERY exit path, not just timeout: a
    pytest that exits cleanly can still leak a background server/worker into
    the group, which would otherwise survive to contaminate later steps
    (codex #1220 r9).
    """
    # ``start_new_session=True`` runs the child through ``setsid()``: it
    # leads a brand-new process group whose PGID EQUALS its PID. Capture that
    # id NOW, not via ``os.getpgid(proc.pid)`` on timeout — by then the leader
    # may already have exited and ``getpgid`` would raise, leaving the group
    # un-killed (codex #1220).
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [*_rlimit_wrap(), *cmd],
            stdout=out_f,
            stderr=err_f,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        pgid = proc.pid  # == PGID under start_new_session (setsid)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Time budget blown: tear the whole group down and re-raise with
            # the captured tail so the caller can log WHAT hung. Disk is
            # handled separately by the kernel RLIMIT (no userspace poll).
            _kill_group(proc, pgid)
            raise subprocess.TimeoutExpired(
                cmd,
                timeout,
                output=_read_capped(out_f),
                stderr=_read_capped(err_f),
            ) from None
        except BaseException:
            # Any other escape (e.g. KeyboardInterrupt): tear the group down
            # so nothing is orphaned, then re-raise unchanged.
            _kill_group(proc, pgid)
            raise
        # Clean leader exit — but SWEEP the group anyway: a cleanly-exited
        # pytest may have leaked a background server/worker still running in
        # the isolated group. Best-effort; an already-empty group is ESRCH.
        _sweep_group(pgid)
        return subprocess.CompletedProcess(
            cmd, proc.returncode, _read_capped(out_f), _read_capped(err_f)
        )


# Wrapper source: a fresh Python process that sets ``RLIMIT_FSIZE`` on ITSELF
# then ``exec``s the real command (argv after the byte count). rlimits survive
# ``execve`` and are inherited across ``fork``, so the real command and every
# descendant it spawns run under the cap. Setting the limit here — post-exec,
# in a brand-new single-threaded process — avoids the ``preexec_fn`` fork/exec
# deadlock window entirely (codex #1220 r12). ``os.execvp`` uses an absolute
# ``argv[0]`` (our commands start with ``sys.executable``) directly.
_RLIMIT_WRAP_SRC = (
    "import os,resource,sys;"
    "n=int(sys.argv[1]);"
    "resource.setrlimit(resource.RLIMIT_FSIZE,(n,n));"
    "os.execvp(sys.argv[2],sys.argv[2:])"
)


def _rlimit_wrap() -> list[str]:
    """Argv prefix that caps the wrapped command's max file size at
    ``_OUTPUT_QUOTA_BYTES`` (the KERNEL disk backstop; see ``_run_group_bounded``).
    Built per-call so a test monkeypatching ``_OUTPUT_QUOTA_BYTES`` takes
    effect. Unix-only, like the rest of this helper (``setsid``/``killpg``)."""
    return [sys.executable, "-c", _RLIMIT_WRAP_SRC, str(_OUTPUT_QUOTA_BYTES)]


def _sweep_group(pgid: int) -> None:
    """Best-effort SIGKILL of any process still lingering in the isolated
    process group after its leader has exited. An empty group raises ESRCH,
    which we ignore. (PID-reuse of ``pgid`` in the microseconds since the
    leader was reaped is theoretically possible but negligible, and no worse
    than any ``killpg``-after-``wait``.)"""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


def _kill_group(proc: subprocess.Popen, pgid: int) -> None:
    """SIGKILL the whole process group, then reap the leader under a bounded
    wait. With output on temp files (no inherited pipes) there is nothing for
    a descendant to hold open, so a plain ``wait`` reaps the SIGKILLed leader
    at once; the bound only guards the near-impossible wedged-leader case so
    we never hang the auto-deploy pipeline."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        # Group already gone / not permitted — fall back to the leader.
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pass  # truly wedged leader (near-impossible post-SIGKILL) — give up


def _read_capped(f, limit: int = _STDOUT_CAP_BYTES) -> str:
    """Read back the LAST ``limit`` bytes of a temp file as text. Bounds
    read-back memory regardless of how much the child wrote, while keeping
    the stream tail that callers actually parse/log (diff-cover footer,
    pytest failure summary). Decodes leniently — the file is raw bytes."""
    f.flush()
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(max(0, size - limit))
    data = f.read().decode("utf-8", "replace")
    if size > limit:
        return f"…[{size - limit} bytes elided]…\n" + data
    return data


def _path_exists(path: Path | None) -> bool:
    """``Path.exists()`` that NEVER raises. ``pathlib.Path.exists`` re-raises
    OSErrors other than ENOENT/ENOTDIR (e.g. EACCES permission-denied, EIO on
    a failing mount), so a bare ``log_path.exists()`` inside ``_skip`` could
    escape ``run``'s handler and become the blocking ``error`` this advisory
    step promises never to produce (codex #1220 r7). Treat any error as
    'not present' — the worst case is a missing artifact link, never a block."""
    if path is None:
        return False
    try:
        return path.exists()
    except OSError:
        return False


def _safe_write(path: Path, text: str) -> None:
    """Best-effort diagnostic write. NEVER raises. An advisory step must
    still return skip/pass even if it can't write its own log (disk full,
    permissions) — otherwise a failing write inside an exception handler
    would escape as a blocking ``error`` (codex review on #1220).

    Writes explicitly as UTF-8 with ``errors="replace"``: the default
    locale encoding can be ASCII on some CI, and our diagnostics contain
    non-ASCII (tracebacks, the ``…`` elision marker), which would raise
    ``UnicodeEncodeError`` — a ``ValueError``, NOT an ``OSError`` — and slip
    past a bare ``except OSError`` to become a blocking error (codex #1220
    r10). We also catch ``UnicodeError`` belt-and-braces."""
    try:
        path.write_text(text, encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        pass


def _tail(text: str | None, limit: int = _LOG_TAIL_CHARS) -> str:
    """Last ``limit`` chars of ``text`` (pytest's failure summary lives at the
    end), with an elision marker when truncated. Never raises."""
    s = text or ""
    if len(s) <= limit:
        return s
    return f"…[{len(s) - limit} chars elided]…\n" + s[-limit:]


def _pytest_dump(cmd: list[str], proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f"# instrumented pytest (exit {proc.returncode})\n\n"
        f"## cmd\n`{' '.join(cmd)}`\n\n"
        "## stdout\n" + _tail(proc.stdout) + "\n## stderr\n" + _tail(proc.stderr)
    )


def _timeout_dump(label: str, cmd: list[str], exc: subprocess.TimeoutExpired) -> str:
    """Diagnostics for a timed-out / output-quota-exceeded child. Preserves
    the captured tail carried on the exception so the log records WHAT hung
    before the temp files are deleted (codex #1220 r10)."""
    return (
        f"# {label} TIMED OUT or exceeded its output quota\n\n"
        f"## cmd\n`{' '.join(cmd)}`\n\n"
        "## stdout (tail)\n"
        + _tail(exc.stdout if isinstance(exc.stdout, str) else "")
        + "\n## stderr (tail)\n"
        + _tail(exc.stderr if isinstance(exc.stderr, str) else "")
    )


# ----------------------------------------------------------------------
# diff-cover output parsing — we parse the stable human-readable text
# rather than a JSON report because the JSON flag name has churned
# across diff-cover versions (``--json-report`` vs ``--format json:``)
# while the text footer (``Total:`` / ``Missing:`` / ``Coverage:``) has
# been stable for years.
# ----------------------------------------------------------------------

_TOTAL_RE = re.compile(r"^Total:\s+(\d+)\s+lines?", re.MULTILINE)
_MISSING_RE = re.compile(r"^Missing:\s+(\d+)\s+lines?", re.MULTILINE)
_NO_LINES_RE = re.compile(r"No lines with coverage information", re.IGNORECASE)

# Distinct from ``None``: diff-cover exited 0 but we recognized NEITHER its
# explicit no-lines message NOR a Total/Missing footer. ``None`` means the
# legitimate "nothing to score" case; this sentinel means the output format
# was unrecognizable — most likely a diff-cover version whose text drifted
# (the ``>=8.0.0`` floor permits future majors). The caller reports the two
# differently so a format break can't masquerade as "no lines" and silently
# kill baseline collection (codex #1220 r8).
_PARSE_FAILED = object()


def _parse_diff_cover(stdout: str) -> tuple[float, int, int] | None | object:
    """Return ``(percent, covered_lines, total_lines)``; ``None`` when
    diff-cover legitimately reports nothing to score; or ``_PARSE_FAILED``
    when its output is unrecognizable (format drift — see the sentinel).

    Percent is computed from the exact ``Total`` / ``Missing`` counts,
    NOT from diff-cover's own ``Coverage:`` line. diff-cover *floors*
    that displayed integer (e.g. 58/1934 = 2.9989 % prints as
    ``Coverage: 2%``), which both loses resolution and can read a point
    below the true value — bad for a baseline we intend to threshold on
    later. So our headline % may read ~1 pt above the number in the
    saved diff-cover log; the ``covered/total`` counts we surface
    alongside it are the unambiguous ground truth.
    """
    text = stdout or ""
    if _NO_LINES_RE.search(text):
        return None  # legitimate: no scorable lines in the diff
    tm = _TOTAL_RE.search(text)
    mm = _MISSING_RE.search(text)
    if not tm or not mm:
        return _PARSE_FAILED  # unrecognized footer — format drift
    total = int(tm.group(1))
    missing = int(mm.group(1))
    if total <= 0:
        return None  # explicit "Total: 0 lines" — nothing to score
    if not 0 <= missing <= total:
        # Missing out of [0, Total] is impossible for real diff-cover output;
        # a footer this malformed means format drift, not a coverage number.
        # Guarding it stops a bogus negative / >100 % from being published
        # (codex #1220 r9).
        return _PARSE_FAILED
    covered = total - missing
    pct = 100.0 * covered / total
    return pct, covered, total
