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
     the gate. The asymmetry is the whole argument: sharing would save
     the cost of one extra instrumented suite run (~3 min), but at the
     risk of a false block on the gate — which costs a maintainer a
     blocked-PR investigation. Isolation wins.
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

import collections
import importlib.util
import os
import re
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from ..base import Step, StepResult
from ..context import Context

# The package we measure. Coverage is scoped to production code only —
# test files aren't the subject of "is this PR's new code tested".
_COV_PACKAGE = "vllm_mlx"

# Bounded so a hung test or a wedged diff-cover can't block the pipeline
# — the whole point of an advisory step is that it never blocks. On
# expiry ``_run_group_bounded`` SIGKILLs the whole process group; we catch
# and skip. The suite timeout is deliberately generous (the instrumented
# full suite is ~3 min today) — a false timeout would just silently drop a
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

# Bounded in-memory capture per stream. A plain ``communicate()`` buffers the
# ENTIRE suite output for up to ``_PYTEST_TIMEOUT_S``, so a test that streams
# gigabytes to stdout could OOM-kill the validator before its advisory handler
# runs (codex #1220 r17). Instead each stream is drained by a reader thread that
# keeps only the last ~``_CAPTURE_TAIL_BYTES`` — memory stays bounded no matter
# the volume, while the END (diff-cover's footer, pytest's ``-q`` summary) is
# always retained. 1 MiB is far more than any well-behaved run emits, yet a
# hard ceiling against a runaway.
_CAPTURE_TAIL_BYTES = 1_048_576
_CAPTURE_BLOCK_BYTES = 65536  # pipe read granularity

# Keep the diagnostic log readable — we tail (not head) subprocess output so
# the most recent lines (pytest's failure summary) always survive truncation.
_LOG_TAIL_CHARS = 4000


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
            # Fresh-file guarantee: unlink any pre-existing log so ``_skip``
            # can never attach a stale ``diff-coverage.log`` (e.g. from a reused
            # artifact dir) that THIS run did not write — an early skip
            # (tooling missing) writes no log, so its artifact list must stay
            # empty (codex #1220 r15). Best-effort: a failed unlink must not
            # itself escape as a blocking error.
            try:
                log_path.unlink(missing_ok=True)
            except OSError:
                pass
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
        # dedicated (per-run) artifact path so an advisory run can NEVER
        # erase/overwrite a developer's own ``.coverage`` database at the repo
        # root (codex #1220 r12). The XML report is still emitted to
        # ``xml_path``; only the intermediate data file moves. Unlink it first
        # so a leftover from an aborted prior run can't be appended to.
        cov_data = ctx.artifact_path("coverage.data")
        cov_data.unlink(missing_ok=True)
        cov_env = {**os.environ, "COVERAGE_FILE": str(cov_data)}
        # Drop ``PYTEST_ADDOPTS`` from the child env: a host that exports it
        # (``--collect-only``, ``-x``, ``--cov-append``, ``-p no:cov`` …) would
        # otherwise silently turn the run into empty / partial / stale coverage
        # that we would then publish as the PR's measurement. The explicit argv
        # below is the whole spec of the run; nothing may override it (codex
        # #1220 r14).
        cov_env.pop("PYTEST_ADDOPTS", None)

        # 2. Instrumented suite — mirrors ``full_unit``'s selection so the
        #    coverage picture matches what we actually gate on.
        #    ``-o addopts=`` clears any repo-level ``addopts`` from
        #    pyproject.toml / pytest.ini: stripping the ENV var alone is not
        #    enough — a configured ``-x`` / ``--maxfail`` would stop the suite
        #    early and hand us partial exit-1 coverage that we would then
        #    publish as the PR's number. With this override the argv here is the
        #    complete, self-contained spec of the run (codex #1220 r17).
        pytest_cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
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
                f"{_PYTEST_TIMEOUT_S}s — process group killed)",
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
                f"{_DIFF_COVER_TIMEOUT_S}s — process group killed)",
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


def _drain_tail(pipe, chunks: collections.deque) -> None:
    """Reader thread: stream ``pipe`` in fixed blocks, retaining only the last
    ~``_CAPTURE_TAIL_BYTES`` in ``chunks``. The MIDDLE of a runaway stream is
    dropped so validator memory stays bounded regardless of total volume (codex
    #1220 r17), while the END — where diff-cover's footer / pytest's ``-q``
    summary live — is always kept.

    Draining concurrently (rather than after the process exits) is also what
    keeps a chatty child from deadlocking on a full 64 KiB kernel pipe buffer.
    Runs to EOF, which the child's exit — or the group SIGKILL closing every
    write end — delivers; a stuck reader (a setsid-escapee still holding the
    write end) is a harmless daemon thread the caller stops ``join``-ing after a
    bound, never a hang."""
    try:
        for block in iter(lambda: pipe.read(_CAPTURE_BLOCK_BYTES), b""):
            chunks.append(block)
            # Drop whole oldest blocks while the retained tail (excluding the
            # newest block) still exceeds the cap. Keep >=1 block so a stream we
            # actually read is never emptied.
            retained = sum(len(c) for c in chunks)
            while len(chunks) > 1 and retained - len(chunks[0]) >= _CAPTURE_TAIL_BYTES:
                retained -= len(chunks.popleft())
    except (OSError, ValueError):
        # Pipe closed under us / read on a closed fd — nothing left to drain.
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def _joined(chunks: collections.deque) -> str:
    """Decode the retained byte tail. ``errors='replace'`` because child output
    isn't guaranteed valid UTF-8 and a block boundary may split a multibyte
    sequence — a torn edge byte becomes U+FFFD, never an exception."""
    return b"".join(chunks).decode("utf-8", "replace")


def _run_group_bounded(
    cmd: list[str], cwd: str, timeout: int, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` with a hard timeout, killing its whole process group on
    expiry, capturing a BOUNDED tail of each stream.

    Like the merge-GATING ``full_unit`` step this runs the SAME pytest suite and
    returns a ``CompletedProcess`` with text stdout/stderr. It adds the two
    properties an ADVISORY step in an auto-deploy pipeline needs and the gate
    does not:

      * **hard timeout + whole-group kill.** ``subprocess``'s own timeout
        SIGKILLs only the direct child, so a timed-out pytest could orphan xdist
        workers / a serve subprocess into later steps. The child leads a new
        session (``start_new_session=True`` → its own process group) and on
        expiry we SIGKILL the WHOLE group (``_kill_group``). The
        ``TimeoutExpired`` is re-raised (with the captured tail attached) for the
        caller's skip-on-timeout handling.
      * **bounded capture + bounded drain.** Two reader threads keep only the
        last ~``_CAPTURE_TAIL_BYTES`` per stream instead of ``communicate()``
        buffering the entire (up-to-30-min) suite output in memory, which a
        runaway test could grow until it OOM-kills the validator before its
        advisory handler runs (codex #1220 r17). The threads also make the
        post-kill drain BOUNDED: we ``join`` them under ``_REAP_TIMEOUT_S`` and
        proceed with whatever tail we have, so a setsid-escaped descendant still
        holding a pipe write-end cannot wedge the never-block contract — the
        unbounded second ``communicate()`` of the prior design could (codex
        #1220 r17). A stuck reader is a daemon thread reaped at interpreter exit.

    Deliberately NOT done, after a long convergence (codex #1220 r8–r17): no
    ``RLIMIT_FSIZE`` file-size cap (inherited by the whole pytest tree, it would
    truncate/corrupt legitimate large writes — e.g. a >256 MiB model shard —
    codex r16), no temp-file/quota/exec-wrapper machinery. A setsid-escaping
    descendant can still SURVIVE the group-kill — a fundamental POSIX limitation
    the gating siblings don't guard against either (codex r11/r13/r16); portable
    containment would need cgroups / job-objects unavailable here.
    """
    # ``start_new_session=True`` runs the child through ``setsid()``: it leads a
    # brand-new process group whose PGID EQUALS its PID. Capture that id NOW,
    # not via ``os.getpgid(proc.pid)`` on timeout — by then the leader may have
    # exited and ``getpgid`` would raise, leaving the group un-killed. Binary
    # pipes (no ``text=``): the reader threads decode the retained tail.
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    pgid = proc.pid  # == PGID under start_new_session (setsid)
    out_chunks: collections.deque = collections.deque()
    err_chunks: collections.deque = collections.deque()
    t_out = threading.Thread(
        target=_drain_tail, args=(proc.stdout, out_chunks), daemon=True
    )
    t_err = threading.Thread(
        target=_drain_tail, args=(proc.stderr, err_chunks), daemon=True
    )
    t_out.start()
    t_err.start()

    def _join_readers() -> None:
        # Bounded on purpose. On clean exit the pipes hit EOF the moment the
        # child dies, so both joins return at once; after a group SIGKILL EOF
        # arrives when the last writer dies. The bound only bites when a
        # setsid-escapee still holds a write end — we then proceed with the
        # partial tail rather than block the pipeline (codex #1220 r17).
        t_out.join(_REAP_TIMEOUT_S)
        t_err.join(_REAP_TIMEOUT_S)

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Time budget blown: SIGKILL the whole group BEFORE reaping the leader
        # (race-free — ``pgid`` can't be recycled while a member is alive), let
        # the readers drain the now-EOF pipes under a bound, and re-raise with
        # the captured tail so the caller can log WHAT hung.
        _kill_group(proc, pgid)
        _join_readers()
        raise subprocess.TimeoutExpired(
            cmd, timeout, output=_joined(out_chunks), stderr=_joined(err_chunks)
        ) from None
    except BaseException:
        # Any other escape (e.g. KeyboardInterrupt): tear the group down so
        # nothing is orphaned, drain under a bound, then re-raise unchanged.
        _kill_group(proc, pgid)
        _join_readers()
        raise
    _join_readers()
    return subprocess.CompletedProcess(
        cmd, proc.returncode, _joined(out_chunks), _joined(err_chunks)
    )


def _kill_group(proc: subprocess.Popen, pgid: int) -> None:
    """SIGKILL the whole process group, then reap the leader under a bounded
    wait. SIGKILL is uncatchable so the leader dies at once; the bound only
    guards the near-impossible wedged-leader case so we never hang the
    auto-deploy pipeline. Called only on the timeout/exception paths, where the
    leader is unreaped when ``killpg`` fires — so ``pgid`` can't have been
    recycled onto an unrelated group (race-free)."""
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
    """Diagnostics for a timed-out child. Preserves the captured (tail-
    truncated) stdout/stderr carried on the ``TimeoutExpired`` so the log
    records WHAT hung."""
    return (
        f"# {label} TIMED OUT\n\n"
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
