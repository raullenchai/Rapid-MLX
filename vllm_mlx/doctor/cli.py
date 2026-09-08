# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for ``rapid-mlx doctor`` — pure env-health probe.

Doctor is now strictly about answering "is my install / environment broken?".
Model-validation tiers (smoke / check / full / benchmark) moved to
``rapid-mlx bench --tier ...`` in PRs #1-#3 of the doctor refactor series;
this PR rips out the dispatch and leaves only the env-health surface.

Output is modelled on ``hermes doctor``: sections of one-line ✓/⚠/✗ probes,
a summary, and an exit code (0 unless any ✗). Built-in I/O is timeout-bounded,
and the runner targets a five-second scheduling budget.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from vllm_mlx import __version__

from .env_health import CheckStatus, Report, Section, run_all

# Removed tiers — referenced from doctor_command's error message so a user
# typing the old subcommand sees the canonical replacement, not a bare
# ``unknown argument`` traceback.
_REMOVED_TIERS = ("smoke", "check", "full", "benchmark")


# Status glyphs. ASCII fallbacks aren't supported — we already require a
# UTF-8 terminal for the section headers (◆), so a stray cp1252 user gets
# mojibake everywhere, not just on the status column.
_GLYPHS = {
    CheckStatus.OK: "✓",  # ✓
    CheckStatus.WARN: "⚠",  # ⚠
    CheckStatus.FAIL: "✗",  # ✗
    CheckStatus.SKIPPED: "○",
}

_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:[\"'])?\b([a-z][a-z0-9_.-]*)\b(?:[\"'])?\s*[=:]\s*"
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)\b(?P<authorization>authorization\s+(?:bearer|basic)\s+)"
)
_AUTH_SCHEME_RE = re.compile(
    r"(?i)(?<![\w-])(?P<authorization>(?:bearer|basic)\s+)(?=\S+)"
)
_SPACE_KEY_VALUE_RE = re.compile(
    r"(?i)(?=\b(?P<key>[a-z][a-z0-9_.-]*)\s+(?P<value>\S+))"
)
_URL_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/?#]*@")
_SECRET_KEY_COMPONENTS = {
    "token",
    "secret",
    "password",
    "passwd",
    "pwd",
    "passphrase",
    "credential",
    "authorization",
    "auth",
    "cookie",
    "privatekey",
    "accesskey",
    "apikey",
    "secretkey",
    "authorizationheader",
    "sessionid",
    "connectionstring",
}
_SECRET_KEY_PAIRS = {
    ("private", "key"),
    ("access", "key"),
    ("api", "key"),
    ("secret", "key"),
    ("authorization", "header"),
    ("session", "id"),
    ("connection", "string"),
}


def _is_secret_key(key: str) -> bool:
    """Recognize sensitive components in snake, kebab, dotted, or camel keys."""
    words = [
        word.lower()
        for word in re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+",
            re.sub(r"[_.-]+", " ", key),
        )
    ]
    if any(word in _SECRET_KEY_COMPONENTS for word in words):
        return True
    return any(pair in _SECRET_KEY_PAIRS for pair in zip(words, words[1:]))


def _open_collector_pipes() -> tuple[int, int, int, int]:
    """Allocate collector lifecycle pipes without leaking on partial setup."""
    allocated: list[int] = []
    try:
        keepalive_read_fd, keepalive_write_fd = os.pipe()
        allocated.extend((keepalive_read_fd, keepalive_write_fd))
        status_read_fd, status_write_fd = os.pipe()
        allocated.extend((status_read_fd, status_write_fd))
        os.set_blocking(status_read_fd, False)
        return (
            keepalive_read_fd,
            keepalive_write_fd,
            status_read_fd,
            status_write_fd,
        )
    except Exception:
        for fd in allocated:
            os.close(fd)
        raise


def _collect_report_isolated(
    *,
    only: set[str] | None,
    skip: set[str] | None,
    deep: bool = False,
    timeout_s: float = 8.0,
) -> Report:
    """Collect a report in a fresh executable whose stdout is discarded."""
    with tempfile.TemporaryDirectory(prefix="rapid-mlx-doctor-") as result_dir:
        request_path = Path(result_dir) / "request.json"
        result_path = Path(result_dir) / "report.json"
        request_path.write_text(
            json.dumps(
                {
                    "only": None if only is None else sorted(only),
                    "skip": None if skip is None else sorted(skip),
                    "deep": deep,
                }
            ),
            encoding="utf-8",
        )
        (
            keepalive_read_fd,
            keepalive_write_fd,
            status_read_fd,
            status_write_fd,
        ) = _open_collector_pipes()
        process: subprocess.Popen | None = None
        try:
            process = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    "-I",
                    "-c",
                    (
                        "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                        "runpy.run_module('vllm_mlx.doctor.json_worker',"
                        "run_name='__main__')"
                    ),
                    str(Path(__file__).resolve().parents[2]),
                    str(request_path),
                    str(result_path),
                    str(keepalive_read_fd),
                    str(status_write_fd),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=(keepalive_read_fd, status_write_fd),
                start_new_session=True,
            )
        except Exception:
            os.close(keepalive_write_fd)
            os.close(status_read_fd)
            raise
        finally:
            os.close(keepalive_read_fd)
            os.close(status_write_fd)
        assert process is not None
        deadline = time.monotonic() + timeout_s
        result_ready = False
        child_exited = False
        try:
            # The worker stays alive on the keepalive pipe after publishing
            # the atomic result, so its process-group ID cannot be reused.
            while time.monotonic() < deadline:
                if result_path.exists():
                    result_ready = True
                    break
                try:
                    if os.read(status_read_fd, 1) == b"":
                        child_exited = True
                        break
                except BlockingIOError:
                    pass
                time.sleep(0.01)
        finally:
            # Kill the disposable group before closing the keepalive or
            # reaping its leader. SIGKILL also covers stuck descendants.
            group_cleanup_error: OSError | None = None
            try:
                _terminate_collector_group(process.pid)
            except OSError as exc:
                # A child that exited before publishing may already have no
                # live process group on macOS; the unreaped leader still
                # protects the PID until the wait below.
                if not child_exited:
                    group_cleanup_error = exc
            finally:
                os.close(keepalive_write_fd)
                os.close(status_read_fd)
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                direct_kill_error: OSError | None = None
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    direct_kill_error = exc
                try:
                    process.wait(timeout=0.5)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise RuntimeError("JSON collector could not be reaped") from exc
                if direct_kill_error is not None:
                    raise RuntimeError("JSON collector could not be killed") from (
                        direct_kill_error
                    )
            if group_cleanup_error is not None:
                raise RuntimeError("JSON collector group cleanup failed") from (
                    group_cleanup_error
                )
        if child_exited:
            raise RuntimeError("JSON report child exited without a report")
        if not result_ready:
            raise RuntimeError("JSON report collection timed out")
        try:
            message = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"JSON report result is unreadable: {exc}") from exc

    if not isinstance(message, dict) or not isinstance(message.get("ok"), bool):
        raise RuntimeError("JSON report child returned an invalid message")
    if not message["ok"]:
        raise RuntimeError(
            f"JSON report collection failed: {message.get('error', '?')}"
        )
    return _report_from_document(message.get("report"))


def _report_from_document(document: object) -> Report:
    """Strictly validate and reconstruct a schema-v1 collector report."""
    try:
        root_keys = {
            "schemaVersion",
            "rapidMlxVersion",
            "status",
            "exitCode",
            "durationMs",
            "summary",
            "sections",
        }
        if not isinstance(document, dict) or set(document) != root_keys:
            raise TypeError("report root fields are invalid")
        if (
            isinstance(document["schemaVersion"], bool)
            or not isinstance(document["schemaVersion"], int)
            or document["schemaVersion"] != 1
        ):
            raise ValueError("schemaVersion must be 1")
        if not isinstance(document["rapidMlxVersion"], str):
            raise TypeError("rapidMlxVersion is invalid")
        if document["status"] not in {"ok", "warn", "fail", "skipped"}:
            raise ValueError("status is invalid")
        if (
            isinstance(document["exitCode"], bool)
            or not isinstance(document["exitCode"], int)
            or document["exitCode"] not in {0, 1}
        ):
            raise ValueError("exitCode is invalid")
        _require_nonnegative_int(document["durationMs"], "durationMs")
        summary = document["summary"]
        if not isinstance(summary, dict) or set(summary) != {
            "ok",
            "warnings",
            "failures",
            "skipped",
        }:
            raise TypeError("summary is invalid")
        for key, value in summary.items():
            _require_nonnegative_int(value, f"summary.{key}")
        if not isinstance(document["sections"], list):
            raise TypeError("sections is invalid")
        report = Report(duration_ms=document["durationMs"])
        for raw_section in document["sections"]:
            if not isinstance(raw_section, dict) or set(raw_section) != {
                "id",
                "title",
                "durationMs",
                "checks",
            }:
                raise TypeError("section is invalid")
            if not isinstance(raw_section["id"], str) or not raw_section["id"]:
                raise TypeError("section id is invalid")
            if not isinstance(raw_section["title"], str) or not raw_section["title"]:
                raise TypeError("section title is invalid")
            _require_nonnegative_int(raw_section["durationMs"], "section duration")
            if not isinstance(raw_section["checks"], list):
                raise TypeError("section checks are invalid")
            section = Section(
                raw_section["title"],
                id=raw_section["id"],
                duration_ms=raw_section["durationMs"],
            )
            for raw_check in raw_section["checks"]:
                if not isinstance(raw_check, dict) or set(raw_check) != {
                    "id",
                    "status",
                    "summary",
                    "detail",
                }:
                    raise TypeError("check is invalid")
                raw_id = raw_check["id"]
                if raw_id is not None and not isinstance(raw_id, str):
                    raise TypeError("check id is invalid")
                if not isinstance(raw_check["summary"], str) or not isinstance(
                    raw_check["detail"], str
                ):
                    raise TypeError("check text is invalid")
                section.add(
                    raw_check["summary"],
                    CheckStatus(str(raw_check["status"])),
                    detail=raw_check["detail"],
                    check_id=raw_id,
                )
            report.sections.append(section)
        expected_summary = {
            "ok": report.n_ok,
            "warnings": report.n_warn,
            "failures": report.n_fail,
            "skipped": report.n_skipped,
        }
        if summary != expected_summary:
            raise ValueError("summary counts do not match checks")
        if document["status"] != report.overall_status:
            raise ValueError("root status does not match checks")
        if document["exitCode"] != report.exit_code:
            raise ValueError("exitCode does not match checks")
        return report
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"JSON report child returned an invalid report: {exc}"
        ) from exc


def _require_nonnegative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{field_name} is invalid")


def _terminate_collector_group(pid: int) -> None:
    """Kill a disposable collector group while its leader is still owned."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _collection_failure_report(exc: Exception) -> Report:
    section = Section("Doctor", id="doctor")
    section.add(
        "JSON report collection failed",
        CheckStatus.FAIL,
        detail=f"{type(exc).__name__}: {exc}",
        check_id="doctor.json.collection",
    )
    return Report(sections=[section])


def doctor_command(args: Any) -> None:
    """Render the env-health report and ``sys.exit`` with 0 or 1.

    ``args`` is the argparse namespace from ``vllm_mlx.cli``. We only read
    ``args.verbose`` (and reject the removed positional ``tier`` argument
    with a clear pointer to ``rapid-mlx bench --tier ...``).
    """
    # Hard removal: PRs #1-#3 deprecated the tier subcommands; this PR closes
    # the door. If a user still types ``rapid-mlx doctor smoke`` they hit
    # this branch and get pointed at the replacement.
    legacy_tier = getattr(args, "tier", None)
    if legacy_tier in _REMOVED_TIERS:
        print(
            f"rapid-mlx doctor {legacy_tier!s} was removed in 0.7.22.\n"
            f"Use:  rapid-mlx bench <model> --tier {legacy_tier}\n"
            "Doctor is now a pure environment-health check; "
            "model-validation tiers live in `rapid-mlx bench`.",
            file=sys.stderr,
        )
        sys.exit(2)

    verbose = bool(getattr(args, "verbose", False))
    json_output = bool(getattr(args, "json", False))
    summary_only = bool(getattr(args, "summary", False))
    raw_only = getattr(args, "only", None)
    raw_skip = getattr(args, "skip", None)
    only = None if raw_only is None else set(raw_only)
    skip = None if raw_skip is None else set(raw_skip)
    fix = bool(getattr(args, "fix", False))
    dry_run = bool(getattr(args, "dry_run", False))
    assume_yes = bool(getattr(args, "yes", False))
    # Selecting the deep section is itself an explicit opt-in. Without this,
    # ``--only deep`` filters out every normal builder while never registering
    # the deep builder, producing an empty successful report.
    deep = bool(getattr(args, "deep", False) or fix or (only and "deep" in only))

    if dry_run and not fix:
        print("error: --dry-run requires --fix", file=sys.stderr)
        sys.exit(2)
    if assume_yes and not fix:
        print("error: --yes requires --fix", file=sys.stderr)
        sys.exit(2)
    if fix and not dry_run and not assume_yes and not sys.stdin.isatty():
        print(
            "error: --fix needs interactive confirmation or explicit --yes",
            file=sys.stderr,
        )
        sys.exit(2)

    def collect_report(*, force_deep: bool = False) -> Report:
        use_deep = deep or force_deep

        def operation() -> Report:
            return (
                run_all(only=only, skip=skip, deep=True)
                if use_deep
                else run_all(only=only, skip=skip)
            )

        if json_output:
            try:
                return _collect_report_isolated(
                    only=only,
                    skip=skip,
                    deep=use_deep,
                    timeout_s=35.0 if use_deep else 8.0,
                )
            except Exception as exc:  # noqa: BLE001 - preserve JSON contract
                return _collection_failure_report(exc)
        return operation()

    def execute() -> Report:
        nonlocal assume_yes
        report = collect_report()
        if fix:
            report.schema_version = 2
            from .repairs import apply_repairs, plan_repairs, service_is_live

            actions = plan_repairs(report)
            if actions and not dry_run and not assume_yes:
                print("Doctor proposes:", file=sys.stderr)
                for action in actions:
                    print(f"  - {_redact(action.summary)}", file=sys.stderr)
                    command = " ".join(_redact(argument) for argument in action.command)
                    print(f"    {command}", file=sys.stderr)
                print(
                    "Apply these repairs? [y/N] ", end="", file=sys.stderr, flush=True
                )
                assume_yes = sys.stdin.readline().strip().lower() in {"y", "yes"}
                if not assume_yes:
                    report.repairs = [
                        {
                            "id": action.id,
                            "status": "not_applied",
                            "summary": action.summary,
                            "command": list(action.command),
                            "detail": "operator declined the proposed repair",
                        }
                        for action in actions
                    ]
            if actions and (dry_run or assume_yes):
                results = apply_repairs(
                    actions,
                    dry_run=dry_run,
                    verify_service=service_is_live,
                )
                if any(result.status == "verified" for result in results):
                    report = collect_report(force_deep=True)
                    report.schema_version = 2
                report.repairs = [result.to_dict() for result in results]
        return report

    report = execute()
    if json_output:
        render_json(report)
    elif summary_only:
        render_summary(report)
    else:
        render(report, verbose=verbose)
    sys.exit(report.exit_code)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(report: Report, *, verbose: bool = False, stream=None) -> None:
    """Write the report to ``stream`` (defaults to stdout)."""
    stream = stream or sys.stdout
    write = stream.write

    write("\n")
    write("┌" + "─" * 57 + "┐\n")
    write("│" + "\U0001fa7a Rapid-MLX Doctor".center(57) + "│\n")
    write("└" + "─" * 57 + "┘\n")
    write("\n")

    for section in report.sections:
        _render_section(section, write=write, verbose=verbose)
        write("\n")

    _render_summary(report, write=write, verbose=verbose)
    if report.repairs:
        write("\nRepairs:\n")
        for repair in report.repairs:
            status = _redact(str(repair["status"]))
            summary = _redact(str(repair["summary"]))
            command = " ".join(_redact(str(item)) for item in repair["command"])
            write(f"  {status}: {summary} ({command})\n")
            if verbose and repair.get("detail"):
                write(f"      ↳ {_redact(str(repair['detail']))}\n")


def _redact(value: str) -> str:
    """Remove common local identifiers and inline credentials from JSON."""
    home = os.path.expanduser("~")
    if home and home != "/":
        value = re.sub(
            rf"{re.escape(home)}(?![\w.-])",
            "~",
            value,
        )
    value = _URL_USERINFO_RE.sub(r"\1[REDACTED]@", value)

    redacted_lines: list[str] = []
    continuation_quote: str | None = None
    for line in value.splitlines(keepends=True):
        if continuation_quote is not None:
            if _contains_unescaped_quote(line, continuation_quote):
                continuation_quote = None
            redacted_lines.append(_line_ending(line))
            continue
        sensitive_assignment = None
        for match in _ASSIGNMENT_RE.finditer(line):
            if _is_secret_key(match.group(1)):
                sensitive_assignment = match
                break
        spaced_secret = None
        for match in _SPACE_KEY_VALUE_RE.finditer(line):
            if not _is_secret_key(match.group("key")):
                continue
            spaced_secret = match
            break
        authorization = _AUTHORIZATION_RE.search(line)
        auth_scheme = _AUTH_SCHEME_RE.search(line)
        candidates = [
            (sensitive_assignment.start(), sensitive_assignment.group(0))
            if sensitive_assignment
            else None,
            (
                spaced_secret.start("key"),
                line[spaced_secret.start("key") : spaced_secret.start("value")],
            )
            if spaced_secret
            else None,
            (authorization.start(), authorization.group("authorization"))
            if authorization
            else None,
            (auth_scheme.start(), auth_scheme.group("authorization"))
            if auth_scheme
            else None,
        ]
        matches = [candidate for candidate in candidates if candidate is not None]
        if matches:
            start, prefix = min(matches, key=lambda candidate: candidate[0])
            tail = line[start + len(prefix) :]
            stripped_tail = tail.lstrip()
            if stripped_tail[:1] in {'"', "'"}:
                quote = stripped_tail[0]
                if not _contains_unescaped_quote(stripped_tail[1:], quote):
                    continuation_quote = quote
            # An unquoted value may contain spaces or delimiters, so redact
            # through this line. A subsequent line is independent unless a
            # quoted value was genuinely left open above.
            redacted_lines.append(
                f"{line[:start]}{prefix}[REDACTED]{_line_ending(line)}"
            )
            continue
        redacted_lines.append(line)
    return "".join(redacted_lines)


def _line_ending(line: str) -> str:
    return line[len(line.rstrip("\r\n")) :]


def _contains_unescaped_quote(value: str, quote: str) -> bool:
    escaped = False
    for character in value:
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            return True
    return False


def report_document(report: Report) -> dict[str, Any]:
    """Return the versioned, redacted machine-readable Doctor contract."""
    schema_version = 2 if report.repairs else report.schema_version
    document = {
        "schemaVersion": schema_version,
        "rapidMlxVersion": __version__,
        "status": report.overall_status,
        "exitCode": report.exit_code,
        "durationMs": report.duration_ms,
        "summary": {
            "ok": report.n_ok,
            "warnings": report.n_warn,
            "failures": report.n_fail,
            "skipped": report.n_skipped,
        },
        "sections": [
            {
                "id": section.id,
                "title": section.title,
                "durationMs": section.duration_ms,
                "checks": [
                    {
                        "id": check.id,
                        "status": check.status.value,
                        "summary": _redact(check.label),
                        "detail": _redact(check.detail),
                    }
                    for check in section.checks
                ],
            }
            for section in report.sections
        ],
    }
    if schema_version >= 2:
        document["repairs"] = [
            {
                key: (
                    _redact(value)
                    if isinstance(value, str)
                    else [
                        _redact(item) if isinstance(item, str) else item
                        for item in value
                    ]
                    if isinstance(value, list)
                    else value
                )
                for key, value in repair.items()
            }
            for repair in report.repairs
        ]
    return document


def render_json(report: Report, *, stream=None) -> None:
    """Write only JSON to stdout so callers can parse it safely."""
    stream = stream or sys.stdout
    json.dump(report_document(report), stream, indent=2, sort_keys=True)
    stream.write("\n")


def render_summary(report: Report, *, stream=None) -> None:
    """Write a single-line status suitable for logs and shell scripts."""
    stream = stream or sys.stdout
    stream.write(
        f"Rapid-MLX Doctor: {report.overall_status} — "
        f"{report.n_ok} ok, {report.n_warn} warnings, "
        f"{report.n_fail} issues, {report.n_skipped} skipped "
        f"({report.duration_ms} ms)\n"
    )


def _render_section(section: Section, *, write, verbose: bool) -> None:
    write(f"◆ {section.title}\n")
    for check in section.checks:
        glyph = _GLYPHS[check.status]
        write(f"  {glyph} {check.label}\n")
        if verbose and check.detail:
            write(f"      ↳ {check.detail}\n")


def _render_summary(report: Report, *, write, verbose: bool) -> None:
    write("─" * 40 + "\n")
    write(
        f"Summary: {report.n_ok} ok, "
        f"{report.n_warn} warnings, "
        f"{report.n_fail} issue"
        f"{'s' if report.n_fail != 1 else ''}"
    )
    if report.n_skipped:
        write(f", {report.n_skipped} skipped")
    write("\n")
    if not verbose and (report.n_warn or report.n_fail):
        write("Run with `--verbose` for details on each check.\n")
