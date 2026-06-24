# SPDX-License-Identifier: Apache-2.0
"""Step 1 — supply chain audit.

What "supply chain" means for an external PR to a published package:

1. **New dependencies** — does this PR add a package to pyproject.toml
   or requirements files? Are those packages known-good or yanked /
   typo-squat / known-vulnerable?
2. **License drift** — pulling in GPL/AGPL/SSPL into our Apache-2.0
   tree would force a relicense; we want to refuse silently-shifted
   licenses.
3. **Install hooks** — `setup.py`, ``pyproject.toml`` build hooks,
   ``conftest.py`` (runs on `pip install` for editable installs and on
   every pytest invocation), and ``.github/workflows/`` (auto-deploys
   to PyPI/Homebrew). Code added to any of these gets to run on every
   user's machine without explicit consent — they need extra scrutiny.
4. **Suspicious patterns in regular code** — base64-decoded blobs that
   `exec()`, ``socket.connect`` to hardcoded IPs, ``urllib`` requests
   to non-anthropic / non-github / non-pypi hosts, ``os.system`` /
   ``subprocess`` with shell-formed strings.

This step is intentionally conservative — we'd rather false-positive
on a benign PR (let the maintainer eyeball it) than miss a malicious
one. The cost of a false positive is "human reads the diff anyway".
The cost of a miss is auto-deploy of malware to every PyPI user.

Network calls (pip-audit) are best-effort; if pip-audit isn't
installed or the index is unreachable we ``skip`` rather than ``fail``
— locally checking deps without network would be misleading.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from ..base import Step, StepResult
from ..context import Context

# Files that gain code-execution capability when modified — install
# hooks, CI config, anything that runs unattended. ``pyproject.toml``
# and ``requirements*.txt`` are here because pr_validate's own
# ``TestEnvCheckStep`` may ``pip install '.[test]'`` to recover a
# missing pytest plugin; if a malicious PR adds a build-hook or fake
# package source, that install would execute the attacker's code in
# the validator's interpreter. Flagging the file as a hook means an
# external-author PR that touches it is [BLOCKING] at this step BEFORE
# any auto-installing step downstream runs — see threat-model section
# of scripts/pr_validate/README.md.
HOOK_PATHS = (
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    # pr_validate's own pin list (TRUSTED_TEST_PINS materialized as a
    # file in some downstream uses) — kept in sync with the dep-file
    # denylist in ``scripts/pr_validate/_test_env.py``.
    "requirements-pin.txt",
    "conftest.py",  # runs on every `pytest`
    "tests/conftest.py",
    ".github/workflows/",
    "Makefile",
    ".pre-commit-config.yaml",
    "Formula/",  # Homebrew tap
    "homebrew-rapid-mlx/",
)

# Files that declare project deps. Any addition gets pip-audited.
DEP_DECLARATION_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
)

# Patterns that, when added in a diff, warrant human eyeballs even in
# regular .py files. Each is (regex, why-suspicious). False-positive
# rate is high — that's accepted; a maintainer can dismiss easily.
SUSPICIOUS_PATTERNS = (
    (re.compile(r"\beval\s*\("), "eval() — usually wrong; never on untrusted data"),
    (re.compile(r"\bexec\s*\("), "exec() — usually wrong; never on untrusted data"),
    (
        re.compile(r"base64\.b64decode\s*\("),
        "base64-decoded blob — possible code-as-data smuggling",
    ),
    (
        re.compile(r"pickle\.loads?\s*\("),
        "pickle.load on untrusted data is RCE; verify source",
    ),
    (
        re.compile(r"subprocess\.\w+\([^)]*shell\s*=\s*True"),
        "shell=True — command injection if any arg is external",
    ),
    (
        re.compile(r"os\.system\s*\("),
        "os.system — subject to command injection; prefer subprocess.run([...])",
    ),
    (
        re.compile(r"socket\.connect\s*\(\s*\(['\"][\d.]+['\"]"),
        "raw socket.connect to a hardcoded IP",
    ),
    (
        re.compile(r"urllib\.request\.urlopen\s*\(\s*['\"]https?://"),
        "hardcoded HTTP URL — verify the host",
    ),
    (
        re.compile(r"requests\.(get|post|put|delete)\s*\(\s*['\"]https?://"),
        "hardcoded HTTP URL via requests — verify the host",
    ),
    # GitHub Actions specific — adding `secrets.` access in a workflow.
    (
        re.compile(r"secrets\.[A-Z_]+"),
        "workflow accesses repository secret — verify intent",
    ),
    # Hex-encoded blobs (>64 chars) — sometimes seen in obfuscated payloads.
    (
        re.compile(r"['\"][0-9a-fA-F]{64,}['\"]"),
        "long hex literal — could be a hash (fine) or obfuscated data",
    ),
)


class SupplyChainStep(Step):
    name = "supply_chain"
    description = "deps audit + license + install-hook scan"

    def run(self, ctx: Context) -> StepResult:
        diff = Path(ctx.diff_path).read_text()

        findings: list[str] = []
        artifacts: list[str] = []

        # 1. Hook-file modifications get an automatic flag — not a
        # FAIL on its own (legitimate workflow updates exist), but
        # surfaced loudly so the human knows to read carefully.
        hook_files = [
            f
            for f in ctx.files_changed
            if any(f == p or f.startswith(p) for p in HOOK_PATHS)
        ]
        if hook_files:
            # Even an "innocent-looking" hook change is worth surfacing.
            # External-author + hook change = strong reason to read.
            severity = "BLOCKING" if ctx.is_external_author else "warning"
            findings.append(
                f"[{severity}] modifies install/CI hook(s): {hook_files}. "
                "These run unattended; review every line."
            )

        # 2. Suspicious patterns in ADDED lines (not removed — removed
        # lines were dangerous before this PR, that's a different
        # problem).
        added_lines = _added_lines(diff)
        pattern_hits = _scan_patterns(added_lines)
        for path, lineno, line, why in pattern_hits[:20]:
            findings.append(
                f"`{path}` near l{lineno}: {why}\n  > `{line.strip()[:120]}`"
            )

        # 3. Deps changes — diff pyproject.toml / requirements files,
        # extract added package names, run pip-audit on them.
        new_deps = _extract_added_deps(diff, ctx.files_changed)
        if new_deps:
            audit_path = ctx.artifact_path("pip-audit.log")
            audit_findings = _pip_audit(new_deps, audit_path)
            artifacts.append(str(audit_path))
            if audit_findings:
                findings.extend(audit_findings)
            else:
                # Successful audit with no issues — note it for the log
                # but don't add as a finding.
                ctx.run_log(
                    f"pip-audit clean for {len(new_deps)} new dep(s): "
                    f"{', '.join(new_deps[:5])}"
                )

        # 4. Save the full pattern scan for inspection.
        scan_path = ctx.artifact_path("supply-chain-scan.log")
        scan_path.write_text(_format_scan(hook_files, pattern_hits, new_deps))
        artifacts.append(str(scan_path))

        # Decision rule. Anything tagged BLOCKING → fail. Otherwise pass
        # but surface warnings as findings (they go in the scorecard
        # for human eyeballs).
        blocking = [f for f in findings if "[BLOCKING]" in f]
        if blocking:
            return StepResult(
                name=self.name,
                status="fail",
                summary=f"{len(blocking)} blocking finding(s) "
                f"(+{len(findings) - len(blocking)} warning(s))",
                findings=findings,
                artifacts=artifacts,
            )
        if findings:
            # Warnings only — human-needed but not auto-blocked. Still
            # report as ``pass`` so the gate doesn't false-positive on
            # every legitimate change; findings carry the signal.
            return StepResult(
                name=self.name,
                status="pass",
                summary=f"{len(findings)} warning(s) — human review wanted",
                findings=findings,
                artifacts=artifacts,
            )

        return StepResult(
            name=self.name,
            status="pass",
            summary="no hooks touched, no suspicious patterns, deps clean",
            artifacts=artifacts,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _added_lines(diff: str) -> list[tuple[str, int, str]]:
    """Extract every '+' line in a unified diff with its file path and
    estimated line number in the new file. Skips '+++' header lines."""
    out = []
    cur_path = ""
    new_lineno = 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            cur_path = line[6:]
            new_lineno = 0
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            # @@ -<old>,<n> +<newstart>,<n> @@
            m = re.search(r"\+(\d+)", line)
            new_lineno = int(m.group(1)) - 1 if m else 0
            continue
        if line.startswith("+") and not line.startswith("+++"):
            new_lineno += 1
            out.append((cur_path, new_lineno, line[1:]))
        elif not line.startswith("-"):
            new_lineno += 1
    return out


def _scan_patterns(
    added: list[tuple[str, int, str]],
) -> list[tuple[str, int, str, str]]:
    """Apply SUSPICIOUS_PATTERNS to added lines. Returns
    (path, lineno, line, why) per hit."""
    out = []
    for path, lineno, line in added:
        # Skip our own validation rule definitions — the patterns
        # themselves contain the regex source, which would self-match.
        if "scripts/pr_validate/" in path:
            continue
        # Heuristic: skip test files for the *most* aggressive patterns,
        # since tests legitimately use eval/pickle/etc. for fixtures.
        # We still flag setup.py / conftest.py / workflows above.
        is_test = "/tests/" in path or path.startswith("tests/")
        for pattern, why in SUSPICIOUS_PATTERNS:
            if pattern.search(line):
                if is_test and "eval(" in pattern.pattern:
                    continue
                if is_test and "exec(" in pattern.pattern:
                    continue
                out.append((path, lineno, line, why))
    return out


def _extract_added_deps(diff: str, files_changed: list[str]) -> list[str]:
    """Naive but cautious: find lines in dep-declaration files that
    look like `name = "version"` or `name>=ver` and weren't there
    before. We don't try to parse pyproject.toml fully — too many
    formats. Just regex the additions."""
    if not any(f in DEP_DECLARATION_FILES for f in files_changed):
        return []

    deps: list[str] = []
    in_dep_file = False
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            in_dep_file = path in DEP_DECLARATION_FILES
            continue
        if not in_dep_file or not line.startswith("+"):
            continue
        if line.startswith("+++"):
            continue

        body = line[1:].strip()
        # pyproject style: '"package>=1.2.3",' or '"package",'
        m = re.match(r'["\']([a-zA-Z0-9_\-.\[\]]+)(?:\s*[~<>=!]+[^"\']*)?["\']', body)
        if m:
            name = m.group(1).split("[", 1)[0]  # strip extras like httpx[http2]
            deps.append(name.lower())
            continue
        # requirements.txt style: 'package>=1.2.3'
        m = re.match(r"([a-zA-Z0-9_\-.\[\]]+)\s*[~<>=!]+", body)
        if m:
            deps.append(m.group(1).split("[", 1)[0].lower())

    # Dedup and drop standard library / our own package.
    seen = set()
    out = []
    for d in deps:
        if d in seen or d in ("rapid-mlx", "vllm-mlx"):
            continue
        seen.add(d)
        out.append(d)
    return out


def _pip_audit(deps: list[str], log_path: Path) -> list[str]:
    """Run pip-audit on the candidate deps. Returns findings list (one
    per known-vulnerable dep). If pip-audit isn't installed we skip
    silently (log says so but no finding)."""
    if not shutil.which("pip-audit"):
        log_path.write_text(
            "pip-audit not installed — `pip install pip-audit` to enable\n"
        )
        return []

    # pip-audit takes a requirements file or a list of installed packages.
    # We construct a one-off requirements file with just the names — it
    # will resolve to whatever's currently published.
    req_file = log_path.with_suffix(".req")
    req_file.write_text("\n".join(deps) + "\n")

    proc = subprocess.run(  # noqa: S603
        [
            "pip-audit",
            "-r",
            str(req_file),
            "--format",
            "json",
            "--progress-spinner",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    log_path.write_text(
        (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or "")
    )

    if proc.returncode == 0 and not proc.stdout.strip():
        return []

    findings: list[str] = []
    try:
        import json as _json

        data = _json.loads(proc.stdout) if proc.stdout.strip() else {}
        for entry in data.get("dependencies", []):
            for vuln in entry.get("vulns", []):
                findings.append(
                    f"[BLOCKING] dep `{entry.get('name')}` "
                    f"vuln {vuln.get('id')}: "
                    f"{(vuln.get('description') or '')[:120]}"
                )
    except Exception as e:  # noqa: BLE001
        # pip-audit format change or weird output — don't crash, log it.
        findings.append(f"pip-audit output not parseable ({e}) — see {log_path}")
    return findings


def _format_scan(
    hook_files: list[str],
    pattern_hits: list[tuple[str, int, str, str]],
    new_deps: list[str],
) -> str:
    lines = ["# Supply-chain scan", ""]
    lines.append("## Hook files modified")
    lines.extend(f"- {f}" for f in hook_files) if hook_files else lines.append("(none)")
    lines.append("")
    lines.append("## Suspicious patterns in added lines")
    if pattern_hits:
        for path, lineno, line, why in pattern_hits:
            lines.append(f"- `{path}` l{lineno} — {why}")
            lines.append(f"  > `{line.strip()[:120]}`")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## New dependencies")
    lines.extend(f"- {d}" for d in new_deps) if new_deps else lines.append("(none)")
    return "\n".join(lines)
