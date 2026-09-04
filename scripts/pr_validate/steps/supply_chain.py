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

from .._test_env import is_dep_declaration_file
from ..base import Step, StepResult
from ..context import Context

# Files that gain code-execution capability when modified — install
# hooks, CI config, anything that runs unattended. ``pyproject.toml``
# and the ``requirements*.txt`` family are NOT listed here directly;
# they're matched via ``_test_env.is_dep_declaration_file`` (shared
# truth source — codex r2 BLOCKING was that diverging lists let
# ``requirements-test.txt`` through this step but not the
# ``test_env_check`` gate). The combined ``_is_hook_file`` matcher
# below is what the SupplyChainStep uses. External-author PRs
# touching any of these get [BLOCKING] BEFORE any downstream
# auto-installing step runs — see threat-model section of
# ``scripts/pr_validate/README.md``.
HOOK_PATHS = (
    "conftest.py",  # runs on every `pytest`
    "tests/conftest.py",
    ".github/workflows/",
    "Makefile",
    ".pre-commit-config.yaml",
    "Formula/",  # Homebrew tap
    "homebrew-rapid-mlx/",
)


def _is_hook_file(path: str) -> bool:
    """Combined matcher: dep-declaration files (shared with
    ``_test_env.is_dep_declaration_file``) PLUS the explicit
    ``HOOK_PATHS`` set. Centralized so we never again have two lists
    that drift apart."""
    if is_dep_declaration_file(path):
        return True
    return any(path == p or path.startswith(p) for p in HOOK_PATHS)


# ---------------------------------------------------------------------------
# Issue #2522 — the "roster-only workflow edit" exception.
#
# The explicit CI test roster in ``.github/workflows/ci.yml`` is a long
# ``tests/test_*.py \`` (shell line-continuation) list. Enrolling a NEW test
# in that list — ``+            tests/test_foo.py \`` and nothing else in any
# workflow file — is the expected shape of "I added a test". Blocking it for
# an external contributor is self-defeating: the gate blocks the exact
# contribution it asks for.
#
# We therefore detect "roster-only" workflow edits: a workflow file counts
# as roster-only when EVERY change in its diff is a pure ADDITION of a
# ``tests/<name>.py`` roster entry (no removed lines, no other added or
# modified lines). The overall hook finding is downgraded from [BLOCKING] to
# [warning] ONLY when EVERY workflow file the PR touches is roster-only and
# there are no other ``.github/workflows/`` edits of any other kind. Any
# other workflow edit — a ``runs-on`` / ``step.run:`` change, a removed
# line, a structural edit — or any non-workflow hook file (conftest.py,
# Makefile, a dep file) keeps [BLOCKING] for external authors.
#
# The path grammar is deliberately narrower than a shell token: allowing
# ``$()``, quotes, ``;``, ``..`` etc. would let the exception itself become
# a workflow-code injection bypass. Only a plain ``tests/<alnum>_./-`` token
# with an optional trailing ``\`` continuation matches. A no-continuation
# entry is accepted only when it is proven to be the final command argument at
# the end of its YAML literal block; see ``_pytest_roster_lines``.
# ---------------------------------------------------------------------------

_WORKFLOW_PREFIX = ".github/workflows/"
_ROSTER_WORKFLOWS = frozenset({".github/workflows/ci.yml"})

# A single roster-enrollment content line (leading ``+`` already stripped).
_ROSTER_ENTRY_RE = re.compile(
    r"^\s*(?P<path>tests/[A-Za-z0-9_./-]+\.py)(?P<cont>\s*\\)?\s*$"
)


def _roster_addition_path(content: str) -> str | None:
    """If *content* (a ``+`` line with the marker stripped) is a roster
    enrollment of the form ``tests/<name>.py \\``, return the test path,
    else ``None``. Rejects anything with traversal (``..``) — that would
    be an injection, not an enrollment."""
    m = _ROSTER_ENTRY_RE.match(content)
    if m is None:
        return None
    path = m.group("path")
    if ".." in Path(path).parts:
        return None
    return path


# An anchor roster line must CONTINUE the shell list — i.e. it ends with a
# backslash continuation, exactly like every entry in the real roster. A
# terminal ``tests/foo.py`` (no backslash) runs as its own command, so a
# ``tests/new.py \`` following it would be a SECOND command, not another
# pytest argument — that is not an enrollment (codex r1 round-2).
_ROSTER_CONTINUE_RE = re.compile(r"^\s*tests/[A-Za-z0-9_./-]+\.py\s*\\\s*$")

# The ``run: |``-block command that OPENS a pytest test roster, e.g.
# ``          pytest \``. It must end with a continuation backslash: a
# complete, non-continuing command like ``pytest -q`` runs as its own shell
# invocation and does NOT open the multi-line ``tests/x.py \`` list (codex
# r1 #2). An enrollment is only trusted when it is anchored to that list —
# the line directly above it is another (continuing) roster entry or this
# opening command.
_PYTEST_ROSTER_CMD = re.compile(r"^\s*pytest\b.*\\\s*$")
_PYTEST_ROSTER_STEP = "- name: Run MLX-dependent tests"


def _pytest_roster_lines(content: str) -> set[int]:
    """Return the (1-based) line numbers in *content* that are roster entries
    of the explicit pytest test list. This is the authoritative, ground-truth
    denominator: the contiguous ``tests/*.py \\`` continuation lines that
    follow a ``pytest \\`` command, plus a no-backslash final test only when
    the next nonblank source line has left that YAML literal block (or EOF was
    reached). Looking merely for another test is unsafe: following pytest
    options would become separate shell commands. Codex r1 round-3 — do not
    trust hunk context (a long non-pytest file list would hide its opener);
    verify each added line against the ACTUAL roster location in the file."""
    roster: set[int] = set()
    lines = content.splitlines()
    # Bind the exception to the one reviewed job step, not merely to any
    # command spelling ``pytest`` somewhere in ci.yml.
    step_indexes = [
        idx for idx, line in enumerate(lines) if line.strip() == _PYTEST_ROSTER_STEP
    ]
    if len(step_indexes) != 1:
        return roster
    step_start = step_indexes[0]
    step_indent = len(lines[step_start]) - len(lines[step_start].lstrip())
    step_end = len(lines)
    for idx in range(step_start + 1, len(lines)):
        stripped = lines[idx].lstrip()
        indent = len(lines[idx]) - len(stripped)
        if stripped.startswith("- ") and indent <= step_indent:
            step_end = idx
            break

    i = step_start + 1
    n = step_end
    while i < n:
        if _PYTEST_ROSTER_CMD.match(lines[i]):
            command_indent = len(lines[i]) - len(lines[i].lstrip())
            j = i + 1
            # A run of continuing roster entries...
            while j < n and _ROSTER_CONTINUE_RE.match(lines[j]):
                roster.add(j + 1)  # 1-based
                j += 1
            # A final test may omit ``\`` only when it genuinely ends the
            # literal shell block. Any later nonblank line at the command's
            # indentation or deeper is another command/argument in that block.
            if j < n and _ROSTER_ENTRY_RE.match(lines[j]):
                next_nonblank = j + 1
                while next_nonblank < n and not lines[next_nonblank].strip():
                    next_nonblank += 1
                left_literal_block = next_nonblank >= n or (
                    len(lines[next_nonblank]) - len(lines[next_nonblank].lstrip())
                    < command_indent
                )
                if left_literal_block:
                    roster.add(j + 1)
                j += 1
            i = j
        else:
            i += 1
    return roster


def _roster_only_workflows(
    diff: str, files_changed: set[str], head_content: dict[str, str]
) -> tuple[set[str], dict[str, list[str]]]:
    """Classify every workflow file touched by the PR.

    ``head_content`` must map each modified workflow path to its HEAD file
    text (the full file the diff will produce once merged) — used to verify
    that each added roster line lands inside the ACTUAL pytest roster, rather
    than trusting diff hunk context.

    Returns ``(roster_only, roster_additions)``:

    * ``roster_only`` — the set of workflow files whose ENTIRE diff is a
      pure roster enrollment (only ``tests/<name>.py \\`` lines added to the
      explicit pytest test list, no removed lines, no other change).
    * ``roster_additions`` — ``{workflow_file: [test paths enrolled]}`` for
      those roster-only files, so the human gets the exact added lines.

    A workflow file that fails any check (removed lines, metadata, a roster
    token added outside the real roster, a file we cannot verify) is absent
    from both — it stays [BLOCKING]."""

    # This is not a general exemption for pytest commands in workflow files.
    # It applies only to the reviewed explicit Apple/MLX roster in ci.yml.
    # Any other workflow remains an executable hook and therefore blocking for
    # an external author.
    workflow_files = {
        f
        for f in files_changed
        if f.startswith(_WORKFLOW_PREFIX) and f in _ROSTER_WORKFLOWS
    }
    if not workflow_files:
        return set(), {}

    # A roster enrollment is only meaningful for a test this PR NEWLY adds
    # (issue #2522: "I added a test" → enroll it). Compute the newly-created
    # files from the diff's ``new file mode`` markers so a PR that merely
    # edits an existing test cannot get the downgrade.
    new_files = _new_files(diff)
    files_changed_set = set(files_changed)
    # Authoritative roster line numbers, from the actual HEAD file.
    roster_lines = {
        path: _pytest_roster_lines(head_content[path])
        for path in workflow_files
        if path in head_content
    }

    # Collect every kind of line (context / added / removed) per workflow file,
    # tracking each added line's absolute NEW-file line number (from the @@
    # hunk headers) so we can match it against the real roster location.
    per_file: dict[str, dict[str, list]] = {
        f: {"added": [], "removed": []} for f in workflow_files
    }
    # Workflow diffs carrying extended metadata (rename/copy/mode change) are
    # structural, not roster enrollments — a roster-only ``+tests/x.py \``
    # cannot accompany a rename or a mode change of the workflow file itself.
    metadata_files: set[str] = set()
    cur_path = ""
    in_hunk = False
    new_lineno = 0
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            cur_path = line.split(" b/", 1)[-1] if " b/" in line else ""
            in_hunk = False
            continue
        if cur_path not in per_file:
            continue
        if line.startswith("@@"):
            # @@ -a,b +c,d @@ → the first new-file line is c.
            in_hunk = True
            m = re.search(r"\+(\d+)", line)
            new_lineno = int(m.group(1)) if m else 0
            continue
        # Extended diff metadata lines (no + / - / space marker): a rename,
        # copy, or mode change of the workflow file itself is disqualifying.
        if line[:1] not in (" ", "+", "-"):
            if line.startswith(
                (
                    "rename from ",
                    "rename to ",
                    "copy from ",
                    "copy to ",
                    "old mode ",
                    "new mode ",
                )
            ):
                metadata_files.add(cur_path)
            continue
        # ``--- /path`` and ``+++ /path`` are file headers ONLY before the
        # first hunk. Inside a hunk a content line may itself begin with
        # ``-- `` / ``++ `` (e.g. a removed ``-- flag`` line) and must be
        # treated as a real change, not skipped as a header.
        if not in_hunk and line.startswith(("--- ", "+++ ")):
            continue
        marker = line[:1]
        if marker == "+":
            per_file[cur_path]["added"].append((new_lineno, line[1:]))
            new_lineno += 1  # an added line occupies a new-file position
        elif marker == "-":
            per_file[cur_path]["removed"].append(line[1:])
        elif marker == " ":
            new_lineno += 1  # a context line also occupies a new-file position

    roster_only: set[str] = set()
    roster_additions: dict[str, list[str]] = {}
    for path, block in per_file.items():
        # Anything deleted from (or structurally altered with) a workflow file
        # is NOT roster-only (#2522).
        if block["removed"] or path in metadata_files:
            continue
        if not block["added"]:
            continue  # no tracked change; stay conservative
        # We must be able to verify the file against its real roster; without
        # head content we cannot prove the additions are roster-only.
        if path not in roster_lines:
            continue
        enrolled: list[str] = []
        ok = True
        for lineno, content in block["added"]:
            p = _roster_addition_path(content)
            # A real roster token for a test this PR NEWLY adds...
            if p is None or p not in new_files or p not in files_changed_set:
                ok = False
                break
            # ...placed at a line that is ACTUALLY a pytest-roster entry in the
            # merged file (ground truth, not hunk context).
            if lineno not in roster_lines[path]:
                ok = False
                break
            enrolled.append(p)
        if ok:
            roster_only.add(path)
            roster_additions[path] = enrolled
    return roster_only, roster_additions


def _new_files(diff: str) -> set[str]:
    """Return the repo-relative paths of files newly created in *diff*
    (``new file mode`` markers). Used to require that a roster enrollment
    point at a test this PR actually adds, not an existing one it edits."""
    new_files: set[str] = set()
    cur_path = ""
    is_new = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if is_new and cur_path:
                new_files.add(cur_path)
            cur_path = line.split(" b/", 1)[-1] if " b/" in line else ""
            is_new = False
            continue
        if line.startswith("new file mode"):
            is_new = True
            continue
    if is_new and cur_path:
        new_files.add(cur_path)
    return new_files


# Backwards-compatibility alias — the dep-declaration matcher now
# lives in ``_test_env.is_dep_declaration_file`` (shared with
# ``test_env_check``); kept as a constant here so existing call
# sites keep working until they're migrated to the matcher. New
# code should call ``is_dep_declaration_file(path)`` directly so
# the ``requirements*.txt`` prefix coverage is picked up
# automatically.
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
        # ``_is_hook_file`` unifies HOOK_PATHS with the dep-
        # declaration matcher from ``_test_env`` so the two lists
        # can't drift (codex r2 BLOCKING).
        hook_files = [f for f in ctx.files_changed if _is_hook_file(f)]
        if hook_files:
            # Even an "innocent-looking" hook change is worth surfacing.
            # External-author + hook change = strong reason to read.
            #
            # Issue #2522 exception: an external PR that ONLY enrolls a new
            # test into the explicit CI roster (pure ``tests/<name>.py \``
            # additions, no removed lines, no other workflow edit) is the
            # expected shape of a "I added a test" contribution. Downgrade
            # that to a WARNING (still surfaced, with the added lines) so it
            # doesn't BLOCK the behavior the gate is meant to encourage. Any
            # other workflow edit, or any non-workflow hook file modified
            # alongside, keeps [BLOCKING] for external authors.
            # Build the HEAD content of each modified workflow file (the PR
            # head is checked out in the working tree) so roster-only can be
            # verified against the file's ACTUAL pytest roster, not diff hunk
            # context (codex r1 round-3). Only external authors need this
            # (internal writers already get [warning]); a file that cannot be
            # read is treated as NOT roster-only → BLOCKING, never a crash.
            head_content: dict[str, str] = {}
            if ctx.is_external_author:
                for wf in ctx.files_changed:
                    if not wf.startswith(_WORKFLOW_PREFIX):
                        continue
                    try:
                        p = ctx.repo_root / wf
                        if p.is_file():
                            head_content[wf] = p.read_text()
                    except (OSError, UnicodeDecodeError):
                        # Unreadable workflow → cannot prove roster-only.
                        continue
            roster_only, roster_additions = _roster_only_workflows(
                diff, set(ctx.files_changed), head_content
            )
            roster_downgrade = bool(
                ctx.is_external_author
                and hook_files
                and all(f.startswith(_WORKFLOW_PREFIX) for f in hook_files)
                and all(f in roster_only for f in hook_files)
            )
            severity = (
                "warning"
                if roster_downgrade
                else ("BLOCKING" if ctx.is_external_author else "warning")
            )
            detail = ""
            if roster_downgrade:
                # Include the exact enrolled test paths (diff hunk) so the
                # human still inspects what was added (issue #2522).
                added = sorted(
                    {  # flatten, dedupe, sort for a stable message
                        p for paths in roster_additions.values() for p in paths
                    }
                )
                detail = " Roster-only test enrollment: " + ", ".join(
                    f"`{p}`" for p in added
                )
            findings.append(
                f"[{severity}] modifies install/CI hook(s): {hook_files}. "
                "These run unattended; review every line." + detail
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
    formats. Just regex the additions.

    Uses ``is_dep_declaration_file`` so every ``requirements*.txt``
    variant gets pip-audited, not just the three legacy names —
    codex r2 BLOCKING was that ``requirements-test.txt`` slipped
    through the previous exact-match list."""
    if not any(is_dep_declaration_file(f) for f in files_changed):
        return []

    deps: list[str] = []
    in_dep_file = False
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            in_dep_file = is_dep_declaration_file(path)
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
