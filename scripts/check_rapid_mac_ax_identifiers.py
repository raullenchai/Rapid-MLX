# SPDX-License-Identifier: Apache-2.0
"""Gate: a PR may not ADD an interactive rapid-mac control without an
``.accessibilityIdentifier(...)``.

Why (grounded, not hypothetical). The desktop app's only real GUI test harness —
``apps/rapid-mac/scripts/gui-golden-flows.sh`` driving ``rapid-ax.swift`` — finds
and presses controls by ``AXIdentifier`` and nothing else (see
``apps/rapid-mac/docs/gui-golden-flows.md``). Its ceiling is therefore *exactly*
the set of controls that carry one. Every feature that ships an unlabelled
control silently lowers that ceiling, and nobody notices because the app still
works by hand. Settings -> Tools is the live proof: three tool toggles, the
web-search backend radio group, its key field + Save button and the browsing
toggle are all real, working ``AXCheckBox``/``AXRadioButton``/``AXTextField``
elements that no automated flow can reach. ``docs/userflows.md`` has also carried
"Approval dialogs lack identifiers" as an open item across several releases.

Design decisions, and why each one is the way it is:

* **Added lines only.** The existing backlog is real and is tracked separately.
  A gate that fails on it would be un-landable, get disabled inside a week, and
  leave us worse off than having no gate. So the unit of enforcement is "this
  diff introduced a new unlabelled control".

* **Carry-over suppression.** A violation whose control declaration is textually
  identical to one already present in the base version of the same file is not
  reported, so reformatting / moving / re-indenting known-bad code does not
  suddenly light up. Matching is a multiset over ``(kind, normalised head
  line)``: copy an unlabelled control a second time and the copy IS reported.

* **Precision over recall.** A gate that cries wolf gets ignored, and an ignored
  gate is worse than none because it manufactures confidence. Where the two
  trade off, this script chooses to miss. Concretely it walks the real postfix
  chain of each control (balanced delimiters over a comment/string-masked copy
  of the source), so an identifier attached five modifiers later still counts,
  and it deliberately skips ``Commands`` scenes (the AX driver never walks the
  menu bar). What it does not detect is listed under "Known blind spots" below.

* **An escape hatch that costs something.** SwiftUI ``confirmationDialog`` /
  ``alert`` buttons genuinely cannot be reached this way. Those get an explicit,
  greppable marker with a written reason on the same line::

      Button("Allow once") { … }  // ax-exempt: confirmationDialog buttons live
                                  // outside the app's AX tree

  ``rg ax-exempt apps/rapid-mac`` enumerates every one of them. A marker with no
  reason (or a one-word reason) is itself a failure, so the cost cannot be
  dodged by typing ``// ax-exempt:`` and moving on.

Known blind spots (deliberate — each would cost more false positives than it
buys):

* interactivity bolted onto a non-control view (``.onTapGesture``, ``.gesture``,
  ``.contextMenu``, ``.swipeActions``, ``.draggable``);
* bespoke ``View`` structs that are interactive without literally naming a
  SwiftUI control type at the point of use (the wrapper's own ``Button`` is
  still checked where the wrapper is *defined*);
* AppKit surfaces bridged through ``NSViewRepresentable`` (the chat composer's
  ``AutosizingTextView`` is the app's real example);
* ``Commands`` / menu-bar items, skipped on purpose;
* a control whose identifier is applied to an enclosing container rather than
  the control itself — reported, because ``AXPress`` needs the control.

Pure-logic core (``mask_source`` / ``find_violations``) is unit-tested in
tests/test_rapid_mac_ax_identifiers.py — no network, no Swift toolchain, no GPU.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Only the app's shipping UI. Tests, scripts and the crash handler are out of
# scope: nothing there renders a control a golden flow needs to drive.
SCOPE_PREFIX = "apps/rapid-mac/Sources/"

# SwiftUI control types a golden flow has to drive or assert on. Static text,
# spacers, shapes and decorative images are intentionally absent — labelling
# those would be noise, and the harness asserts on their *text*, not their id.
CONTROL_KINDS = (
    "Button",
    "Toggle",
    "Picker",
    "DatePicker",
    "ColorPicker",
    "TextField",
    "SecureField",
    "TextEditor",
    "Menu",
    "Stepper",
    "Slider",
    "NavigationLink",
    "Link",
)

# ``(?<![A-Za-z0-9_.])`` keeps ``SendButton``, ``pickerStyle`` and
# ``Foo.Button`` out; the trailing ``[({]`` keeps type references
# (``Button<Label>``, ``struct Button: View``) out. What is left is a control
# being *constructed*.
_CONTROL_RE = re.compile(r"(?<![A-Za-z0-9_.])(" + "|".join(CONTROL_KINDS) + r")\s*[({]")

# Scenes whose contents the AX driver never walks. ``rapid-ax.swift`` skips
# every non-window child of the application element precisely so the global menu
# bar stays out of the dumps, and ``gui-golden-flows.sh`` drives menus by title
# through ``peekaboo menu click``. An identifier on a command item would be
# decoration, so requiring one would be a pure false positive.
_COMMAND_SCOPE_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?:CommandGroup|CommandMenu)\s*\(|\.commands\s*\{"
)

# ``func makeBody(configuration:)`` is the ButtonStyle / ToggleStyle /
# LabelStyle protocol requirement. A control built in there is the *rendering*
# of whatever control the caller declared — ``TrailingSettingsToggleStyle`` is
# the app's live example. An identifier here would be stamped onto every toggle
# that adopts the style, which is worse than none, so the caller is the only
# correct place to put it and this scope is skipped.
_STYLE_BODY_RE = re.compile(
    r"(?<![A-Za-z0-9_.])func\s+makeBody\s*(\()\s*configuration\s*:"
)

_IDENTIFIER_RE = re.compile(r"\.accessibilityIdentifier\s*\(")

EXEMPT_MARKER = "ax-exempt:"
# No ``$`` anchor: ``.`` already stops at a newline, and a block comment's
# captured text can carry one, which an anchored pattern would refuse to match.
_EXEMPT_RE = re.compile(re.escape(EXEMPT_MARKER) + r"(.*)")

# A reason has to actually be a reason. Ten characters is roughly "two words",
# which is the floor at which the marker starts carrying information for the
# next reviewer instead of just silencing the gate.
MIN_EXEMPT_REASON_CHARS = 10

_IDENT_START = re.compile(r"[A-Za-z_]")
_TRAILING_LABEL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*:\s*\{")
_STRING_OPEN_RE = re.compile(r'(#*)("""|")')

_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = {")", "]", "}"}


@dataclass(frozen=True)
class Violation:
    """One added interactive control with no reachable identifier."""

    path: str
    line: int  # 1-based, the line the control is declared on
    kind: str
    source: str  # the declaration line, stripped
    reason: str  # why it failed ("no accessibility identifier" / bad marker)

    @property
    def key(self) -> tuple[str, str]:
        """Identity used for carry-over suppression against the base revision.

        Whitespace is collapsed so a re-indent or a line re-wrap does not read
        as a brand-new control.
        """
        return (self.kind, re.sub(r"\s+", " ", self.source).strip())

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind} — {self.reason}"


# --------------------------------------------------------------------------
# Lexing: mask comments and string literals so the structural scan below can
# trust every brace, paren and dot it sees.
# --------------------------------------------------------------------------


def mask_source(src: str) -> tuple[str, dict[int, str]]:
    """Blank out comments and string literals, preserving offsets and newlines.

    Returns ``(masked, comments_by_line)``. ``masked`` has the exact same length
    and line structure as ``src`` — every comment/string character becomes a
    space — so a match offset in ``masked`` is a valid offset in ``src``.
    ``comments_by_line`` maps a 1-based line number to the comment text found on
    it, which is where ``ax-exempt:`` markers are looked for.

    Handles Swift's line comments, *nested* block comments, escapes, string
    interpolation (treated as string content: we never need to find a control
    inside ``\\(…)``), multi-line ``\"\"\"`` literals, and raw strings with any
    number of ``#`` delimiters.
    """
    out = list(src)
    comments: dict[int, list[str]] = {}
    n = len(src)
    i = 0
    line = 1
    block_depth = 0
    # Context stack: ("str", hashes, multiline) or ("interp", paren_depth).
    stack: list[tuple] = []

    def blank(start: int, stop: int) -> None:
        nonlocal line
        for k in range(start, min(stop, n)):
            if src[k] == "\n":
                line += 1
            else:
                out[k] = " "

    def in_string() -> bool:
        return any(frame[0] == "str" for frame in stack)

    while i < n:
        if block_depth > 0:
            if src.startswith("/*", i):
                block_depth += 1
                blank(i, i + 2)
                i += 2
                continue
            if src.startswith("*/", i):
                block_depth -= 1
                blank(i, i + 2)
                i += 2
                continue
            comments.setdefault(line, []).append(src[i])
            blank(i, i + 1)
            i += 1
            continue

        top = stack[-1] if stack else None
        if top is not None and top[0] == "str":
            _, hashes, multiline = top
            escape = "\\" + "#" * hashes
            if src.startswith(escape + "(", i):
                blank(i, i + len(escape) + 1)
                stack.append(("interp", 1))
                i += len(escape) + 1
                continue
            if src.startswith(escape, i):
                blank(i, i + len(escape) + 1)
                i += len(escape) + 1
                continue
            closer = ('"""' if multiline else '"') + "#" * hashes
            if src.startswith(closer, i):
                blank(i, i + len(closer))
                stack.pop()
                i += len(closer)
                continue
            blank(i, i + 1)
            i += 1
            continue

        # Normal code — either at top level or inside a ``\(…)`` interpolation.
        # Interpolation contents are still blanked (``in_string()`` is true),
        # but they are lexed properly so a nested literal cannot desync us.
        if src.startswith("//", i):
            end = src.find("\n", i)
            end = n if end == -1 else end
            comments.setdefault(line, []).append(src[i:end])
            blank(i, end)
            i = end
            continue
        if src.startswith("/*", i):
            block_depth = 1
            blank(i, i + 2)
            i += 2
            continue

        opener = _STRING_OPEN_RE.match(src, i)
        if opener:
            hashes = len(opener.group(1))
            multiline = opener.group(2) == '"""'
            blank(i, opener.end())
            stack.append(("str", hashes, multiline))
            i = opener.end()
            continue

        ch = src[i]
        if top is not None and top[0] == "interp":
            if ch == "(":
                stack[-1] = ("interp", top[1] + 1)
            elif ch == ")":
                if top[1] <= 1:
                    stack.pop()
                else:
                    stack[-1] = ("interp", top[1] - 1)

        if in_string():
            blank(i, i + 1)
        elif ch == "\n":
            line += 1
        i += 1

    # Block comments accumulate one character per entry while line comments
    # arrive whole, so the parts are joined with nothing between them — a
    # separator here would shred "ax-exempt:" inside a /* … */ into letters and
    # make the marker silently unrecognisable in block-comment form.
    return "".join(out), {ln: "".join(parts) for ln, parts in comments.items()}


# --------------------------------------------------------------------------
# Structural scan over the masked source.
# --------------------------------------------------------------------------


def _consume_balanced(masked: str, start: int) -> int:
    """Return the offset just past the delimiter group opening at ``start``."""
    stack = [_OPENERS[masked[start]]]
    i = start + 1
    n = len(masked)
    while i < n and stack:
        ch = masked[i]
        if ch in _OPENERS:
            stack.append(_OPENERS[ch])
        elif ch in _CLOSERS:
            if ch == stack[-1]:
                stack.pop()
            else:
                # Unbalanced (or something the mask got wrong). Bail rather
                # than run to EOF and swallow the rest of the file.
                return i + 1
        i += 1
    return i


def _skip_space(masked: str, i: int) -> int:
    n = len(masked)
    while i < n and masked[i].isspace():
        i += 1
    return i


def _expression_end(masked: str, delim: int) -> int:
    """Walk a control's whole postfix expression: trailing closures + modifiers.

    ``delim`` is the ``(`` or ``{`` that opens the constructor. The walk stops
    at the first token that cannot continue a postfix chain, which in a
    ``@ViewBuilder`` body is the next sibling view or the closing brace. Erring
    toward consuming slightly too much is deliberate: over-consuming can only
    cause a miss, under-consuming causes a false alarm.
    """
    i = _consume_balanced(masked, delim)
    n = len(masked)
    while True:
        j = _skip_space(masked, i)
        if j >= n:
            return i
        ch = masked[j]
        if ch in _OPENERS:
            i = _consume_balanced(masked, j)
            continue
        if ch in "?!":
            i = j + 1
            continue
        if ch == ".":
            k = j + 1
            if k < n and _IDENT_START.match(masked[k]):
                while k < n and (masked[k].isalnum() or masked[k] == "_"):
                    k += 1
                i = k
                continue
            return i
        # Swift's second (and later) trailing closure: ``} label: { … }``.
        label = _TRAILING_LABEL_RE.match(masked, j)
        if label:
            i = _consume_balanced(masked, label.end() - 1)
            continue
        return i


def _skipped_scopes(masked: str) -> list[tuple[int, int]]:
    """Char ranges whose controls are deliberately not required to be labelled."""
    spans: list[tuple[int, int]] = []
    for m in _COMMAND_SCOPE_RE.finditer(masked):
        delim = m.end() - 1
        if masked[delim] in _OPENERS:
            spans.append((m.start(), _expression_end(masked, delim)))
    for m in _STYLE_BODY_RE.finditer(masked):
        after_args = _consume_balanced(masked, m.start(1))
        body = masked.find("{", after_args)
        if body != -1:
            spans.append((m.start(), _consume_balanced(masked, body)))
    return spans


def _exemption(comments: dict[int, str], head_line: int) -> tuple[bool, str | None]:
    """Look for an ``ax-exempt:`` marker on the control's line or the line above.

    Returns ``(found, reason)``. ``reason`` is ``None`` when the marker is there
    but the written justification is missing or too thin — which is a failure in
    its own right, not a pass.
    """
    for candidate in (head_line, head_line - 1):
        text = comments.get(candidate)
        if not text:
            continue
        m = _EXEMPT_RE.search(text)
        if not m:
            continue
        reason = m.group(1).strip().lstrip("-—:").strip()
        if len(reason) < MIN_EXEMPT_REASON_CHARS:
            return True, None
        return True, reason
    return False, None


def find_violations(path: str, src: str) -> list[Violation]:
    """Every interactive control in ``src`` with no reachable identifier."""
    masked, comments = mask_source(src)
    lines = src.splitlines()
    skipped = _skipped_scopes(masked)
    violations: list[Violation] = []

    for m in _CONTROL_RE.finditer(masked):
        start = m.start()
        if any(lo <= start < hi for lo, hi in skipped):
            continue
        delim = m.end() - 1
        end = _expression_end(masked, delim)
        if _IDENTIFIER_RE.search(masked, start, end):
            continue

        head_line = masked.count("\n", 0, start) + 1
        marked, reason = _exemption(comments, head_line)
        if marked and reason:
            continue

        source = lines[head_line - 1].strip() if head_line <= len(lines) else ""
        if marked:
            why = (
                f"'{EXEMPT_MARKER}' marker with no usable reason — write at least "
                f"{MIN_EXEMPT_REASON_CHARS} characters saying why this control "
                "cannot carry an identifier"
            )
        else:
            why = "no .accessibilityIdentifier(…) on this control"
        violations.append(
            Violation(
                path=path,
                line=head_line,
                kind=m.group(1),
                source=source,
                reason=why,
            )
        )
    return violations


# --------------------------------------------------------------------------
# git plumbing
# --------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _merge_base(base_ref: str, head_ref: str) -> str:
    return _git("merge-base", base_ref, head_ref).strip()


def changed_swift_files(base: str, head: str) -> dict[str, str]:
    """Map each changed in-scope Swift file to the path it had on ``base``.

    Rename-aware (``-M``): a file renamed in the PR is compared against its
    pre-rename blob, so moving a panel full of known-unlabelled controls is not
    read as adding a panel full of new ones. Deletions are dropped — there is
    nothing left to label.
    """
    out = _git(
        "diff",
        "--name-status",
        "-M",
        "--diff-filter=ACMR",
        base,
        head,
        "--",
        SCOPE_PREFIX,
    )
    paths: dict[str, str] = {}
    for raw in out.splitlines():
        fields = raw.split("\t")
        status = fields[0]
        if status.startswith("R") and len(fields) >= 3:
            old, new = fields[1], fields[2]
        else:
            old = new = fields[-1]
        if new.endswith(".swift"):
            paths[new] = old
    return paths


_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def added_lines(base: str, head: str, path: str) -> set[int]:
    """1-based line numbers added to ``path`` between ``base`` and ``head``."""
    diff = _git("diff", "--unified=0", "-M", base, head, "--", path)
    added: set[int] = set()
    for raw in diff.splitlines():
        m = _HUNK_RE.match(raw)
        if m:
            start = int(m.group(1))
            count = 1 if m.group(2) is None else int(m.group(2))
            added.update(range(start, start + count))
    return added


def _blob(rev: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def suppress_carried(
    head_violations: list[Violation],
    base_violations: list[Violation],
    touched: set[int],
) -> list[Violation]:
    """Report the head violations this diff is actually responsible for.

    Multiset arithmetic over the WHOLE file, not just the added lines: the base
    revision funds a fixed number of violations per ``(kind, normalised line)``,
    and only the excess is new. Move or re-indent an existing unlabelled control
    and it is absorbed; duplicate one and the copy is reported, because the base
    only funds one of the two.

    Untouched occurrences claim the base budget first. They are carried over by
    definition, so letting an *added* line spend the budget instead — which is
    what happens with a naive in-order walk when the copy lands above the
    original — would launder a genuine duplication into a pass.
    """
    budget = Counter(v.key for v in base_violations)
    for v in head_violations:
        if v.line not in touched and budget[v.key] > 0:
            budget[v.key] -= 1

    kept: list[Violation] = []
    for v in head_violations:
        if v.line not in touched:
            continue
        if budget[v.key] > 0:
            budget[v.key] -= 1
            continue
        kept.append(v)
    return kept


def new_violations(base: str, head: str, path: str, base_path: str) -> list[Violation]:
    """Violations this diff is responsible for, in one file.

    Two filters, in order: the control must be declared on a line this diff
    added, and it must not be a carry-over of an identical violation that
    already existed in the base revision of the same file.
    """
    head_src = _blob(head, path)
    if head_src is None:
        return []
    touched = added_lines(base, head, path)
    head_violations = find_violations(path, head_src)
    candidates = [v for v in head_violations if v.line in touched]
    if not candidates:
        return []

    base_src = _blob(base, base_path)
    if base_src is None:  # new file — everything in it is this PR's doing
        return candidates
    return suppress_carried(head_violations, find_violations(path, base_src), touched)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _report(violations: list[Violation]) -> None:
    for v in violations:
        print(f"::error file={v.path},line={v.line}::{v.kind}: {v.reason}")
        print(f"  {v.path}:{v.line}")
        print(f"    {v.source}")
        print(f"    -> {v.reason}")


_FAIL_ADVICE = f"""
[ax-identifier-gate] BLOCKED — the controls above are new and cannot be reached
by apps/rapid-mac/scripts/gui-golden-flows.sh, which finds every element through
AXIdentifier (see apps/rapid-mac/docs/gui-golden-flows.md). Shipping them
unlabelled quietly lowers what the GUI suite is able to cover.

Fix — attach a stable identifier to the control itself, matching the existing
'<Surface>.<Thing>' convention inventoried in apps/rapid-mac/docs/userflows.md:

    Toggle("Browse", isOn: $on)
        .accessibilityIdentifier("Settings.Tools.BrowseToggle")

If the control genuinely cannot carry one — SwiftUI confirmationDialog / alert
buttons are the known case — opt out explicitly, with a reason, on the control's
line or the line directly above it:

    // {EXEMPT_MARKER} confirmationDialog buttons render outside the app's AX tree
    Button("Allow once") {{ approve() }}

Every opt-out is greppable: rg '{EXEMPT_MARKER}' apps/rapid-mac
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when a PR adds an interactive rapid-mac control with no "
            ".accessibilityIdentifier(…)."
        )
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="branch the PR targets (default: origin/main)",
    )
    parser.add_argument(
        "--head-ref",
        default="HEAD",
        help=(
            "revision under test (default: HEAD). Together with --base-ref this "
            "replays the gate over any historical commit: "
            "--base-ref <sha>~1 --head-ref <sha>"
        ),
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help=(
            "ignore the diff and report EVERY unlabelled control under "
            f"{SCOPE_PREFIX} — the existing backlog, not the gate"
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="with --audit, restrict the sweep to these files",
    )
    args = parser.parse_args(argv)

    if args.audit:
        files = args.paths or sorted(
            str(p.relative_to(REPO_ROOT))
            for p in (REPO_ROOT / SCOPE_PREFIX).rglob("*.swift")
        )
        found: list[Violation] = []
        for path in files:
            src = (REPO_ROOT / path).read_text(encoding="utf-8")
            found.extend(find_violations(path, src))
        for v in found:
            print(str(v))
        print(f"\n[ax-identifier-gate] audit: {len(found)} unlabelled control(s)")
        return 1 if found else 0

    base = _merge_base(args.base_ref, args.head_ref)
    files = changed_swift_files(base, args.head_ref)
    if not files:
        print(
            f"[ax-identifier-gate] no {SCOPE_PREFIX} Swift files changed "
            "— nothing to check."
        )
        return 0

    print(f"[ax-identifier-gate] base {base[:12]}, {len(files)} changed file(s):")
    for path in sorted(files):
        print(f"    {path}")

    violations: list[Violation] = []
    for path, base_path in sorted(files.items()):
        violations.extend(new_violations(base, args.head_ref, path, base_path))

    if not violations:
        print(
            "[ax-identifier-gate] PASS — every control this PR adds is "
            "reachable by AXIdentifier."
        )
        return 0

    _report(violations)
    print(_FAIL_ADVICE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
