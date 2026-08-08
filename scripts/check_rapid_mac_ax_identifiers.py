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

* **This diff's doing, not the backlog's.** The existing backlog is real and is
  tracked separately. A gate that fails on it would be un-landable, get disabled
  inside a week, and leave us worse off than having no gate. Two things count as
  this diff's doing: declaring an unlabelled control on a line it added, and
  taking the identifier off one that had it — which lowers the harness's ceiling
  just as much, and is the easier accident, since the declaration line does not
  change and an added-lines filter never looks at it.

* **Carry-over suppression is per hunk.** A violation identical to one the SAME
  hunk removed is absorbed, so re-indenting / re-wrapping / re-ordering
  known-bad code does not suddenly light up. Deliberately not a file-wide
  budget: that let labelling a control in one place pay for a genuinely new
  unlabelled one somewhere else, and the PR came out green having both fixed and
  broken one. The cost is that moving an unlabelled control across a file reads
  as new — the same answer this gate already gave for extracting it into a new
  file, and the fix it asks for is the one we want anyway. Identity is the
  declaration line with trailing comments removed, so re-wording a comment does
  not rename a control.

* **Precision over recall.** A gate that cries wolf gets ignored, and an ignored
  gate is worse than none because it manufactures confidence. Where the two
  trade off, this script chooses to miss. Concretely it walks the real postfix
  chain of each control (balanced delimiters over a comment/string-masked copy
  of the source), so an identifier attached five modifiers later still counts,
  and it deliberately skips ``Commands`` scenes (the AX driver never walks the
  menu bar). What it does not detect is listed under "Known blind spots" below.

* **An escape hatch that costs something.** A control that genuinely cannot
  carry an identifier gets an explicit, greppable marker with a written reason
  on the same line::

      Button("Allow once") { … }  // ax-exempt: <what you measured that shows
                                  // the identifier cannot be reached>

  No such control is known on the current surface. ``confirmationDialog`` /
  ``alert`` buttons were the standing suspicion, and it was measured rather than
  inherited: the presented dialog is an ``AXSheet`` whose ``AXButton`` children
  DO carry the identifiers declared at the call site (see ``docs/userflows.md``).
  So ``rg ax-exempt apps/rapid-mac`` finding nothing is the expected state, and
  an exemption has to bring new evidence. A marker with no reason (or a one-word
  reason) is itself a failure, so the cost cannot be dodged by typing
  ``// ax-exempt:`` and moving on.

Known blind spots (deliberate — each would cost more false positives than it
buys):

* interactivity bolted onto a non-control view (``.onTapGesture``, ``.gesture``,
  ``.contextMenu``, ``.swipeActions``, ``.draggable``);
* a control built through Swift's *contextual* member lookup, where the type
  and the constructor are in different places::

      let saveButton: Button<Text> = .init(action: save) { Text("Save") }

  ``Button<Text>`` there is a type annotation and ``.init`` names no control,
  so neither half looks like a construction. Detecting it means resolving types,
  which this lint does not do, and guessing at a bare ``.init(`` would fire on
  every non-control initializer in the app;
* an EXPLICITLY generic control built with a trailing closure and no parens
  (``Button<Text> { save() } label: { … }``). After a generic argument list only
  ``(`` counts, because ``var b: Button<Text> {`` is a return type followed by a
  property body and reading that brace as a constructor blocked PRs whose real
  control was labelled. Spelling the generic out is unnecessary and
  non-idiomatic here — SwiftUI infers it — so the miss costs less than the
  false alarm did;
* bespoke ``View`` structs that are interactive without literally naming a
  SwiftUI control type at the point of use (the wrapper's own ``Button`` is
  still checked where the wrapper is *defined*);
* AppKit surfaces bridged through ``NSViewRepresentable`` (the chat composer's
  ``AutosizingTextView`` is the app's real example);
* ``Commands`` / menu-bar items, skipped on purpose;
* an identifier attached to a *parenthesised* control
  (``(Button(…)).accessibilityIdentifier(…)``) — the balanced walk stops at the
  enclosing ``)``, so the control is reported. Making those parens transparent
  requires deciding whether a ``(`` is grouping or an argument list, which the
  preceding character cannot answer (a closure call ends in ``}``; ``return``
  ends in a letter), and both attempts produced something worse than the miss —
  including crediting ``Card(Button(…))``'s identifier to the Button, the exact
  defect this gate exists to catch. Attach the identifier to the control;
* moving a file AND adding a new one at its old path in the same PR. ``-M``
  cannot pair that (the source path still exists), and ``-C`` would — but
  ``-C`` cannot tell it apart from *adding a copy of an existing file*, which is
  genuinely new unlabelled surface. The moved file's backlog is reported. Label
  it, or split the move and the replacement into two PRs;
* laundering a new gap through a fix in the SAME hunk. Carry-over is a count
  within one hunk, so replacing an unlabelled control with a labelled one AND
  an unlabelled one, all inside the same run of changed lines, nets to zero and
  passes. Distinguishing "the same control moved" from "one fixed, one added"
  needs per-line identity that a text diff does not carry. The window is
  narrow — with zero context a hunk is exactly the changed lines, so the two
  edits have to be adjacent — and the file-wide version of this hole (fix
  anywhere funds a regression anywhere) is closed;
* a control whose identifier is applied to an enclosing container rather than
  the control itself — reported, because ``AXPress`` needs the control. The
  mirror image is reported too: an identifier on a control NESTED inside an
  unlabelled one does not label the parent, because the parent's own postfix
  chain is what is searched.

Pure-logic core (``mask_source`` / ``find_violations``) is unit-tested in
tests/test_rapid_mac_ax_identifiers.py — no network, no Swift toolchain, no GPU.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
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
    "DisclosureGroup",
    "Stepper",
    "Slider",
    "NavigationLink",
    "Link",
    # First-party controls that render a real, pressable element. Omitting them
    # made the docstring's claim — that everything undetected is listed under
    # "Known blind spots" — untrue: ``ShareLink(item:)`` is an ordinary button
    # to a user and was invisible to the gate.
    "ShareLink",
    "SettingsLink",
    "HelpLink",
    "PasteButton",
    "EditButton",
    "RenameButton",
    "MultiDatePicker",
)

# ``(?<![A-Za-z0-9_])`` keeps ``SendButton`` and ``pickerStyle`` out; the
# trailing ``[({]`` keeps type references (``Button<Label>``, ``struct Button:
# View``) out. What is left is a control being *constructed*.
#
# The dot is deliberately NOT in the lookbehind. ``SwiftUI.Button(…)`` IS the
# control, and a dot-rejecting lookbehind let a fully qualified construction
# walk straight past the gate — but spelling the qualifier into the pattern
# instead (``(?:SwiftUI\s*\.\s*)?``) is worse than useless: the optional group
# simply does not participate, the match starts at ``Button`` either way, and
# ``Chrome . Button`` matches too (whitespace may surround Swift member access),
# falsely blocking somebody else's type that happens to share a name. So the
# pattern is permissive and ``_qualifier_before`` decides in code.
# ``Button<Text>(action:)`` and ``Button.init("Save", action:)`` are ordinary
# constructions that a name-then-delimiter pattern walks straight past. One
# level of nested generics is enough for real SwiftUI (``Button<Label<Text>>``);
# beyond that the pattern gives up rather than guess.
#
# After a generic argument list ONLY ``(`` counts. ``{`` there is a property
# body, not a constructor::
#
#     private var saveButton: Button<Text> {   // <- return type, then a body
#         Button("Save") { save() }
#             .accessibilityIdentifier("Toolbar.Save")
#     }
#
# Reading that as a construction reported a phantom unlabelled Button and
# blocked a PR whose real control was correctly labelled. Nobody writes
# ``Button<Text> { … }`` as a construction — the generic is inferred — so
# requiring the paren costs nothing and removes the whole false-positive class.
_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(CONTROL_KINDS)
    + r")\s*(?:"
    + r"<(?:[^<>]|<[^<>]*>)*>\s*(?:\.\s*init\s*)?\("  # generic: paren only
    + r"|(?:\.\s*init\s*)?[({]"  # bare: paren or trailing closure
    + r")"
)

# The only namespace whose ``X.Button(…)`` is the SwiftUI control.
QUALIFIER_ALLOWED = "SwiftUI"

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

# The one modifier that makes a control reachable. Named ONCE: every message,
# every docstring and the matcher itself derive from this, so there is no
# second literal to fall out of step with the check.
IDENTIFIER_MODIFIER = "accessibilityIdentifier"

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
    """``(masked, comments_by_line)`` — see ``_mask`` for the full contract."""
    masked, comments, _ = _mask(src)
    return masked, comments


def _mask(src: str) -> tuple[str, dict[int, str], dict[int, set[int]]]:
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
    # Every column (0-based) a comment occupies, per line. The carry-over
    # identity below is the declaration line's *code*, so the comment has to
    # come out of it — and as a set of columns rather than a truncation point,
    # because a comment can also LEAD the line::
    #
    #     /* old explanation */ Button("Save") { save() }
    #
    # Truncating at its start would leave the empty string, which every other
    # control then matches; keeping the line whole (the earlier compromise)
    # meant re-wording that comment renamed the control and reported a
    # long-standing gap as new. Deleting exactly the comment's columns is right
    # in both positions, and for more than one comment on a line.
    comment_cols: dict[int, set[int]] = {}

    def note_comment_span(start_at: int, stop_at: int) -> None:
        # Walks characters so a span crossing a newline lands on the right
        # line: a block comment is one span in the source and several in the
        # per-line view this returns.
        ln = line
        col = start_at - (src.rfind("\n", 0, start_at) + 1)
        for k in range(start_at, min(stop_at, n)):
            if src[k] == "\n":
                ln += 1
                col = 0
                continue
            comment_cols.setdefault(ln, set()).add(col)
            col += 1

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
                note_comment_span(i, i + 2)
                blank(i, i + 2)
                i += 2
                continue
            if src.startswith("*/", i):
                block_depth -= 1
                note_comment_span(i, i + 2)
                blank(i, i + 2)
                i += 2
                continue
            comments.setdefault(line, []).append(src[i])
            note_comment_span(i, i + 1)
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
            note_comment_span(i, end)
            comments.setdefault(line, []).append(src[i:end])
            blank(i, end)
            i = end
            continue
        if src.startswith("/*", i):
            note_comment_span(i, i + 2)
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
    return (
        "".join(out),
        {ln: "".join(parts) for ln, parts in comments.items()},
        comment_cols,
    )


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


def _skip_space_back(masked: str, i: int) -> int:
    while i >= 0 and masked[i].isspace():
        i -= 1
    return i


def _has_own_identifier(masked: str, delim: int, end: int) -> bool:
    """Does the control's OWN postfix chain carry ``.accessibilityIdentifier``?

    Searching the whole ``[start, end)`` span instead lets a *child* label its
    unlabelled parent, because the span includes trailing-closure bodies::

        Menu("Actions") {                       // <- unlabelled, must fail
            Button("Rename") { … }
                .accessibilityIdentifier("Sidebar.Rename")
        }

    The identifier there belongs to the Button. ``AXPress`` on the Menu has
    nothing to aim at, so a flow still cannot open it — exactly the blind spot
    this gate exists to close, silently satisfied by the child.

    So walk the same postfix chain ``_expression_end`` walks, and look only at
    the ``.name`` tokens it visits at depth 0, stepping OVER every nested
    delimiter group rather than into it.
    """
    i = _consume_balanced(masked, delim)
    n = len(masked)
    while i < end:
        j = _skip_space(masked, i)
        if j >= end:
            return False
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
                name_start = k
                while k < n and (masked[k].isalnum() or masked[k] == "_"):
                    k += 1
                if masked[name_start:k] == IDENTIFIER_MODIFIER:
                    return True
                i = k
                continue
            return False
        label = _TRAILING_LABEL_RE.match(masked, j)
        if label:
            i = _consume_balanced(masked, label.end() - 1)
            continue
        return False
    return False


def _qualifier_before(masked: str, start: int) -> str | None:
    """The member-access qualifier immediately left of ``start``, if any.

    ``None`` means the control is named bare. Whitespace may surround Swift's
    member-access dot, so this walks backwards rather than pattern-matching.
    """
    dot = _skip_space_back(masked, start - 1)
    if dot < 0 or masked[dot] != ".":
        return None
    end = _skip_space_back(masked, dot - 1)
    i = end
    while i >= 0 and (masked[i].isalnum() or masked[i] == "_"):
        i -= 1
    name = masked[i + 1 : end + 1] if i < end else ""
    # The WHOLE qualifier, not its last component: ``Chrome.SwiftUI.Button`` is
    # somebody's nested namespace, and returning just ``SwiftUI`` would treat it
    # as the real control and block it.
    if _skip_space_back(masked, i) >= 0 and masked[_skip_space_back(masked, i)] == ".":
        return f"<nested>.{name}"
    return name


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


def _strip_comments(raw: str, columns: set[int] | None) -> str:
    """``raw`` with every comment character on the line removed.

    The carry-over identity below is the declaration line's text, so a comment
    on that line is part of it — meaning *editing the comment* renames the
    control as far as this gate is concerned, and a long-standing unlabelled
    control lights up as brand new because somebody reworded a note. Comments
    have no bearing on whether ``AXPress`` can reach a control, so they are cut
    out of the identity, wherever on the line they sit.
    """
    if not columns:
        return raw
    return "".join(ch for col, ch in enumerate(raw) if col not in columns)


def find_violations(path: str, src: str) -> list[Violation]:
    """Every interactive control in ``src`` with no reachable identifier."""
    masked, comments, comment_cols = _mask(src)
    lines = src.splitlines()
    skipped = _skipped_scopes(masked)
    violations: list[Violation] = []

    for m in _CONTROL_RE.finditer(masked):
        start = m.start()
        qualifier = _qualifier_before(masked, start)
        if qualifier is not None and qualifier != QUALIFIER_ALLOWED:
            continue
        if any(lo <= start < hi for lo, hi in skipped):
            continue
        delim = m.end() - 1
        end = _expression_end(masked, delim)
        if _has_own_identifier(masked, delim, end):
            continue

        head_line = masked.count("\n", 0, start) + 1
        marked, reason = _exemption(comments, head_line)
        if marked and reason:
            continue

        raw = lines[head_line - 1] if head_line <= len(lines) else ""
        source = _strip_comments(raw, comment_cols.get(head_line)).strip()
        if marked:
            why = (
                f"'{EXEMPT_MARKER}' marker with no usable reason — write at least "
                f"{MIN_EXEMPT_REASON_CHARS} characters saying why this control "
                "cannot carry an identifier"
            )
        else:
            why = f"no .{IDENTIFIER_MODIFIER}(…) on this control"
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

    ``-M`` only, deliberately — NOT ``-C``. Copy detection would also pair the
    case ``-M`` cannot see (move ``Panel.swift`` to ``Moved/Panel.swift`` AND
    add a new ``Panel.swift``, where the source path still exists so git records
    a modify plus an add), and it does. But it cannot tell that apart from
    *adding a copy of an existing file*, which is genuinely new unlabelled
    surface — same ``C src dst`` record, same "source still exists in head",
    opposite correct answer. Between a false positive on an unusual refactor
    and silently admitting a whole new panel of unreachable controls, this gate
    takes the false positive: it is the one failure a contributor can see and
    fix in seconds, and admitting new unlabelled surface is the exact thing the
    gate exists to prevent. Listed under "Known blind spots".
    """
    out = _git(
        "diff",
        "--name-status",
        "-M",
        "-z",
        "--diff-filter=ACMR",
        base,
        head,
        "--",
        SCOPE_PREFIX,
    )
    # ``-z``: NUL-delimited, so paths arrive verbatim. Line-oriented output
    # QUOTES anything non-ASCII — ``Résumé.swift`` comes back as
    # ``"R\303\251sum\303\251.swift"``, whose trailing character is a quote,
    # so the ``.swift`` test failed and the file was never examined at all.
    # With ``-z`` a rename is three fields (status, old, new) and everything
    # else is two, in one flat stream.
    fields = [f for f in out.split("\0") if f]
    paths: dict[str, str] = {}
    i = 0
    while i < len(fields):
        status = fields[i]
        if status.startswith("R") and i + 2 < len(fields):
            old, new = fields[i + 1], fields[i + 2]
            i += 3
        elif i + 1 < len(fields):
            old = new = fields[i + 1]
            i += 2
        else:
            break
        if new.endswith(".swift"):
            paths[new] = old
    return paths


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _unquote_path(raw: str) -> str:
    """Decode git's C-style quoting of a patch-header path.

    ``changed_swift_files`` reads paths from ``-z`` output, which is verbatim,
    but a patch HEADER has no NUL form: git writes
    ``+++ "b/…/R\303\251sum\303\251.swift"`` for anything non-ASCII. Comparing
    that literal against the decoded path attributed no hunks to the file, so a
    line inserted above a pre-existing gap in ``Résumé.swift`` shifted it, the
    carry-over match missed, and the gate announced an identifier removal that
    never happened.
    """
    if not (raw.startswith('"') and raw.endswith('"') and len(raw) >= 2):
        return raw
    # Octal escapes are UTF-8 BYTES: decode the escapes to latin-1 code points,
    # take those back to bytes, then decode as UTF-8.
    try:
        return (
            raw[1:-1]
            .encode("latin-1", "backslashreplace")
            .decode("unicode_escape")
            .encode("latin-1")
            .decode("utf-8")
        )
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw


@dataclass(frozen=True)
class Hunk:
    """One ``--unified=0`` hunk: the base lines it removed paired with the head
    lines it put in their place. Either side may be empty."""

    base_start: int
    base_count: int
    head_start: int
    head_count: int

    def has_base(self, line: int) -> bool:
        return self.base_start <= line < self.base_start + self.base_count

    def has_head(self, line: int) -> bool:
        return self.head_start <= line < self.head_start + self.head_count


def diff_hunks(
    base: str, head: str, path: str, base_path: str | None = None
) -> list[Hunk]:
    """The ``--unified=0`` hunks for one file, in order.

    Zero context is what makes a hunk a usable unit of blame: each one is
    exactly the lines this diff removed paired with exactly the lines it added
    in their place, with no shared context blurring the boundary.

    BOTH paths go in the pathspec when the file moved, exactly as
    ``changed_swift_files`` resolved them. ``-M`` can only pair a rename it can
    see, and a pathspec naming just the destination hides the source: git emits
    ``new file mode`` with one ``@@ -0,0 +1,N @@``, so moving a panel full of
    known gaps is reported as adding every one of them.

    Naming two paths means the output can carry two file sections — a move plus
    a same-named replacement produces both — so hunks are attributed by their
    ``+++ b/…`` header rather than swept up together. Merging them mixes one
    file's added lines into another file's budget, which is how a move beside a
    replacement reported the moved file's entire backlog as new.
    """
    paths = [path] if base_path in (None, path) else [base_path, path]
    diff = _git("diff", "--unified=0", "-M", base, head, "--", *paths)
    hunks: list[Hunk] = []
    current: str | None = None
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = _unquote_path(raw[4:])
            current = target[2:] if target.startswith("b/") else target
            continue
        m = _HUNK_RE.match(raw)
        if not m or current != path:
            continue
        hunks.append(
            Hunk(
                base_start=int(m.group(1)),
                base_count=1 if m.group(2) is None else int(m.group(2)),
                head_start=int(m.group(3)),
                head_count=1 if m.group(4) is None else int(m.group(4)),
            )
        )
    return hunks


def added_lines(
    base: str, head: str, path: str, base_path: str | None = None
) -> set[int]:
    """1-based line numbers added to ``path`` between ``base`` and ``head``."""
    added: set[int] = set()
    for h in diff_hunks(base, head, path, base_path):
        added.update(range(h.head_start, h.head_start + h.head_count))
    return added


def head_line_to_base(hunks: list[Hunk], line: int) -> int:
    """Where an UNCHANGED head line sat in the base revision.

    Only meaningful for a line no hunk added; every hunk that ends before it
    shifts it by however many lines that hunk grew or shrank the file.

    ``max(head_count, 1)`` is load-bearing. A pure deletion is ``@@ -7,2 +6,0 @@``
    — head 6 is the line the deletion follows, NOT a line the hunk covers — so
    ``head_start + head_count`` is 6 and a naive ``<=`` shifts line 6 itself.
    That mapped a surviving carried-over violation to the wrong base line, found
    nothing there, and reported it as an identifier this diff had removed:
    deleting any labelled control lit up an unrelated control above it.
    """
    delta = 0
    for h in hunks:
        if line >= h.head_start + max(h.head_count, 1):
            delta += h.head_count - h.base_count
    return line - delta


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
    hunks: list[Hunk],
) -> list[Violation]:
    """Report the head violations this diff is actually responsible for.

    Carry-over is decided PER HUNK, not per file. A hunk is the lines this diff
    removed paired with the lines it put in their place, so an unlabelled
    control that was re-indented, re-wrapped, re-ordered or had its trailing
    comment edited is funded by the identical violation the same hunk removed,
    and is absorbed. Two things a file-wide budget got wrong, each of which let
    a real regression through:

    * **Fixing an old gap must not fund a new one.** Labelling a control in one
      part of the file removes a violation from the file-wide tally, and the
      freed budget then absorbed a genuinely new unlabelled control somewhere
      else — the PR came out green having both fixed and broken one. Budget is
      local to the hunk that freed it, so a fix in hunk A cannot pay for a
      regression in hunk B.

    * **Editing a comment must not rename a control.** Identity is the
      declaration line's text, so re-wording a trailing comment used to make a
      years-old unlabelled control read as brand new. ``Violation.source`` now
      excludes trailing comments, so the base and head copies match.

    Untouched head lines are the caller's problem, not this function's: they map
    back to a base line, and whether *that* line was already a violation is the
    whole question (see ``_identifier_removals``). They neither claim nor need
    budget here.

    The deliberate cost: moving an unlabelled control from one part of a file to
    another lands in two different hunks and IS reported. That is the same
    answer this gate already gave for extracting it into a new file, and the fix
    — label the control you just moved, or write an ``ax-exempt:`` reason — is
    the outcome the gate wants anyway.
    """
    budget: Counter[tuple[int, tuple[str, str]]] = Counter()
    for v in base_violations:
        for i, h in enumerate(hunks):
            if h.has_base(v.line):
                budget[(i, v.key)] += 1
                break

    kept: list[Violation] = []
    for v in head_violations:
        slot = next((i for i, h in enumerate(hunks) if h.has_head(v.line)), None)
        if slot is None:
            continue
        if budget[(slot, v.key)] > 0:
            budget[(slot, v.key)] -= 1
            continue
        kept.append(v)
    return kept


def _identifier_removals(
    head_violations: list[Violation],
    base_violations: list[Violation],
    hunks: list[Hunk],
    touched: set[int],
) -> list[Violation]:
    """Controls this diff UNLABELLED without touching their declaration line.

    Deleting a ``.accessibilityIdentifier(…)`` takes a control out of the golden
    flows' reach just as surely as never adding one, and it is the likelier
    accident: when the modifier sits on its own line — the dominant style in
    this app — the declaration line does not change at all, so an added-lines
    filter never even looks at it.

    The test is exact rather than statistical: an unchanged head line sits at a
    known base line and holds the same text. If the control there is unlabelled
    now but was not a violation then, this diff is what unlabelled it.

    Matched on ``(base line, key)`` and consumed, not on the line number alone.
    One line can hold two controls — ``Menu("x") { Button("y") { … } }`` — and a
    line-only test let the pre-existing gap vouch for its neighbour: delete the
    identifier off the Menu and the Button already on that line marked the line
    "known bad", so the newly unreachable Menu was suppressed.
    """
    carried: Counter[tuple[int, tuple[str, str]]] = Counter(
        (v.line, v.key) for v in base_violations
    )
    removals: list[Violation] = []
    for v in head_violations:
        if v.line in touched:
            continue
        slot = (head_line_to_base(hunks, v.line), v.key)
        if carried[slot] > 0:
            carried[slot] -= 1
            continue
        removals.append(
            replace(
                v,
                reason=(
                    f"this diff removed the .{IDENTIFIER_MODIFIER}(…) that made "
                    "this control reachable"
                ),
            )
        )
    return removals


def new_violations(base: str, head: str, path: str, base_path: str) -> list[Violation]:
    """Violations this diff is responsible for, in one file.

    Two ways to be responsible: declare a new unlabelled control on a line this
    diff added, or take the identifier off one that already had it.
    """
    head_src = _blob(head, path)
    if head_src is None:
        return []
    head_violations = find_violations(path, head_src)
    if not head_violations:
        return []

    touched = added_lines(base, head, path, base_path)
    base_src = _blob(base, base_path)
    if base_src is None:
        # New file — everything in it is this PR's doing. If the diff produced
        # no hunks at all (``.gitattributes`` marking ``*.swift`` binary is the
        # way that happens), ``touched`` is empty and filtering by it would
        # report nothing: a whole new panel of unreachable controls, admitted
        # in silence. A file that exists only in head has every line added by
        # definition, so fall back to that rather than to zero.
        return [v for v in head_violations if not touched or v.line in touched]

    hunks = diff_hunks(base, head, path, base_path)
    base_violations = find_violations(path, base_src)
    found = suppress_carried(head_violations, base_violations, hunks)
    found.extend(_identifier_removals(head_violations, base_violations, hunks, touched))
    return sorted(found, key=lambda v: v.line)


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
        .{IDENTIFIER_MODIFIER}("Settings.Tools.BrowseToggle")

No control on this surface is currently known to be unable to carry one — the
confirmationDialog / alert doubt was measured and closed (docs/userflows.md). If
you have found a real one, opt out explicitly, with the evidence, on the
control's line or the line directly above it:

    // {EXEMPT_MARKER} <what you measured that shows it cannot be reached>
    Button("Allow once") {{ approve() }}

Every opt-out is greppable: rg '{EXEMPT_MARKER}' apps/rapid-mac
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when a PR adds an interactive rapid-mac control with no "
            f".{IDENTIFIER_MODIFIER}(…)."
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
