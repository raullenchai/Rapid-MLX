#!/usr/bin/env python3
"""Structural AX baselines for the Rapid-MLX Desktop golden flows.

``rapid-ax dump`` serialises the whole accessibility tree of the running app.
The golden flows already collect those dumps; this tool turns one into a
*stable* structural fingerprint that can be committed and diffed, so a change
that silently drops a button, reparents a control, flips an enabled state or
renames an identifier shows up as a reviewable diff instead of passing
unnoticed.

Scope, stated plainly: this is structure only. It cannot see colour, spacing,
typography, ordering *within* a text run, or anything else that does not reach
the accessibility layer. The PNG snapshots under
``Tests/RapidTests/__Snapshots__`` remain the pixel-level check.

What the normalizer keeps
  * nesting depth and parent/child relationships (rendered as indentation)
  * ``AXRole`` and ``AXSubrole``
  * ``accessibilityIdentifier``
  * ``AXDescription`` / ``AXHelp``, and ``AXTitle`` only when the element
    publishes no ``AXDescription`` (scrubbed, see below)
  * ``AXEnabled``
  * the *kind* of ``AXValue`` — ``bool:true`` / ``bool:false`` for toggles,
    ``number`` / ``text`` / ``empty`` for everything else
  * sibling order below the window level

What it drops or rewrites, and why
  * ``bounds`` — absolute screen coordinates move whenever the window opens at
    a different origin; two consecutive launches produced 715,107 and 725,145.
    Layout size is equally volatile across displays.
  * ``pid`` — new every launch.
  * top-level window order — AX returns windows in z-order, so opening
    Settings reorders the roots. Windows are sorted by identifier/title.
  * value *contents* — a token/s figure, an on-disk size, a model name or a
    streamed transcript is data, not structure.
  * version strings, byte sizes, token rates, durations, dates, clock times,
    UUIDs and ``/Users/<name>`` paths inside titles/descriptions/identifiers.
    ``Settings.App.UpToDate`` legitimately carries the release version, and a
    conversation row identifier legitimately carries a fresh UUID.
  * caller-supplied tokens (``--scrub``) — the golden flows pass the fake
    model alias so a rename of the fixture is not a baseline change.
  * ``AXTitle`` on an element that also publishes ``AXDescription`` — the
    description is the label the app authored, the title is AppKit's own
    synthesis from how the control is drawn, and that synthesis differs
    between macOS releases (see ``render_node``). A baseline that pins it
    cannot be regenerated on one macOS and used on another, in either
    direction.
  * ``AXSelected`` — dumped by ``rapid-ax.swift`` (a flow has to be able to ask
    which model card the user chose, see #1653) but deliberately NOT rendered
    here. Which of several equal-looking cards is highlighted is state, not
    structure, and it legitimately differs between a fresh persona and a
    relaunched one. Flows assert it against the raw dump instead.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

SCHEMA_HEADER = "# rapid-mac AX structural baseline v1"

# Applied in order to identifiers, titles, descriptions and help text. Order
# matters: "1.2 GB" has to be consumed by the size rule before the version
# rule can mistake "1.2" for a release number.
_SCRUBBERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}"
            r"-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b"
        ),
        "<uuid>",
    ),
    (re.compile(r"/Users/[^/\s\"]+"), "/Users/<user>"),
    (
        # Multi-character units only. A bare "B" would swallow "Sub-1B
        # models", where the B is a parameter count, not a byte count.
        #
        # The spelled-out units (…"gigabytes") are how VoiceOver reads the
        # footer memory gauge — "12.3 gigabytes used out of 32 gigabytes" — so
        # both the used reading AND the machine's total collapse here. Without
        # them the total (an integer, no decimal) slips past every rule and a
        # baseline written on a 32 GB Mac can never match a 7 GB runner.
        re.compile(
            r"\b\d+(?:[.,]\d+)?\s*"
            r"(?:KB|MB|GB|TB|PB|KiB|MiB|GiB|TiB"
            r"|kilobytes?|megabytes?|gigabytes?|terabytes?|petabytes?"
            r"|bytes?)\b",
        ),
        "<size>",
    ),
    (re.compile(r"~?\s*\d+(?:\.\d+)?\s*(?:tok/s|tokens/s|t/s)"), "<rate>"),
    # Footer telemetry is live state, not structure. Once #1588 mounted the
    # footer, consecutive dumps legitimately observed CPU 89%, 99%, and 100%.
    (re.compile(r"\b\d+(?:\.\d+)?\s*percent\b"), "<percent>"),
    # The memory-pressure label is derived from live utilisation and therefore
    # legitimately differs between a developer Mac and a hosted runner even
    # when the UI is identical. Keep the Memory gauge and its wording, but
    # collapse only the volatile pressure bucket.
    (
        re.compile(r"\bMemory (?:normal|tight|critical):"),
        "Memory <pressure>:",
    ),
    (
        re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|µs|us|ns|s|min|h)\b"),
        "<duration>",
    ),
    (
        re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?"),
        "<date>",
    ),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b"), "<time>"),
    (
        re.compile(r"\bv?\d+\.\d+(?:\.\d+)*(?:-[0-9A-Za-z.]+)?\b"),
        "<version>",
    ),
    # Apple-silicon marketing name, as the model-management header prints it
    # ("Recommended for your <size> · M2 Pro"). It is the machine's chip, so a
    # baseline written on an M2 Pro can never match an M1/M3/M4 runner — the
    # same portability trap as the RAM total beside it. Scoped to that header
    # (matched after the size rule has already run) so a stray "M2" in a model
    # name or user text elsewhere keeps its own value.
    (
        re.compile(
            r"(Recommended for your\b[^·]*·\s*)M[1-9]\d?(?:\s+(?:Pro|Max|Ultra))?\b"
        ),
        r"\1<chip>",
    ),
    # GitHub's hosted Apple-silicon runners append ``(Virtual)`` to the chip
    # name. The preceding rule has already replaced the actual chip, leaving
    # this runner-only suffix behind. Keep this scoped to the recommendation
    # header: "Virtual" in model names or user-authored text remains visible.
    (
        re.compile(r"(Recommended for your\b[^·]*·\s*<chip>)\s+\(Virtual\)$"),
        r"\1",
    ),
    # Conversation-list date headings. The transcript a flow just created is
    # filed under "Today" — until the run straddles local midnight, at which
    # point the same unchanged UI reports "Yesterday" and every baseline
    # holding one of these headings goes red for no product reason.
    #
    # Anchored and case-sensitive, and deliberately NOT including "Tomorrow":
    # a conversation filed in the future is an impossible state worth failing
    # on, and collapsing it would hide exactly that bug. Only the two headings
    # a legitimate midnight crossing can produce are rewritten.
    (re.compile(r"^(?:Today|Yesterday)$"), "<relative-day>"),
)

# Identifiers whose trailing segment is data (a model alias, a status string).
# The prefix is what carries the structural meaning: "there is a Download
# button on a model row", not "there is a Download button on qwen3-4b".
_DYNAMIC_ID_PREFIXES: tuple[tuple[str, str], ...] = (
    ("Quickstart.Choice.", "<model>"),
    ("Settings.ModelManagement.Recommended.Cancel.", "<model>"),
    ("Settings.ModelManagement.Recommended.Delete.", "<model>"),
    ("Settings.ModelManagement.Recommended.Download.", "<model>"),
    ("Settings.ModelManagement.Recommended.Retry.", "<model>"),
    ("Settings.ModelManagement.Cancel.", "<model>"),
    ("Settings.ModelManagement.Delete.", "<model>"),
    ("Settings.ModelManagement.Download.", "<model>"),
    ("Settings.ModelManagement.Favorite.", "<model>"),
    ("Settings.ModelManagement.Retry.", "<model>"),
    ("Settings.ModelManagement.Row.", "<model>"),
    ("Settings.ModelManagement.Status.", "<text>"),
)

# Roles whose AXValue is a two-state control rather than a magnitude. For
# these the concrete 0/1 is the state we want to catch flipping; everywhere
# else a number is a measurement and only its presence is structural.
_TOGGLE_ROLES = frozenset(
    {"AXCheckBox", "AXRadioButton", "AXDisclosureTriangle", "AXMenuItem"}
)
_TOGGLE_SUBROLES = frozenset({"AXSwitch", "AXToggle"})

# OS-owned window chrome. The buttons themselves stay in the fingerprint; what
# gets dropped is everything *below* them, which belongs to AppKit and is
# realized lazily rather than in response to anything the app does.
_WINDOW_CONTROL_SUBROLES = frozenset(
    {
        "AXCloseButton",
        "AXMinimizeButton",
        "AXZoomButton",
        "AXFullScreenButton",
        "AXToolbarButton",
    }
)

# AppKit exposes the same SwiftUI ``Picker`` as either role depending on the
# macOS release. Both are the same semantic popup control, so fingerprints
# use one stable spelling instead of treating an OS upgrade as a product diff.
_ROLE_EQUIVALENTS = {
    "AXMenuButton": "AXPopUpButton",
}


def is_window_control(record: dict) -> bool:
    """True for a traffic-light / toolbar button whose subtree is AppKit's."""
    return record.get("subrole") in _WINDOW_CONTROL_SUBROLES


def is_system_sidebar_button(node: Node) -> bool:
    """True for AppKit's session-dependent window-toolbar sidebar control.

    AppKit may return this system-owned button before or after the app-authored
    toolbar items depending on the display/session.  It has no application
    identifier; keep the control itself in the fingerprint, but canonicalise
    only its position relative to our controls.  The order among app-authored
    siblings remains untouched and therefore remains regression-sensitive.
    """
    record = node.record
    return (
        record.get("role") == "AXButton"
        and not record.get("identifier")
        and record.get("description") in {"Hide Sidebar", "Show Sidebar"}
    )


def normalize_toolbar_children(children: list[Node]) -> list[Node]:
    """Place AppKit's sidebar toggle first without reordering app controls."""
    sidebar = [child for child in children if is_system_sidebar_button(child)]
    authored = [child for child in children if not is_system_sidebar_button(child)]
    return [*sidebar, *authored]


# The footer's GPU gauge has no stable cross-machine shape. Apple Silicon
# publishes a live utilisation reading (``AXUnknown desc="GPU 47 percent"``);
# Intel Macs and sandboxed apps can't probe AGXAccelerator at all and instead
# publish a static "unavailable" note (``AXStaticText help="GPU probe
# unavailable — …" value=text``). SAME footer item, different role, different
# attribute, different text. Its reading is live state, not structure — exactly
# like the CPU and memory gauges beside it — so collapse the whole element to
# one token. Otherwise a baseline written on either machine turns the other red
# on the GPU line alone, which is precisely how the golden flows regressed.
#
# The two exact shapes SystemPills.swift publishes, each keyed to the attribute
# it lands in and full-matched end to end. The live reading is an
# accessibilityLabel ("GPU <int> percent", an AXDescription); the note is a
# .help string that names AGXAccelerator (an AXHelp). Full-matching both — and
# never a bare "GPU " prefix or loose substring — means an unrelated control
# ("GPU 47 percent limit", "GPU probe unavailable settings", a "GPU settings"
# button) keeps its own structure, so a real regression there still fails.
_GPU_READING = re.compile(r"GPU \d+ percent")
_GPU_UNAVAILABLE = re.compile(r"GPU probe unavailable\b.*AGXAccelerator.*", re.DOTALL)


def is_gpu_telemetry(record: dict) -> bool:
    """True for the footer GPU utilisation readout, in either platform shape.

    Keyed to the exact (role, attribute) pair each shape uses and full-matched,
    so only the gauge itself is collapsed. A gauge whose role ever changes stops
    matching and surfaces as a visible golden-flow diff, never a silent rewrite.
    """
    role = record.get("role")
    if role == "AXUnknown":
        return bool(_GPU_READING.fullmatch(record.get("description") or ""))
    if role == "AXStaticText":
        return bool(_GPU_UNAVAILABLE.fullmatch(record.get("help") or ""))
    return False


class Node:
    """One accessibility element plus its children."""

    __slots__ = ("record", "children")

    def __init__(self, record: dict) -> None:
        self.record = record
        self.children: list[Node] = []


def is_lazy_button_wrapper(node: Node) -> bool:
    """True for an ``AXButton`` that only wraps a re-published copy of itself.

    macOS 15 realizes a SwiftUI toolbar ``Button`` as an outer AXButton *plus*
    a lazily-created inner AXButton that carries the button's own identity — the
    same identifier and description, with the tooltip surfacing as ``AXHelp`` on
    the child. macOS 26 publishes the button flat, with no inner copy. Keeping
    the outer button (so a vanished control is still a diff) and dropping the
    OS-owned duplicate makes a baseline generated on one release match the
    other, exactly as ``is_window_control`` already does for the traffic lights.

    #1845 pinned this for the system "Hide/Show Sidebar" toggle by matching its
    description; keying on the STRUCTURE instead absorbs it for every toolbar
    button — app-authored ones like ``Toolbar.SearchChats`` (#1879) included —
    so a new toolbar button no longer has to be added to a hand-kept allowlist.

    The predicate matches only the exact shape observed in the real cross-OS
    dumps (``tests/fixtures/ax_baseline``): an ``AXButton`` directly under an
    ``AXToolbar`` (``render`` checks the parent) with an identity of its own,
    whose SINGLE child is a LEAF ``AXButton`` re-publishing that same identifier
    and description. Every clause narrows away a way a real control could be
    hidden:

    * an *anonymous* button (no identifier and no description) has no identity to
      re-publish, so a button nested in it is a distinct control, never a copy;
    * a real nested control brings its own children or siblings, so requiring a
      single leaf child leaves any richer structure — and any change to it —
      visible in the diff;
    * a child with a different identifier or description is a different control
      and is kept.

    Only AppKit's lazy self-copy, which no product change can introduce, matches.
    """
    record = node.record
    if record.get("role") != "AXButton" or len(node.children) != 1:
        return False
    identifier = record.get("identifier") or ""
    description = record.get("description") or ""
    if not identifier and not description:
        return False
    (child,) = node.children
    child_record = child.record
    return (
        not child.children
        and child_record.get("role") == "AXButton"
        and (child_record.get("identifier") or "") == identifier
        and (child_record.get("description") or "") == description
    )


def scrub(text: str, extra_tokens: tuple[str, ...] = ()) -> str:
    """Replace every volatile substring in ``text`` with a placeholder."""
    for token in extra_tokens:
        if token:
            text = text.replace(token, "<model>")
    for pattern, replacement in _SCRUBBERS:
        text = pattern.sub(replacement, text)
    return text


def normalize_identifier(raw: str, extra_tokens: tuple[str, ...]) -> str:
    for prefix, placeholder in _DYNAMIC_ID_PREFIXES:
        if raw.startswith(prefix):
            return prefix + placeholder
    return scrub(raw, extra_tokens)


def value_kind(record: dict) -> str | None:
    """Reduce AXValue to its control kind.

    Booleans keep their state — a toggle flipping is exactly the class of
    regression these baselines exist to catch. Every other value collapses to
    ``number`` / ``text`` / ``empty`` because the content is data: a token
    rate, a splitter offset in points, a streamed assistant turn.
    """
    if "value" not in record:
        return None
    value = record["value"]
    role = record.get("role", "")
    subrole = record.get("subrole", "")
    toggle = role in _TOGGLE_ROLES or subrole in _TOGGLE_SUBROLES
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, (int, float)):
        if toggle and value in (0, 1):
            return f"bool:{str(bool(value)).lower()}"
        return "number"
    if isinstance(value, str):
        return "text" if value.strip() else "empty"
    return "other"


def build_tree(records: list[dict]) -> Node | None:
    """Rebuild the parent/child tree from the flat pre-order dump."""
    root: Node | None = None
    stack: list[Node] = []
    for record in records:
        depth = record.get("depth")
        if not isinstance(depth, int) or depth < 0:
            continue
        node = Node(record)
        if depth == 0:
            if root is None:
                root = node
                stack = [node]
            continue
        if depth > len(stack):
            # Malformed / truncated dump: the walker caps depth and record
            # count, so a subtree can be cut mid-descent. Attach to the
            # deepest known parent rather than dropping the node silently.
            depth = len(stack)
        parent = stack[depth - 1]
        parent.children.append(node)
        del stack[depth:]
        stack.append(node)
    return root


def window_sort_key(node: Node) -> tuple[str, str, str, str]:
    record = node.record
    # Sort on exactly what gets RENDERED, including the title rule in
    # `render_node`. Ordering windows by an attribute the baseline then hides
    # would let two machines emit identical lines in a different order — the
    # same un-regenerable baseline this normalizer exists to prevent, only
    # harder to read.
    title = "" if record.get("description") else (record.get("title", "") or "")
    return (
        record.get("identifier", "") or "",
        title,
        record.get("subrole", "") or "",
        record.get("role", "") or "",
    )


def quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


# The footer version pill states its own version AND the updater's verdict
# about it ("· up to date" / "· update X.Y.Z available" / bare when the
# check is inconclusive), with the tooltip phrased around the same three
# cases. That verdict compares the running build against the latest
# PUBLISHED release — network state, not UI state — and it INVERTS on
# every version-bump PR: the bumped app is newer than any release until
# the release it is cutting exists, so the pill drops to its unknown
# state and every baseline holding "up to date" goes red for no product
# reason (first hit: the 0.12.11 bump).
#
# Collapsed by IDENTIFIER rather than by a text scrubber so the same
# phrasing elsewhere — notably Settings → App, whose version line the
# flows assert against separately — keeps its own value. What survives:
# the pill exists, is a button, carries this identifier, is enabled. Its
# three-state wording is covered by DesktopVersionPillTests.
_UPDATE_VERDICT_IDS = frozenset({"Footer.DesktopVersionPill"})
_UPDATE_VERDICT_TEXT = "Rapid-MLX <version> <update-state>"


def render_node(node: Node, extra_tokens: tuple[str, ...]) -> str:
    record = node.record
    if is_gpu_telemetry(record):
        # One canonical line for both platform shapes (see is_gpu_telemetry).
        parts = ["AXUnknown", 'desc="GPU <gpu>"']
        if "enabled" in record:
            parts.append(f"enabled={str(bool(record['enabled'])).lower()}")
        return " ".join(parts)
    raw_role = record.get("role", "AXUnknown")
    parts = [_ROLE_EQUIVALENTS.get(raw_role, raw_role)]
    subrole = record.get("subrole")
    if subrole:
        parts.append(f"subrole={subrole}")
    identifier = record.get("identifier")
    if identifier:
        normalized = normalize_identifier(identifier, extra_tokens)
        if normalized:
            parts.append(f"id={quote(normalized)}")
    # AXTitle is dropped whenever AXDescription is published on the same
    # element.
    #
    # SwiftUI's accessibilityLabel lands in AXDescription: that is the label
    # the APP authored. AppKit may additionally synthesise an AXTitle for the
    # same control out of how it happens to be drawn, and that synthesis is
    # not stable across macOS releases. Measured on one commit, two machines:
    # the conversation "..." menu publishes
    #     AXPopUpButton id="Sidebar.Conversation.Menu.<uuid>"
    #         title="More" desc="Conversation actions"
    # on macOS 15 (hosted runner) and the same line WITHOUT the title on
    # macOS 26 (dev machine). Pinning that makes a baseline un-regenerable in
    # both directions — whoever writes it turns the other OS red, and
    # `--update-baselines` becomes a trap rather than a tool.
    #
    # Deliberately narrow. A title with NO description beside it is the only
    # label the element has and is kept, which preserves every title the
    # committed baselines actually rely on: AXApplication and AXWindow
    # titles, and controls such as `Settings.ModelManagement.SortMenu`
    # (title="Sort", no description) that are otherwise indistinguishable
    # from a neighbouring popup.
    #
    # What this gives up, stated plainly: a change to a control's VISIBLE text
    # while its accessibility label stays put. That was never this file's job
    # — see the scope note at the top of the module; visible text belongs to
    # the PNG snapshots in Tests/RapidTests.
    description = record.get("description")
    verdict_bearing = identifier in _UPDATE_VERDICT_IDS
    for key, label in (("title", "title"), ("description", "desc"), ("help", "help")):
        if key == "title" and description:
            continue
        text = record.get(key)
        if text:
            value = (
                _UPDATE_VERDICT_TEXT if verdict_bearing else scrub(text, extra_tokens)
            )
            parts.append(f"{label}={quote(value)}")
    kind = value_kind(record)
    if kind is not None:
        parts.append(f"value={kind}")
    if "enabled" in record:
        parts.append(f"enabled={str(bool(record['enabled'])).lower()}")
    return " ".join(parts)


def render(root: Node, extra_tokens: tuple[str, ...]) -> list[str]:
    lines: list[str] = []

    def walk(
        node: Node, depth: int, sort_children: bool, parent_is_toolbar: bool
    ) -> None:
        lines.append("  " * depth + render_node(node, extra_tokens))
        if is_window_control(node.record) or (
            parent_is_toolbar and is_lazy_button_wrapper(node)
        ):
            # The traffic-light buttons are AppKit's, not ours. Their anonymous
            # AXGroup descendants are lazily realized: two dumps taken seconds
            # apart in the SAME run recorded one group under AXZoomButton in
            # settings-root and two in models-idle. Keeping the button (so a
            # missing close box is still a diff) and dropping its private
            # innards removes a whole class of false failure that no product
            # change could ever cause. The same holds for a toolbar button whose
            # only child is macOS 15's lazily-realized copy of itself — scoped to
            # a button OWNED by the toolbar so an identical-identity nesting
            # deeper in the tree still surfaces as a diff (see
            # ``is_lazy_button_wrapper``).
            #
            # The dropped copy carries the button's tooltip as ``AXHelp``, and it
            # is dropped, not lifted onto the parent, ON PURPOSE: macOS 26 does
            # not publish that tooltip on the toolbar button at all (the flat
            # button has ``help=None`` in the fixtures), so lifting it would make
            # the two releases disagree on ``help`` and reintroduce exactly the
            # un-regenerable baseline this normalizer exists to prevent. A tooltip
            # that only one OS exposes is not cross-OS structure; the PNG
            # snapshots under ``Tests/RapidTests/__Snapshots__`` remain the check
            # for user-visible tooltip text.
            return
        children = node.children
        if sort_children:
            # Only the window level is reordered. AX hands back the app's
            # windows in z-order, so merely focusing Settings would otherwise
            # rewrite the baseline. Deeper sibling order is the view order and
            # is exactly what we want to detect changing.
            children = sorted(children, key=window_sort_key)
        # AppKit's lazy button self-copy sits DIRECTLY under the toolbar, so the
        # collapse in ``is_lazy_button_wrapper`` is armed only for a toolbar's
        # own children, never for buttons deeper in the subtree.
        node_is_toolbar = node.record.get("role") == "AXToolbar"
        if node_is_toolbar:
            children = normalize_toolbar_children(children)
        for child in children:
            walk(
                child, depth + 1, sort_children=False, parent_is_toolbar=node_is_toolbar
            )

    walk(root, 0, sort_children=True, parent_is_toolbar=False)
    return lines


def normalize_dump(path: Path, extra_tokens: tuple[str, ...]) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("data", {}).get("ui_elements", [])
    if not records:
        raise SystemExit(f"ax-baseline: no ui_elements in {path}")
    root = build_tree(records)
    if root is None:
        raise SystemExit(f"ax-baseline: no root element in {path}")
    return [SCHEMA_HEADER, *render(root, extra_tokens)]


def describe_diff(baseline: list[str], observed: list[str]) -> list[str]:
    """A change list a reviewer can read, not a JSON dump.

    Every hunk is labelled with what actually happened to the tree — a node
    appeared, disappeared, or changed in place — with the baseline line number
    so it can be found in the committed file.
    """
    report: list[str] = []
    matcher = difflib.SequenceMatcher(a=baseline, b=observed, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            report.append(f"  changed at baseline line {i1 + 1}:")
            for line in baseline[i1:i2]:
                report.append(f"    - {line}")
            for line in observed[j1:j2]:
                report.append(f"    + {line}")
        elif tag == "delete":
            report.append(f"  disappeared at baseline line {i1 + 1}:")
            for line in baseline[i1:i2]:
                report.append(f"    - {line}")
        elif tag == "insert":
            report.append(f"  appeared after baseline line {i1}:")
            for line in observed[j1:j2]:
                report.append(f"    + {line}")
    return report


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_normalize(args: argparse.Namespace) -> int:
    lines = normalize_dump(Path(args.dump), tuple(args.scrub))
    if args.output:
        write_lines(Path(args.output), lines)
    else:
        sys.stdout.write("\n".join(lines) + "\n")
    return 0


def command_check(args: argparse.Namespace) -> int:
    dump = Path(args.dump)
    baseline_path = Path(args.baseline)
    observed = normalize_dump(dump, tuple(args.scrub))

    if args.observed:
        write_lines(Path(args.observed), observed)

    if args.update:
        verb = "updated" if baseline_path.exists() else "recorded"
        write_lines(baseline_path, observed)
        print(f"ax-baseline: {verb} {baseline_path} ({len(observed) - 1} elements)")
        return 0

    if not baseline_path.exists():
        # Recording on absence would make a typo'd snapshot name — or a
        # baseline someone forgot to commit — pass CI green while comparing
        # against nothing. A gate that cannot fail is not a gate.
        print(
            f"ax-baseline: no committed baseline for {baseline_path.stem}",
            file=sys.stderr,
        )
        print(f"  expected: {baseline_path}", file=sys.stderr)
        if args.observed:
            print(f"  observed: {args.observed}", file=sys.stderr)
        print(
            "\n  If this snapshot is new, record it with --update-baselines "
            "and commit the result.",
            file=sys.stderr,
        )
        return 1

    expected = baseline_path.read_text(encoding="utf-8").splitlines()
    if expected == observed:
        return 0

    print(f"ax-baseline: structural mismatch for {baseline_path.stem}", file=sys.stderr)
    print(f"  baseline: {baseline_path}", file=sys.stderr)
    if args.observed:
        print(f"  observed: {args.observed}", file=sys.stderr)
    print(f"  source dump: {dump}", file=sys.stderr)
    for line in describe_diff(expected, observed):
        print(line, file=sys.stderr)
    print(
        "\n  If this is an intended UI change, re-run with --update-baselines "
        "and commit the new baseline.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ax-baseline.py",
        description="Normalize and compare rapid-ax AX dumps.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("dump", help="rapid-ax dump JSON")
    common.add_argument(
        "--scrub",
        action="append",
        default=[],
        metavar="TOKEN",
        help="literal token to replace with <model> (repeatable)",
    )

    normalize = sub.add_parser(
        "normalize", parents=[common], help="print the normalized tree"
    )
    normalize.add_argument("-o", "--output", help="write to a file instead of stdout")
    normalize.set_defaults(func=command_normalize)

    check = sub.add_parser(
        "check", parents=[common], help="compare a dump against a committed baseline"
    )
    check.add_argument("--baseline", required=True, help="committed baseline file")
    check.add_argument("--observed", help="also write the normalized tree here")
    check.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline instead of comparing",
    )
    check.set_defaults(func=command_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
