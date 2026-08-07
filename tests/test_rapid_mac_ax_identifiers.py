# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the rapid-mac accessibility-identifier gate.

Pure-logic — no git, no network, no Swift toolchain, no GPU. The gate is a lint
over Swift source text, so everything here is a string in and a list of
violations out. The last section runs the real ``apps/rapid-mac/Sources`` tree
through the masker to catch a lexer desync on Swift the fixtures do not model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load the script directly (scripts/ is not an importable package). Registering
# it in sys.modules is load-bearing: the module defines a @dataclass, and
# dataclasses resolves field types through sys.modules[cls.__module__].
_SPEC = importlib.util.spec_from_file_location(
    "check_rapid_mac_ax_identifiers",
    _REPO_ROOT / "scripts" / "check_rapid_mac_ax_identifiers.py",
)
gate = importlib.util.module_from_spec(_SPEC)
sys.modules["check_rapid_mac_ax_identifiers"] = gate
_SPEC.loader.exec_module(gate)


def flags(src: str) -> list[tuple[str, int]]:
    """(kind, line) for every violation the gate reports in ``src``."""
    return [(v.kind, v.line) for v in gate.find_violations("Fixture.swift", src)]


# --------------------------------------------------------------------------
# The core contract: added interactive control without an identifier.
# --------------------------------------------------------------------------


def test_unlabelled_button_is_flagged():
    src = """\
struct Panel: View {
    var body: some View {
        Button("Save") { save() }
    }
}
"""
    assert flags(src) == [("Button", 3)]


def test_identifier_directly_on_the_control_passes():
    src = """\
struct Panel: View {
    var body: some View {
        Button("Save") { save() }
            .accessibilityIdentifier("Settings.Tools.Save")
    }
}
"""
    assert flags(src) == []


def test_identifier_further_down_the_modifier_chain_passes():
    """The whole point of walking the chain instead of grepping one line."""
    src = """\
struct Panel: View {
    var body: some View {
        Toggle(isOn: $on) {
            VStack {
                Text("Approve every page automatically")
                Text("Skips the confirmation for unattended use.")
            }
        }
        .toggleStyle(TrailingSettingsToggleStyle())
        .disabled(locked)
        .help("Browsing approval")
        .accessibilityLabel("Approve every page automatically")
        .accessibilityHint("Applies to the browse tool only.")
        .accessibilityIdentifier("Settings.Tools.BrowseAutoApproveToggle")
    }
}
"""
    assert flags(src) == []


def test_identifier_on_a_sibling_does_not_cover_this_control():
    """Over-consuming the chain would silently credit the wrong control."""
    src = """\
struct Panel: View {
    var body: some View {
        HStack {
            Button("Save") { save() }
            Button("Cancel") { cancel() }
                .accessibilityIdentifier("Settings.Tools.Cancel")
        }
    }
}
"""
    assert flags(src) == [("Button", 4)]


def test_identifier_on_the_container_does_not_cover_the_control():
    """Documented behaviour: AXPress needs the control, not its HStack."""
    src = """\
struct Panel: View {
    var body: some View {
        HStack {
            Button("Save") { save() }
        }
        .accessibilityIdentifier("Settings.Tools.Row")
    }
}
"""
    assert flags(src) == [("Button", 4)]


def test_every_declared_control_kind_is_detected():
    body = "\n".join(f"        {kind}()" for kind in gate.CONTROL_KINDS)
    src = f"struct Panel: View {{\n    var body: some View {{\n{body}\n    }}\n}}\n"
    assert [k for k, _ in flags(src)] == list(gate.CONTROL_KINDS)


def test_static_and_decorative_views_are_out_of_scope():
    src = """\
struct Panel: View {
    var body: some View {
        VStack {
            Text("Tools")
            Image(systemName: "wrench")
            Spacer()
            Divider()
            ProgressView()
        }
    }
}
"""
    assert flags(src) == []


def test_multiple_trailing_closures_keep_one_chain():
    """`Menu { … } label: { … }` is one expression, identifier at the end."""
    src = """\
struct Panel: View {
    var body: some View {
        Menu {
            Button("Rename") { rename() }
                .accessibilityIdentifier("Sidebar.Row.Rename")
        } label: {
            Image(systemName: "ellipsis")
        }
        .menuStyle(.borderlessButton)
        .accessibilityIdentifier("Sidebar.Row.Menu")
    }
}
"""
    assert flags(src) == []


# --------------------------------------------------------------------------
# Shapes that must NOT be mistaken for a control (false-positive guards).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "SendButton(action: send)",  # a custom view whose name ends in Button
        "MyToggleGroup(items: items)",
        "        .buttonStyle(.plain)",
        "        .pickerStyle(.radioGroup)",
        "        .menuStyle(.borderlessButton)",
        "    let style: Button<Text>.Type = Button<Text>.self",
        "struct Button: View { var body: some View { EmptyView() } }",
        "        Chrome.Button.render()",
    ],
)
def test_non_construction_tokens_are_not_controls(line):
    src = f"struct Panel: View {{\n    var body: some View {{\n{line}\n    }}\n}}\n"
    assert flags(src) == []


def test_controls_named_inside_comments_and_strings_are_ignored():
    src = '''\
struct Panel: View {
    /// Historically this row shipped a `Button("Retry") { retry() }` with no
    /// identifier — see docs/userflows.md.
    var body: some View {
        // Button("Ghost") { }
        Text("Button(\\"Quoted\\") { }")
        Text("""
            Button("Fenced") {
                still not code
            }
            """)
    }
}
'''
    assert flags(src) == []


def test_raw_string_delimiters_do_not_desync_the_lexer():
    src = """\
struct Panel: View {
    var body: some View {
        Text(#"a raw "quoted" Button("Nope") { } string"#)
        Button("Real") { fire() }
    }
}
"""
    assert flags(src) == [("Button", 4)]


def test_string_interpolation_containing_braces_is_survivable():
    src = """\
struct Panel: View {
    var body: some View {
        Text("Model: \\(alias.isEmpty ? "none" : alias) ready")
        Button("Real") { fire() }
    }
}
"""
    assert flags(src) == [("Button", 4)]


def test_nested_block_comments_are_survivable():
    src = """\
struct Panel: View {
    var body: some View {
        /* outer /* inner Button("Ghost") { } */ still comment */
        Button("Real") { fire() }
    }
}
"""
    assert flags(src) == [("Button", 4)]


def test_command_menu_items_are_out_of_scope():
    """rapid-ax.swift never walks the menu bar; an id there is decoration."""
    src = """\
struct RapidApp: App {
    var body: some Scene {
        Window("Rapid-MLX", id: "main") { ContentView() }
            .commands {
                CommandGroup(replacing: .appSettings) {
                    Button("Settings…") { openSettings() }
                }
                CommandMenu("Conversation") {
                    Button("Open in New Window") { pop() }
                }
            }
    }
}
"""
    assert flags(src) == []


def test_style_makebody_is_out_of_scope():
    """An id here would be stamped on every caller adopting the style."""
    src = """\
struct TrailingSettingsToggleStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        Toggle(isOn: binding) { configuration.label }
            .toggleStyle(.switch)
    }
}
"""
    assert flags(src) == []


def test_style_makebody_scope_closes_at_its_own_body():
    """The skip must not leak into the next view in the same file."""
    src = """\
struct PlainStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        Button("inner") { }
    }
}

struct Panel: View {
    var body: some View {
        Button("outer") { fire() }
    }
}
"""
    assert flags(src) == [("Button", 9)]


# --------------------------------------------------------------------------
# The escape hatch has to cost something.
# --------------------------------------------------------------------------


def test_exempt_marker_with_a_reason_on_the_same_line_passes():
    src = """\
struct Panel: View {
    var body: some View {
        Button("Allow once") { approve() }  // ax-exempt: confirmationDialog buttons render outside the app's AX tree
    }
}
"""
    assert flags(src) == []


def test_exempt_marker_on_the_line_above_passes():
    src = """\
struct Panel: View {
    var body: some View {
        // ax-exempt: confirmationDialog buttons render outside the app's AX tree
        Button("Allow once") { approve() }
    }
}
"""
    assert flags(src) == []


def test_exempt_marker_two_lines_above_does_not_reach():
    src = """\
struct Panel: View {
    var body: some View {
        // ax-exempt: confirmationDialog buttons render outside the app's AX tree

        Button("Allow once") { approve() }
    }
}
"""
    assert flags(src) == [("Button", 5)]


@pytest.mark.parametrize(
    "marker", ["// ax-exempt:", "// ax-exempt: dialog", "// ax-exempt:  -- "]
)
def test_exempt_marker_without_a_real_reason_is_itself_a_failure(marker):
    src = f"""\
struct Panel: View {{
    var body: some View {{
        {marker}
        Button("Allow once") {{ approve() }}
    }}
}}
"""
    violations = gate.find_violations("Fixture.swift", src)
    assert len(violations) == 1
    assert gate.EXEMPT_MARKER in violations[0].reason


def test_exempt_marker_in_block_comment_form_is_recognised():
    """Regression: a per-character join shredded the marker into letters."""
    src = """\
struct Panel: View {
    var body: some View {
        /* ax-exempt: confirmationDialog buttons live outside the AX tree */
        Button("Allow once") { approve() }
    }
}
"""
    assert flags(src) == []


def test_marker_must_sit_on_the_last_comment_line_before_the_control():
    """The lookup is deliberately narrow — the control's line or the one
    directly above it. A marker stranded on the first line of a multi-line
    comment does not reach, which keeps the rule the same one everyone already
    knows from `# noqa` / `swiftlint:disable:next` instead of a bespoke
    proximity heuristic."""
    stranded = """\
struct Panel: View {
    var body: some View {
        /* ax-exempt: confirmationDialog buttons live outside the AX tree
           so no golden flow can ever press this one. */
        Button("Allow once") { approve() }
    }
}
"""
    assert flags(stranded) == [("Button", 5)]

    adjacent = """\
struct Panel: View {
    var body: some View {
        /* No golden flow can ever press this one:
           ax-exempt: confirmationDialog buttons live outside the AX tree */
        Button("Allow once") { approve() }
    }
}
"""
    assert flags(adjacent) == []


def test_exempt_marker_inside_a_string_is_not_a_marker():
    src = """\
struct Panel: View {
    var body: some View {
        Text("ax-exempt: this is prose, not a suppression")
        Button("Allow once") { approve() }
    }
}
"""
    assert flags(src) == [("Button", 4)]


# --------------------------------------------------------------------------
# Carry-over suppression: reformatting known-bad code must not light up.
# --------------------------------------------------------------------------


def _violation(line: int, source: str) -> gate.Violation:
    return gate.Violation(
        path="Fixture.swift", line=line, kind="Button", source=source, reason="x"
    )


def test_carried_violation_is_suppressed_across_a_reindent():
    """The declaration line is 'added' by the diff, but it is the same control."""
    base = [_violation(10, 'Button("Save") { save() }')]
    head = [_violation(42, 'Button("Save")   { save() }')]
    assert gate.suppress_carried(head, base, {42}) == []


def test_untouched_violations_are_never_reported():
    base = [_violation(10, 'Button("Save") { save() }')]
    head = [_violation(10, 'Button("Save") { save() }')]
    assert gate.suppress_carried(head, base, set()) == []


def test_duplicating_a_carried_violation_reports_the_copy():
    """The base funds ONE unlabelled Save button; the second copy is new."""
    base = [_violation(10, 'Button("Save") { save() }')]
    head = [
        _violation(10, 'Button("Save") { save() }'),  # untouched original
        _violation(24, 'Button("Save") { save() }'),  # the pasted copy
    ]
    kept = gate.suppress_carried(head, base, {24})
    assert [v.line for v in kept] == [24]


def test_copy_pasted_above_the_original_still_reports():
    """Regression: an in-order walk let the added copy spend the base budget,
    leaving the untouched original to absorb the blame and the diff to pass."""
    base = [_violation(30, 'Button("Save") { save() }')]
    head = [
        _violation(10, 'Button("Save") { save() }'),  # the pasted copy, added
        _violation(30, 'Button("Save") { save() }'),  # untouched original
    ]
    kept = gate.suppress_carried(head, base, {10})
    assert [v.line for v in kept] == [10]


def test_a_genuinely_new_control_survives_suppression():
    base = [_violation(10, 'Button("Save") { save() }')]
    head = [
        _violation(10, 'Button("Save") { save() }'),
        _violation(42, 'Button("Delete") { delete() }'),
    ]
    assert [v.line for v in gate.suppress_carried(head, base, {42})] == [42]


# --------------------------------------------------------------------------
# The real tree. Not a coverage assertion — a lexer soak test.
# --------------------------------------------------------------------------


def _real_sources() -> list[Path]:
    root = _REPO_ROOT / gate.SCOPE_PREFIX
    return sorted(root.rglob("*.swift")) if root.is_dir() else []


def test_masker_preserves_offsets_and_line_structure_on_the_real_tree():
    files = _real_sources()
    assert files, "apps/rapid-mac/Sources should contain Swift files"
    for path in files:
        src = path.read_text(encoding="utf-8")
        masked, _ = gate.mask_source(src)
        assert len(masked) == len(src), path
        assert masked.count("\n") == src.count("\n"), path
        # Masking only ever replaces a character with a space.
        assert all(m == c or m == " " for m, c in zip(masked, src)), path


def test_gate_runs_over_the_real_tree_without_exploding():
    """Every reported line must exist, and every control the app already
    labels must stay unreported — the tree is the fixture the gate ships
    against."""
    for path in _real_sources():
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        for v in gate.find_violations(str(path), src):
            assert 1 <= v.line <= len(lines)
            assert v.source == lines[v.line - 1].strip()
            assert ".accessibilityIdentifier" not in v.source
