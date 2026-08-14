# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the rapid-mac accessibility-identifier gate.

Pure-logic — no git, no network, no Swift toolchain, no GPU. The gate is a lint
over Swift source text, so everything here is a string in and a list of
violations out. The last section runs the real ``apps/rapid-mac/Sources`` tree
through the masker to catch a lexer desync on Swift the fixtures do not model.
"""

from __future__ import annotations

import importlib.util
import re
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


def _hunk(base_start, base_count, head_start, head_count) -> gate.Hunk:
    return gate.Hunk(base_start, base_count, head_start, head_count)


def test_carried_violation_is_suppressed_across_a_reindent():
    """The declaration line is 'added' by the diff, but it is the same control:
    the very hunk that added it is the one that removed the original."""
    base = [_violation(10, 'Button("Save") { save() }')]
    head = [_violation(42, 'Button("Save")   { save() }')]
    assert gate.suppress_carried(head, base, [_hunk(10, 1, 42, 1)]) == []


def test_untouched_violations_are_never_reported_here():
    """Nothing outside a hunk is reported by this function; whether an untouched
    control was just unlabelled is _identifier_removals' question."""
    base = [_violation(10, 'Button("Save") { save() }')]
    head = [_violation(10, 'Button("Save") { save() }')]
    assert gate.suppress_carried(head, base, []) == []


def test_duplicating_a_carried_violation_reports_the_copy():
    """The hunk that added line 24 removed nothing, so nothing funds the copy."""
    base = [_violation(10, 'Button("Save") { save() }')]
    head = [
        _violation(10, 'Button("Save") { save() }'),  # untouched original
        _violation(24, 'Button("Save") { save() }'),  # the pasted copy
    ]
    kept = gate.suppress_carried(head, base, [_hunk(23, 0, 24, 1)])
    assert [v.line for v in kept] == [24]


def test_copy_pasted_above_the_original_still_reports():
    """Regression: a file-wide budget let the added copy spend it, leaving the
    untouched original to absorb the blame and the diff to pass."""
    base = [_violation(30, 'Button("Save") { save() }')]
    head = [
        _violation(10, 'Button("Save") { save() }'),  # the pasted copy, added
        _violation(30, 'Button("Save") { save() }'),  # untouched original
    ]
    kept = gate.suppress_carried(head, base, [_hunk(9, 0, 10, 1)])
    assert [v.line for v in kept] == [10]


def test_a_genuinely_new_control_survives_suppression():
    base = [_violation(10, 'Button("Save") { save() }')]
    head = [
        _violation(10, 'Button("Save") { save() }'),
        _violation(42, 'Button("Delete") { delete() }'),
    ]
    kept = gate.suppress_carried(head, base, [_hunk(41, 0, 42, 1)])
    assert [v.line for v in kept] == [42]


def test_fixing_one_gap_does_not_fund_a_new_one_elsewhere():
    """Labelling the Save button at line 10 removes a violation from the
    file-wide tally; a file-wide budget then absorbed the brand-new unlabelled
    Save button at line 80, and the PR came out green having both fixed and
    broken one. Budget is local to the hunk that freed it."""
    base = [_violation(10, 'Button("Save") { save() }')]
    head = [_violation(80, 'Button("Save") { save() }')]
    hunks = [
        _hunk(10, 1, 10, 1),  # the fix: the line rewritten with an identifier
        _hunk(79, 0, 80, 1),  # the regression: a new unlabelled control
    ]
    assert [v.line for v in gate.suppress_carried(head, base, hunks)] == [80]


def test_same_hunk_laundering_is_a_known_blind_spot():
    """Pinned, not endorsed. Inside ONE hunk carry-over is still a count, so
    replacing an unlabelled control with a labelled one AND a new unlabelled one
    nets to zero. Telling that apart from "the same control moved" needs
    per-line identity a text diff does not carry; the file-wide version of the
    hole — a fix anywhere funding a regression anywhere — is closed above.
    Change this test if the gate ever learns to tell them apart."""
    base = [_violation(10, 'Button("Save") { save() }')]
    head = [_violation(11, 'Button("Save") { save() }')]
    assert gate.suppress_carried(head, base, [_hunk(10, 1, 10, 2)]) == []


def test_editing_a_comment_does_not_rename_a_control():
    """Identity was the raw declaration line, so re-wording a trailing comment
    made a years-old unlabelled control read as brand new. Comments have no
    bearing on whether AXPress can reach it."""
    src_base = """struct V: View {
    var body: some View {
        Button("Save") { save() }  // TODO: wire this up
    }
}
"""
    src_head = src_base.replace("// TODO: wire this up", "// FIXME: still pending")
    base = gate.find_violations("Fixture.swift", src_base)
    head = gate.find_violations("Fixture.swift", src_head)
    assert [v.line for v in base] == [3] and [v.line for v in head] == [3]
    assert gate.suppress_carried(head, base, [_hunk(3, 1, 3, 1)]) == []


def test_a_comment_that_starts_the_line_is_kept_in_the_identity():
    """Truncating there would collapse the identity to the empty string, which
    every other control would then match."""
    v = gate.find_violations(
        "Fixture.swift",
        """struct V: View {
    var body: some View {
        /* x */ Button("Save") { save() }
    }
}
""",
    )
    assert v and "Button" in v[0].key[1]


# --------------------------------------------------------------------------
# Identifier removal: taking a label off is as damaging as never adding one.
# --------------------------------------------------------------------------


def test_deleting_an_identifier_is_reported_though_no_line_was_added():
    """The declaration line does not change when the modifier lived on its own
    line, so an added-lines-only filter never even looks at it."""
    head = [_violation(3, 'Button("Save") { save() }')]
    # The base's only violation was at line 9 — NOT the control at head line 3,
    # which was labelled and no longer is.
    base = [_violation(9, 'Button("Other") { other() }')]
    removals = gate._identifier_removals(head, base, [_hunk(4, 1, 4, 0)], set())
    assert [v.line for v in removals] == [3]
    assert gate.IDENTIFIER_MODIFIER in removals[0].reason


def test_an_untouched_carry_over_is_not_read_as_a_removal():
    head = [_violation(3, 'Button("Save") { save() }')]
    base = [_violation(3, 'Button("Save") { save() }')]
    assert gate._identifier_removals(head, base, [], set()) == []


def test_a_gap_on_the_same_line_does_not_vouch_for_its_neighbour():
    """One line can hold two controls — ``Menu("x") { Button("y") { … } }``.
    Matching on the line number alone let the pre-existing Button gap mark the
    line "known bad", so deleting the Menu's identifier was suppressed."""
    line = 'Menu("Actions") { Button("Rename") { rename() } }'
    base = [gate.Violation("F.swift", 3, "Button", line, "x")]
    head = [
        gate.Violation("F.swift", 3, "Button", line, "x"),  # the old gap
        gate.Violation("F.swift", 3, "Menu", line, "x"),  # just unlabelled
    ]
    removals = gate._identifier_removals(head, base, [_hunk(4, 1, 4, 0)], set())
    assert [v.kind for v in removals] == ["Menu"]


def test_head_line_to_base_undoes_the_shift_of_earlier_hunks():
    assert gate.head_line_to_base([_hunk(4, 0, 5, 2)], 20) == 18  # 2 inserted
    assert gate.head_line_to_base([_hunk(5, 2, 4, 0)], 20) == 22  # 2 deleted


def test_a_pure_deletion_does_not_shift_the_line_it_follows():
    """``@@ -7,2 +6,0 @@`` — head 6 is the line the deletion FOLLOWS, not a line
    the hunk covers. Shifting it mapped a surviving carried-over violation to the
    wrong base line, found nothing there, and reported it as an identifier this
    diff had removed: deleting any labelled control lit up the control above it.
    """
    deletion = [_hunk(7, 2, 6, 0)]
    assert gate.head_line_to_base(deletion, 6) == 6  # before the cut: unmoved
    assert gate.head_line_to_base(deletion, 7) == 9  # after it: shifted by 2


def test_deleting_a_labelled_control_does_not_blame_its_neighbour():
    head = [_violation(6, 'Button("Save") { save() }')]
    base = [_violation(6, 'Button("Save") { save() }')]
    removals = gate._identifier_removals(head, base, [_hunk(7, 2, 6, 0)], set())
    assert removals == []


# --------------------------------------------------------------------------
# A child's identifier must not satisfy its unlabelled parent.
# --------------------------------------------------------------------------


def test_a_child_identifier_does_not_label_its_parent():
    """The parent's postfix span includes its trailing closure, so a search over
    the whole span found the CHILD's identifier and passed the Menu. AXPress on
    the Menu still has nothing to aim at."""
    src = """struct V: View {
    var body: some View {
        Menu("Actions") {
            Button("Rename") { rename() }
                .accessibilityIdentifier("Sidebar.Rename")
        }
    }
}
"""
    assert [(v.kind, v.line) for v in gate.find_violations("F.swift", src)] == [
        ("Menu", 3)
    ]


def test_the_parents_own_modifier_chain_still_counts():
    src = """struct V: View {
    var body: some View {
        Menu("Actions") {
            Button("Rename") { rename() }
                .accessibilityIdentifier("Sidebar.Rename")
        }
        .menuStyle(.borderlessButton)
        .accessibilityIdentifier("Sidebar.Actions")
    }
}
"""
    assert gate.find_violations("F.swift", src) == []


def test_an_identifier_five_modifiers_later_still_counts():
    src = """struct V: View {
    var body: some View {
        Button("Save") { save() }
            .buttonStyle(.borderless)
            .disabled(false)
            .help("Save it")
            .keyboardShortcut("s")
            .accessibilityIdentifier("Toolbar.Save")
    }
}
"""
    assert gate.find_violations("F.swift", src) == []


def test_a_swiftui_qualified_control_is_still_a_control():
    """``SwiftUI.Button(…)`` IS the control; the qualifier-rejecting lookbehind
    let a fully qualified construction walk straight past the gate."""
    src = """\
struct V: View {
    var body: some View {
        SwiftUI.Button("Delete") { delete() }
    }
}
"""
    assert [(v.kind, v.line) for v in gate.find_violations("F.swift", src)] == [
        ("Button", 3)
    ]


def test_another_namespaces_button_is_still_not_a_control():
    src = """\
struct V: View {
    var body: some View {
        Chrome.Button("Delete") { delete() }
    }
}
"""
    assert gate.find_violations("F.swift", src) == []


def test_bare_grouping_parentheses_are_a_known_blind_spot():
    """Pinned, not endorsed. The balanced walk stops at the enclosing ``)``, so
    an identifier attached to a parenthesised control is missed and the control
    is reported. Two attempts to make those parens transparent both produced
    something worse — the first credited ``Card(Button(…))``'s identifier to the
    Button, the second turned ``return (Button(…))`` into a false alarm — because
    "is this paren an argument list or grouping?" cannot be answered from the
    character in front of it: a closure call ends in ``}``, and ``return`` ends
    in a letter. A gate that reports an unusual shape is recoverable in seconds;
    one that credits a wrapper's identifier to the control inside it is the
    exact defect the gate exists to catch. So the walk stays literal, and the
    fix is to attach the identifier to the control itself."""
    src = """\
struct V: View {
    var body: some View {
        (
            Button("Save") { save() }
        )
        .accessibilityIdentifier("Toolbar.Save")
    }
}
"""
    assert [(v.kind, v.line) for v in gate.find_violations("F.swift", src)] == [
        ("Button", 4)
    ]


def test_a_wrappers_identifier_is_never_credited_to_the_control_it_holds():
    """The direction that must never regress: an identifier on the enclosing
    thing gives ``AXPress`` nothing to press."""
    for wrapper in [
        'Card(Button("Save") { save() })',
        '{ child in HStack { child } }(Button("Save") { save() })',
    ]:
        src = f"""\
struct V: View {{
    var body: some View {{
        {wrapper}
            .accessibilityIdentifier("Row")
    }}
}}
"""
        assert [v.kind for v in gate.find_violations("F.swift", src)] == ["Button"], (
            wrapper
        )


def test_a_qualified_control_with_spaces_around_the_dot_is_still_theirs():
    """Whitespace may surround Swift member access, so a lookbehind cannot
    decide the qualifier: ``Chrome . Button`` matched at ``Button`` with only a
    space behind it and was falsely blocked."""
    for spelling in ["Chrome.Button", "Chrome . Button", "Chrome\n            .Button"]:
        src = f"""\
struct V: View {{
    var body: some View {{
        {spelling}("Delete") {{ delete() }}
    }}
}}
"""
        assert gate.find_violations("F.swift", src) == [], spelling


def test_generic_and_init_constructions_are_still_constructions():
    """A name-then-delimiter pattern walked straight past both of these, so a
    PR could add an unreachable SwiftUI button and the gate stayed green."""
    for spelling in [
        'Button<Text>(action: save) { Text("Save") }',
        'Button.init("Save", action: save)',
        'SwiftUI.Button.init("Save", action: save)',
        'Button<Label<Text, Image>>(action: save) { Text("Save") }',
    ]:
        src = f"""\
struct V: View {{
    var body: some View {{
        {spelling}
    }}
}}
"""
        assert [v.kind for v in gate.find_violations("F.swift", src)] == ["Button"], (
            spelling
        )


def test_a_generic_return_type_is_not_a_construction():
    """``private var saveButton: Button<Text> {`` is a return type followed by a
    property body. Reading that ``{`` as a constructor reported a phantom
    unlabelled Button and blocked a PR whose real control was labelled."""
    src = """\
struct V: View {
    private var saveButton: Button<Text> {
        Button("Save") { save() }
            .accessibilityIdentifier("Toolbar.Save")
    }
    var body: some View { saveButton }
}
"""
    assert gate.find_violations("F.swift", src) == []


def test_a_generic_type_reference_is_still_not_a_construction():
    """The trailing delimiter is what separates the two, and it has to keep
    doing that now that generics are allowed in between."""
    src = """\
struct V: View {
    let style: Button<Text>.Type = Button<Text>.self
    var body: some View { EmptyView() }
}
"""
    assert gate.find_violations("F.swift", src) == []


def test_first_party_controls_are_in_the_inventory():
    """Omitting these made the docstring's claim — that everything undetected
    is listed under "Known blind spots" — untrue. ShareLink is an ordinary
    button to a user."""
    for kind in ("ShareLink", "PasteButton", "EditButton", "MultiDatePicker"):
        assert kind in gate.CONTROL_KINDS, kind
    src = """\
struct V: View {
    var body: some View {
        ShareLink(item: reportURL)
    }
}
"""
    assert [v.kind for v in gate.find_violations("F.swift", src)] == ["ShareLink"]


def test_a_nested_namespace_ending_in_swiftui_is_still_theirs():
    """Reading only the qualifier's last component made ``Chrome.SwiftUI.Button``
    look like the real control and falsely blocked it."""
    src = """\
struct V: View {
    var body: some View {
        Chrome.SwiftUI.Button("Delete") { delete() }
    }
}
"""
    assert gate.find_violations("F.swift", src) == []


def test_swiftui_qualified_with_spaces_is_still_a_control():
    src = """\
struct V: View {
    var body: some View {
        SwiftUI . Button("Delete") { delete() }
    }
}
"""
    assert [v.kind for v in gate.find_violations("F.swift", src)] == ["Button"]


def test_a_real_argument_list_is_not_treated_as_grouping():
    """Only parens holding nothing but the control are transparent."""
    src = """\
struct V: View {
    var body: some View {
        wrap(Button("Save") { save() }, other)
            .accessibilityIdentifier("Toolbar.Wrapper")
    }
}
"""
    assert [(v.kind, v.line) for v in gate.find_violations("F.swift", src)] == [
        ("Button", 3)
    ]


def test_rewording_a_leading_block_comment_does_not_rename_the_control():
    """The earlier compromise kept the whole line when a comment LED it, so
    re-wording that comment reported a long-standing gap as brand new."""
    src_base = """\
struct V: View {
    var body: some View {
        /* old explanation */ Button("Save") { save() }
    }
}
"""
    src_head = src_base.replace("old explanation", "a different note entirely")
    base = gate.find_violations("F.swift", src_base)
    head = gate.find_violations("F.swift", src_head)
    assert [v.line for v in base] == [3] and [v.line for v in head] == [3]
    assert base[0].key == head[0].key
    assert gate.suppress_carried(head, base, [_hunk(3, 1, 3, 1)]) == []


def test_stripping_comments_still_leaves_a_distinguishing_identity():
    """Deleting only the comment's columns, rather than truncating the line, is
    what keeps two different controls from collapsing to the same key."""
    src = """\
struct V: View {
    var body: some View {
        /* a */ Button("Save") { save() }
        /* b */ Button("Cancel") { cancel() }
    }
}
"""
    keys = [v.key for v in gate.find_violations("F.swift", src)]
    assert len(keys) == 2 and keys[0] != keys[1]
    assert all("/*" not in k[1] for k in keys)


def test_disclosure_group_is_a_control():
    src = """struct V: View {
    var body: some View {
        DisclosureGroup("Advanced") { Text("x") }
    }
}
"""
    assert [(v.kind, v.line) for v in gate.find_violations("F.swift", src)] == [
        ("DisclosureGroup", 3)
    ]


# --------------------------------------------------------------------------
# The real tree. Lexer soak plus the paid-down backlog contract.
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


def test_shipping_tree_has_no_unlabelled_native_controls():
    """The diff gate deliberately grandfathered the old backlog. That backlog
    is now zero, so keep it zero: every shipping native SwiftUI control must be
    reachable by the assembled-app AX harness, not only controls added by the
    current PR."""
    violations = []
    for path in _real_sources():
        violations.extend(
            gate.find_violations(str(path), path.read_text(encoding="utf-8"))
        )
    assert violations == [], "\n" + "\n".join(str(v) for v in violations)


def test_shipping_control_wrappers_are_identified_at_each_call_site():
    """Native controls inside these wrappers are intentionally identified by
    the caller, where the surface/entity name is known. Pin that second half of
    the contract so the wrapper exemptions cannot hide an unreachable use."""
    wrapper = re.compile(
        r"(?<![A-Za-z0-9_])(QuietIconButton|SheetCloseButton|RapidTextField)\s*([({])"
    )
    missing = []
    for path in _real_sources():
        src = path.read_text(encoding="utf-8")
        masked, _, _ = gate._mask(src)
        for match in wrapper.finditer(masked):
            delimiter = match.end() - 1
            end = gate._expression_end(masked, delimiter)
            if not gate._has_own_identifier(masked, delimiter, end):
                line = masked.count("\n", 0, match.start()) + 1
                missing.append(f"{path}:{line}: {match.group(1)}")
    assert missing == [], "\n" + "\n".join(missing)
