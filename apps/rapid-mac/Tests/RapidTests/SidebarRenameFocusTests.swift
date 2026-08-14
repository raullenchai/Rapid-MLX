import Foundation
import Testing
@testable import Rapid

/// Regression guard: the sidebar's inline rename editor must actually take
/// keyboard focus.
///
/// As shipped in #1568 the editor had no ``@FocusState`` and no
/// `.focused(...)`, so nothing ever moved first responder to it. Live, that
/// meant: the field appeared but every keystroke went to the chat composer at
/// the bottom of the window (the title the user typed silently became a
/// message draft); `.onSubmit` and `.onExitCommand` — which only fire for the
/// FOCUSED view — were unreachable, so Return could not commit and Escape
/// could not dismiss; and the row stayed in edit mode with no way out. On top
/// of that the bare ``TextField`` is only its intrinsic ~16pt tall inside a
/// 30pt row, so a click on the visible pill mostly landed on dead space,
/// resigning first responder and focusing nothing.
///
/// Focus routing does not appear in a view-model test — the proof of the fix
/// is the live run. These are the cheap tripwires that stop the wiring being
/// deleted again, in the source-grep shape used by
/// ``SidebarConversationDeleteConfirmationTests`` (ViewInspector is not in
/// this target — #1492).
@Suite("Sidebar rename editor takes focus")
struct SidebarRenameFocusTests {
    private static func source(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
        return try String(
            contentsOf: root.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }

    private func sidebarSource() throws -> String {
        try Self.source("Sources/Rapid/UI/SidebarView.swift")
    }

    /// The body of ``SidebarView.renameField(_:)``, comment- and
    /// whitespace-stripped. Sliced out of the file first so assertions about
    /// the editor cannot be satisfied by an identically-shaped modifier
    /// somewhere else in the view (``.contentShape`` in particular is also
    /// used by the ordinary row chrome).
    private func renameFieldBody() throws -> String {
        let source = try sidebarSource()
        let start = try #require(
            source.range(of: "private func renameField("),
            "SidebarView no longer declares renameField(_:) — update this guard."
        )
        let end = try #require(
            source.range(of: "private func cancelRename(", range: start.upperBound..<source.endIndex),
            "renameField(_:) is no longer followed by cancelRename() — update this guard."
        )
        return CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(
            String(source[start.upperBound..<end.lowerBound])
        )
    }

    // MARK: - The field is focusable and gets focused

    @Test("A FocusState is declared and bound to the rename field")
    func focusStateIsBound() throws {
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(try sidebarSource())
        #expect(
            stripped.contains("@FocusStateprivatevarrenameFieldFocused:Bool"),
            "SidebarView must declare a @FocusState for the inline rename editor."
        )
        #expect(
            try renameFieldBody().contains(".focused($renameFieldFocused)"),
            "The rename TextField must be bound with .focused($renameFieldFocused) — without it nothing ever moves first responder to the field and every keystroke goes to the chat composer."
        )
    }

    /// Asserted as ONE contiguous stripped literal rather than as separate
    /// substrings: `renameFieldFocused=true` also appears in the tap handler,
    /// so a split assertion would still pass with the focus request deleted
    /// from the task.
    @Test("Focus is requested from .task, not inline in the render pass")
    func focusIsRequestedAfterTheFieldExists() throws {
        #expect(
            try renameFieldBody().contains(
                ".task(id:renameSession){renameFieldDidFocus=falseawaitTask.yield()guard!Task.isCancelled,renamingID==conv.idelse{return}renameFieldFocused=true}"
            ),
            "The focus request must hang off .task(id: renameSession) — keyed on the EDIT, not the row, so re-opening Rename on the row already being edited re-focuses too — and must land after a yield (a same-update request reaches no AppKit field), guarded so a superseded task cannot focus the wrong row."
        )
    }

    @Test("Clicking anywhere in the row's pill focuses the field")
    func clickAnywhereFocuses() throws {
        let body = try renameFieldBody()
        #expect(
            body.contains(".contentShape("),
            "The rename editor needs a .contentShape covering the drawn pill: a bare TextField is only ~16pt tall inside the 30pt row, so most of the visible target was not hit-testable."
        )
        #expect(
            body.contains(".onTapGesture{renameFieldFocused=true}"),
            "A click on the rename row must focus the field."
        )
    }

    // MARK: - The documented contract: Return commits, Escape / blur cancels

    @Test("Escape and focus loss both cancel — the documented contract")
    func escapeAndBlurCancel() throws {
        let body = try renameFieldBody()
        #expect(
            body.contains(".onExitCommand{cancelRename()}"),
            "Escape must dismiss the editor."
        )
        #expect(
            body.contains(".onChange(of:renameFieldFocused)"),
            "renameField's doc comment promises that losing focus cancels; that requires observing renameFieldFocused."
        )
        // One contiguous literal: `cancelRename()` also appears in
        // .onExitCommand, so asserting it separately would still pass with the
        // blur branch's body emptied out. The gate is asserted in the same
        // literal because the two are only correct together — the editor's
        // opening `false` (it exists but has not been given first responder
        // yet) must NOT be read as the user clicking away, or the rename would
        // cancel itself the instant it opened.
        #expect(
            body.contains("guardrenameFieldDidFocuselse{return}cancelRename()"),
            "Cancel-on-focus-loss must cancel, and must be gated on focus having actually been held once."
        )
    }

    @Test("Return commits through the model and tears the editor down")
    func returnCommits() throws {
        let body = try renameFieldBody()
        #expect(
            body.contains(".onSubmit{chat.renameConversation(conv.id,to:renameDraft)endRename()}"),
            "Return must commit the rename and close the editor."
        )
        // The pre-fix shape cleared renamingID directly and left the focus
        // flags stale, which is what made the follow-up focus loss look like
        // a second cancel.
        #expect(
            !body.contains(".onExitCommand{renamingID=nil}"),
            "The editor must route Escape through cancelRename(), not clear renamingID in place."
        )
    }

    @Test("Teardown clears BOTH focus flags, so the trailing blur is inert")
    func teardownClearsFocusState() throws {
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(try sidebarSource())
        #expect(
            stripped.contains(
                #"privatefuncendRename(){renamingID=nilrenameDraft=""renameFieldDidFocus=falserenameFieldFocused=false}"#
            ),
            "endRename() must clear renamingID, the draft AND both focus flags — leaving renameFieldDidFocus set makes the focus loss that follows a commit look like a fresh cancel."
        )
        // Starting a rename while another one is open has to end the first
        // cycle outright: leaving renameFieldFocused true would deny the new
        // editor the false -> true transition its gate watches for, and that
        // editor's own blur would then never cancel. The session bump is what
        // makes the editor's .task re-run for the new edit.
        #expect(
            stripped.contains(
                "endRename()renameDraft=conv.titlerenamingID=conv.idrenameSession&+=1"
            ),
            "The Rename menu item must tear down any rename already in flight and bump the edit session before opening a new one."
        )
    }

    @Test("Every path that removes the editor without a blur resolves the edit")
    func nonBlurDismissalsResolveTheEdit() throws {
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(try sidebarSource())
        #expect(
            stripped.contains("SidebarRow(isSelected:isActive){cancelRename()onSelectConversation(conv.id)}"),
            "Opening another conversation must resolve a pending rename — otherwise a row that never took focus stays in edit mode forever."
        )
        #expect(
            stripped.contains("SidebarRow(isSelected:isSelected,action:{cancelRename()action()})"),
            "New Chat / Launch must resolve a pending rename too."
        )
        #expect(
            stripped.contains("Button{cancelRename()showArchived.toggle()}"),
            "Collapsing the Archived group takes an archived row's editor — and its focus observer — off screen together, so it must cancel explicitly."
        )
        // Pin and Archive move a row between sections, restructuring the list
        // an open editor lives in. Both call sites of the pin toggle (the row's
        // hover button and the menu item) plus Archive must resolve the edit.
        #expect(
            stripped.contains("cancelRename()chat.setConversationArchived(conv.id,!conv.isArchived)"),
            "Archive / Unarchive must resolve a rename in progress — it relocates the row between sections."
        )
        let pinCancels = stripped.components(
            separatedBy: "cancelRename()chat.setConversationPinned(conv.id,!conv.isPinned)"
        ).count - 1
        #expect(
            pinCancels == 2,
            "Both pin toggles (hover button + menu item) must resolve a rename in progress; found \(pinCancels) of 2."
        )
    }

    // MARK: - The neighbouring surface: editing a sent message in ChatView

    /// ``MessageRow``'s edit-a-sent-message editor is the same shape of
    /// affordance, and — verified live on the same build — it had the same
    /// defect in a subtler form: it DID own a ``@FocusState``, but requested
    /// focus from a `.task(id: isEditing)` on the enclosing bubble, which runs
    /// in the update that flips ``isEditing`` — before the ``TextEditor``
    /// exists. The editor opened unfocused and typing went to the composer.
    /// The request now hangs off the editor itself.
    @Test("ChatView's sent-message editor requests focus from the editor itself")
    func chatViewEditorRequestsFocusOnTheEditor() throws {
        let source = try Self.source("Sources/Rapid/UI/ChatView.swift")
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(source)
        #expect(
            stripped.contains("@FocusStateprivatevareditFieldFocused:Bool"),
            "MessageRow must keep a @FocusState for the sent-message editor."
        )
        #expect(
            stripped.contains(
                ".focused($editFieldFocused).accessibilityIdentifier(actionIdentifier(\"EditField\")).task{awaitTask.yield()guard!Task.isCancelledelse{return}editFieldFocused=true}"
            ),
            "The sent-message editor must stay AX-addressable and request focus from a .task attached to the editor, after a yield — a request made while the bubble is still switching branches is dropped and the editor opens unfocused."
        )
        #expect(
            !stripped.contains(".task(id:isEditing){guardisEditingelse{return}editFieldFocused=true}"),
            "The focus request must not sit on the enclosing bubble: it runs before the TextEditor is in the responder chain."
        )
    }
}
