import Foundation
import Testing
@testable import Rapid

/// Regression guard: deleting a conversation from the sidebar must go through
/// a confirmation. A conversation is a HARD delete — ``ConversationStore.save``
/// atomically overwrites the on-disk store, with no trash and no undo — so
/// unlike a cached MODEL (re-downloadable, and already gated by a dialog), a
/// right-click "Delete" that fired immediately meant one misclick permanently
/// destroyed a whole chat history. The fix stages the deletion into
/// ``pendingDeletion`` and only removes it once the confirmation dialog's
/// destructive button is pressed.
///
/// ViewInspector was removed from this target (#1492), so the wiring is pinned
/// by a source-grep guard (same shape as the capability-chip / bidi suites)
/// plus a behavioural check on the pure title helper.
@Suite("Sidebar conversation delete requires confirmation")
struct SidebarConversationDeleteConfirmationTests {
    // MARK: - Behavioural: the confirmation copy

    @Test("Title helper fronts the conversation's name")
    func titleIncludesConversationName() {
        let conv = ChatConversation(
            id: UUID(),
            title: "Quarterly planning notes",
            messages: [],
            createdAt: Date(),
            updatedAt: Date()
        )
        let title = SidebarView.deleteConfirmationTitle(for: conv)
        #expect(title.contains("Quarterly planning notes"))
    }

    @Test("Title helper falls back when there is no name (nil or blank)")
    func titleFallsBackWhenBlank() {
        #expect(SidebarView.deleteConfirmationTitle(for: nil) == "Delete this conversation?")
        let blank = ChatConversation(
            id: UUID(),
            title: "   ",
            messages: [],
            createdAt: Date(),
            updatedAt: Date()
        )
        #expect(SidebarView.deleteConfirmationTitle(for: blank) == "Delete this conversation?")
    }

    // MARK: - Source guard: the delete is gated

    private func sidebarSource() throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
        let url = root.appendingPathComponent("Sources/Rapid/UI/SidebarView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    @Test("Context-menu Delete STAGES a pending deletion — it does not delete immediately")
    func contextMenuStagesRatherThanDeletes() throws {
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(try sidebarSource())
        // The context-menu destructive button must set pendingDeletion, never
        // call the model's delete directly. This literal shape is exactly what
        // the old immediate-delete build lacked, so it fails against a revert.
        #expect(
            stripped.contains(#"Button("Delete",role:.destructive){pendingDeletion=conv}"#),
            "SidebarView's context-menu Delete must stage `pendingDeletion = conv`, not delete on the spot."
        )
        // And the pre-fix immediate shape must be gone from the context menu.
        #expect(
            !stripped.contains(#".contextMenu{Button("Delete",role:.destructive){chat.deleteConversation"#),
            "SidebarView still deletes a conversation straight from the context menu with no confirmation."
        )
    }

    @Test("A confirmation dialog is wired and is the only path to the delete")
    func confirmationDialogGatesTheDelete() throws {
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(try sidebarSource())
        #expect(
            stripped.contains(".confirmationDialog("),
            "SidebarView must present a confirmationDialog before removing a conversation."
        )
        // The removal must live INSIDE the confirmation's destructive button —
        // the one that also clears `pendingDeletion` (a shape only the dialog
        // button has; the context-menu button sets `pendingDeletion = conv`).
        // Asserting the full button body pins the delete to the confirmed path,
        // so moving it back to an immediate context-menu action fails here even
        // if a confirmationDialog still exists elsewhere.
        #expect(
            stripped.contains(#"role:.destructive){chat.deleteConversation(conv.id)pendingDeletion=nil}"#),
            "chat.deleteConversation must be the body of the confirmation dialog's destructive button (which then clears pendingDeletion), not an unconfirmed call site."
        )
        // And it is the ONLY delete call site.
        let deleteCallCount = stripped.components(separatedBy: "chat.deleteConversation(").count - 1
        #expect(
            deleteCallCount == 1,
            "Expected exactly one chat.deleteConversation call site (the confirmed one); found \(deleteCallCount)."
        )
    }
}
