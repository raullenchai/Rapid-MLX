import AppKit
import SwiftUI
import Testing
@testable import Rapid

@MainActor
@Suite("Transient editors cancel on Escape", .serialized)
struct PopoverEscapeTests {
    @Test("Escape from the focused instruction editor cancels without saving")
    func instructionEscape() async throws {
        var draft = "Unsaved replacement"
        var saved = "Saved prompt"
        var cancelled = false
        let existingWindows = Set(NSApplication.shared.windows.map(\.windowNumber))
        let content = ConversationInstructionsPopover(
            draft: Binding(get: { draft }, set: { draft = $0 }),
            global: "",
            onSave: { saved = $0 },
            onCancel: { cancelled = true }
        )
        let host = NSHostingView(rootView: Color.clear
            .frame(width: 440, height: 500)
            .popover(isPresented: .constant(true)) { content }
        )
        let window = NSWindow(
            contentRect: NSRect(x: -8000, y: -8000, width: 440, height: 500),
            styleMask: [.borderless], backing: .buffered, defer: false
        )
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.orderBack(nil)
        defer { window.orderOut(nil); window.close() }
        host.layoutSubtreeIfNeeded()
        func textView(in view: NSView) -> NSTextView? {
            if let text = view as? NSTextView { return text }
            return view.subviews.lazy.compactMap { textView(in: $0) }.first
        }
        func presentedWindow() -> NSWindow? {
            NSApplication.shared.windows.first {
                !existingWindows.contains($0.windowNumber) && $0 !== window
                    && $0.contentView.flatMap { textView(in: $0) } != nil
            }
        }
        for _ in 0..<100 {
            if presentedWindow() != nil { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        let popoverWindow = try #require(presentedWindow())
        let editor = try #require(popoverWindow.contentView.flatMap { textView(in: $0) })
        #expect(popoverWindow.makeFirstResponder(editor))
        // Allow SwiftUI to register the focused editor's command handlers.
        try await Task.sleep(for: .milliseconds(200))
        let escape = try #require(NSEvent.keyEvent(
            with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0,
            windowNumber: popoverWindow.windowNumber, context: nil,
            characters: "\u{1b}", charactersIgnoringModifiers: "\u{1b}",
            isARepeat: false, keyCode: 53
        ))
        editor.keyDown(with: escape)
        #expect(cancelled)
        #expect(saved == "Saved prompt")
    }

    private func source(_ name: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(
            try String(contentsOf: root.appendingPathComponent("Sources/Rapid/UI/\(name).swift"), encoding: .utf8)
        )
    }

    @Test("Reopening instructions restores saved text and Cancel never commits")
    func instructionDraftLifecycle() throws {
        let chat = try source("ChatView")
        #expect(chat.contains("conversationInstructionsDraft=viewModel.conversationInstructions"))
        #expect(chat.contains("onCancel:{showsConversationInstructions=false}"))
        let popover = try source("InstructionTextEditor")
        #expect(popover.contains(".onExitCommand(perform:onCancel)"))
        #expect(popover.contains("Button(\"Cancel\",action:onCancel)"))
    }

    @Test("Memory and dictation sheets map Escape to their existing Cancel actions")
    func sheetCancelShortcuts() throws {
        #expect(try source("SettingsMemoryPanel").contains(
            "Button(\"Cancel\"){editingEntry=nil}.keyboardShortcut(.cancelAction)"
        ))
        #expect(try source("DictationView").contains(
            "Button(\"Cancel\"){dismiss()}.keyboardShortcut(.cancelAction)"
        ))
    }
}
