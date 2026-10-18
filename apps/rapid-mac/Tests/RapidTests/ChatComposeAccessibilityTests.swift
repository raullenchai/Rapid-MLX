import AppKit
import Testing
@testable import Rapid

/// Bug 3-A residual P2: AppleScript / cliclick / VoiceOver target
/// NSTextView by ``accessibilityIdentifier``. NSTextView itself ships
/// with no label or identifier so external tooling can't tell the
/// compose field apart from the system-prompt editor or search bar.
///
/// These tests pin the three attributes ``applyComposeAccessibility``
/// sets so a future refactor that drops the call (or rewrites the
/// IDs) can't silently break the cliclick integration that powers
/// external automation scripts.
@MainActor
@Suite("ChatCompose accessibility shape")
struct ChatComposeAccessibilityTests {
    @Test("Conversation identity changes include entering and leaving New Chat")
    func conversationIdentityTruthTable() {
        let a = AnyHashable(UUID())
        let b = AnyHashable(UUID())
        for (previous, current, changed) in [
            (nil, nil, false), (nil, a, true), (a, nil, true),
            (a, a, false), (a, b, true),
        ] as [(AnyHashable?, AnyHashable?, Bool)] {
            #expect(ComposeTextEditor.conversationIdentityChanged(
                previous: previous, current: current
            ) == changed)
        }
    }

    private func editor(identity: AnyHashable? = nil, text: String = "") -> ComposeTextEditor {
        ComposeTextEditor(
            text: .constant(text),
            conversationIdentity: identity,
            focusToken: 7,
            isStreaming: false,
            onSubmit: {},
            onCancel: {},
            onPasteAttachments: { false },
            onDropAttachments: nil,
            onRecallLastUser: { nil },
            onMeasuredHeight: { _ in }
        )
    }

    @Test("Shared composer call sites default to no conversation identity")
    func sharedComposerDefaultsToNoConversation() {
        let field = ComposeField(
            text: .constant("image prompt"),
            focusToken: 0,
            isStreaming: false,
            onSubmit: {},
            onCancel: {}
        )
        #expect(field.conversationIdentity == nil)
    }

    @Test("Initial and unchanged identities preserve editor state")
    func coordinatorSeedsConversationIdentity() {
        for identity in [nil, AnyHashable(UUID())] {
            let editor = editor(identity: identity, text: "binding draft")
            let coordinator = editor.makeCoordinator()
            #expect(coordinator.lastConversationIdentity == identity)
            #expect(coordinator.lastFocusToken == 7)
            let view = AutosizingTextView()
            view.string = "editor draft"
            editor.applyConversationIdentity(to: view, coordinator: coordinator)
            #expect(view.string == "editor draft")
        }
    }

    @MainActor
    private final class UndoDelegate: NSObject, NSTextViewDelegate {
        let manager = UndoManager()
        func undoManager(for view: NSTextView) -> UndoManager? { manager }
    }

    @Test("Conversation switches discard marked text and undo before applying the draft")
    func conversationSwitchResetsEditorInPlace() throws {
        let oldEditor = editor(identity: UUID(), text: "old draft")
        let coordinator = oldEditor.makeCoordinator()
        let view = AutosizingTextView()
        let delegate = UndoDelegate()
        view.delegate = delegate
        view.allowsUndo = true
        // Close the group explicitly: an open event group can report canUndo
        // even after all its targeted actions have been removed.
        delegate.manager.groupsByEvent = false
        delegate.manager.beginUndoGrouping()
        var states: [Bool] = []
        view.onComposingChange = { states.append($0) }
        view.setMarkedText(
            "nihao",
            selectedRange: NSRange(location: 5, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
        let undo = try #require(view.undoManager)
        undo.registerUndo(
            withTarget: view, selector: #selector(setter: NSTextView.string), object: "old draft"
        )
        undo.endUndoGrouping()
        #expect(view.hasMarkedText())
        #expect(undo.canUndo)

        let nextEditor = editor(identity: UUID(), text: "restored draft")
        nextEditor.applyConversationIdentity(to: view, coordinator: coordinator)

        #expect(!view.hasMarkedText())
        #expect(!undo.canUndo)
        #expect(!undo.canRedo)
        #expect(undo.isUndoRegistrationEnabled)
        #expect(view.string == "restored draft")
        #expect(states.first == true)
        #expect(states.last == false)
        #expect(coordinator.lastConversationIdentity == nextEditor.conversationIdentity)
        #expect(coordinator.text.wrappedValue == "old draft")
    }

    @Test("A conversation switch keeps an owner's disabled undo registration")
    func conversationSwitchPreservesOwnerUndoDisable() {
        let coordinator = editor(identity: UUID(), text: "old draft").makeCoordinator()
        let view = AutosizingTextView()
        let delegate = UndoDelegate()
        view.delegate = delegate
        view.allowsUndo = true
        delegate.manager.disableUndoRegistration()
        view.setMarkedText(
            "nihao",
            selectedRange: NSRange(location: 5, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )

        editor(identity: UUID(), text: "restored draft")
            .applyConversationIdentity(to: view, coordinator: coordinator)

        #expect(!delegate.manager.isUndoRegistrationEnabled)
        #expect(view.string == "restored draft")
        delegate.manager.enableUndoRegistration()
        #expect(delegate.manager.isUndoRegistrationEnabled)
    }

    @Test("A conversation switch puts the caret at the end of the restored draft")
    func conversationSwitchMovesCaretToDraftEnd() {
        let coordinator = editor(identity: UUID(), text: "a much longer old draft").makeCoordinator()
        let view = AutosizingTextView()
        view.string = "a much longer old draft"
        view.setSelectedRange(NSRange(location: 2, length: 4))

        editor(identity: UUID(), text: "next ✓")
            .applyConversationIdentity(to: view, coordinator: coordinator)

        #expect(view.string == "next ✓")
        #expect(view.selectedRange() == NSRange(location: ("next ✓" as NSString).length, length: 0))
    }

    @Test("Chat passes conversation identity without replacing the composer")
    func chatComposerPreservesAccessibilityElement() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/UI/ChatView.swift"),
            encoding: .utf8
        )
        let stripped = CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(source)
        #expect(stripped.contains(
            "ComposeField(text:draftBinding,conversationIdentity:viewModel.activeConversationID,"
        ))
        #expect(stripped.contains(
            "ComposeTextEditor(text:$text,conversationIdentity:conversationIdentity,"
        ))
        // Only the pre-existing ConversationInstructionsPopover keeps this ID.
        #expect(stripped.components(separatedBy: ".id(viewModel.activeConversationID)").count - 1 == 1)
        #expect(stripped.contains(
            "messages.last(where:{$0.role==.user})?.content})composerControls"
        ))
        #expect(stripped.contains(
            "applyConversationIdentity(to:view,coordinator:context.coordinator)"
        ))
        let reset = try #require(stripped.range(of:
            "applyConversationIdentity(to:view,coordinator:context.coordinator)"
        ))
        let rebind = try #require(stripped.range(of: "context.coordinator.text=$text"))
        #expect(reset.lowerBound < rebind.lowerBound)
    }

    @Test("A fresh editor consumes old focus requests instead of replaying them")
    func freshCoordinatorDoesNotReplayFocusToken() {
        let editor = ComposeTextEditor(
            text: .constant(""),
            focusToken: 7,
            isStreaming: false,
            onSubmit: {},
            onCancel: {},
            onPasteAttachments: { false },
            onDropAttachments: nil,
            onRecallLastUser: { nil },
            onMeasuredHeight: { _ in }
        )
        #expect(editor.makeCoordinator().lastFocusToken == 7)
    }

    @Test("Marked text reports composition until AppKit unmarks it")
    func markedTextReportsCompositionLifecycle() {
        let tv = AutosizingTextView()
        var states: [Bool] = []
        tv.onComposingChange = { states.append($0) }

        tv.setMarkedText(
            "nihao",
            selectedRange: NSRange(location: 5, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )

        #expect(tv.hasMarkedText())
        #expect(tv.string == "nihao")
        #expect(states == [true])

        tv.unmarkText()

        #expect(!tv.hasMarkedText())
        #expect(states == [true, false])
    }

    @Test("SwiftUI binding never overwrites input-method pre-edit text")
    func bindingSyncYieldsToMarkedText() {
        #expect(!ComposeTextEditor.shouldApplyBindingText(
            viewHasMarkedText: true,
            editorText: "nihao",
            bindingText: ""
        ))
        #expect(ComposeTextEditor.shouldApplyBindingText(
            viewHasMarkedText: false,
            editorText: "stale draft",
            bindingText: "restored draft"
        ))
        #expect(!ComposeTextEditor.shouldApplyBindingText(
            viewHasMarkedText: false,
            editorText: "same draft",
            bindingText: "same draft"
        ))
    }

    @Test("Compose configurator sets label, identifier, role description")
    func applyComposeAccessibilitySetsAllThreeAttributes() {
        let tv = AutosizingTextView()
        // Sanity: NSTextView ships with no compose-specific attrs.
        #expect(tv.accessibilityLabel() == nil || tv.accessibilityLabel()!.isEmpty)
        #expect(tv.accessibilityIdentifier().isEmpty)

        AutosizingTextView.applyComposeAccessibility(tv)

        #expect(tv.accessibilityLabel() == AutosizingTextView.composeAccessibilityLabel)
        #expect(tv.accessibilityIdentifier() == AutosizingTextView.composeAccessibilityIdentifier)
        #expect(tv.accessibilityRoleDescription() == AutosizingTextView.composeAccessibilityRoleDescription)
    }

    @Test("Accessibility identifier is stable for external tooling")
    func identifierMatchesPublishedContract() {
        // External cliclick / AppleScript snippets reference the literal
        // string "rapid.chat.compose". If someone renames the constant
        // without updating those scripts, the renamer needs to see this
        // test fail and decide whether to coordinate the rename.
        #expect(AutosizingTextView.composeAccessibilityIdentifier == "rapid.chat.compose")
    }

    @Test("The Images composer is a different element from the chat composer")
    func imagePromptHasItsOwnIdentity() {
        // ``ComposeField`` is shared, so before the configurator took
        // arguments the Images tab's editor announced itself as
        // "rapid.chat.compose": one identifier on two surfaces. Anything
        // driving a text field by identifier — VoiceOver, cliclick, the
        // `image-generation` golden flow — then either hit the chat field or
        // hit ``Images.Prompt``, which resolves to the placeholder static text
        // rather than the NSTextView, reports a successful set-value, and
        // changes nothing.
        let tv = AutosizingTextView()
        AutosizingTextView.applyComposeAccessibility(
            tv,
            identifier: AutosizingTextView.imagePromptAccessibilityIdentifier,
            label: AutosizingTextView.imagePromptAccessibilityLabel,
            roleDescription: AutosizingTextView.imagePromptAccessibilityRoleDescription
        )
        #expect(tv.accessibilityIdentifier() == "rapid.images.compose")
        #expect(tv.accessibilityLabel() == AutosizingTextView.imagePromptAccessibilityLabel)
        #expect(
            AutosizingTextView.imagePromptAccessibilityIdentifier
                != AutosizingTextView.composeAccessibilityIdentifier
        )
    }

    @Test("The configurator still defaults to chat when given no identity")
    func defaultsRemainTheChatComposer() {
        // The new parameters must not move the default: every existing call
        // site, and the external tooling pinned to "rapid.chat.compose",
        // depends on the no-argument form staying exactly as it was.
        let tv = AutosizingTextView()
        AutosizingTextView.applyComposeAccessibility(tv)
        #expect(tv.accessibilityIdentifier() == "rapid.chat.compose")
    }

    @Test("NSTextView role stays at AXTextArea after configurator runs")
    func textAreaRolePreserved() {
        // Pin AppKit's NSTextView default so VoiceOver still narrates
        // "text area". The configurator only touches label /
        // identifier / roleDescription — never role — so if a future
        // edit adds a setAccessibilityRole(...) call this test fires.
        let tv = AutosizingTextView()
        AutosizingTextView.applyComposeAccessibility(tv)
        #expect(tv.accessibilityRole() == .textArea)
    }
}
