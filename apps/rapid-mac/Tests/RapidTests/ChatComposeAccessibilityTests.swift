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
