import AppKit
import Carbon.HIToolbox

/// Delivers finished transcripts into whatever app has focus.
///
/// Synthesising ⌘V is the only approach that works everywhere: typing the text
/// character by character (`CGEvent(keyboardEventSource:virtualKey:)` per glyph)
/// is far slower, mangles CJK because dead keys and input methods intercept the
/// stream, and loses the text entirely if focus moves mid-way. The clipboard
/// round-trip is atomic from the target app's point of view.
@MainActor
enum DictationInjector {
    /// The transcript is left on the clipboard on purpose — it is the fallback
    /// when the frontmost app refuses a synthetic paste (some secure fields do),
    /// and users have come to expect dictation results to be re-pastable.
    static func deliver(_ text: String, paste: Bool) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)

        guard paste, canPaste else { return }
        synthesizePaste()
    }

    /// Pasting drives other applications, which is an Accessibility-gated
    /// capability — the same permission the hotkey tap needs.
    static var canPaste: Bool { AXIsProcessTrusted() }

    private static func synthesizePaste() {
        // `.combinedSessionState` inherits the real keyboard's modifier state,
        // which matters because the user may still be physically holding the
        // trigger modifier when the transcript lands.
        guard let source = CGEventSource(stateID: .combinedSessionState) else { return }

        // Suppress the local keyboard's own modifiers for these synthetic
        // events; otherwise a still-held ⌘ or ⌥ merges into the flags below and
        // the target app sees ⌘⌥V instead of ⌘V.
        source.setLocalEventsFilterDuringSuppressionState(
            [.permitLocalMouseEvents, .permitSystemDefinedEvents],
            state: .eventSuppressionStateSuppressionInterval
        )

        let key = CGKeyCode(kVK_ANSI_V)
        let down = CGEvent(keyboardEventSource: source, virtualKey: key, keyDown: true)
        let up = CGEvent(keyboardEventSource: source, virtualKey: key, keyDown: false)
        down?.flags = .maskCommand
        up?.flags = .maskCommand
        down?.post(tap: .cgAnnotatedSessionEventTap)
        up?.post(tap: .cgAnnotatedSessionEventTap)
    }
}
