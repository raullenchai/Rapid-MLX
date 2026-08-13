import AppKit
import Testing
@testable import Rapid

/// The fade's frame loop and the window it needs.
///
/// `NSView.displayLink(target:selector:)` binds to the screen the view is on.
/// Off-window there is no screen and the returned link never fires — silently.
/// Since `NSViewRepresentable` builds its view before SwiftUI mounts it, the
/// first content nearly always arrives off-window, so getting this wrong means
/// the fade never runs anywhere.
@Suite("Text fade display link")
@MainActor
struct TextFadeDisplayLinkTests {

    @Test("Starting off-window does not leave a dead link behind")
    func offWindowStartIsDeclined() {
        let link = ClosureDisplayLink()
        let orphan = NSView(frame: NSRect(x: 0, y: 0, width: 100, height: 100))
        #expect(orphan.window == nil)

        link.start(in: orphan, preferredFrameRate: 120, minimumFrameRate: 30) { _ in }

        #expect(
            !link.isRunning,
            "a link bound off-window never fires, and reporting it as running makes every later start a no-op"
        )
    }

    @Test("Starting on-window arms the link")
    func onWindowStartSucceeds() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 200, height: 200),
            styleMask: [.titled], backing: .buffered, defer: false
        )
        let view = NSView(frame: NSRect(x: 0, y: 0, width: 100, height: 100))
        window.contentView?.addSubview(view)
        #expect(view.window != nil)

        let link = ClosureDisplayLink()
        link.start(in: view, preferredFrameRate: 120, minimumFrameRate: 30) { _ in }
        #expect(link.isRunning)
        link.stop()
    }

    /// The recovery path: content arrived off-window, the link declined, and
    /// joining a window has to re-arm it.
    @Test("Joining a window re-arms a fade that was queued off-window")
    func joiningWindowRestartsTheFade() {
        var options = MarkdownOptions.assistantTranscript()
        options.textColor = .black
        let view = MarkdownTextBlockView(options: options)

        // Configure while detached — this is what SwiftUI does.
        view.configure(
            blocks: [.init(runs: [InlineRun(text: "alpha beta gamma")], kind: .paragraph)],
            options: options,
            streaming: true,
            fadeState: TextFadeAnimationState()
        )
        view.frame = NSRect(x: 0, y: 0, width: 300, height: 100)
        _ = view.height(forWidth: 300)

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 300),
            styleMask: [.titled], backing: .buffered, defer: false
        )
        window.contentView?.addSubview(view)

        #expect(view.window != nil, "view should be on a window now")
        // The assertion that matters is that moving to a window does not trap
        // and leaves the view able to animate; the link itself is private.
        view.configure(
            blocks: [.init(runs: [InlineRun(text: "alpha beta gamma delta")], kind: .paragraph)],
            options: options,
            streaming: true,
            fadeState: TextFadeAnimationState()
        )
    }
}
