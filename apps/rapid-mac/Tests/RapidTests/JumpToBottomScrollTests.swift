import AppKit
import SwiftUI
import Testing
@testable import Rapid

/// Pressing "jump to latest" must actually move the transcript.
///
/// The regression this exists for: the button set `isPinnedToBottom = true`
/// and trusted `attach` to do the scrolling. `attach` only anchors on a NEW
/// attachment (#1877), and by the time a reader presses the button the scroll
/// view has been attached for the whole session — so nothing moved. Following
/// then rested entirely on the document-frame notification, which fires while
/// an answer streams and never again once it settles. The button therefore
/// worked mid-stream and did nothing at all on a finished transcript, which is
/// the state a reader is most likely to press it in.
///
/// Driven against a real `NSScrollView` rather than a source grep: the bug was
/// never visible in the source of either file on its own. Both halves read
/// correctly; only their composition was wrong.
@Suite("Jump to bottom")
@MainActor
struct JumpToBottomScrollTests {

    /// A flipped document taller than its clip, matching the transcript.
    private func makeScrollView() -> (NSScrollView, FlippedView, NSView) {
        let scrollView = NSScrollView(frame: NSRect(x: 0, y: 0, width: 400, height: 200))
        let document = FlippedView(frame: NSRect(x: 0, y: 0, width: 400, height: 2_000))
        scrollView.documentView = document
        let probe = NSView(frame: .zero)
        document.addSubview(probe)
        scrollView.layoutSubtreeIfNeeded()
        return (scrollView, document, probe)
    }

    private func makeCoordinator(
        pinned: Binding<Bool>
    ) -> TranscriptScrollPositionProbe.Coordinator {
        TranscriptScrollPositionProbe.Coordinator(
            isPinnedToBottom: pinned, bottomResumeSlack: 24
        )
    }

    /// Drains the coalescing hop `requestScrollToBottom` schedules.
    private func settle() async {
        await Task.yield()
        try? await Task.sleep(nanoseconds: 60_000_000)
    }

    @Test("A changed request token scrolls an already-attached transcript")
    func requestTokenScrollsWhenAlreadyAttached() async {
        var pinned = false
        let binding = Binding(get: { pinned }, set: { pinned = $0 })
        let (scrollView, _, probe) = makeScrollView()
        let coordinator = makeCoordinator(pinned: binding)

        // Attach first, then move away from the bottom — this is the state a
        // reader is in when the button appears.
        coordinator.attach(to: probe)
        coordinator.honourScrollRequest(0)
        scrollView.contentView.scroll(to: NSPoint(x: 0, y: 0))
        scrollView.reflectScrolledClipView(scrollView.contentView)
        #expect(scrollView.contentView.bounds.minY == 0, "fixture must start away from the bottom")

        // What the button does.
        pinned = true
        coordinator.attach(to: probe)
        coordinator.honourScrollRequest(1)
        await settle()

        #expect(
            scrollView.contentView.bounds.minY > 0,
            "the transcript did not move — re-pinning alone never scrolled an already-attached view"
        )
    }

    /// The coalescing this sits next to exists because `updateNSView` runs for
    /// every streamed mutation. An unchanged token must stay silent or the
    /// scroll returns on every keystroke.
    @Test("An unchanged token does not scroll")
    func unchangedTokenIsSilent() async {
        var pinned = true
        let binding = Binding(get: { pinned }, set: { pinned = $0 })
        let (scrollView, _, probe) = makeScrollView()
        let coordinator = makeCoordinator(pinned: binding)

        coordinator.attach(to: probe)
        coordinator.honourScrollRequest(7)
        await settle()

        scrollView.contentView.scroll(to: NSPoint(x: 0, y: 0))
        scrollView.reflectScrolledClipView(scrollView.contentView)

        coordinator.honourScrollRequest(7)
        await settle()

        #expect(
            scrollView.contentView.bounds.minY == 0,
            "an unchanged token scrolled anyway — this would fight the reader on every streamed frame"
        )
    }

    /// The first render arrives with the token already at some value, and
    /// `attach` is what anchors that one. Treating it as a request would scroll
    /// a transcript the reader had deliberately left scrolled up.
    @Test("The first token seen is recorded, not acted on")
    func firstTokenIsNotARequest() async {
        var pinned = false
        let binding = Binding(get: { pinned }, set: { pinned = $0 })
        let (scrollView, _, probe) = makeScrollView()
        let coordinator = makeCoordinator(pinned: binding)

        coordinator.attach(to: probe)
        scrollView.contentView.scroll(to: NSPoint(x: 0, y: 0))
        scrollView.reflectScrolledClipView(scrollView.contentView)

        coordinator.honourScrollRequest(3)
        await settle()

        #expect(scrollView.contentView.bounds.minY == 0)
    }
}

/// The transcript's document view is flipped; the probe's bottom maths depends
/// on it, so the fixture has to be too.
final class FlippedView: NSView {
    override var isFlipped: Bool { true }
}
