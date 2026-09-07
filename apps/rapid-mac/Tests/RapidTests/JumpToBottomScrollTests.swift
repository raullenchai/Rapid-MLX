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

    /// Drains the target-update hop `requestScrollToBottom` schedules.
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
        coordinator.advanceScrollFrame(duration: 1.0 / 60.0)

        #expect(
            scrollView.contentView.bounds.minY > 0,
            "the transcript did not move — re-pinning alone never scrolled an already-attached view"
        )
        #expect(
            scrollView.contentView.bounds.minY < 1_800,
            "the first frame jumped directly to the target instead of moving smoothly"
        )

        for _ in 0..<30 {
            coordinator.advanceScrollFrame(duration: 1.0 / 60.0)
        }
        #expect(abs(scrollView.contentView.bounds.minY - 1_800) < 1)
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

    @Test("Unchanged document heights do not schedule follow scrolling")
    func unchangedDocumentHeightIsIgnored() {
        #expect(!TranscriptScrollPositionProbe.Coordinator.documentHeightChanged(
            from: 2_000, to: 2_000
        ))
        #expect(!TranscriptScrollPositionProbe.Coordinator.documentHeightChanged(
            from: 2_000, to: 2_000.4
        ))
        #expect(TranscriptScrollPositionProbe.Coordinator.documentHeightChanged(
            from: 2_000, to: 2_001
        ))
        #expect(TranscriptScrollPositionProbe.Coordinator.documentHeightChanged(
            from: nil, to: 2_000
        ))
    }

    @Test("Frame interpolation moves monotonically and converges")
    func frameInterpolationConverges() {
        var current: CGFloat = 0
        var offsets: [CGFloat] = []

        for _ in 0..<30 {
            current = TranscriptScrollPositionProbe.Coordinator.nextScrollOffset(
                current: current,
                target: 100,
                duration: 1.0 / 60.0
            )
            offsets.append(current)
        }

        #expect(offsets[0] > 0 && offsets[0] < 100)
        #expect(zip(offsets, offsets.dropFirst()).allSatisfy { $0.0 <= $0.1 })
        #expect(offsets[0] < offsets[1])
        #expect(current == 100)
    }

    @Test("A user scroll cancels an in-flight bottom target")
    func userScrollCancelsTarget() async {
        var pinned = false
        let binding = Binding(get: { pinned }, set: { pinned = $0 })
        let (scrollView, _, probe) = makeScrollView()
        let coordinator = makeCoordinator(pinned: binding)

        coordinator.attach(to: probe)
        coordinator.honourScrollRequest(0)
        pinned = true
        coordinator.honourScrollRequest(1)
        await settle()

        NotificationCenter.default.post(
            name: NSScrollView.willStartLiveScrollNotification,
            object: scrollView
        )
        let originBeforeFrame = scrollView.contentView.bounds.origin
        coordinator.advanceScrollFrame(duration: 1.0 / 60.0)

        #expect(!pinned)
        #expect(scrollView.contentView.bounds.origin == originBeforeFrame)
    }

    @Test("A long streaming answer releases after one viewport")
    func longAnswerReleasesFollowing() async {
        var pinned = true
        let binding = Binding(get: { pinned }, set: { pinned = $0 })
        let (scrollView, document, probe) = makeScrollView()
        let coordinator = makeCoordinator(pinned: binding)

        coordinator.setStreaming(true, contentLength: 1)
        coordinator.attach(to: probe)
        await settle()
        #expect(abs(scrollView.contentView.bounds.minY - 1_800) < 1)

        document.setFrameSize(NSSize(width: 400, height: 4_000))
        await settle()
        coordinator.advanceScrollFrame(duration: 1.0 / 60.0)

        #expect(!pinned)
        #expect(abs(scrollView.contentView.bounds.minY - 1_800) < 1)
    }

    @Test("A measured assistant row, not older document layout, controls release")
    func measuredAssistantRowControlsRelease() async {
        var pinned = true
        let binding = Binding(get: { pinned }, set: { pinned = $0 })
        let (scrollView, _, probe) = makeScrollView()
        defer { withExtendedLifetime(scrollView) {} }
        let coordinator = makeCoordinator(pinned: binding)
        let messageID = UUID()

        coordinator.attach(to: probe)
        await settle()
        coordinator.setStreaming(
            true,
            messageID: messageID,
            contentLength: 20,
            answerHeight: 200
        )
        #expect(pinned, "an answer exactly one viewport tall should keep following")

        coordinator.setStreaming(
            true,
            messageID: messageID,
            contentLength: 21,
            answerHeight: 201
        )
        #expect(!pinned, "a measured answer taller than the viewport should release following")
    }

    @Test("Answer growth threshold is one viewport")
    func answerGrowthThreshold() {
        #expect(!TranscriptScrollPositionProbe.Coordinator.answerOutgrewViewport(
            documentHeight: 2_200,
            documentHeightAtStreamStart: 2_000,
            viewportHeight: 200
        ))
        #expect(TranscriptScrollPositionProbe.Coordinator.answerOutgrewViewport(
            documentHeight: 2_201,
            documentHeightAtStreamStart: 2_000,
            viewportHeight: 200
        ))
    }

    @Test("Only transcript navigation keys count as scroll intent")
    func transcriptScrollKeys() {
        for keyCode: UInt16 in [115, 116, 119, 121, 125, 126] {
            #expect(TranscriptScrollPositionProbe.Coordinator.isTranscriptScrollKey(keyCode))
        }
        #expect(!TranscriptScrollPositionProbe.Coordinator.isTranscriptScrollKey(0))
        #expect(!TranscriptScrollPositionProbe.Coordinator.isTranscriptScrollKey(nil))
    }
}

/// The transcript's document view is flipped; the probe's bottom maths depends
/// on it, so the fixture has to be too.
final class FlippedView: NSView {
    override var isFlipped: Bool { true }
}
