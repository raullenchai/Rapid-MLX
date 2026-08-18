import AppKit
import SwiftUI

/// Owns transcript follow-mode at the AppKit layer. User live-scroll gestures
/// pause following until they return to the bottom; document frame changes
/// keep a followed transcript pinned through every stage of SwiftUI layout.
struct TranscriptScrollPositionProbe: NSViewRepresentable {
    @Binding var isPinnedToBottom: Bool
    let bottomResumeSlack: CGFloat
    /// Whether an answer is currently being written. Drives the one-shot
    /// release described on ``Coordinator/releaseIfAnswerOutgrewViewport()``.
    var isStreaming: Bool = false
    /// Bumped by anything outside that wants the transcript moved to the
    /// bottom right now, ``JumpToBottomButton`` being the only caller today.
    ///
    /// Setting ``isPinnedToBottom`` back to true is NOT enough on its own, and
    /// used to be: ``attach`` only anchors on a NEW attachment (#1877), and by
    /// the time the reader presses the button the scroll view has long been
    /// attached. Following then depends entirely on the document-frame
    /// notification, which fires while an answer streams and never again once
    /// it settles — so the button worked mid-stream and did nothing at all on
    /// a finished transcript. A token the caller changes makes the request
    /// explicit rather than a side effect of re-pinning.
    var scrollToBottomRequest: Int = 0

    func makeCoordinator() -> Coordinator {
        Coordinator(
            isPinnedToBottom: $isPinnedToBottom,
            bottomResumeSlack: bottomResumeSlack
        )
    }

    func makeNSView(context: Context) -> NSView {
        let probe = NSView(frame: .zero)
        DispatchQueue.main.async {
            context.coordinator.attach(to: probe)
        }
        return probe
    }

    func updateNSView(_ probe: NSView, context: Context) {
        context.coordinator.update(
            isPinnedToBottom: $isPinnedToBottom,
            bottomResumeSlack: bottomResumeSlack
        )
        context.coordinator.setStreaming(isStreaming)
        context.coordinator.attach(to: probe)
        // After ``attach``: a first render arrives with the token already at
        // its initial value, and attaching is what anchors that one.
        context.coordinator.honourScrollRequest(scrollToBottomRequest)
    }

    static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) {
        coordinator.detach()
    }

    @MainActor
    final class Coordinator: NSObject {
        private var isPinnedToBottom: Binding<Bool>
        private var bottomResumeSlack: CGFloat
        private weak var scrollView: NSScrollView?
        private weak var documentView: NSView?
        private var isLiveScrolling = false
        private var bottomScrollScheduled = false
        /// Last token acted on. Starts at `nil` so the initial value — whatever
        /// it happens to be — is recorded rather than treated as a request.
        private var lastScrollRequest: Int?

        init(
            isPinnedToBottom: Binding<Bool>,
            bottomResumeSlack: CGFloat
        ) {
            self.isPinnedToBottom = isPinnedToBottom
            self.bottomResumeSlack = bottomResumeSlack
        }

        func update(
            isPinnedToBottom: Binding<Bool>,
            bottomResumeSlack: CGFloat
        ) {
            self.isPinnedToBottom = isPinnedToBottom
            self.bottomResumeSlack = bottomResumeSlack
        }

        func attach(to probe: NSView) {
            guard let enclosingScrollView = probe.enclosingScrollView else { return }
            var attachmentChanged = false
            if enclosingScrollView !== scrollView {
                detach()
                scrollView = enclosingScrollView
                observeScrollView(enclosingScrollView)
                attachmentChanged = true
            }
            attachmentChanged = observeDocumentViewIfNeeded() || attachmentChanged
            if isStreaming, documentHeightAtStreamStart == nil {
                documentHeightAtStreamStart = documentView?.bounds.height
            }
            // updateNSView runs for every streamed mutation. Document-frame
            // notifications own steady-state following; only a new attachment
            // needs an explicit initial anchor (#1877).
            if attachmentChanged, isPinnedToBottom.wrappedValue {
                requestScrollToBottom()
            }
        }

        /// Scroll if the caller's token moved since we last looked.
        ///
        /// Deliberately compares rather than tests for non-zero: the token is a
        /// counter the caller owns and may wrap, reset, or start anywhere.
        func honourScrollRequest(_ token: Int) {
            defer { lastScrollRequest = token }
            guard let previous = lastScrollRequest, previous != token else { return }
            requestScrollToBottom()
        }

        func detach() {
            NotificationCenter.default.removeObserver(self)
            scrollView = nil
            documentView = nil
            isLiveScrolling = false
            bottomScrollScheduled = false
        }

        @objc private func liveScrollWillStart(_ notification: Notification) {
            isLiveScrolling = true
            // User intent wins before the first wheel/trackpad delta, so a
            // simultaneous streamed layout cannot pull the gesture back.
            setPinned(false)
        }

        @objc private func liveScrollDidEnd(_ notification: Notification) {
            isLiveScrolling = false
            if isAtBottom { setPinned(true) }
        }

        @objc private func boundsDidChange(_ notification: Notification) {
            // Our own ``scrollToBottom`` emits this too; it is not user intent.
            guard !isProgrammaticScroll else { return }
            if !isAtBottom {
                // Any move away from the bottom releases the pin — whether or
                // not AppKit bracketed it. A legacy mouse wheel can post bounds
                // changes with NO willStartLiveScroll/didEndLiveScroll pair, and
                // gating release on ``isLiveScrolling`` meant such a scroll never
                // paused following: the next streamed frame yanked the reader
                // straight back to the bottom.
                setPinned(false)
            } else if !isLiveScrolling {
                // Back at the bottom on an UNBRACKETED scroll: no
                // ``didEndLiveScroll`` is coming, so resume here. Deliberately
                // not done mid-gesture — ``isAtBottom`` is a slack comparison,
                // so re-pinning during a gesture would let a sub-slack gentle
                // scroll be dragged back on every event.
                setPinned(true)
            }
        }

        /// The viewport itself resized (the composer grew from one line to
        /// several, the window was resized). The document frame is unchanged,
        /// so ``documentFrameDidChange`` never fires — yet a shorter viewport
        /// means the newest content has slid out of sight while we still claim
        /// to be following it. Re-anchor.
        @objc private func clipFrameDidChange(_ notification: Notification) {
            guard isPinnedToBottom.wrappedValue else { return }
            requestScrollToBottom()
        }

        @objc private func documentFrameDidChange(_ notification: Notification) {
            guard isPinnedToBottom.wrappedValue else { return }
            if releaseIfAnswerOutgrewViewport() { return }
            requestScrollToBottom()
        }

        /// Whether this stream has already handed control back to the reader.
        private var didReleaseForCurrentStream = false
        private var isStreaming = false
        private var documentHeightAtStreamStart: CGFloat?

        func setStreaming(_ streaming: Bool) {
            guard streaming != isStreaming else { return }
            isStreaming = streaming
            if streaming {
                didReleaseForCurrentStream = false
                documentHeightAtStreamStart = documentView?.bounds.height
            } else {
                documentHeightAtStreamStart = nil
            }
        }

        /// Stop following the moment a streamed answer grows taller than the
        /// viewport, once per stream.
        ///
        /// Following unconditionally is right while the answer still fits: the
        /// text simply fills space the reader can already see, and nothing
        /// appears to move. Past that point every batch drags the viewport
        /// down, so the reader is held at the newest words and cannot read the
        /// answer from its start without fighting the scroll — which is what
        /// "keeps slamming into the bottom" describes.
        ///
        /// Releasing the pin instead leaves the text where the reader is
        /// looking and surfaces ``JumpToBottomButton``, so catching up becomes
        /// something they ask for rather than something done to them. One shot
        /// per stream: after the reader presses the button they are pinned
        /// again deliberately, and that choice is theirs to keep.
        ///
        /// Returns true when it released, meaning the caller must not scroll.
        private func releaseIfAnswerOutgrewViewport() -> Bool {
            guard isStreaming, !didReleaseForCurrentStream else { return false }
            guard let scrollView, let documentView,
                  let startingHeight = documentHeightAtStreamStart else { return false }
            let viewportHeight = scrollView.contentView.bounds.height
            guard Self.answerOutgrewViewport(
                documentHeight: documentView.bounds.height,
                documentHeightAtStreamStart: startingHeight,
                viewportHeight: viewportHeight
            ) else { return false }
            didReleaseForCurrentStream = true
            // Not `setPinned(false)` — that helper is reserved for the
            // user-gesture path (`liveScrollWillStart` / `boundsDidChange`).
            // This is the app deciding to stop following, and routing it
            // through the same helper would blur who released the pin.
            if isPinnedToBottom.wrappedValue {
                isPinnedToBottom.wrappedValue = false
            }
            return true
        }

        nonisolated static func answerOutgrewViewport(
            documentHeight: CGFloat,
            documentHeightAtStreamStart: CGFloat,
            viewportHeight: CGFloat
        ) -> Bool {
            viewportHeight > 0
                && documentHeight - documentHeightAtStreamStart > viewportHeight
        }

        private func observeScrollView(_ scrollView: NSScrollView) {
            scrollView.contentView.postsBoundsChangedNotifications = true
            scrollView.contentView.postsFrameChangedNotifications = true
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(clipFrameDidChange(_:)),
                name: NSView.frameDidChangeNotification,
                object: scrollView.contentView
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(boundsDidChange(_:)),
                name: NSView.boundsDidChangeNotification,
                object: scrollView.contentView
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(liveScrollWillStart(_:)),
                name: NSScrollView.willStartLiveScrollNotification,
                object: scrollView
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(liveScrollDidEnd(_:)),
                name: NSScrollView.didEndLiveScrollNotification,
                object: scrollView
            )
        }

        @discardableResult
        private func observeDocumentViewIfNeeded() -> Bool {
            guard let nextDocumentView = scrollView?.documentView else { return false }
            guard nextDocumentView !== documentView else { return false }

            if let documentView {
                NotificationCenter.default.removeObserver(
                    self,
                    name: NSView.frameDidChangeNotification,
                    object: documentView
                )
            }
            documentView = nextDocumentView
            nextDocumentView.postsFrameChangedNotifications = true
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(documentFrameDidChange(_:)),
                name: NSView.frameDidChangeNotification,
                object: nextDocumentView
            )
            return true
        }

        private var isAtBottom: Bool {
            guard let scrollView, let documentView else { return false }
            let clipBounds = scrollView.contentView.bounds
            let distance: CGFloat
            if documentView.isFlipped {
                distance = documentView.bounds.maxY
                    + scrollView.contentInsets.bottom
                    - clipBounds.maxY
            } else {
                distance = clipBounds.minY
                    - (documentView.bounds.minY - scrollView.contentInsets.bottom)
            }
            return distance <= bottomResumeSlack
        }

        /// True while ``scrollToBottom`` is driving the clip view, so the
        /// bounds notification it emits is not mistaken for the user moving.
        private var isProgrammaticScroll = false
        /// SwiftUI can resize the document several times for one streamed
        /// token. Collapse those notifications into one end-of-run-loop scroll.
        private func requestScrollToBottom() {
            guard !bottomScrollScheduled else { return }
            bottomScrollScheduled = true
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.bottomScrollScheduled = false
                guard self.isPinnedToBottom.wrappedValue, !self.isLiveScrolling else {
                    return
                }
                self.scrollToBottom()
            }
        }

        private func scrollToBottom() {
            guard let scrollView, let documentView else { return }
            isProgrammaticScroll = true
            defer { isProgrammaticScroll = false }
            let clipView = scrollView.contentView
            let targetY: CGFloat
            if documentView.isFlipped {
                // SwiftUI's full-size macOS window contributes the transparent
                // titlebar safe area through contentInsets. For a short
                // transcript, zero is therefore below the natural top edge.
                targetY = max(
                    documentView.bounds.minY - scrollView.contentInsets.top,
                    documentView.bounds.maxY
                        + scrollView.contentInsets.bottom
                        - clipView.bounds.height
                )
            } else {
                targetY = documentView.bounds.minY - scrollView.contentInsets.bottom
            }
            // Clamp through AppKit rather than scrolling to the raw target.
            // `constrainBoundsRect` is the scroll view's own answer to "where
            // is this allowed to stop" — it picks up content insets and
            // elasticity, and is defensive against a target that overshoots
            // the document for any reason.
            let proposed = NSRect(
                origin: NSPoint(x: clipView.bounds.minX, y: targetY),
                size: clipView.bounds.size
            )
            let constrained = clipView.constrainBoundsRect(proposed).origin
            // Scrolling to the current origin still emits AppKit notifications
            // and schedules SwiftUI layout. At streaming cadence that no-op can
            // become a self-sustaining AttributeGraph loop (#1877).
            guard abs(clipView.bounds.minY - constrained.y) > 0.5 else { return }
            clipView.scroll(to: constrained)
            scrollView.reflectScrolledClipView(clipView)
        }

        private func setPinned(_ pinned: Bool) {
            if pinned != isPinnedToBottom.wrappedValue {
                isPinnedToBottom.wrappedValue = pinned
            }
            if pinned { requestScrollToBottom() }
        }
    }
}
