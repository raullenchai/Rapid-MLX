import AppKit
import SwiftUI

/// Owns transcript follow-mode at the AppKit layer. User live-scroll gestures
/// pause following until they return to the bottom; document frame changes
/// keep a followed transcript pinned through every stage of SwiftUI layout.
struct TranscriptScrollPositionProbe: NSViewRepresentable {
    @Binding var isPinnedToBottom: Bool
    let bottomResumeSlack: CGFloat

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
        context.coordinator.attach(to: probe)
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
            if enclosingScrollView !== scrollView {
                detach()
                scrollView = enclosingScrollView
                observeScrollView(enclosingScrollView)
            }
            observeDocumentViewIfNeeded()
            if isPinnedToBottom.wrappedValue { scrollToBottom() }
        }

        func detach() {
            NotificationCenter.default.removeObserver(self)
            scrollView = nil
            documentView = nil
            isLiveScrolling = false
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
            guard isLiveScrolling else { return }
            // Mid-gesture this may only RELEASE the pin, never restore it.
            // ``isAtBottom`` is a slack comparison, so a gentle scroll whose
            // per-event delta is within ``bottomResumeSlack`` still reads as
            // "at the bottom" — re-pinning on it would let the next streamed
            // frame yank the user straight back, which is precisely the
            // hijacking this probe exists to stop, and the user could never
            // escape by scrolling softly. Resuming is decided once the user
            // has settled, in ``liveScrollDidEnd``.
            if !isAtBottom { setPinned(false) }
        }

        @objc private func documentFrameDidChange(_ notification: Notification) {
            guard isPinnedToBottom.wrappedValue else { return }
            scrollToBottom()
            DispatchQueue.main.async { [weak self] in
                guard let self, self.isPinnedToBottom.wrappedValue else { return }
                self.scrollToBottom()
            }
        }

        private func observeScrollView(_ scrollView: NSScrollView) {
            scrollView.contentView.postsBoundsChangedNotifications = true
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

        private func observeDocumentViewIfNeeded() {
            guard let nextDocumentView = scrollView?.documentView else { return }
            guard nextDocumentView !== documentView else { return }

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
        }

        private var isAtBottom: Bool {
            guard let scrollView, let documentView else { return false }
            let clipBounds = scrollView.contentView.bounds
            let distance: CGFloat
            if documentView.isFlipped {
                distance = documentView.bounds.maxY - clipBounds.maxY
            } else {
                distance = clipBounds.minY - documentView.bounds.minY
            }
            return distance <= bottomResumeSlack
        }

        private func scrollToBottom() {
            guard let scrollView, let documentView else { return }
            let clipView = scrollView.contentView
            let targetY: CGFloat
            if documentView.isFlipped {
                targetY = max(
                    documentView.bounds.minY,
                    documentView.bounds.maxY - clipView.bounds.height
                )
            } else {
                targetY = documentView.bounds.minY
            }
            clipView.scroll(to: NSPoint(x: clipView.bounds.minX, y: targetY))
            scrollView.reflectScrolledClipView(clipView)
        }

        private func setPinned(_ pinned: Bool) {
            if pinned != isPinnedToBottom.wrappedValue {
                isPinnedToBottom.wrappedValue = pinned
            }
            if pinned { scrollToBottom() }
        }
    }
}
