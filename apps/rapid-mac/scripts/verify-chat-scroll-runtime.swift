import AppKit
import SwiftUI

// Runtime regression for transcript follow mode. This mounts a real SwiftUI
// ScrollView and drives its backing NSScrollView through repeated user-scroll
// and content-growth cycles.

@MainActor
final class HarnessModel: ObservableObject {
    @Published var rowCount = 120
    @Published var isPinnedToBottom = true
}

@MainActor
final class ScrollCapture {
    weak var scrollView: NSScrollView?
}

private struct HarnessView: View {
    @ObservedObject var model: HarnessModel
    let capture: ScrollCapture

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 8) {
                ForEach(0..<model.rowCount, id: \.self) { row in
                    Text("row \(row)")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .frame(height: 24)
                }
                Color.clear.frame(height: 1)
            }
            .padding(24)
            .background(
                TranscriptScrollPositionProbe(
                    isPinnedToBottom: $model.isPinnedToBottom,
                    bottomResumeSlack: 2
                )
            )
            .overlay(ScrollCaptureProbe(capture: capture).frame(width: 0, height: 0))
        }
        .frame(width: 420, height: 320)
    }
}

private struct ScrollCaptureProbe: NSViewRepresentable {
    let capture: ScrollCapture

    func makeNSView(context: Context) -> NSView {
        let probe = NSView(frame: .zero)
        DispatchQueue.main.async { capture.scrollView = probe.enclosingScrollView }
        return probe
    }

    func updateNSView(_ probe: NSView, context: Context) {
        if capture.scrollView == nil {
            DispatchQueue.main.async { capture.scrollView = probe.enclosingScrollView }
        }
    }
}

@main
@MainActor
struct ChatScrollRuntimeCheck {
    static func pump(_ seconds: TimeInterval = 0.2) {
        RunLoop.main.run(until: Date().addingTimeInterval(seconds))
    }

    static func metrics(_ scroll: NSScrollView) -> String {
        let document = scroll.documentView
        let parts = [
            "doc.bounds=\(document?.bounds.debugDescription ?? "nil")",
            "doc.frame=\(document?.frame.debugDescription ?? "nil")",
            "doc.safe=\(String(describing: document?.safeAreaInsets))",
            "clip.bounds=\(scroll.contentView.bounds.debugDescription)",
            "clip.docRect=\(scroll.contentView.documentRect.debugDescription)",
            "visible=\(scroll.documentVisibleRect.debugDescription)",
            "insets=\(String(describing: scroll.contentInsets))",
            "scrollerInsets=\(String(describing: scroll.scrollerInsets))"
        ]
        return parts.joined(separator: " ")
    }

    static func move(_ scroll: NSScrollView, toY y: CGFloat) {
        scroll.contentView.scroll(to: NSPoint(x: 0, y: y))
        scroll.reflectScrolledClipView(scroll.contentView)
        pump()
    }

    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let model = HarnessModel()
        let capture = ScrollCapture()
        let host = NSHostingView(rootView: HarnessView(model: model, capture: capture))
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 420, height: 320),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.contentView = host
        window.orderFrontRegardless()
        pump(0.8)

        guard let scroll = capture.scrollView, let document = scroll.documentView else {
            fputs("FAIL: probe did not attach to a scroll view\n", stderr)
            exit(1)
        }

        var bottomY = max(0, document.bounds.height - scroll.contentView.bounds.height)
        move(scroll, toY: bottomY)
        guard model.isPinnedToBottom else {
            fputs("FAIL: initial transcript did not follow the bottom\n", stderr)
            exit(1)
        }

        for cycle in 1...3 {
            NotificationCenter.default.post(name: NSScrollView.willStartLiveScrollNotification, object: scroll)
            move(scroll, toY: max(0, bottomY - 240))
            NotificationCenter.default.post(name: NSScrollView.didEndLiveScrollNotification, object: scroll)
            pump()
            guard !model.isPinnedToBottom else {
                fputs("FAIL: cycle \(cycle) upward scroll did not pause following\n", stderr)
                fputs("\(metrics(scroll))\n", stderr)
                exit(1)
            }

            let pausedY = scroll.contentView.bounds.minY
            for _ in 0..<4 {
                model.rowCount += 1
                pump(0.03)
            }
            pump(0.2)
            guard !model.isPinnedToBottom,
                  abs(scroll.contentView.bounds.minY - pausedY) <= 0.5
            else {
                fputs("FAIL: cycle \(cycle) streamed content moved a paused transcript\n", stderr)
                fputs("\(metrics(scroll))\n", stderr)
                exit(1)
            }

            bottomY = max(0, document.bounds.height - scroll.contentView.bounds.height)
            NotificationCenter.default.post(name: NSScrollView.willStartLiveScrollNotification, object: scroll)
            move(scroll, toY: bottomY)
            NotificationCenter.default.post(name: NSScrollView.didEndLiveScrollNotification, object: scroll)
            pump()
            guard model.isPinnedToBottom else {
                fputs("FAIL: cycle \(cycle) returning to bottom did not resume following\n", stderr)
                fputs("\(metrics(scroll))\n", stderr)
                exit(1)
            }

            for _ in 0..<8 {
                model.rowCount += 1
                pump(0.03)
            }
            pump(0.2)
            let distance = document.bounds.height - scroll.contentView.bounds.maxY
            guard model.isPinnedToBottom && distance <= 2.5 else {
                fputs("FAIL: cycle \(cycle) streamed content was not followed after resuming\n", stderr)
                fputs("distance=\(distance) \(metrics(scroll))\n", stderr)
                exit(1)
            }
            bottomY = max(0, document.bounds.height - scroll.contentView.bounds.height)
        }

        // A GENTLE scroll must escape too. The cycles above jump 240 pt in one
        // move, far outside `bottomResumeSlack`, so they never exercised the
        // case where each per-event delta lands INSIDE the slack. If the
        // mid-gesture handler is allowed to re-pin on a slack comparison, every
        // small step reads as "still at the bottom", the next streamed frame
        // snaps the transcript back down, and the user can never scroll away by
        // moving softly — the exact hijacking this probe exists to prevent.
        move(scroll, toY: bottomY)
        pump()
        guard model.isPinnedToBottom else {
            fputs("FAIL: could not re-pin before the gentle-scroll check\n", stderr)
            exit(1)
        }
        NotificationCenter.default.post(name: NSScrollView.willStartLiveScrollNotification, object: scroll)
        for _ in 0..<8 {
            // Step 1 pt above WHERE WE CURRENTLY ARE (not a precomputed
            // absolute Y): if a streamed frame drags us back to the bottom,
            // the next step starts from the bottom again and no distance ever
            // accumulates — which is exactly the failure being detected.
            let currentY = scroll.contentView.bounds.minY
            move(scroll, toY: max(0, currentY - 1))   // 1 pt: inside the 2 pt slack
            model.rowCount += 1                        // stream while the user reads
            pump(0.03)
        }
        NotificationCenter.default.post(name: NSScrollView.didEndLiveScrollNotification, object: scroll)
        pump(0.2)
        let gentleDistance = document.bounds.height - scroll.contentView.bounds.maxY
        guard !model.isPinnedToBottom, gentleDistance > 2 else {
            fputs("FAIL: a gentle (sub-slack) scroll was dragged back to the bottom\n", stderr)
            fputs("distance=\(gentleDistance) pinned=\(model.isPinnedToBottom) \(metrics(scroll))\n", stderr)
            exit(1)
        }

        print("PASS: paused scrolling and bottom-resume following survived 3 streaming cycles + gentle scroll")
        window.close()
    }
}
