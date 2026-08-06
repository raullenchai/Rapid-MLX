import AppKit
import SwiftUI

// Runtime regression for transcript follow mode. This mounts a real SwiftUI
// ScrollView and drives its backing NSScrollView through repeated user-scroll
// and content-growth cycles.

@MainActor
final class HarnessModel: ObservableObject {
    @Published var rowCount: Int
    @Published var isPinnedToBottom = true

    init(rowCount: Int = 120) {
        self.rowCount = rowCount
    }
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
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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

    static func bottomTargetY(_ scroll: NSScrollView, document: NSView) -> CGFloat {
        if document.isFlipped {
            return max(
                document.bounds.minY - scroll.contentInsets.top,
                document.bounds.maxY
                    + scroll.contentInsets.bottom
                    - scroll.contentView.bounds.height
            )
        }
        return document.bounds.minY - scroll.contentInsets.bottom
    }

    static func bottomDistance(_ scroll: NSScrollView, document: NSView) -> CGFloat {
        if document.isFlipped {
            return document.bounds.maxY
                + scroll.contentInsets.bottom
                - scroll.contentView.bounds.maxY
        }
        return scroll.contentView.bounds.minY
            - (document.bounds.minY - scroll.contentInsets.bottom)
    }

    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let model = HarnessModel(rowCount: 2)
        let capture = ScrollCapture()
        let host = NSHostingView(
            rootView: NavigationSplitView {
                Text("Rapid-MLX")
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .navigationSplitViewColumnWidth(180)
            } detail: {
                HarnessView(model: model, capture: capture)
                    .frame(minWidth: 440, minHeight: 320)
            }
            .frame(width: 900, height: 560)
        )
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 900, height: 560),
            styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "Rapid-MLX"
        window.titlebarAppearsTransparent = true
        window.toolbar = NSToolbar(identifier: "Rapid.ChatScrollCheck")
        window.contentView = host
        window.orderFrontRegardless()
        pump(0.8)

        guard let scroll = capture.scrollView, let document = scroll.documentView else {
            fputs("FAIL: probe did not attach to a scroll view\n", stderr)
            exit(1)
        }

        // SwiftUI's full-size macOS window contributes a titlebar content
        // inset to the transcript scroll view. Reproduce it explicitly so
        // this check remains deterministic outside a full App scene.
        scroll.automaticallyAdjustsContentInsets = false
        scroll.contentInsets = NSEdgeInsets(top: 48, left: 0, bottom: 0, right: 0)
        model.rowCount = 3
        pump(0.2)
        model.rowCount = 2
        pump(0.2)

        // A short conversation must remain wholly visible. This is the exact
        // shape where an eager bottom anchor can leave the first user bubble
        // above the clip view even though most of the transcript is blank;
        // trying to reveal it then only produces AppKit's elastic snap-back.
        let shortDocumentRect = document.bounds
        let shortVisibleRect = scroll.documentVisibleRect
        guard shortDocumentRect.height <= shortVisibleRect.height + 0.5,
              shortVisibleRect.contains(shortDocumentRect)
        else {
            fputs("FAIL: short transcript hid content above the viewport\n", stderr)
            fputs("\(metrics(scroll))\n", stderr)
            exit(1)
        }
        let expectedShortY = document.bounds.minY - scroll.contentInsets.top
        guard abs(scroll.contentView.bounds.minY - expectedShortY) <= 0.5 else {
            fputs("FAIL: short transcript was pushed beneath its top content inset\n", stderr)
            fputs(
                "expectedY=\(expectedShortY) actualY=\(scroll.contentView.bounds.minY) "
                    + "\(metrics(scroll))\n",
                stderr
            )
            exit(1)
        }

        // A SwiftUI Window uses a full-size content view: the scroll view may
        // geometrically extend beneath the translucent titlebar, but its
        // scroll content must be inset by the overlap. Otherwise the first
        // message is visible to NSScrollView yet blurred behind the titlebar,
        // and attempting to reveal it only hits the elastic top boundary.
        let scrollRectInHost = scroll.convert(scroll.bounds, to: host)
        let safeContentTop = host.bounds.maxY - host.safeAreaInsets.top
        let titlebarOverlap = max(0, scrollRectInHost.maxY - safeContentTop)
        guard scroll.contentInsets.top + 0.5 >= titlebarOverlap else {
            fputs("FAIL: transcript content extends beneath the titlebar\n", stderr)
            fputs(
                "overlap=\(titlebarOverlap) host.safe=\(host.safeAreaInsets) "
                    + "scroll.frame=\(scrollRectInHost) \(metrics(scroll))\n",
                stderr
            )
            exit(1)
        }

        model.rowCount = 120
        pump(0.4)

        var bottomY = bottomTargetY(scroll, document: document)
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

            bottomY = bottomTargetY(scroll, document: document)
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
            let distance = bottomDistance(scroll, document: document)
            guard model.isPinnedToBottom && distance <= 2.5 else {
                fputs("FAIL: cycle \(cycle) streamed content was not followed after resuming\n", stderr)
                fputs("distance=\(distance) \(metrics(scroll))\n", stderr)
                exit(1)
            }
            bottomY = bottomTargetY(scroll, document: document)
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
        let gentleDistance = bottomDistance(scroll, document: document)
        guard !model.isPinnedToBottom, gentleDistance > 2 else {
            fputs("FAIL: a gentle (sub-slack) scroll was dragged back to the bottom\n", stderr)
            fputs("distance=\(gentleDistance) pinned=\(model.isPinnedToBottom) \(metrics(scroll))\n", stderr)
            exit(1)
        }

        print("PASS: titlebar insets, paused scrolling, bottom resume, and gentle scrolling")
        window.close()
    }
}
