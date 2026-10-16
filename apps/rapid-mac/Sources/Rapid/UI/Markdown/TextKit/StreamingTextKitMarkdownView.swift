import AppKit
import SwiftUI

/// The streaming message's body.
///
/// Reads compiled blocks from ``StreamingMarkdownStore`` rather than taking
/// the raw string as a parameter. That indirection is the point: a
/// `let content: String` on this view changes on every coalesced SSE batch,
/// so SwiftUI rebuilds the row around it ~20× a second. The store instead
/// publishes stable blocks and a mutable tail at display cadence; see its doc
/// comment for the measurements that motivated the change.
struct StreamingTextKitMarkdownView: View {
    @Bindable var store: StreamingMarkdownStore
    let messageID: UUID

    /// One timeline per message. Segment boundaries are an implementation
    /// detail and must not restart the reveal as stable blocks are committed.
    @State private var fadeState = TextFadeAnimationState()

    @ScaledMetric(relativeTo: .body) private var basePointSize: CGFloat = 15

    var body: some View {
        let options = TextKitMarkdownView.options(basePointSize: basePointSize)
        VStack(alignment: .leading, spacing: options.interContentSpacing) {
            ForEach(store.segments(for: messageID)) { segment in
                StreamingMarkdownSegmentView(
                    segment: segment,
                    basePointSize: basePointSize,
                    uiFontScale: UIFontSettings.shared.scale,
                    fadeState: fadeState,
                    fadeConfiguration: Self.fadeConfiguration
                )
                .equatable()
            }
        }
        .chatLinkSafetyFilter()
    }

    private static let fadeConfiguration: TextFadeConfiguration = {
        if UserDefaults.standard.bool(forKey: "rapid.chat.fade.disabled") { return .off }
        if NSWorkspace.shared.accessibilityDisplayShouldReduceMotion { return .off }
        return TextFadeConfiguration()
    }()
}

/// A committed segment stops changing and is skipped by SwiftUI's diffing.
/// The mutable segment keeps the same ID when committed, preserving its
/// TextKit views while a new mutable segment is mounted after it.
private struct StreamingMarkdownSegmentView: View, Equatable {
    let segment: StreamingMarkdownDocument.Segment
    let basePointSize: CGFloat
    /// Part of equality so a committed segment re-renders on a text-size change.
    let uiFontScale: Double
    let fadeState: TextFadeAnimationState
    let fadeConfiguration: TextFadeConfiguration

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.segment == rhs.segment && lhs.basePointSize == rhs.basePointSize
            && lhs.uiFontScale == rhs.uiFontScale
    }

    var body: some View {
        MarkdownBlockStack(
            result: segment.result,
            options: TextKitMarkdownView.options(basePointSize: basePointSize),
            isStreaming: segment.isMutable,
            fadeState: fadeState,
            fadeConfiguration: fadeConfiguration
        )
    }
}
