import AppKit
import SwiftUI

/// The streaming message's body.
///
/// Reads compiled blocks from ``StreamingMarkdownStore`` rather than taking
/// the raw string as a parameter. That indirection is the point: a
/// `let content: String` on this view changes on every coalesced SSE batch,
/// so SwiftUI rebuilds the row around it ~20× a second. The store publishes
/// on the compiler's 100 ms beat instead — see its doc comment for the
/// measurements that motivated the change.
struct StreamingTextKitMarkdownView: View {
    @Bindable var store: StreamingMarkdownStore

    /// Shared across every block of one message so the reveal runs on a
    /// single timeline. Without it each block would restart the animation at
    /// its own boundary, and a long answer would visibly re-fade at every
    /// paragraph.
    @State private var fadeState = TextFadeAnimationState()

    @ScaledMetric(relativeTo: .body) private var basePointSize: CGFloat = 15

    var body: some View {
        MarkdownBlockStack(
            result: store.result,
            options: TextKitMarkdownView.options(basePointSize: basePointSize),
            isStreaming: true,
            fadeState: fadeState,
            fadeConfiguration: Self.fadeConfiguration
        )
        .chatLinkSafetyFilter()
    }

    /// Word-by-word reveal, driven by a display link against the layout
    /// manager's rendering attributes.
    ///
    /// This is what makes streamed text read as text arriving rather than as
    /// a buffer being redrawn: `setRenderingAttributes` changes a glyph's
    /// alpha without invalidating layout, so a word can fade in without
    /// moving anything around it.
    ///
    /// Off under Reduce Motion, and behind a defaults key for anyone who
    /// wants the old instant paint.
    private static let fadeConfiguration: TextFadeConfiguration = {
        if UserDefaults.standard.bool(forKey: "rapid.chat.fade.disabled") { return .off }
        if NSWorkspace.shared.accessibilityDisplayShouldReduceMotion { return .off }
        return TextFadeConfiguration()
    }()
}
