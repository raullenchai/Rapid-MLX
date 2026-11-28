import AppKit
import SwiftUI

extension View {
    /// Puts the artifact panel to the right of this view and lets the code
    /// blocks inside it open artifacts into the panel.
    func artifactPanel(model: ArtifactPanelModel, messages: [ChatMessage]) -> some View {
        modifier(ArtifactPanelHost(model: model, messages: messages))
    }
}

private struct ArtifactPanelHost: ViewModifier {
    @Bindable var model: ArtifactPanelModel
    let messages: [ChatMessage]
    @State private var containerWidth: CGFloat = 0
    /// The transcript's width the last time it was visible. Full screen keeps
    /// laying it out at this width while it takes no room, so a long
    /// transcript is not re-measured down to nothing and back.
    @State private var transcriptWidth: CGFloat = 0

    /// The transcript keeps at least this much, however wide the panel is
    /// dragged.
    private static let minimumTranscriptWidth: CGFloat = 360

    private var isFullScreen: Bool { model.isOpen && model.isFullScreen }

    private var panelWidth: CGFloat {
        // Until the first measurement arrives there is nothing to clamp to.
        guard containerWidth > 0 else { return model.panelWidth }
        let room = containerWidth - Self.minimumTranscriptWidth
        let lower = ArtifactPanelModel.widthRange.lowerBound
        return min(model.panelWidth, max(lower, room))
    }

    func body(content: Content) -> some View {
        HStack(spacing: 0) {
            content
                .environment(\.artifactOpener, model.opener)
                .background(
                    GeometryReader { proxy in
                        Color.clear.preference(key: ArtifactTranscriptWidthKey.self, value: proxy.size.width)
                    }
                )
                .frame(width: isFullScreen ? max(transcriptWidth, 1) : nil)
                .frame(width: isFullScreen ? 0 : nil, alignment: .leading)
                .clipped()
                .opacity(isFullScreen ? 0 : 1)
                .allowsHitTesting(!isFullScreen)
                .accessibilityHidden(isFullScreen)
            if model.isOpen {
                if !isFullScreen {
                    ArtifactPanelResizeHandle(model: model, effectiveWidth: panelWidth)
                }
                ArtifactPanelView(model: model)
                    .frame(
                        minWidth: isFullScreen ? 0 : panelWidth,
                        maxWidth: isFullScreen ? .infinity : panelWidth
                    )
            }
        }
        .background(
            GeometryReader { proxy in
                Color.clear.preference(key: ArtifactContainerWidthKey.self, value: proxy.size.width)
            }
        )
        .onPreferenceChange(ArtifactContainerWidthKey.self) { containerWidth = $0 }
        .onPreferenceChange(ArtifactTranscriptWidthKey.self) { width in
            if !isFullScreen, width > 0 { transcriptWidth = width }
        }
        .onChange(of: ArtifactMessageFingerprint.of(messages), initial: true) { _, _ in
            model.messagesDidChange(messages)
        }
    }
}

private struct ArtifactContainerWidthKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

private struct ArtifactTranscriptWidthKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

/// The draggable seam between transcript and panel.
///
/// AppKit rather than a SwiftUI drag gesture: the drag is tracked in window
/// coordinates, so it keeps working while the seam itself moves under the
/// cursor, and it opts out of window dragging.
private struct ArtifactPanelResizeHandle: NSViewRepresentable {
    let model: ArtifactPanelModel
    let effectiveWidth: CGFloat

    func makeNSView(context: Context) -> ArtifactResizeSeam {
        let seam = ArtifactResizeSeam()
        seam.model = model
        seam.currentWidth = effectiveWidth
        return seam
    }

    func updateNSView(_ seam: ArtifactResizeSeam, context: Context) {
        seam.model = model
        seam.currentWidth = effectiveWidth
    }

    func sizeThatFits(
        _ proposal: ProposedViewSize, nsView: ArtifactResizeSeam, context: Context
    ) -> CGSize? {
        CGSize(width: ArtifactResizeSeam.hitWidth, height: proposal.height ?? 0)
    }
}

final class ArtifactResizeSeam: NSView {
    static let hitWidth: CGFloat = 9
    private static let keyboardStep: CGFloat = 40

    weak var model: ArtifactPanelModel?
    var currentWidth: CGFloat = 0
    private var dragStartX: CGFloat = 0
    private var dragStartWidth: CGFloat = 0

    override var mouseDownCanMoveWindow: Bool { false }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func resetCursorRects() {
        addCursorRect(bounds, cursor: .resizeLeftRight)
    }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.separatorColor.setFill()
        NSRect(x: bounds.midX - 0.5, y: bounds.minY, width: 1, height: bounds.height).fill()
    }

    override func mouseDown(with event: NSEvent) {
        dragStartX = event.locationInWindow.x
        dragStartWidth = currentWidth
    }

    override func mouseDragged(with event: NSEvent) {
        let delta = event.locationInWindow.x - dragStartX
        model?.setPanelWidth(dragStartWidth - delta)
    }

    override func isAccessibilityElement() -> Bool { true }
    override func accessibilityRole() -> NSAccessibility.Role? { .splitter }
    override func accessibilityLabel() -> String? { String(localized: "Resize artifacts panel") }
    override func accessibilityValue() -> Any? {
        String(localized: "\(Int(currentWidth)) points wide")
    }
    override func accessibilityIdentifier() -> String { "Artifact.Panel.Resize" }

    override func accessibilityPerformIncrement() -> Bool {
        model?.setPanelWidth(currentWidth + Self.keyboardStep)
        return true
    }

    override func accessibilityPerformDecrement() -> Bool {
        model?.setPanelWidth(currentWidth - Self.keyboardStep)
        return true
    }
}
