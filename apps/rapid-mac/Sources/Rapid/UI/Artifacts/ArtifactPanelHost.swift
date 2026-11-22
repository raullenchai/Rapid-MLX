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

    /// The transcript keeps at least this much, however wide the panel is
    /// dragged.
    private static let minimumTranscriptWidth: CGFloat = 360

    private var panelWidth: CGFloat {
        let room = containerWidth - Self.minimumTranscriptWidth
        let lower = ArtifactPanelModel.widthRange.lowerBound
        return min(model.panelWidth, max(lower, room))
    }

    func body(content: Content) -> some View {
        HStack(spacing: 0) {
            content
                .environment(\.artifactOpener, model.opener)
            if model.isOpen {
                ArtifactPanelResizeHandle(model: model, effectiveWidth: panelWidth)
                ArtifactPanelView(model: model)
                    .frame(width: panelWidth)
            }
        }
        .background(
            GeometryReader { proxy in
                Color.clear.preference(key: ArtifactContainerWidthKey.self, value: proxy.size.width)
            }
        )
        .onPreferenceChange(ArtifactContainerWidthKey.self) { containerWidth = $0 }
        .onChange(of: ArtifactMessageFingerprint.of(messages), initial: true) { _, _ in
            model.messagesDidChange(messages)
        }
    }
}

private struct ArtifactContainerWidthKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

/// The draggable seam between transcript and panel.
private struct ArtifactPanelResizeHandle: View {
    let model: ArtifactPanelModel
    let effectiveWidth: CGFloat
    @State private var widthAtDragStart: CGFloat?

    private static let keyboardStep: CGFloat = 40

    var body: some View {
        Rectangle()
            .fill(RapidTheme.hairline)
            .frame(width: 1)
            .padding(.horizontal, 3)
            .contentShape(Rectangle())
            .onHover { inside in
                if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
            }
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { value in
                        let start = widthAtDragStart ?? effectiveWidth
                        widthAtDragStart = start
                        model.setPanelWidth(start - value.translation.width)
                    }
                    .onEnded { _ in widthAtDragStart = nil }
            )
            .accessibilityElement()
            .accessibilityLabel("Resize artifacts panel")
            .accessibilityValue(String(localized: "\(Int(effectiveWidth)) points wide"))
            .accessibilityAdjustableAction { direction in
                switch direction {
                case .increment: model.setPanelWidth(effectiveWidth + Self.keyboardStep)
                case .decrement: model.setPanelWidth(effectiveWidth - Self.keyboardStep)
                @unknown default: break
                }
            }
            .accessibilityIdentifier("Artifact.Panel.Resize")
    }
}
