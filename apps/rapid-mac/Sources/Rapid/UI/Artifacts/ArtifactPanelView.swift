import AppKit
import SwiftUI

/// The side panel: the artifact the reader opened from the transcript, with
/// the other artifacts of this branch listed under its header.
struct ArtifactPanelView: View {
    @Bindable var model: ArtifactPanelModel

    var body: some View {
        Group {
            if let item = model.selected {
                ArtifactViewer(item: item) {
                    controls
                } accessory: {
                    if model.items.count > 1 {
                        list
                        Divider()
                    }
                }
                .id("\(item.id)|\(item.source.hashValue)")
            } else {
                VStack(spacing: 0) {
                    HStack {
                        Text("Artifacts")
                            .font(.rapid(.subheadline, weight: .semibold))
                        Spacer(minLength: RapidTheme.Space.sm)
                        controls
                    }
                    .padding(.horizontal, RapidTheme.Space.md)
                    .padding(.vertical, RapidTheme.Space.sm)
                    Divider()
                    emptyState
                }
            }
        }
        .background(RapidTheme.surfaceCanvas)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Artifacts")
        .accessibilityIdentifier("Artifact.Panel")
    }

    /// Full screen hides the chat until the panel is closed or collapsed.
    @ViewBuilder
    private var controls: some View {
        QuietIconButton(
            symbol: model.isFullScreen
                ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right",
            label: model.isFullScreen
                ? String(localized: "Show chat beside the panel")
                : String(localized: "Make the panel full screen")
        ) { model.toggleFullScreen() }
            .accessibilityIdentifier("Artifact.Panel.FullScreen")
        QuietIconButton(
            symbol: "xmark", label: String(localized: "Close artifacts panel")
        ) { model.close() }
            .accessibilityIdentifier("Artifact.Panel.Close")
    }

    private var list: some View {
        List(
            selection: Binding(
                get: { model.selectedID },
                set: { if let id = $0 { model.select(id) } }
            )
        ) {
            ForEach(model.items) { item in
                HStack(spacing: RapidTheme.Space.sm) {
                    Image(systemName: ArtifactKind.iconName(for: item.kind))
                        .foregroundStyle(.secondary)
                        .frame(width: 16)
                        .accessibilityHidden(true)
                    Text(item.title).lineLimit(1)
                    Spacer(minLength: 0)
                    Text(item.sizeDescription)
                        .font(.rapid(.caption))
                        .foregroundStyle(.secondary)
                }
                .font(.rapid(.footnote))
                .tag(item.id)
            }
        }
        .listStyle(.plain)
        .frame(height: min(CGFloat(model.items.count) * 28 + 8, 140))
        .accessibilityIdentifier("Artifact.Panel.List")
    }

    private var emptyState: some View {
        VStack(spacing: RapidTheme.Space.xs) {
            Text("No artifacts yet")
                .font(.rapid(.subheadline, weight: .semibold))
            Text("HTML pages, SVG images and diagrams in this conversation appear here.")
                .font(.rapid(.footnote))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(RapidTheme.Space.lg)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

extension ArtifactKind {
    static func iconName(for kind: ArtifactKind) -> String {
        switch kind {
        case .html: "globe"
        case .svg: "photo"
        case .mermaid: "point.3.connected.trianglepath.dotted"
        }
    }
}
