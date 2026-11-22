import AppKit
import SwiftUI

/// The side panel: the artifacts in this branch of the conversation, and the
/// one being shown.
struct ArtifactPanelView: View {
    @Bindable var model: ArtifactPanelModel
    @State private var copiedRecently = false
    @State private var actionError: String?
    /// Zoom-to-fit for SVG and Mermaid; off shows the picture at its own size.
    @State private var fitsToPanel = true
    /// The diagram the image preview last drew, kept so Save can write a PNG.
    @State private var renderedDiagram: NSImage?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if model.items.count > 1 {
                list
                Divider()
            }
            if let actionError {
                errorBanner(actionError)
            }
            content
        }
        .background(RapidTheme.surfaceCanvas)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Artifacts")
        .accessibilityIdentifier("Artifact.Panel")
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: RapidTheme.Space.xs) {
            VStack(alignment: .leading, spacing: 0) {
                Text(model.selected?.title ?? String(localized: "Artifacts"))
                    .font(.rapid(.subheadline, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.tail)
                if let item = model.selected {
                    Text(item.sizeDescription)
                        .font(.rapid(.caption))
                        .foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: RapidTheme.Space.sm)
            if let item = model.selected {
                toolbar(for: item)
            }
            QuietIconButton(
                symbol: "xmark", label: String(localized: "Close artifacts panel")
            ) { model.close() }
                .accessibilityIdentifier("Artifact.Panel.Close")
        }
        .padding(.horizontal, RapidTheme.Space.md)
        .padding(.vertical, RapidTheme.Space.sm)
    }

    @ViewBuilder
    private func toolbar(for item: ArtifactItem) -> some View {
        if item.kind != .html {
            QuietIconButton(
                symbol: fitsToPanel
                    ? "arrow.up.left.and.arrow.down.right" : "arrow.down.right.and.arrow.up.left",
                label: fitsToPanel ? String(localized: "Show at actual size") : String(localized: "Zoom to fit")
            ) { fitsToPanel.toggle() }
                .accessibilityIdentifier("Artifact.Panel.Zoom")
        }
        QuietIconButton(
            symbol: copiedRecently ? "checkmark" : "doc.on.doc",
            label: String(localized: "Copy source"),
            tint: copiedRecently ? RapidTheme.utilityActionSuccess : nil
        ) {
            ArtifactPanelActions.copySource(of: item)
            copiedRecently = true
        }
        .accessibilityIdentifier("Artifact.Panel.Copy")
        .task(id: copiedRecently) {
            guard copiedRecently else { return }
            try? await Task.sleep(for: .seconds(1.2))
            copiedRecently = false
        }
        saveControl(for: item)
        if item.kind == .html {
            QuietIconButton(
                symbol: "arrow.clockwise", label: String(localized: "Reload preview")
            ) {
                actionError = nil
                model.reload()
            }
            .accessibilityIdentifier("Artifact.Panel.Reload")
            QuietIconButton(
                symbol: "safari", label: String(localized: "Open in browser")
            ) { actionError = ArtifactPanelActions.openInBrowser(item) }
                .accessibilityIdentifier("Artifact.Panel.OpenInBrowser")
        }
    }

    @ViewBuilder
    private func saveControl(for item: ArtifactItem) -> some View {
        if item.kind == .mermaid {
            Menu {
                Button("Save source (.mmd)") {
                    actionError = ArtifactPanelActions.save(item)
                }
                .accessibilityIdentifier("Artifact.Panel.Save.Source")
                Button("Save image (PNG)") {
                    actionError = ArtifactPanelActions.save(
                        item, format: .png,
                        png: renderedDiagram.flatMap(ArtifactPanelActions.pngData(for:))
                    )
                }
                .disabled(renderedDiagram == nil)
                .accessibilityIdentifier("Artifact.Panel.Save.PNG")
            } label: {
                Image(systemName: "square.and.arrow.down")
                    .font(.rapid(size: 11))
                    .foregroundStyle(RapidTheme.utilityActionLabel)
                    .frame(width: RapidTheme.ControlHeight.small, height: RapidTheme.ControlHeight.small)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .accessibilityLabel("Save as…")
            .accessibilityIdentifier("Artifact.Panel.Save")
        } else {
            QuietIconButton(
                symbol: "square.and.arrow.down", label: String(localized: "Save as…")
            ) { actionError = ArtifactPanelActions.save(item) }
                .accessibilityIdentifier("Artifact.Panel.Save")
        }
    }

    // MARK: List

    private var list: some View {
        List(
            selection: Binding(
                get: { model.selectedID },
                set: { if let id = $0 { model.select(id) } }
            )
        ) {
            ForEach(model.items) { item in
                HStack(spacing: RapidTheme.Space.sm) {
                    Image(systemName: Self.iconName(for: item.kind))
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

    private static func iconName(for kind: ArtifactKind) -> String {
        switch kind {
        case .html: "globe"
        case .svg: "photo"
        case .mermaid: "point.3.connected.trianglepath.dotted"
        }
    }

    // MARK: Content

    @ViewBuilder
    private var content: some View {
        if let item = model.selected {
            switch item.kind {
            case .html:
                ArtifactHTMLPreview(item: item)
                    .id("\(item.id)|\(item.source.hashValue)|\(model.reloadToken)")
            case .svg, .mermaid:
                ArtifactImagePreview(
                    item: item, fitsToPanel: fitsToPanel, rendered: $renderedDiagram
                )
                .id("\(item.id)|\(item.source.hashValue)")
            }
        } else {
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

    private func errorBanner(_ message: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(message)
                .font(.rapid(.footnote))
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("Dismiss") { actionError = nil }
                .buttonStyle(.link)
                .font(.rapid(.footnote))
                .accessibilityIdentifier("Artifact.Panel.DismissError")
        }
        .padding(.horizontal, RapidTheme.Space.md)
        .padding(.vertical, RapidTheme.Space.xs)
    }
}

// MARK: - HTML

/// The live page. One of these at a time: it is built when it appears and
/// torn down when it goes, so a long transcript never holds a web process.
private struct ArtifactHTMLPreview: View {
    let item: ArtifactItem
    @State private var controller: ArtifactPreviewController

    init(item: ArtifactItem) {
        self.item = item
        _controller = State(initialValue: ArtifactPreviewController(source: item.source))
    }

    var body: some View {
        VStack(spacing: 0) {
            if item.referencesRemoteResources {
                Text("Online resources are blocked in previews.")
                    .font(.rapid(.footnote))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, RapidTheme.Space.md)
                    .padding(.vertical, RapidTheme.Space.xs)
                    .background(RapidTheme.surfaceBand)
                    .accessibilityIdentifier("Artifact.Panel.BlockedNotice")
            }
            ZStack {
                ArtifactWebHost(controller: controller)
                overlay
            }
        }
        .onAppear { controller.start() }
        .onDisappear { controller.teardown() }
    }

    @ViewBuilder
    private var overlay: some View {
        switch controller.state {
        case .idle, .loading:
            ProgressView().controlSize(.small)
        case .ready:
            EmptyView()
        case .failed(let failure):
            VStack(spacing: RapidTheme.Space.sm) {
                Text(Self.message(for: failure))
                    .font(.rapid(.footnote))
                    .multilineTextAlignment(.center)
                Button("Reload") { controller.reload() }
                    .accessibilityIdentifier("Artifact.Panel.ReloadFailed")
            }
            .padding(RapidTheme.Space.lg)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(RapidTheme.surfaceCanvas)
        }
    }

    private static func message(for failure: ArtifactPreviewController.Failure) -> String {
        switch failure {
        case .sandboxUnavailable:
            String(localized: "The preview sandbox couldn't be set up, so nothing was loaded.")
        case .loadFailed: String(localized: "The preview couldn't be loaded.")
        case .timedOut: String(localized: "The preview took too long to load.")
        case .unresponsive: String(localized: "The preview stopped responding.")
        case .contentProcessTerminated: String(localized: "The preview stopped unexpectedly.")
        }
    }
}

private struct ArtifactWebHost: NSViewRepresentable {
    let controller: ArtifactPreviewController

    func makeNSView(context: Context) -> NSView { controller.hostView }
    func updateNSView(_ view: NSView, context: Context) {}
}

// MARK: - SVG and Mermaid

/// A static picture at panel size. Interactive pages are the only thing that
/// gets a web view.
private struct ArtifactImagePreview: View {
    let item: ArtifactItem
    let fitsToPanel: Bool
    @Binding var rendered: NSImage?
    @Environment(\.colorScheme) private var colorScheme
    @State private var phase: Phase = .loading

    private enum Phase {
        case loading
        case ready(NSImage)
        case failed
    }

    var body: some View {
        Group {
            switch phase {
            case .loading:
                ProgressView().controlSize(.small)
            case .failed:
                Text("This artifact can't be previewed.")
                    .font(.rapid(.footnote))
                    .foregroundStyle(.secondary)
            case .ready(let image):
                picture(image)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: colorScheme) { await load() }
    }

    @ViewBuilder
    private func picture(_ image: NSImage) -> some View {
        GeometryReader { proxy in
            if fitsToPanel {
                let scale = ArtifactPanelFit.scale(
                    image: image.size, available: proxy.size, vector: item.kind == .svg
                )
                Image(nsImage: image)
                    .resizable()
                    .frame(width: image.size.width * scale, height: image.size.height * scale)
                    .background(plate)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView([.horizontal, .vertical]) {
                    Image(nsImage: image)
                        .resizable()
                        .frame(width: image.size.width, height: image.size.height)
                        .background(plate)
                        .padding(RapidTheme.Space.md)
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(item.title))
        .accessibilityAddTraits(.isImage)
    }

    private var plate: some View { Color(nsColor: .textBackgroundColor) }

    private func load() async {
        rendered = nil
        switch item.kind {
        case .svg:
            if let image = SVGPreview.image(from: item.source) {
                phase = .ready(image)
            } else {
                phase = .failed
            }
        case .mermaid:
            phase = .loading
            let theme: MermaidRenderer.Theme = colorScheme == .dark ? .dark : .light
            if let image = await MermaidRenderer.shared.image(source: item.source, theme: theme) {
                rendered = image
                phase = .ready(image)
            } else {
                phase = .failed
            }
        case .html:
            phase = .failed
        }
    }
}

enum ArtifactPanelFit {
    /// Fit inside the panel with a margin. A diagram is a 2x bitmap, so it is
    /// not blown up past that; vector art can grow further.
    nonisolated static func scale(image: CGSize, available: CGSize, vector: Bool) -> CGFloat {
        guard image.width > 0, image.height > 0 else { return 1 }
        let margin: CGFloat = 32
        let width = max(1, available.width - margin)
        let height = max(1, available.height - margin)
        return min(width / image.width, height / image.height, vector ? 8 : 2)
    }
}
