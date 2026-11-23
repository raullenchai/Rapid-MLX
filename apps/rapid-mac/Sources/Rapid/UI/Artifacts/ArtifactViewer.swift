import AppKit
import SwiftUI

/// One artifact at full size: title, toolbar, and the content. Shared by the
/// side panel and the library's full-window view, which differ only in what
/// sits beside the toolbar and under the header.
struct ArtifactViewer<Trailing: View, Accessory: View>: View {
    let item: ArtifactItem
    let trailing: Trailing
    let accessory: Accessory

    @State private var copiedRecently = false
    @State private var actionError: String?
    /// Zoom-to-fit for SVG and Mermaid; off shows the picture at its own size.
    @State private var fitsToPanel = true
    /// The diagram the image preview last drew, kept so Save can write a PNG.
    @State private var renderedDiagram: NSImage?
    @State private var reloadToken = 0

    init(
        item: ArtifactItem,
        @ViewBuilder trailing: () -> Trailing,
        @ViewBuilder accessory: () -> Accessory
    ) {
        self.item = item
        self.trailing = trailing()
        self.accessory = accessory()
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            accessory
            if let actionError {
                errorBanner(actionError)
            }
            content
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: RapidTheme.Space.xs) {
            VStack(alignment: .leading, spacing: 0) {
                Text(item.title)
                    .font(.rapid(.subheadline, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.tail)
                Text(item.sizeDescription)
                    .font(.rapid(.caption))
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: RapidTheme.Space.sm)
            toolbar
            trailing
        }
        .padding(.horizontal, RapidTheme.Space.md)
        .padding(.vertical, RapidTheme.Space.sm)
    }

    @ViewBuilder
    private var toolbar: some View {
        if item.kind != .html {
            QuietIconButton(
                symbol: fitsToPanel ? "1.magnifyingglass" : "arrow.up.left.and.down.right.magnifyingglass",
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
        saveControl
        if item.kind == .html {
            QuietIconButton(
                symbol: "arrow.clockwise", label: String(localized: "Reload preview")
            ) {
                actionError = nil
                reloadToken &+= 1
            }
            .accessibilityIdentifier("Artifact.Panel.Reload")
            QuietIconButton(
                symbol: "safari", label: String(localized: "Open in browser")
            ) { actionError = ArtifactPanelActions.openInBrowser(item) }
                .accessibilityIdentifier("Artifact.Panel.OpenInBrowser")
        }
    }

    @ViewBuilder
    private var saveControl: some View {
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

    // MARK: Content

    @ViewBuilder
    private var content: some View {
        switch item.kind {
        case .html:
            ArtifactHTMLPreview(item: item)
                .id("\(item.id)|\(item.source.hashValue)|\(reloadToken)")
        case .svg, .mermaid:
            ArtifactImagePreview(item: item, fitsToPanel: fitsToPanel, rendered: $renderedDiagram)
                .id("\(item.id)|\(item.source.hashValue)")
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
struct ArtifactHTMLPreview: View {
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
struct ArtifactImagePreview: View {
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
                ArtifactFittedImage(image: image, label: item.title, fits: fitsToPanel, vector: item.kind == .svg)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: colorScheme) { await load() }
    }

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

/// A picture on a neutral plate, either fitted to the space or at its own size.
struct ArtifactFittedImage: View {
    let image: NSImage
    let label: String
    let fits: Bool
    let vector: Bool

    var body: some View {
        GeometryReader { proxy in
            if fits {
                let scale = ArtifactPanelFit.scale(image: image.size, available: proxy.size, vector: vector)
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
        .accessibilityLabel(Text(label))
        .accessibilityAddTraits(.isImage)
    }

    private var plate: some View { Color(nsColor: .textBackgroundColor) }
}

enum ArtifactPanelFit {
    /// Fit inside the space with a margin. A diagram is a 2x bitmap, so it is
    /// not blown up past that; vector art can grow further.
    nonisolated static func scale(image: CGSize, available: CGSize, vector: Bool) -> CGFloat {
        guard image.width > 0, image.height > 0 else { return 1 }
        let margin: CGFloat = 32
        let width = max(1, available.width - margin)
        let height = max(1, available.height - margin)
        return min(width / image.width, height / image.height, vector ? 8 : 2)
    }
}
