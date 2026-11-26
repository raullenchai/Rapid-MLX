import AVFoundation
import AppKit
import SwiftUI

/// The Artifacts page: everything kept from chats, image generation and
/// speech, in one list. Choosing a row shows it across the whole window.
struct ArtifactLibraryView: View {
    let library: ArtifactLibrary
    let conversations: [ChatConversation]
    var onOpenConversation: (UUID) -> Void = { _ in }

    private enum Filter: CaseIterable, Identifiable {
        case all, pages, graphics, images, audio

        var id: Self { self }

        var label: String {
            switch self {
            case .all: String(localized: "All")
            case .pages: String(localized: "Pages")
            case .graphics: String(localized: "Diagrams")
            case .images: String(localized: "Images")
            case .audio: String(localized: "Audio")
            }
        }

        func includes(_ kind: ArtifactLibraryItem.Kind) -> Bool {
            switch self {
            case .all: true
            case .pages: kind == .page
            case .graphics: kind == .svg || kind == .diagram
            case .images: kind == .image
            case .audio: kind == .audio
            }
        }
    }

    @State private var selection: ArtifactLibraryItem.ID?
    @State private var opened: ArtifactLibraryItem?
    @State private var filter: Filter = .all
    @State private var sortOrder = [KeyPathComparator(\ArtifactLibraryItem.date, order: .reverse)]

    /// Reload when a chat changes or when the page appears, which is also when
    /// an image or speech result made elsewhere becomes visible.
    private var refreshKey: [Date] { conversations.map(\.updatedAt) }

    private var rows: [ArtifactLibraryItem] {
        library.items.filter { filter.includes($0.kind) }.sorted(using: sortOrder)
    }

    var body: some View {
        Group {
            if let opened {
                ArtifactLibraryDetailView(
                    item: opened, library: library,
                    onBack: {
                        self.opened = nil
                        selection = nil
                    },
                    onOpenConversation: onOpenConversation
                )
            } else {
                list
            }
        }
        .background(RapidTheme.surfaceCanvas)
        .task(id: refreshKey) { await library.refresh(conversations: conversations) }
        .accessibilityIdentifier("Artifact.Library")
    }

    private var list: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Artifacts")
                    .font(.rapid(.title3, weight: .semibold))
                Spacer()
                Picker("Show", selection: $filter) {
                    ForEach(Filter.allCases) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 380)
                .accessibilityIdentifier("Artifact.Library.Filter")
            }
            .padding(.horizontal, RapidTheme.Space.lg)
            .padding(.vertical, RapidTheme.Space.md)
            Divider()
            if rows.isEmpty {
                emptyState
            } else {
                table
            }
        }
    }

    private var table: some View {
        Table(rows, selection: $selection, sortOrder: $sortOrder) {
            TableColumn("Name", value: \.title) { item in
                HStack(spacing: RapidTheme.Space.sm) {
                    Image(systemName: item.kind.iconName)
                        .foregroundStyle(.secondary)
                        .frame(width: 16)
                        .accessibilityHidden(true)
                    Text(item.title).lineLimit(1)
                }
            }
            TableColumn("Type", value: \.kindLabel) { item in
                Text(item.kindLabel).foregroundStyle(.secondary)
            }
            .width(min: 60, ideal: 80, max: 120)
            TableColumn("Source", value: \.originLabel) { item in
                Text(item.originLabel).lineLimit(1).foregroundStyle(.secondary)
            }
            TableColumn("Date", value: \.date) { item in
                Text(item.date, format: .dateTime.year().month().day().hour().minute())
                    .foregroundStyle(.secondary)
            }
            .width(min: 110, ideal: 150, max: 200)
            TableColumn("Size", value: \.byteCount) { item in
                Text(ByteCountFormatter.string(fromByteCount: Int64(item.byteCount), countStyle: .file))
                    .foregroundStyle(.secondary)
            }
            .width(min: 60, ideal: 80, max: 110)
        }
        .onChange(of: selection) { _, id in
            guard let id, let item = library.items.first(where: { $0.id == id }) else { return }
            opened = item
        }
        .accessibilityIdentifier("Artifact.Library.Table")
    }

    private var emptyState: some View {
        VStack(spacing: RapidTheme.Space.xs) {
            if library.isLoading {
                ProgressView().controlSize(.small)
            } else {
                Text("No artifacts yet")
                    .font(.rapid(.subheadline, weight: .semibold))
                Text("HTML pages, SVG images and diagrams from your chats, generated images and speech are kept here.")
                    .font(.rapid(.footnote))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(RapidTheme.Space.lg)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Detail

/// One library item across the whole window, with a way back.
struct ArtifactLibraryDetailView: View {
    let item: ArtifactLibraryItem
    let library: ArtifactLibrary
    var onBack: () -> Void
    var onOpenConversation: (UUID) -> Void

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Divider()
            content
        }
        .accessibilityIdentifier("Artifact.Library.Detail")
    }

    private var topBar: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Button(action: onBack) {
                Label("Artifacts", systemImage: "chevron.left")
            }
            .buttonStyle(.link)
            .accessibilityIdentifier("Artifact.Library.Back")
            Spacer(minLength: RapidTheme.Space.sm)
            Text(item.originLabel)
                .font(.rapid(.footnote))
                .foregroundStyle(.secondary)
                .lineLimit(1)
            if case .conversation(let id, _) = item.origin {
                Button("Open chat") { onOpenConversation(id) }
                    .buttonStyle(.link)
                    .font(.rapid(.footnote))
                    .accessibilityIdentifier("Artifact.Library.OpenChat")
            }
        }
        .padding(.horizontal, RapidTheme.Space.lg)
        .padding(.vertical, RapidTheme.Space.sm)
    }

    @ViewBuilder
    private var content: some View {
        switch item.payload {
        case .artifact(let artifact):
            ArtifactViewer(item: artifact) {
            } accessory: {
            }
            .id(item.id)
        case .image(let id, let prompt, let size, let isEdit):
            GeneratedImageDetail(
                url: library.imageURL(for: id), prompt: prompt, size: size,
                isEdit: isEdit, model: item.originLabel
            )
        case .audio(let entry):
            GeneratedAudioDetail(entry: entry, url: library.audioURL(for: entry))
        }
    }
}

private struct GeneratedImageDetail: View {
    let url: URL
    let prompt: String
    let size: String
    let isEdit: Bool
    let model: String

    @State private var image: NSImage?
    @State private var failed = false
    @State private var fits = true
    @State private var saveError: String?

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: RapidTheme.Space.xs) {
                VStack(alignment: .leading, spacing: 0) {
                    Text(prompt.isEmpty ? String(localized: "Generated image") : prompt)
                        .font(.rapid(.subheadline, weight: .semibold))
                        .lineLimit(2)
                    Text(size.isEmpty ? model : "\(model) · \(size)")
                        .font(.rapid(.caption))
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: RapidTheme.Space.sm)
                QuietIconButton(
                    symbol: fits ? "1.magnifyingglass" : "arrow.up.left.and.down.right.magnifyingglass",
                    label: fits ? String(localized: "Show at actual size") : String(localized: "Zoom to fit")
                ) { fits.toggle() }
                    .accessibilityIdentifier("Artifact.Library.Zoom")
                QuietIconButton(
                    symbol: "square.and.arrow.down", label: String(localized: "Save as…")
                ) { saveError = ArtifactPanelActions.saveCopy(of: url, suggestedName: "rapid-image.png") }
                    .accessibilityIdentifier("Artifact.Library.Save")
            }
            .padding(.horizontal, RapidTheme.Space.lg)
            .padding(.vertical, RapidTheme.Space.sm)
            Divider()
            if let saveError {
                Text(saveError)
                    .font(.rapid(.footnote))
                    .foregroundStyle(.secondary)
                    .padding(RapidTheme.Space.sm)
            }
            Group {
                if let image {
                    ArtifactFittedImage(image: image, label: prompt, fits: fits, vector: false)
                } else if failed {
                    Text("This artifact can't be previewed.")
                        .font(.rapid(.footnote))
                        .foregroundStyle(.secondary)
                } else {
                    ProgressView().controlSize(.small)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .task(id: url) {
            let location = url
            let data = await Task.detached { try? Data(contentsOf: location) }.value
            let loaded = data.flatMap(NSImage.init(data:))
            image = loaded
            failed = loaded == nil
        }
    }
}

private struct GeneratedAudioDetail: View {
    let entry: AudioArtifactStore.Entry
    let url: URL

    @State private var player = ArtifactAudioPlayer()
    @State private var saveError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            HStack(spacing: RapidTheme.Space.sm) {
                Button {
                    player.toggle(url)
                } label: {
                    Label(
                        player.isPlaying ? String(localized: "Stop") : String(localized: "Play"),
                        systemImage: player.isPlaying ? "stop.fill" : "play.fill"
                    )
                }
                .controlSize(.large)
                .accessibilityIdentifier("Artifact.Library.Play")
                Text("\(entry.voice) · \(entry.modelAlias)")
                    .font(.rapid(.footnote))
                    .foregroundStyle(.secondary)
                Spacer()
                QuietIconButton(
                    symbol: "square.and.arrow.down", label: String(localized: "Save as…")
                ) {
                    saveError = ArtifactPanelActions.saveCopy(
                        of: url, suggestedName: "rapid-speech.\(entry.fileExtension)"
                    )
                }
                .accessibilityIdentifier("Artifact.Library.Save")
            }
            if let message = player.errorMessage ?? saveError {
                Text(message)
                    .font(.rapid(.footnote))
                    .foregroundStyle(.secondary)
            }
            Divider()
            ScrollView {
                Text(entry.text)
                    .font(.rapid(.body))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(RapidTheme.Space.lg)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onDisappear { player.stop() }
    }
}

@MainActor
@Observable
private final class ArtifactAudioPlayer {
    private var player: AVAudioPlayer?
    private var monitor: Task<Void, Never>?
    private(set) var isPlaying = false
    private(set) var errorMessage: String?

    func toggle(_ url: URL) {
        if isPlaying {
            stop()
            return
        }
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            guard player.prepareToPlay(), player.play() else {
                errorMessage = String(localized: "Playback could not start.")
                return
            }
            errorMessage = nil
            self.player = player
            isPlaying = true
            monitor = Task { [weak self, weak player] in
                while !Task.isCancelled, player?.isPlaying == true {
                    try? await Task.sleep(for: .milliseconds(100))
                }
                guard !Task.isCancelled else { return }
                self?.isPlaying = false
            }
        } catch {
            errorMessage = String(localized: "Playback could not start.")
        }
    }

    func stop() {
        monitor?.cancel()
        monitor = nil
        player?.stop()
        player = nil
        isPlaying = false
    }
}
