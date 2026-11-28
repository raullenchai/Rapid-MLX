import Foundation
import Observation

/// One row of the Artifacts library.
struct ArtifactLibraryItem: Identifiable, Equatable, Sendable {

    enum Kind: String, CaseIterable, Sendable {
        case page
        case svg
        case diagram
        case image
        case audio

        var label: String {
            switch self {
            case .page: String(localized: "Page")
            case .svg: String(localized: "SVG")
            case .diagram: String(localized: "Diagram")
            case .image: String(localized: "Image")
            case .audio: String(localized: "Audio")
            }
        }

        var iconName: String {
            switch self {
            case .page: "globe"
            case .svg: "photo"
            case .diagram: "point.3.connected.trianglepath.dotted"
            case .image: "photo.artframe"
            case .audio: "waveform"
            }
        }
    }

    /// Where it came from.
    enum Origin: Equatable, Sendable {
        case conversation(id: UUID, title: String)
        case imageGeneration(model: String)
        case speech(model: String, voice: String)

        var label: String {
            switch self {
            case .conversation(_, let title):
                title.isEmpty ? String(localized: "Untitled chat") : title
            case .imageGeneration(let model):
                model.isEmpty
                    ? String(localized: "Image generation")
                    : String(localized: "Image generation · \(model)")
            case .speech(let model, _):
                model.isEmpty
                    ? String(localized: "Speech")
                    : String(localized: "Speech · \(model)")
            }
        }
    }

    enum Payload: Equatable, Sendable {
        case artifact(ArtifactItem)
        case image(id: UUID, prompt: String, size: String, isEdit: Bool)
        case audio(AudioArtifactStore.Entry)
    }

    let id: String
    let title: String
    let origin: Origin
    let date: Date
    let byteCount: Int
    let payload: Payload

    var kind: Kind {
        switch payload {
        case .artifact(let item):
            switch item.kind {
            case .html: .page
            case .svg: .svg
            case .mermaid: .diagram
            }
        case .image: .image
        case .audio: .audio
        }
    }

    var originLabel: String { origin.label }
    var kindLabel: String { kind.label }
}

/// Every artifact the app has kept: code blocks in stored chats, generated
/// images, and synthesized speech. Derived from what is already stored, so
/// it has nothing of its own on disk.
@MainActor
@Observable
final class ArtifactLibrary {

    struct Block: Equatable, Sendable {
        let kind: ArtifactKind
        let source: String
    }

    private(set) var items: [ArtifactLibraryItem] = []
    private(set) var isLoading = false

    @ObservationIgnored private let imageStore: ImageArtifactStore
    @ObservationIgnored private let audioStore: AudioArtifactStore
    @ObservationIgnored private var blockCache: [UUID: (content: String, blocks: [Block])] = [:]
    @ObservationIgnored private var generation = 0

    init(
        imageStore: ImageArtifactStore = ImageArtifactStore(),
        audioStore: AudioArtifactStore = AudioArtifactStore()
    ) {
        self.imageStore = imageStore
        self.audioStore = audioStore
    }

    var imageArtifacts: ImageArtifactStore { imageStore }

    /// Rebuild the list. The newest call wins; an earlier one still in flight
    /// is dropped rather than overwriting it.
    func refresh(conversations: [ChatConversation]) async {
        generation += 1
        let current = generation
        isLoading = true
        defer { if generation == current { isLoading = false } }

        let chatItems = await chatItems(in: conversations)
        let images = (try? await imageStore.metadata()) ?? []
        let audio = (try? await audioStore.entries()) ?? []
        guard generation == current else { return }

        var merged = chatItems
        merged += images.map { image in
            ArtifactLibraryItem(
                id: "image:\(image.id.uuidString)",
                title: Self.excerpt(image.prompt, fallback: String(localized: "Generated image")),
                origin: .imageGeneration(model: image.modelAlias),
                date: image.createdAt,
                byteCount: image.byteCount,
                payload: .image(id: image.id, prompt: image.prompt, size: image.size, isEdit: image.isEdit)
            )
        }
        merged += audio.map { entry in
            ArtifactLibraryItem(
                id: "audio:\(entry.id.uuidString)",
                title: Self.excerpt(entry.text, fallback: String(localized: "Generated speech")),
                origin: .speech(model: entry.modelAlias, voice: entry.voice),
                date: entry.createdAt,
                byteCount: entry.byteCount,
                payload: .audio(entry)
            )
        }
        items = merged.sorted { $0.date > $1.date }
    }

    func audioURL(for entry: AudioArtifactStore.Entry) -> URL { audioStore.fileURL(for: entry) }
    func imageURL(for id: UUID) -> URL { imageStore.imageURL(id) }

    // MARK: Chats

    private func chatItems(in conversations: [ChatConversation]) async -> [ArtifactLibraryItem] {
        var pending: [(UUID, String)] = []
        var live: Set<UUID> = []
        for conversation in conversations {
            for message in conversation.messages where Self.contributes(message) {
                live.insert(message.id)
                if blockCache[message.id]?.content != message.content {
                    pending.append((message.id, message.content))
                }
            }
        }
        if !pending.isEmpty {
            let work = pending
            let computed = await Task.detached(priority: .userInitiated) {
                work.map { ($0.0, $0.1, ArtifactPanelModel.blocks(in: $0.1)) }
            }.value
            for (id, content, blocks) in computed {
                blockCache[id] = (content, blocks)
            }
        }
        blockCache = blockCache.filter { live.contains($0.key) }

        var found: [ArtifactLibraryItem] = []
        for conversation in conversations {
            let origin = ArtifactLibraryItem.Origin.conversation(
                id: conversation.id, title: conversation.title
            )
            for message in conversation.messages where Self.contributes(message) {
                for (index, block) in (blockCache[message.id]?.blocks ?? []).enumerated() {
                    let item = ArtifactItem(
                        id: "\(message.id.uuidString)#\(index)", kind: block.kind, source: block.source
                    )
                    found.append(ArtifactLibraryItem(
                        id: "chat:\(conversation.id.uuidString):\(item.id)",
                        title: item.title, origin: origin, date: message.createdAt,
                        byteCount: block.source.utf8.count, payload: .artifact(item)
                    ))
                }
            }
        }
        return found
    }

    private static func contributes(_ message: ChatMessage) -> Bool {
        message.role == .assistant && message.status != .streaming
    }

    private static func excerpt(_ text: String, fallback: String) -> String {
        let collapsed = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard !collapsed.isEmpty else { return fallback }
        return collapsed.count > 80 ? String(collapsed.prefix(80)) + "…" : collapsed
    }
}
