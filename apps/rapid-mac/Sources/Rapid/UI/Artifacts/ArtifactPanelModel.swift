import Foundation
import Observation
import SwiftUI

/// What the transcript's code blocks call to open an artifact. A stable
/// reference, so putting it in the environment does not invalidate every
/// Markdown row each time the chat view re-evaluates.
@MainActor
final class ArtifactOpener {
    private weak var model: ArtifactPanelModel?

    fileprivate init(model: ArtifactPanelModel) { self.model = model }

    func open(kind: ArtifactKind, source: String) {
        model?.open(ArtifactItem(id: "detached", kind: kind, source: source))
    }
}

private struct ArtifactOpenerKey: EnvironmentKey {
    static let defaultValue: ArtifactOpener? = nil
}

extension EnvironmentValues {
    /// Nil outside a chat transcript: a card then has no "Open in panel".
    var artifactOpener: ArtifactOpener? {
        get { self[ArtifactOpenerKey.self] }
        set { self[ArtifactOpenerKey.self] = newValue }
    }
}

/// The artifacts in the visible branch of the conversation, and which one the
/// panel is showing.
///
/// Nothing here is persisted: artifacts are derived from message text, so the
/// list follows whichever branch the transcript shows, and a regenerated or
/// edited answer is simply a new entry.
@MainActor
@Observable
final class ArtifactPanelModel {

    static let widthRange: ClosedRange<CGFloat> = 300...1600

    private(set) var isOpen = false
    var panelWidth: CGFloat = 560
    private(set) var items: [ArtifactItem] = []
    private(set) var selectedID: String?
    /// Shown when the reader opened a block the list does not hold yet, which
    /// is a block of a reply that is still streaming.
    private(set) var detached: ArtifactItem?
    /// The panel fills the chat area and the transcript is hidden.
    private(set) var isFullScreen = false

    @ObservationIgnored private var messages: [ChatMessage] = []
    @ObservationIgnored private var isStale = true
    @ObservationIgnored private var cache: [UUID: (content: String, items: [ArtifactItem])] = [:]
    @ObservationIgnored private(set) lazy var opener = ArtifactOpener(model: self)

    var selected: ArtifactItem? {
        if let detached, detached.id == selectedID { return detached }
        return items.first { $0.id == selectedID }
    }

    // MARK: Intents

    func open(_ item: ArtifactItem) {
        isOpen = true
        refreshIfNeeded()
        if let match = items.first(where: { $0.kind == item.kind && $0.source == item.source }) {
            detached = nil
            selectedID = match.id
        } else {
            detached = item
            selectedID = item.id
        }
    }

    /// Closing also leaves full screen, so the chat comes back.
    func close() {
        isOpen = false
        isFullScreen = false
    }

    func toggleFullScreen() { isFullScreen.toggle() }

    func select(_ id: String) {
        guard items.contains(where: { $0.id == id }) else { return }
        detached = nil
        selectedID = id
    }

    /// The transcript changed. Cheap while closed: the work waits for the
    /// panel to open.
    func messagesDidChange(_ messages: [ChatMessage]) {
        self.messages = messages
        isStale = true
        if isOpen { refreshIfNeeded() }
    }

    func setPanelWidth(_ width: CGFloat) {
        panelWidth = min(max(width, Self.widthRange.lowerBound), Self.widthRange.upperBound)
    }

    // MARK: Derivation

    private func refreshIfNeeded() {
        guard isStale else { return }
        isStale = false

        var live: Set<UUID> = []
        var derived: [ArtifactItem] = []
        for message in messages where message.role == .assistant && message.status != .streaming {
            live.insert(message.id)
            derived.append(contentsOf: artifacts(in: message))
        }
        cache = cache.filter { live.contains($0.key) }
        items = derived

        // A streaming reply is the only reason to keep an artifact the list
        // does not hold; anything else means the branch moved on.
        let isStreaming = messages.contains { $0.status == .streaming }
        if let detached {
            if let match = items.first(where: {
                $0.kind == detached.kind && $0.source == detached.source
            }) {
                self.detached = nil
                selectedID = match.id
            } else if !isStreaming {
                self.detached = nil
            }
        }
        if selected == nil {
            selectedID = items.first?.id
        }
    }

    private func artifacts(in message: ChatMessage) -> [ArtifactItem] {
        if let hit = cache[message.id], hit.content == message.content { return hit.items }
        let derived = Self.artifacts(in: message.content, messageID: message.id)
        cache[message.id] = (message.content, derived)
        return derived
    }

    /// Every artifact block in `content`, in order. Walks the same compiled
    /// blocks the transcript renders, so the list and the cards agree.
    nonisolated static func blocks(in content: String) -> [ArtifactLibrary.Block] {
        guard content.contains("```") || content.contains("~~~") else { return [] }
        return MarkdownCompiler().compile(content).items.compactMap { item in
            guard case .code(let block) = item,
                  let kind = ArtifactItem.kind(code: block.code, language: block.language),
                  // A page cut off by a stopped stream is not a page.
                  kind != .html || block.isClosed
            else { return nil }
            return ArtifactLibrary.Block(kind: kind, source: block.code)
        }
    }

    static func artifacts(in content: String, messageID: UUID) -> [ArtifactItem] {
        blocks(in: content).enumerated().map { index, block in
            ArtifactItem(
                id: "\(messageID.uuidString)#\(index)", kind: block.kind, source: block.source
            )
        }
    }
}

/// Cheap enough to compute on every chat view update: it never reads message
/// text, only its length.
struct ArtifactMessageFingerprint: Equatable {
    let id: UUID
    let status: ChatMessage.Status
    let length: Int

    static func of(_ messages: [ChatMessage]) -> [ArtifactMessageFingerprint] {
        messages.map { .init(id: $0.id, status: $0.status, length: $0.content.utf8.count) }
    }
}
