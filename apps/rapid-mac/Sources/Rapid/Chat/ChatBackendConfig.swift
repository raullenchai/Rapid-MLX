import Foundation
import Observation

enum ChatBackend: String, CaseIterable, Sendable {
    case legacy
    case ownerV1
}

/// Dormant experimental preference. Production chat does not read this setting
/// until owner bootstrap and the isolated new-conversation adapter are ready.
@MainActor
@Observable
final class ChatBackendConfig {
    nonisolated static let storageKey = "Rapid.experimental.chatBackend"
    private let defaults: UserDefaults

    var chatBackend: ChatBackend {
        didSet { defaults.set(chatBackend.rawValue, forKey: Self.storageKey) }
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        chatBackend = defaults.string(forKey: Self.storageKey)
            .flatMap(ChatBackend.init(rawValue:)) ?? .legacy
    }
}
