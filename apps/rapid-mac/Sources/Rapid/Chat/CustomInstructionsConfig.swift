import Foundation
import Observation

/// App-wide custom instructions shared by every chat. Conversation-specific
/// instructions live on ``ChatConversation`` because they travel with history;
/// this value owns only the global layer stored in preferences.
@MainActor
@Observable
final class CustomInstructionsConfig {
    static let storageKey = "rapid.custom-instructions.global.v1"

    private let defaults: UserDefaults

    var global: String {
        didSet {
            if global.isEmpty {
                defaults.removeObject(forKey: Self.storageKey)
            } else {
                defaults.set(global, forKey: Self.storageKey)
            }
        }
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.global = defaults.string(forKey: Self.storageKey) ?? ""
    }

    nonisolated static func normalized(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
