import Foundation
import Observation

/// App-wide custom instructions shared by every chat. Conversation-specific
/// instructions live on ``ChatConversation`` because they travel with history;
/// this value owns only the global layer stored in preferences.
@MainActor
@Observable
final class CustomInstructionsConfig {
    nonisolated static let storageKey = "rapid.custom-instructions.global.v1"
    nonisolated static let maximumLength = 4_000

    private let defaults: UserDefaults
    private var storedGlobal: String

    var global: String {
        get { storedGlobal }
        set {
            let value = Self.limited(newValue)
            storedGlobal = value
            if value.isEmpty {
                defaults.removeObject(forKey: Self.storageKey)
            } else {
                defaults.set(value, forKey: Self.storageKey)
            }
        }
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.storedGlobal = Self.limited(defaults.string(forKey: Self.storageKey) ?? "")
    }

    nonisolated static func normalized(_ value: String) -> String? {
        let trimmed = limited(value).trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    nonisolated static func limited(_ value: String) -> String {
        String(value.prefix(maximumLength))
    }
}
