import Foundation
import Observation

/// Install-wide claim; visibility lasts only for the current app session.
@MainActor
@Observable
final class ChatNextStepHint {
    static let shared = ChatNextStepHint()
    private static let consumedKey = "chatConnectionNextStepConsumed"
    private let defaults: UserDefaults
    private(set) var isVisible = false

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    func replySucceeded() {
        guard !defaults.bool(forKey: Self.consumedKey) else { return }
        defaults.set(true, forKey: Self.consumedKey)
        isVisible = true
    }

    /// Opening Launch by any route also consumes the hint, even before chat.
    func dismiss() {
        defaults.set(true, forKey: Self.consumedKey)
        isVisible = false
    }
}
