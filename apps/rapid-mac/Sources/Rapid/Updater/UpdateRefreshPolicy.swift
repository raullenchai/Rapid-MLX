import Foundation

/// In-memory manifest freshness; download/install scheduling remains Sparkle's job.
struct UpdateRefreshPolicy {
    // Keep aligned with SUScheduledCheckInterval in Resources/Info.plist.
    static let refreshInterval: TimeInterval = 21_600
    static let maximumRetryDelay: TimeInterval = 3_600
    private(set) var lastSuccessAt: Date?
    private(set) var lastAttemptAt: Date?
    private(set) var consecutiveFailures = 0

    var retryDelay: TimeInterval {
        min(Self.maximumRetryDelay, 60 * pow(2, Double(min(max(0, consecutiveFailures - 1), 6))))
    }

    func delay(at now: Date) -> TimeInterval {
        if consecutiveFailures > 0, let lastAttemptAt {
            return max(0, lastAttemptAt.addingTimeInterval(retryDelay).timeIntervalSince(now))
        }
        guard let anchor = lastSuccessAt ?? lastAttemptAt else { return 0 }
        return max(0, anchor.addingTimeInterval(Self.refreshInterval).timeIntervalSince(now))
    }

    mutating func succeeded(at now: Date) {
        lastAttemptAt = now
        lastSuccessAt = now
        consecutiveFailures = 0
    }

    mutating func failed(at now: Date) {
        lastAttemptAt = now
        consecutiveFailures = min(consecutiveFailures + 1, 7)
    }

    mutating func cancelled(at now: Date) {
        // Cancellation is not a failure and must not cause an immediate retry loop.
        lastAttemptAt = now
    }
}
