import Foundation

/// In-memory manifest freshness; download/install scheduling remains Sparkle's job.
struct UpdateRefreshPolicy {
    // Keep aligned with SUScheduledCheckInterval in Resources/Info.plist.
    static let refreshInterval: TimeInterval = 21_600
    static let maximumRetryDelay: TimeInterval = 3_600
    static let baseRetryDelay: TimeInterval = 60
    private(set) var lastSuccessAt: Date?
    private(set) var lastAttemptAt: Date?
    private(set) var lastCancelledAt: Date?
    private(set) var consecutiveFailures = 0

    var retryDelay: TimeInterval {
        min(Self.maximumRetryDelay, Self.baseRetryDelay * pow(2, Double(min(max(0, consecutiveFailures - 1), 6))))
    }

    func delay(at now: Date) -> TimeInterval {
        if let lastCancelledAt {
            return max(0, lastCancelledAt.addingTimeInterval(Self.baseRetryDelay).timeIntervalSince(now))
        }
        if consecutiveFailures > 0, let lastAttemptAt {
            return max(0, lastAttemptAt.addingTimeInterval(retryDelay).timeIntervalSince(now))
        }
        guard let anchor = lastSuccessAt ?? lastAttemptAt else { return 0 }
        return max(0, anchor.addingTimeInterval(Self.refreshInterval).timeIntervalSince(now))
    }

    mutating func succeeded(at now: Date) {
        lastCancelledAt = nil
        lastAttemptAt = now
        lastSuccessAt = now
        consecutiveFailures = 0
    }

    mutating func failed(at now: Date) {
        lastCancelledAt = nil
        lastAttemptAt = now
        consecutiveFailures = min(consecutiveFailures + 1, 7)
    }

    mutating func cancelled(at now: Date) {
        // Retry cancellation after the base delay, independently of stale success
        // or existing failure backoff, without counting it as another failure.
        lastCancelledAt = now
        lastAttemptAt = now
    }
}
