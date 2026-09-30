import Foundation

// MARK: - Screen state

/// Which composition the Share tab is showing.
///
/// A projection of ``ShareComputeManager/State`` rather than a second source of
/// truth: the manager owns the process, this names what the user should see.
/// Keeping the mapping in one pure function stops each subview from
/// re-deriving "are we sharing yet?" and disagreeing.
enum ShareComputeScreen: Equatable, Sendable {
    /// No session. Model choice and the primary action.
    case ready
    /// Steps 1–5 are running.
    case preparing
    /// The node is in the pool and serving.
    case online
    /// The provider lost its connection and is retrying. Still an active
    /// session — the model stays locked.
    case reconnecting
    /// Stop was pressed (or the provider is tearing down).
    case stopping
    /// The session ended and a receipt is on screen.
    case complete
    /// Something went wrong. Carries the provider's own message.
    case failed(String)

    /// True while a session owns the Mac. Model selection is locked and the
    /// picker must not open.
    var locksModelSelection: Bool {
        switch self {
        case .preparing, .online, .reconnecting, .stopping: return true
        case .ready, .complete, .failed: return false
        }
    }

    /// True when Stop Sharing is a valid thing to press. Stopping is already
    /// stopping, so offering it again would be a no-op the user could read as
    /// a stuck button.
    var allowsStop: Bool {
        switch self {
        case .preparing, .online, .reconnecting: return true
        case .ready, .stopping, .complete, .failed: return false
        }
    }

    /// The mapping. `hasCompletedSession` is the view's memory of a receipt it
    /// has already shown: an idle manager means "ready" on first open and
    /// "complete" immediately after a session ends, and only the view knows
    /// which of those it is.
    static func make(
        state: ShareComputeManager.State,
        hasCompletedSession: Bool
    ) -> ShareComputeScreen {
        switch state {
        case .idle:
            return hasCompletedSession ? .complete : .ready
        case .preparing, .registering, .starting, .warming, .connecting:
            return .preparing
        case .online:
            return .online
        case .reconnecting:
            return .reconnecting
        case .stopping:
            return .stopping
        case .failed(let message):
            return .failed(message)
        }
    }
}

// MARK: - Preparing steps

/// One line of the five-step preparation sequence.
enum ShareComputePreparationStep: Int, CaseIterable, Identifiable, Sendable {
    case pauseCurrentModel
    case registerThisMac
    case startSharedModel
    case warmUp
    case joinPool

    var id: Int { rawValue }

    /// 1-based, for the numbered circle.
    var number: Int { rawValue + 1 }

    var title: String {
        switch self {
        case .pauseCurrentModel: return String(localized: "Pause current model")
        case .registerThisMac: return String(localized: "Register this Mac")
        case .startSharedModel: return String(localized: "Start shared model")
        case .warmUp: return String(localized: "Warm up")
        case .joinPool: return String(localized: "Join compute pool")
        }
    }

    /// The provider phase that means this step is the one currently running.
    fileprivate var runningPhase: ShareComputeManager.State {
        switch self {
        case .pauseCurrentModel: return .preparing
        case .registerThisMac: return .registering
        case .startSharedModel: return .starting
        case .warmUp: return .warming
        case .joinPool: return .connecting
        }
    }
}

/// What one step is doing right now.
///
/// ``alreadyRegistered`` is a distinct case rather than a flavour of
/// ``complete`` because it answers a different question: the step did not run
/// at all this time, because a prior session already registered this Mac for
/// this model and worker. Collapsing it into "Complete" would claim work
/// happened that did not.
enum ShareComputePreparationStatus: Equatable, Sendable {
    case waiting
    case inProgress
    case complete
    case alreadyRegistered
    case failed

    var title: String {
        switch self {
        case .waiting: return String(localized: "Waiting")
        case .inProgress: return String(localized: "In progress")
        case .complete: return String(localized: "Complete")
        case .alreadyRegistered: return String(localized: "Already registered")
        case .failed: return String(localized: "Failed")
        }
    }

    /// Steps carry a glyph as well as a colour so status survives a
    /// monochrome or colour-blind reading.
    var isSettled: Bool { self == .complete || self == .alreadyRegistered }
}

struct ShareComputePreparationRow: Equatable, Identifiable, Sendable {
    let step: ShareComputePreparationStep
    let status: ShareComputePreparationStatus

    var id: Int { step.id }
}

enum ShareComputePreparationPlan {
    /// Projects the provider's current phase onto the five visible steps.
    ///
    /// The rule is strictly ordered and derived only from the phase the
    /// provider actually published: every step BEFORE the running one is
    /// complete, the running one is in progress, everything after is waiting.
    /// Nothing is optimistically marked done — a step only reads "Complete"
    /// because the provider has moved past it.
    ///
    /// `isAlreadyRegistered` comes from the on-disk registration marker
    /// (``ShareComputeManager/registrationMatches(model:worker:home:)``), which
    /// is why the register step can be settled before the session ever reaches
    /// the `registering` phase.
    static func rows(
        state: ShareComputeManager.State,
        isAlreadyRegistered: Bool
    ) -> [ShareComputePreparationRow] {
        let steps = ShareComputePreparationStep.allCases

        func settledStatus(for step: ShareComputePreparationStep) -> ShareComputePreparationStatus {
            step == .registerThisMac && isAlreadyRegistered ? .alreadyRegistered : .complete
        }

        // Online (or beyond) means every step finished.
        switch state {
        case .online, .reconnecting, .stopping, .idle:
            return steps.map { .init(step: $0, status: settledStatus(for: $0)) }
        case .failed:
            // The provider does not say which step failed, so the honest
            // reading is: whatever had not settled is where it stopped. The
            // first unsettled step carries the failure, the rest stay waiting.
            var rows: [ShareComputePreparationRow] = []
            var markedFailure = false
            for step in steps {
                if step == .registerThisMac, isAlreadyRegistered {
                    rows.append(.init(step: step, status: .alreadyRegistered))
                } else if markedFailure {
                    rows.append(.init(step: step, status: .waiting))
                } else {
                    markedFailure = true
                    rows.append(.init(step: step, status: .failed))
                }
            }
            return rows
        default:
            break
        }

        guard let runningIndex = steps.firstIndex(where: { $0.runningPhase == state }) else {
            return steps.map { .init(step: $0, status: .waiting) }
        }
        return steps.enumerated().map { index, step in
            if index < runningIndex {
                return .init(step: step, status: settledStatus(for: step))
            }
            // The registration marker is on disk before the sequence starts,
            // so this step is settled wherever the sequence currently is —
            // including while an EARLIER step is still running. A registering
            // phase on an already-registered Mac is still "already
            // registered": the provider re-checks the marker, it does not
            // re-register.
            if step == .registerThisMac, isAlreadyRegistered {
                return .init(step: step, status: .alreadyRegistered)
            }
            if index == runningIndex {
                return .init(step: step, status: .inProgress)
            }
            return .init(step: step, status: .waiting)
        }
    }
}

// MARK: - Reward tracking

/// What the interface may claim about reward tracking at this moment.
///
/// The rule from the handoff is absolute: nothing is counted until QuickSilver
/// accepts the node. "Accepted" is observable locally — the provider only
/// publishes the `online` phase once the pool has taken the node — so this is
/// a real signal rather than an optimistic one.
enum ShareComputeRewardTracking: Equatable, Sendable {
    /// Preparing. No reward is counted yet.
    case waitingForPool
    /// The node is in the pool; QuickSilver is tracking accepted work.
    case trackedByProvider
    /// The session is over.
    case sessionEnded
    /// Not sharing.
    case notSharing

    static func make(screen: ShareComputeScreen) -> Self {
        switch screen {
        case .preparing: return .waitingForPool
        case .online, .reconnecting: return .trackedByProvider
        case .stopping, .complete: return .sessionEnded
        case .ready, .failed: return .notSharing
        }
    }

    var headline: String {
        switch self {
        case .waitingForPool: return String(localized: "Waiting for pool connection.")
        case .trackedByProvider: return String(localized: "Reward activity is managed in QuickSilver.")
        case .sessionEnded: return String(localized: "Reward activity continues in QuickSilver.")
        case .notSharing: return String(localized: "Reward eligible when work is accepted.")
        }
    }

    var detail: String {
        switch self {
        case .waitingForPool:
            return String(localized: "Tracking begins after QuickSilver accepts the node.")
        case .trackedByProvider:
            return String(localized: "Final usage and rewards appear in QuickSilver.")
        case .sessionEnded:
            return String(localized: "Payment timing and destination remain managed by the provider.")
        case .notSharing:
            return String(localized: "QuickSilver verifies accepted work and finalizes reward activity.")
        }
    }

    /// The "No reward is counted yet" line only belongs on the preparing
    /// surface; repeating it while online would contradict the tracking state.
    var showsNoRewardCountedYet: Bool { self == .waitingForPool }
}

// MARK: - Duration formatting

/// Session durations, formatted the way Paper writes them.
///
/// Two shapes, because the two surfaces ask different questions: a history row
/// wants a scannable `1h 42m`, and the selected-session panel is a receipt, so
/// it shows the exact `1h 42m 18s`. Both are monospaced at the call site so a
/// live-ticking value does not jitter its own width.
enum ShareComputeDuration {
    static func short(_ interval: TimeInterval) -> String {
        let total = Int(max(0, interval).rounded())
        let hours = total / 3_600
        let minutes = (total % 3_600) / 60
        if hours > 0 {
            return String(format: "%dh %02dm", hours, minutes)
        }
        if minutes > 0 {
            return String(format: "%dm", minutes)
        }
        return String(format: "%ds", total % 60)
    }

    static func precise(_ interval: TimeInterval) -> String {
        let total = Int(max(0, interval).rounded())
        let hours = total / 3_600
        let minutes = (total % 3_600) / 60
        let seconds = total % 60
        if hours > 0 {
            return String(format: "%dh %02dm %02ds", hours, minutes, seconds)
        }
        return String(format: "%dm %02ds", minutes, seconds)
    }

    /// The live clock on the Online surface: `1:42:18`.
    static func clock(_ interval: TimeInterval) -> String {
        let total = Int(max(0, interval).rounded())
        return String(
            format: "%d:%02d:%02d",
            total / 3_600,
            (total % 3_600) / 60,
            total % 60
        )
    }

    /// Aggregate time across every session — `18h 47m`, and `47m` below an
    /// hour so the recognition metric never reads `0h 47m`.
    static func total(_ interval: TimeInterval) -> String {
        short(interval)
    }
}
