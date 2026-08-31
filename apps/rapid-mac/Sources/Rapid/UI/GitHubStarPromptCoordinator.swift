import AppKit
import Foundation
import Observation

/// Owns the local-only, post-value invitation to visit Rapid-MLX on GitHub.
///
/// The prompt is deliberately workload-based rather than launch-based: it is
/// eligible only after successful product outcomes, waits for a quiet window,
/// and backs off exponentially when the user asks to see it later.
@MainActor
@Observable
final class GitHubStarPromptCoordinator {
    struct PresentationContext: Equatable {
        var isBusy: Bool
        var hasBlockingSurface: Bool

        static let ready = PresentationContext(isBusy: false, hasBlockingSurface: false)
    }

    enum Keys {
        static let completed = "Rapid.githubStar.completed.v1"
        static let totalSuccessfulActions = "Rapid.githubStar.totalSuccessfulActions.v1"
        static let baselineSuccessfulActions = "Rapid.githubStar.baselineSuccessfulActions.v1"
        static let nextWorkloadThreshold = "Rapid.githubStar.nextWorkloadThreshold.v1"
        static let deferredUntil = "Rapid.githubStar.deferredUntil.v1"
    }

    static let initialWorkloadThreshold = 35
    static let deferralInterval: TimeInterval = 3 * 24 * 60 * 60
    static let quietWindow: Duration = .milliseconds(1_200)

    private(set) var isPresented = false
    private(set) var isStarring = false

    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let now: () -> Date
    @ObservationIgnored private let quietWindow: Duration
    @ObservationIgnored private let starExecutor: @Sendable (URL) async throws -> Void
    @ObservationIgnored private let waitForQuietWindow: @MainActor (Duration) async -> Void
    @ObservationIgnored private var context: PresentationContext = .ready
    @ObservationIgnored private var presentationPending = false
    @ObservationIgnored private var presentationTask: Task<Void, Never>?
    @ObservationIgnored private var keyboardMonitor: Any?
    @ObservationIgnored private var presentationActive: Bool

    init(
        defaults: UserDefaults = .standard,
        now: @escaping () -> Date = Date.init,
        quietWindow: Duration = GitHubStarPromptCoordinator.quietWindow,
        presentationActive: Bool = false,
        starExecutor: @escaping @Sendable (URL) async throws -> Void = { url in
            try await GitHubStarCLI.star(url)
        },
        waitForQuietWindow: @escaping @MainActor (Duration) async -> Void = { duration in
            try? await Task.sleep(for: duration)
        }
    ) {
        self.defaults = defaults
        self.now = now
        self.quietWindow = quietWindow
        self.starExecutor = starExecutor
        self.presentationActive = presentationActive
        self.waitForQuietWindow = waitForQuietWindow
    }

    /// Records one delivered chat reply, dictation transcript, or generated
    /// image. No prompt state, content, or GitHub account data leaves the Mac.
    func productValueDelivered(_ kind: ProductValueKind) {
        guard !defaults.bool(forKey: Keys.completed) else { return }

        let previousTotal = defaults.integer(forKey: Keys.totalSuccessfulActions)
        let total = previousTotal == Int.max ? Int.max : previousTotal + 1
        defaults.set(total, forKey: Keys.totalSuccessfulActions)

        guard !isPresented, !presentationPending, isWorkloadEligible(total: total) else { return }
        presentationPending = true
        schedulePresentationCheck()
    }

    /// Keeps the prompt out of active generation and competing dialogs. A
    /// pending invitation is retained and retried once the workspace is calm.
    func updatePresentationContext(_ newContext: PresentationContext) {
        guard context != newContext else { return }
        context = newContext
        if newContext.isBusy || newContext.hasBlockingSurface {
            presentationTask?.cancel()
            presentationTask = nil
            if isPresented {
                isPresented = false
                presentationPending = true
            }
        } else if presentationPending {
            schedulePresentationCheck()
        }
    }

    /// Restarts the quiet window after a local key press without consuming or
    /// intercepting the event. This prevents the card arriving mid-sentence.
    func noteUserActivity() {
        guard presentationPending, !isPresented else { return }
        schedulePresentationCheck()
    }

    func startMonitoringUserActivity() {
        presentationActive = true
        if keyboardMonitor == nil {
            keyboardMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { [weak self] event in
                self?.noteUserActivity()
                return event
            }
        }
        if presentationPending { schedulePresentationCheck() }
    }

    func stopMonitoringUserActivity() {
        presentationActive = false
        presentationTask?.cancel()
        presentationTask = nil
        if isPresented {
            isPresented = false
            presentationPending = true
        }
        if let keyboardMonitor {
            NSEvent.removeMonitor(keyboardMonitor)
            self.keyboardMonitor = nil
        }
    }

    /// Close and Later are intentionally equivalent: both respect the user's
    /// timing, establish a three-day floor, and double the next workload gate.
    func deferPrompt() {
        guard isPresented else { return }
        isPresented = false
        presentationPending = false

        let total = defaults.integer(forKey: Keys.totalSuccessfulActions)
        let currentThreshold = workloadThreshold
        let (doubled, overflow) = currentThreshold.multipliedReportingOverflow(by: 2)
        defaults.set(total, forKey: Keys.baselineSuccessfulActions)
        defaults.set(overflow ? Int.max : doubled, forKey: Keys.nextWorkloadThreshold)
        defaults.set(now().addingTimeInterval(Self.deferralInterval), forKey: Keys.deferredUntil)
    }

    /// A successfully handed-off browser open completes this invitation for
    /// the install. Rapid does not probe the user's GitHub account to verify it.
    func repositoryOpened() {
        guard isPresented else { return }
        markCompleted()
    }

    /// Tries the GitHub CLI first so developers can complete the invitation
    /// without a browser round-trip. A missing or unauthenticated CLI is not an
    /// error surface: the caller falls back to the existing browser handoff.
    func attemptDirectStar() async -> Bool {
        guard isPresented, !isStarring else { return false }
        isStarring = true
        defer { isStarring = false }

        do {
            try await starExecutor(GitHubCommunity.repositoryURL)
            guard isPresented || presentationPending else { return false }
            markCompleted()
            return true
        } catch {
            return false
        }
    }

    private func markCompleted() {
        isPresented = false
        presentationPending = false
        defaults.set(true, forKey: Keys.completed)
    }

    private var workloadThreshold: Int {
        let stored = defaults.integer(forKey: Keys.nextWorkloadThreshold)
        return stored > 0 ? stored : Self.initialWorkloadThreshold
    }

    private func isWorkloadEligible(total: Int) -> Bool {
        let baseline = defaults.integer(forKey: Keys.baselineSuccessfulActions)
        let completedWork = max(0, total - baseline)
        guard completedWork >= workloadThreshold else { return false }
        guard let deferredUntil = defaults.object(forKey: Keys.deferredUntil) as? Date else { return true }
        return now() >= deferredUntil
    }

    private func schedulePresentationCheck() {
        presentationTask?.cancel()
        guard presentationActive,
              presentationPending,
              !context.isBusy,
              !context.hasBlockingSurface else {
            presentationTask = nil
            return
        }

        if quietWindow == .zero {
            presentIfReady()
            return
        }

        presentationTask = Task { [weak self, quietWindow, waitForQuietWindow] in
            await waitForQuietWindow(quietWindow)
            guard !Task.isCancelled else { return }
            self?.presentIfReady()
        }
    }

    private func presentIfReady() {
        presentationTask = nil
        guard presentationPending,
              !isPresented,
              presentationActive,
              !context.isBusy,
              !context.hasBlockingSurface,
              !defaults.bool(forKey: Keys.completed),
              isWorkloadEligible(total: defaults.integer(forKey: Keys.totalSuccessfulActions)) else { return }
        presentationPending = false
        isPresented = true
    }
}

enum GitHubStarCLIError: Error {
    case executableNotFound
    case invalidRepository
    case timedOut
    case commandFailed
}

/// Runs one narrowly-scoped GitHub CLI request. The URL is Rapid-MLX's own
/// canonical repository, so drag-and-dropped or copied URLs cannot change it.
enum GitHubStarCLI {
    static let timeout: Duration = .seconds(8)

    static func star(
        _ repositoryURL: URL,
        executableURL overrideExecutableURL: URL? = nil,
        timeout requestedTimeout: Duration = timeout
    ) async throws {
        guard repositoryURL == GitHubCommunity.repositoryURL else {
            throw GitHubStarCLIError.invalidRepository
        }
        guard let executable = overrideExecutableURL ?? executableURL() else {
            throw GitHubStarCLIError.executableNotFound
        }

        let process = Process()
        process.executableURL = executable
        process.arguments = apiArguments()

        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice

        try process.run()
        try await waitUntilExit(process, timeout: requestedTimeout)

        guard process.terminationStatus == 0 else {
            throw GitHubStarCLIError.commandFailed
        }
    }

    static func apiArguments() -> [String] {
        [
            "api", "--method", "PUT", "--silent",
            "--hostname", "github.com",
            "user/starred/raullenchai/Rapid-MLX"
        ]
    }

    static func executableURL() -> URL? {
        let searchPaths = [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            ProcessInfo.processInfo.environment["PATH"] ?? "",
            "/usr/bin",
            "/bin"
        ]

        return searchPaths
            .flatMap { $0.split(separator: ":").map(String.init) }
            .map { URL(fileURLWithPath: $0).appendingPathComponent("gh") }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    private static func waitUntilExit(_ process: Process, timeout: Duration) async throws {
        enum Outcome {
            case exited(Int32)
            case timedOut
        }

        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        let outcome = try await withThrowingTaskGroup(of: Outcome.self) { group in
            group.addTask {
                await Task.detached {
                    process.waitUntilExit()
                }.value
                // Once the deadline has passed, timeout is authoritative even
                // if SIGKILL makes the waiter observe a nonzero exit first.
                guard clock.now < deadline else { return .timedOut }
                return .exited(process.terminationStatus)
            }
            group.addTask {
                try await clock.sleep(until: deadline)
                if process.isRunning {
                    kill(process.processIdentifier, SIGKILL)
                }
                return .timedOut
            }

            guard let first = try await group.next() else {
                throw GitHubStarCLIError.commandFailed
            }
            group.cancelAll()
            return first
        }

        switch outcome {
        case .exited(0):
            return
        case .exited:
            throw GitHubStarCLIError.commandFailed
        case .timedOut:
            throw GitHubStarCLIError.timedOut
        }
    }
}
