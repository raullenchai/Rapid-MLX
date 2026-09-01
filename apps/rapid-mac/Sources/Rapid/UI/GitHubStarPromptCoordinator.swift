import AppKit
import Foundation
import Observation

enum GitHubStarAttemptResult: Equatable {
    case starred
    case unavailable
    case cancelled
}

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
    func attemptDirectStar() async -> GitHubStarAttemptResult {
        guard isPresented, !isStarring else { return .unavailable }
        isStarring = true
        defer { isStarring = false }

        do {
            try await starExecutor(GitHubCommunity.repositoryURL)
            guard isPresented || presentationPending else { return .unavailable }
            markCompleted()
            return .starred
        } catch is CancellationError {
            return .cancelled
        } catch {
            return .unavailable
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
    static let trustedExecutablePaths = [
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        "/usr/bin/gh"
    ]

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

        let child = try GitHubStarChild.spawn(
            executableURL: executable,
            arguments: apiArguments()
        )
        let terminationStatus = try await waitUntilExit(child, timeout: requestedTimeout)

        guard terminationStatus == 0 else {
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
        trustedExecutablePaths
            .map { URL(fileURLWithPath: $0) }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    private static func waitUntilExit(_ child: GitHubStarChild, timeout: Duration) async throws -> Int32 {
        enum Outcome {
            case exited(Int32)
            case timedOut
        }

        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        let outcome = try await withTaskCancellationHandler {
            try await withThrowingTaskGroup(of: Outcome.self) { group in
                group.addTask {
                    let status = await child.waitUntilExit()
                    return .exited(status)
                }
                group.addTask {
                    try await clock.sleep(until: deadline)
                    // Reap an already-exited child before declaring timeout. The
                    // dispatch exit callback may be delayed even though `gh`
                    // completed successfully before the deadline.
                    if let status = child.terminationStatusIfExited() {
                        return .exited(status)
                    }
                    child.killIfRunning()
                    return .timedOut
                }

                guard let first = try await group.next() else {
                    throw GitHubStarCLIError.commandFailed
                }
                group.cancelAll()
                return first
            }
        } onCancel: {
            // Cancelling the caller also cancels the timeout sleeper. Kill the
            // child here so the detached waiter can reap it before the task
            // group propagates CancellationError.
            child.killIfRunning()
        }

        switch outcome {
        case let .exited(status):
            return status
        case .timedOut:
            throw GitHubStarCLIError.timedOut
        }
    }
}

/// Owns launch, signalling, and reaping for the short-lived `gh` child.
/// Foundation `Process.waitUntilExit()` can race a raw PID signal after the
/// process has been reaped. This owner serializes SIGKILL with the only
/// `waitpid`, so an exited PID cannot be reused before signalling decides.
private final class GitHubStarChild: @unchecked Sendable {
    private static let exitQueue = DispatchQueue(
        label: "com.rapidmlx.desktop.github-star-exit",
        qos: .userInitiated
    )

    private let processIdentifier: pid_t
    private let lock = NSLock()
    private let reapLock = NSLock()
    private var running = true
    private var terminationStatus: Int32?
    private var waiters: [CheckedContinuation<Int32, Never>] = []
    private var exitSource: (any DispatchSourceProcess)?

    private init(processIdentifier: pid_t) {
        self.processIdentifier = processIdentifier
    }

    func killIfRunning() {
        reapLock.lock()
        lock.lock()
        let shouldSignal = running
        lock.unlock()
        if shouldSignal {
            _ = kill(-processIdentifier, SIGKILL)
        }
        reapLock.unlock()
    }

    func waitUntilExit() async -> Int32 {
        await withCheckedContinuation { continuation in
            lock.lock()
            if let terminationStatus {
                lock.unlock()
                continuation.resume(returning: terminationStatus)
            } else {
                waiters.append(continuation)
                lock.unlock()
            }
        }
    }

    func terminationStatusIfExited() -> Int32? {
        _ = reapExitedProcess(waitOptions: WNOHANG)
        lock.lock()
        let status = terminationStatus
        lock.unlock()
        return status
    }

    static func spawn(executableURL: URL, arguments: [String]) throws -> GitHubStarChild {
        var fileActions: posix_spawn_file_actions_t?
        var attributes: posix_spawnattr_t?
        guard posix_spawn_file_actions_init(&fileActions) == 0 else {
            throw GitHubStarCLIError.commandFailed
        }
        defer { posix_spawn_file_actions_destroy(&fileActions) }
        guard posix_spawnattr_init(&attributes) == 0 else {
            throw GitHubStarCLIError.commandFailed
        }
        defer { posix_spawnattr_destroy(&attributes) }
        guard posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP)) == 0,
            posix_spawnattr_setpgroup(&attributes, 0) == 0
        else {
            throw GitHubStarCLIError.commandFailed
        }

        guard posix_spawn_file_actions_addopen(
            &fileActions,
            STDIN_FILENO,
            "/dev/null",
            O_RDONLY,
            0
        ) == 0,
            posix_spawn_file_actions_addopen(
                &fileActions,
                STDOUT_FILENO,
                "/dev/null",
                O_WRONLY,
                0
            ) == 0,
            posix_spawn_file_actions_addopen(
                &fileActions,
                STDERR_FILENO,
                "/dev/null",
                O_WRONLY,
                0
            ) == 0
        else {
            throw GitHubStarCLIError.commandFailed
        }

        let argv = [executableURL.path] + arguments
        let environment = ProcessInfo.processInfo.environment
            .map { "\($0.key)=\($0.value)" }
            .sorted()
        var pid: pid_t = 0
        let spawnResult = argv.withGitHubStarCStringArray { argvPointer in
            environment.withGitHubStarCStringArray { environmentPointer in
                executableURL.path.withCString { executablePath in
                    posix_spawn(
                        &pid,
                        executablePath,
                        &fileActions,
                        &attributes,
                        argvPointer,
                        environmentPointer
                    )
                }
            }
        }
        guard spawnResult == 0 else {
            throw GitHubStarCLIError.commandFailed
        }

        let child = GitHubStarChild(processIdentifier: pid)
        child.startExitMonitor()
        return child
    }

    private func startExitMonitor() {
        let source = DispatchSource.makeProcessSource(
            identifier: processIdentifier,
            eventMask: .exit,
            queue: Self.exitQueue
        )
        source.setEventHandler { [weak self] in
            self?.reapExitedProcess(waitOptions: 0)
        }
        lock.lock()
        exitSource = source
        lock.unlock()
        source.activate()

        // Close the very-short-lived-child race before the source is armed.
        _ = reapExitedProcess(waitOptions: WNOHANG)
    }

    @discardableResult
    private func reapExitedProcess(waitOptions: Int32) -> Bool {
        reapLock.lock()
        lock.lock()
        guard running else {
            lock.unlock()
            reapLock.unlock()
            return true
        }
        lock.unlock()

        var waitStatus: Int32 = 0
        var waited: pid_t
        repeat {
            waited = waitpid(processIdentifier, &waitStatus, waitOptions)
        } while waited == -1 && errno == EINTR
        guard waited == processIdentifier else {
            reapLock.unlock()
            return false
        }

        let statusCode = waitStatus & 0x7f
        let status = statusCode == 0 ? (waitStatus >> 8) & 0xff : statusCode
        lock.lock()
        running = false
        terminationStatus = status
        let continuations = waiters
        waiters.removeAll()
        let source = exitSource
        exitSource = nil
        lock.unlock()
        reapLock.unlock()

        source?.cancel()
        for continuation in continuations {
            continuation.resume(returning: status)
        }
        return true
    }
}

private extension Array where Element == String {
    func withGitHubStarCStringArray<Result>(
        _ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) throws -> Result
    ) rethrows -> Result {
        let strings = map { strdup($0) }
        defer {
            for string in strings {
                free(string)
            }
        }
        let pointer = UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
            .allocate(capacity: count + 1)
        defer { pointer.deallocate() }
        for index in indices {
            pointer[index] = strings[index]
        }
        pointer[count] = nil
        return try body(pointer)
    }
}
