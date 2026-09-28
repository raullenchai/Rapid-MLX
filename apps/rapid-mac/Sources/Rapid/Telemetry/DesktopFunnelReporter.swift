import Darwin
import Foundation

/// Sends identifier-free, aggregate milestones for the Desktop first-run
/// funnel. The wire body is deliberately limited to app version + a closed
/// milestone enum; no telemetry identity is read or created.
actor DesktopFunnelReporter {
    enum Milestone: String, CaseIterable, Sendable {
        case onboardingShown = "onboarding_shown"
        case modelDownloadStarted = "model_download_started"
        case modelDownloadCompleted = "model_download_completed"
        case modelDownloadFailed = "model_download_failed"
        case engineReady = "engine_ready"
        case engineStartFailed = "engine_start_failed"
        case firstChatReply = "first_chat_reply"
    }

    typealias Enabled = @Sendable () -> Bool
    typealias Send = @Sendable (URLRequest) async -> Int?
    typealias ClaimMarker = @Sendable (URL) -> Bool

    static let shared = DesktopFunnelReporter()
    nonisolated static let endpoint = URL(string: "https://rapidmlx.com/api/desktop-funnel")!
    nonisolated static let productionBundleIdentifier = "com.rapidmlx.rapid"
    nonisolated static let releaseBuildInfoKey = "RapidDesktopFunnelReleaseBuild"
    nonisolated static let maxBodyBytes = 256
    nonisolated static let cohortMarkerName = "desktop_funnel_cohort"

    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieAcceptPolicy = .never
        configuration.httpShouldSetCookies = false
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        configuration.timeoutIntervalForRequest = 3
        configuration.timeoutIntervalForResource = 3
        return URLSession(
            configuration: configuration,
            delegate: NoRedirectDelegate(),
            delegateQueue: nil
        )
    }()

    private let isEnabled: Enabled
    private let send: Send
    private let markerDirectory: URL
    private let version: String
    private let claimMarkerOverride: ClaimMarker?
    private var attemptedThisProcess: Set<Milestone> = []

    private nonisolated static let processState = DesktopFunnelProcessState()
    private nonisolated static let engineAttemptGate = DesktopFunnelEngineAttemptGate()

    init(
        isEnabled: @escaping Enabled = { DesktopFunnelReporter.productionEnabled() },
        send: @escaping Send = DesktopFunnelReporter.sendRequest,
        markerDirectory: URL = TelemetryIdentity.sharedTelemetryDirectory(),
        version: String = TelemetryClient.currentVersion(),
        claimMarker: ClaimMarker? = nil
    ) {
        self.isEnabled = isEnabled
        self.send = send
        self.markerDirectory = markerDirectory
        self.version = version
        self.claimMarkerOverride = claimMarker
    }

    /// The UI-facing entry point. Creating the task is immediate; eligibility,
    /// disk access, JSON encoding, and networking all happen away from the main
    /// actor and can never delay a view or lifecycle transition.
    nonisolated static func enqueue(_ milestone: Milestone) {
        Task.detached(priority: .utility) {
            await shared.report(milestone)
        }
    }

    /// The Settings switch calls this synchronously before it schedules the
    /// shared consent-file write. Once a user has declined in this process we
    /// stay fail-closed until relaunch, even if persistence fails or they flip
    /// the control back while an earlier write is still queued.
    nonisolated static func latchProcessOptOut() {
        processState.latchOptOut()
    }

    nonisolated static func resetProcessStateForTesting() {
        processState.reset()
    }

    /// Arms exactly the first engine start initiated by onboarding. The token
    /// lets its caller disarm only its own attempt when `ServerManager.start`
    /// returns without a ready/failure terminal state.
    @discardableResult
    nonisolated static func armOnboardingEngineAttempt(alias: String) -> UUID {
        engineAttemptGate.arm(alias: alias)
    }

    nonisolated static func disarmOnboardingEngineAttempt(_ token: UUID) {
        engineAttemptGate.disarm(token: token)
    }

    /// ServerManager routes every lifecycle terminal through this seam. Only a
    /// terminal matching the currently armed onboarding start is consumed and
    /// emitted; later manual starts, restarts, and model switches are inert.
    nonisolated static func enqueueEngineOutcomeIfArmed(
        _ milestone: Milestone,
        alias: String
    ) {
        guard engineAttemptGate.consume(milestone: milestone, alias: alias) else { return }
        enqueue(milestone)
    }

    /// Enrols only a genuinely new install. The empty cohort marker is local
    /// state, written before and independently of consent or network delivery,
    /// so later milestones cannot accidentally include upgrading installs.
    nonisolated static func enqueueOnboardingShown(isFirstRun: Bool) {
        Task.detached(priority: .utility) {
            await shared.reportOnboardingShown(isFirstRun: isFirstRun)
        }
    }

    func reportOnboardingShown(isFirstRun: Bool) async {
        guard isFirstRun else { return }
        let cohort = cohortMarkerURL
        guard FileManager.default.fileExists(atPath: cohort.path)
                || claimMarker(at: cohort) else { return }
        await report(.onboardingShown)
    }

    func report(_ milestone: Milestone) async {
        // `onboardingShown` is reached only through
        // `reportOnboardingShown(isFirstRun:)`, which creates this marker.
        // Every downstream event must belong to the same locally enrolled
        // cohort; an upgrade that never saw first-run setup stays silent.
        guard FileManager.default.fileExists(atPath: cohortMarkerURL.path) else { return }
        guard !attemptedThisProcess.contains(milestone),
              Self.processState.allowsSending,
              isEnabled() else { return }

        let marker = markerURL(for: milestone)
        if FileManager.default.fileExists(atPath: marker.path)
            || Self.processState.isResolved(marker.path) {
            attemptedThisProcess.insert(milestone)
            return
        }

        // Claim the process-local attempt before suspending in the sender.
        // Repeated UI notifications therefore cannot create a retry loop.
        attemptedThisProcess.insert(milestone)
        guard let request = Self.request(version: version, milestone: milestone) else { return }
        guard Self.processState.allowsSending, isEnabled() else { return }
        let statusCode = await send(request)
        switch Self.delivery(for: statusCode) {
        case .accepted, .discard:
            // Permanent client/protocol rejections are resolved just like an
            // accepted response: retrying an unchanged request every launch
            // cannot succeed. Transient failures deliberately remain open.
            // SingleInstanceGuard prevents another Desktop process racing this
            // request. Remember the resolution process-wide even if the durable
            // write fails, so a second reporter cannot re-send this launch.
            // A crash/relaunch after a 2xx but before a successful marker write
            // remains an intentional at-least-once delivery edge.
            Self.processState.resolve(marker.path)
            _ = claimMarker(at: marker)
        case .retry:
            return
        }
    }

    enum Delivery: Equatable {
        case accepted
        case discard
        case retry
    }

    nonisolated static func delivery(for statusCode: Int?) -> Delivery {
        guard let statusCode else { return .retry }
        if (200..<300).contains(statusCode) { return .accepted }
        if (400..<500).contains(statusCode), statusCode != 408, statusCode != 429 {
            return .discard
        }
        return .retry
    }

    nonisolated static func request(version: String, milestone: Milestone) -> URLRequest? {
        struct Payload: Encodable {
            let v: String
            let m: String
        }

        guard let body = try? JSONEncoder().encode(Payload(v: version, m: milestone.rawValue)),
              body.count <= maxBodyBytes else {
            return nil
        }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 3
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Rapid-Desktop/\(version)", forHTTPHeaderField: "User-Agent")
        request.httpBody = body
        return request
    }

    /// Pure gate used by the production resolver and exhaustive unit tests.
    /// `telemetryDeclined == false` includes both undecided and opted-in users.
    nonisolated static func gatesAllowSending(
        telemetryDeclined: Bool,
        updateChecksEnabled: Bool,
        environment: [String: String],
        bundleIdentifier: String?,
        releaseBuild: Bool
    ) -> Bool {
        guard !telemetryDeclined,
              updateChecksEnabled,
              !TelemetryConfig.killSwitchActive(environment: environment),
              !isTestOrHarness(environment: environment),
              bundleIdentifier == productionBundleIdentifier,
              releaseBuild else {
            return false
        }
        return true
    }

    nonisolated static func isTestOrHarness(environment: [String: String]) -> Bool {
        let exactKeys = [
            "XCTestConfigurationFilePath",
            "XCTestBundlePath",
            "XCInjectBundle",
            "XCInjectBundleInto",
        ]
        if exactKeys.contains(where: { !(environment[$0] ?? "").isEmpty }) {
            return true
        }
        let prefixes = ["RAPID_GUI_", "RAPID_DEV_", "RAPID_XCUI_", "RAPID_SIMULATED_"]
        return environment.contains { key, value in
            !value.isEmpty && prefixes.contains(where: key.hasPrefix)
        }
    }

    nonisolated private static func productionEnabled() -> Bool {
        let environment = ProcessInfo.processInfo.environment
        let markerDirectory = TelemetryIdentity.sharedTelemetryDirectory(environment: environment)
        return processState.allowsSending && gatesAllowSending(
            telemetryDeclined: TelemetryConsent.hasExplicitlyDeclined(
                telemetryDirectory: markerDirectory
            ),
            updateChecksEnabled: UpdateChecker.updateChecksEnabled(
                environment: environment
            ),
            environment: environment,
            bundleIdentifier: Bundle.main.bundleIdentifier,
            releaseBuild: Bundle.main.object(
                forInfoDictionaryKey: releaseBuildInfoKey
            ) as? Bool == true
        )
    }

    nonisolated private static func sendRequest(_ request: URLRequest) async -> Int? {
        do {
            let (_, response) = try await session.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode
        } catch {
            return nil
        }
    }

    private func markerURL(for milestone: Milestone) -> URL {
        markerDirectory.appendingPathComponent(
            "desktop_funnel_\(milestone.rawValue)",
            isDirectory: false
        )
    }

    private var cohortMarkerURL: URL {
        markerDirectory.appendingPathComponent(Self.cohortMarkerName, isDirectory: false)
    }

    private func claimMarker(at url: URL) -> Bool {
        if let claimMarkerOverride { return claimMarkerOverride(url) }
        do {
            try FileManager.default.createDirectory(
                at: markerDirectory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
        } catch {
            return false
        }

        let descriptor = url.path.withCString {
            open($0, O_WRONLY | O_CREAT | O_EXCL, mode_t(0o600))
        }
        guard descriptor >= 0 else { return false }
        close(descriptor)
        return true
    }
}

/// Small lock-protected process state shared by every reporter instance.
/// Tests use unique marker paths, so resolved delivery state cannot leak
/// between cases even though production intentionally keeps it for app life.
final class DesktopFunnelProcessState: @unchecked Sendable {
    private let lock = NSLock()
    private var optedOut = false
    private var resolvedMarkers: Set<String> = []

    var allowsSending: Bool { lock.withLock { !optedOut } }

    func latchOptOut() {
        lock.withLock { optedOut = true }
    }

    func isResolved(_ markerPath: String) -> Bool {
        lock.withLock { resolvedMarkers.contains(markerPath) }
    }

    func resolve(_ markerPath: String) {
        lock.withLock { _ = resolvedMarkers.insert(markerPath) }
    }

    func reset() {
        lock.withLock {
            optedOut = false
            resolvedMarkers.removeAll()
        }
    }
}

/// Causal gate for the one engine lifecycle attributable to onboarding.
final class DesktopFunnelEngineAttemptGate: @unchecked Sendable {
    private struct Attempt {
        let token: UUID
        let alias: String
    }

    private let lock = NSLock()
    private var attempt: Attempt?

    @discardableResult
    func arm(alias: String) -> UUID {
        let token = UUID()
        lock.withLock {
            attempt = Attempt(token: token, alias: Self.normalized(alias))
        }
        return token
    }

    func disarm(token: UUID) {
        lock.withLock {
            if attempt?.token == token { attempt = nil }
        }
    }

    func consume(milestone: DesktopFunnelReporter.Milestone, alias: String) -> Bool {
        guard milestone == .engineReady || milestone == .engineStartFailed else { return false }
        return lock.withLock {
            guard let current = attempt,
                  current.alias == Self.normalized(alias) else { return false }
            attempt = nil
            return true
        }
    }

    private static func normalized(_ alias: String) -> String {
        alias.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }
}
