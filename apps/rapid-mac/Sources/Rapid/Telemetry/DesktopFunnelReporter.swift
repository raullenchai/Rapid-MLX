import Darwin
import Foundation

/// Sends identifier-free, aggregate milestones for the Desktop first-run
/// funnel. The wire body is deliberately limited to app version + a closed
/// milestone enum; no telemetry identity is read or created.
actor DesktopFunnelReporter {
    struct FlowToken: Hashable, Sendable {
        let id: UUID
    }

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
    typealias Send = @Sendable (URLRequest) async -> Void
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
    nonisolated static func enqueue(
        _ milestone: Milestone,
        flowToken: FlowToken? = nil
    ) {
        Task.detached(priority: .utility) {
            await shared.report(milestone, flowToken: flowToken)
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

    /// Enrols one genuine first-run flow in this process. A remounted view can
    /// recover the same token, but a later process or re-onboarding flow cannot
    /// derive one from the durable cohort marker alone.
    nonisolated static func beginFirstRunFlow(isFirstRun: Bool) -> FlowToken? {
        guard isFirstRun else { return nil }
        let directory = TelemetryIdentity.sharedTelemetryDirectory()
        let cohort = directory.appendingPathComponent(cohortMarkerName, isDirectory: false)
        return processState.enrollFlow(cohortPath: cohort.path) {
            claimMarker(at: cohort, in: directory)
        }
    }

    /// Arms exactly the first engine start initiated by the active first-run
    /// flow. Re-onboarding has no flow token and therefore cannot arm it.
    @discardableResult
    nonisolated static func armOnboardingEngineAttempt(
        alias: String,
        flowToken: FlowToken?
    ) -> UUID? {
        guard let flowToken, processState.contains(flowToken) else { return nil }
        return engineAttemptGate.arm(alias: alias, flowToken: flowToken)
    }

    nonisolated static func disarmOnboardingEngineAttempt(_ token: UUID) {
        engineAttemptGate.disarm(token: token)
    }

    nonisolated static func retainOnboardingEngineAttempt(_ token: UUID) {
        engineAttemptGate.retain(token: token)
    }

    nonisolated static func releaseOnboardingEngineAttempt(_ token: UUID) {
        engineAttemptGate.release(token: token)
    }

    /// ServerManager routes every lifecycle terminal through this seam. Only a
    /// terminal matching the currently armed onboarding start is consumed and
    /// emitted; later manual starts, restarts, and model switches are inert.
    nonisolated static func enqueueEngineOutcomeIfArmed(
        _ milestone: Milestone,
        alias: String
    ) {
        guard let flowToken = engineAttemptGate.consume(
            milestone: milestone,
            alias: alias
        ) else { return }
        enqueue(milestone, flowToken: flowToken)
    }

    /// Enrols only a genuinely new install. The empty cohort marker is local
    /// state, written before and independently of consent or network delivery,
    /// so later milestones cannot accidentally include upgrading installs.
    nonisolated static func enqueueOnboardingShown(flowToken: FlowToken) {
        Task.detached(priority: .utility) {
            await shared.report(.onboardingShown, flowToken: flowToken)
        }
    }

    func beginFirstRunFlow(isFirstRun: Bool) -> FlowToken? {
        guard isFirstRun else { return nil }
        let cohort = cohortMarkerURL
        return Self.processState.enrollFlow(cohortPath: cohort.path) {
            claimMarker(at: cohort)
        }
    }

    func report(_ milestone: Milestone, flowToken: FlowToken? = nil) async {
        guard FileManager.default.fileExists(atPath: cohortMarkerURL.path) else { return }
        if milestone != .firstChatReply {
            guard let flowToken,
                  Self.processState.matches(
                      flowToken,
                      cohortPath: cohortMarkerURL.path
                  ) else { return }
        }
        guard Self.processState.allowsSending,
              isEnabled() else { return }

        let marker = markerURL(for: milestone)
        guard !FileManager.default.fileExists(atPath: marker.path) else { return }
        guard let request = Self.request(version: version, milestone: milestone) else { return }
        guard Self.processState.allowsSending, isEnabled() else { return }
        // The durable claim is the delivery boundary. Once it succeeds this
        // milestone is never attempted again, even if transport fails.
        guard claimMarker(at: marker) else { return }
        await send(request)
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

    nonisolated private static func sendRequest(_ request: URLRequest) async {
        do {
            _ = try await session.data(for: request)
        } catch {
            return
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
        return Self.claimMarker(at: url, in: markerDirectory)
    }

    nonisolated private static func claimMarker(at url: URL, in directory: URL) -> Bool {
        do {
            try FileManager.default.createDirectory(
                at: directory,
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
final class DesktopFunnelProcessState: @unchecked Sendable {
    private let lock = NSLock()
    private var optedOut = false
    private var flowsByCohortPath: [String: DesktopFunnelReporter.FlowToken] = [:]

    var allowsSending: Bool { lock.withLock { !optedOut } }

    func latchOptOut() {
        lock.withLock { optedOut = true }
    }

    func enrollFlow(
        cohortPath: String,
        claim: () -> Bool
    ) -> DesktopFunnelReporter.FlowToken? {
        lock.withLock {
            if let existing = flowsByCohortPath[cohortPath] { return existing }
            guard claim() else { return nil }
            let token = DesktopFunnelReporter.FlowToken(id: UUID())
            flowsByCohortPath[cohortPath] = token
            return token
        }
    }

    func contains(_ token: DesktopFunnelReporter.FlowToken) -> Bool {
        lock.withLock { flowsByCohortPath.values.contains(token) }
    }

    func matches(_ token: DesktopFunnelReporter.FlowToken, cohortPath: String) -> Bool {
        lock.withLock { flowsByCohortPath[cohortPath] == token }
    }

    func reset() {
        lock.withLock {
            optedOut = false
            flowsByCohortPath.removeAll()
        }
    }
}

/// Causal gate for the one engine lifecycle attributable to onboarding.
final class DesktopFunnelEngineAttemptGate: @unchecked Sendable {
    private struct Attempt {
        let token: UUID
        let alias: String
        let flowToken: DesktopFunnelReporter.FlowToken
        var retained = false
    }

    private let lock = NSLock()
    private var attempt: Attempt?

    @discardableResult
    func arm(alias: String, flowToken: DesktopFunnelReporter.FlowToken) -> UUID {
        let token = UUID()
        lock.withLock {
            attempt = Attempt(
                token: token,
                alias: Self.normalized(alias),
                flowToken: flowToken
            )
        }
        return token
    }

    func disarm(token: UUID) {
        lock.withLock {
            if attempt?.token == token, attempt?.retained == false { attempt = nil }
        }
    }

    func retain(token: UUID) {
        lock.withLock {
            if attempt?.token == token { attempt?.retained = true }
        }
    }

    func release(token: UUID) {
        lock.withLock {
            if attempt?.token == token { attempt = nil }
        }
    }

    func consume(
        milestone: DesktopFunnelReporter.Milestone,
        alias: String
    ) -> DesktopFunnelReporter.FlowToken? {
        guard milestone == .engineReady || milestone == .engineStartFailed else { return nil }
        return lock.withLock {
            guard let current = attempt,
                  current.alias == Self.normalized(alias) else { return nil }
            attempt = nil
            return current.flowToken
        }
    }

    private static func normalized(_ alias: String) -> String {
        alias.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }
}
