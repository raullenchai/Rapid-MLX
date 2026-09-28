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

    static let shared = DesktopFunnelReporter()
    nonisolated static let endpoint = URL(string: "https://rapidmlx.com/api/desktop-funnel")!
    nonisolated static let productionBundleIdentifier = "com.rapidmlx.rapid"
    nonisolated static let maxBodyBytes = 256

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
    private var attemptedThisProcess: Set<Milestone> = []

    init(
        isEnabled: @escaping Enabled = { DesktopFunnelReporter.productionEnabled() },
        send: @escaping Send = DesktopFunnelReporter.sendRequest,
        markerDirectory: URL = TelemetryIdentity.sharedTelemetryDirectory(),
        version: String = TelemetryClient.currentVersion()
    ) {
        self.isEnabled = isEnabled
        self.send = send
        self.markerDirectory = markerDirectory
        self.version = version
    }

    /// The UI-facing entry point. Creating the task is immediate; eligibility,
    /// disk access, JSON encoding, and networking all happen away from the main
    /// actor and can never delay a view or lifecycle transition.
    nonisolated static func enqueue(_ milestone: Milestone) {
        Task.detached(priority: .utility) {
            await shared.report(milestone)
        }
    }

    func report(_ milestone: Milestone) async {
        guard !attemptedThisProcess.contains(milestone), isEnabled() else { return }

        let marker = markerURL(for: milestone)
        if FileManager.default.fileExists(atPath: marker.path) {
            attemptedThisProcess.insert(milestone)
            return
        }

        // Claim the process-local attempt before suspending in the sender.
        // Repeated UI notifications therefore cannot create a retry loop.
        attemptedThisProcess.insert(milestone)
        guard let request = Self.request(version: version, milestone: milestone) else { return }
        guard isEnabled() else { return }
        guard let statusCode = await send(request),
              (200..<300).contains(statusCode) else { return }
        _ = claimMarker(at: marker)
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
        bundleIdentifier: String?
    ) -> Bool {
        guard !telemetryDeclined,
              updateChecksEnabled,
              !TelemetryConfig.killSwitchActive(environment: environment),
              !isTestOrHarness(environment: environment),
              bundleIdentifier == productionBundleIdentifier else {
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
        return gatesAllowSending(
            telemetryDeclined: TelemetryConsent.hasExplicitlyDeclined(
                telemetryDirectory: markerDirectory
            ),
            updateChecksEnabled: UpdateChecker.updateChecksEnabled(
                environment: environment
            ),
            environment: environment,
            bundleIdentifier: Bundle.main.bundleIdentifier
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

    private func claimMarker(at url: URL) -> Bool {
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
