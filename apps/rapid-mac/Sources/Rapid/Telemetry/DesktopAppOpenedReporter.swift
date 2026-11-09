import Foundation

/// A launch is attempted once per consented process, even on same-day relaunch.
/// Later foreground use claims a new UTC date at most once. Claims precede
/// best-effort delivery: a failed helper/network request loses the event rather
/// than retrying it on every click. No consent means no identity or marker work.
@MainActor
final class DesktopAppOpenedReporter {
    enum Trigger: String { case launch, dayRollover = "day_rollover" }
    static let shared = DesktopAppOpenedReporter()
    static let lastDateKey = "com.rapidmlx.rapid.telemetry.appOpenedUTCDate"

    private let defaults: UserDefaults
    private let enabled: () -> Bool
    private let now: () -> Date
    private let send: (Trigger) async -> Void
    private var launched = false
    private var lastDate: String?

    init(
        defaults: UserDefaults = .standard,
        enabled: @escaping () -> Bool = { TelemetryConfig.isEnabled },
        now: @escaping () -> Date = { Date() },
        send: @escaping (Trigger) async -> Void = { await DesktopAppOpenedReporter.sendUsingEngine($0) }
    ) {
        self.defaults = defaults
        self.enabled = enabled
        self.now = now
        self.send = send
    }

    func foregroundUsed() async {
        guard enabled() else { return }
        let formatter = ISO8601DateFormatter()
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.formatOptions = [.withFullDate]
        let today = formatter.string(from: now())
        let trigger: Trigger
        if !launched {
            trigger = .launch
            launched = true
        } else {
            guard today > max(lastDate ?? "", defaults.string(forKey: Self.lastDateKey) ?? "") else { return }
            trigger = .dayRollover
        }
        lastDate = today
        defaults.set(max(today, defaults.string(forKey: Self.lastDateKey) ?? ""), forKey: Self.lastDateKey)
        await send(trigger)
    }

    nonisolated static func engineEnvironment(_ ambient: [String: String]) -> [String: String] {
        var environment = ServerManager.serveEnvironmentAdditions(bearer: "", ambient: ambient)
        // Keep the engine's own opt-out check authoritative as well as Swift's.
        for key in [TelemetryConfig.killSwitchEnvironmentKey, TelemetryConfig.doNotTrackEnvironmentKey]
            + TelemetryConfig.ciEnvironmentKeys {
            environment[key] = ambient[key]
        }
        return EngineProcessEnvironment.sidecar(environment)
    }

    /// Resolve the signed bundled engine independently of ServerManager/model
    /// selection. Reuse its role, trust boundary, consent and PostHog transport.
    private static func sendUsingEngine(_ trigger: Trigger) async {
        let rawTrigger = trigger.rawValue
        await Task.detached(priority: .utility) {
            guard TelemetryConfig.isEnabled,
                  let binary = ServerLocator.find() else { return }
            let process = Process()
            process.executableURL = binary
            process.arguments = ["telemetry", "desktop-opened", "--trigger", rawTrigger]
            process.environment = engineEnvironment(ProcessInfo.processInfo.environment)
            process.standardInput = FileHandle.nullDevice
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            do {
                try DesktopRuntimeTrust.run(process)
                let deadline = Date().addingTimeInterval(10)
                while process.isRunning, Date() < deadline {
                    try? await Task.sleep(for: .milliseconds(50))
                }
                if process.isRunning { process.terminate() }
            } catch {
                // Telemetry is best effort and must never affect app startup.
            }
        }.value
    }
}
