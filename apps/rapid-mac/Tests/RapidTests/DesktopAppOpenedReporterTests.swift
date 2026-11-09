import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Desktop app_opened")
struct DesktopAppOpenedReporterTests {
    private func isolatedDefaults() -> UserDefaults {
        UserDefaults(suiteName: "DesktopAppOpenedTests.\(UUID().uuidString)")!
    }

    private func date(_ value: String) -> Date {
        ISO8601DateFormatter().date(from: value)!
    }

    @Test("No event or date claim before consent, then one launch after consent")
    func consentGating() async {
        let defaults = isolatedDefaults()
        defer { defaults.removeObject(forKey: DesktopAppOpenedReporter.lastDateKey) }
        var enabled = false
        var triggers: [DesktopAppOpenedReporter.Trigger] = []
        let reporter = DesktopAppOpenedReporter(
            defaults: defaults, enabled: { enabled },
            send: { triggers.append($0) }
        )
        await reporter.foregroundUsed()
        #expect(triggers.isEmpty)
        #expect(defaults.object(forKey: DesktopAppOpenedReporter.lastDateKey) == nil)
        enabled = true
        await reporter.foregroundUsed()
        await reporter.foregroundUsed()
        #expect(triggers == [.launch])
    }

    @Test("Existing consent checks preserve environment opt-outs", arguments: [
        ["DO_NOT_TRACK": "1"], ["RAPID_MLX_TELEMETRY": "0"], ["CI": "1"]
    ])
    func environmentGating(environment: [String: String]) async {
        let defaults = isolatedDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        defer { defaults.removeObject(forKey: TelemetryConfig.enabledKey) }
        var triggers: [DesktopAppOpenedReporter.Trigger] = []
        let reporter = DesktopAppOpenedReporter(
            defaults: defaults,
            enabled: { TelemetryConfig.isEnabled(defaults: defaults, environment: environment) },
            send: { triggers.append($0) }
        )
        await reporter.foregroundUsed()
        #expect(triggers.isEmpty)
        #expect(defaults.object(forKey: DesktopAppOpenedReporter.lastDateKey) == nil)
    }

    @Test("UTC rollover requires foreground use; repeated input and relaunches deduplicate")
    func launchAndDailyDedup() async {
        let defaults = isolatedDefaults()
        defer { defaults.removeObject(forKey: DesktopAppOpenedReporter.lastDateKey) }
        var moment = date("2026-11-09T23:59:59Z")
        var triggers: [DesktopAppOpenedReporter.Trigger] = []
        func makeReporter() -> DesktopAppOpenedReporter {
            DesktopAppOpenedReporter(
                defaults: defaults, enabled: { true }, now: { moment },
                send: { triggers.append($0) }
            )
        }
        let reporter = makeReporter()
        await reporter.foregroundUsed()
        await reporter.foregroundUsed()
        #expect(triggers == [.launch])
        #expect(defaults.string(forKey: DesktopAppOpenedReporter.lastDateKey) == "2026-11-09")
        moment = date("2026-11-10T00:00:01Z")
        #expect(triggers == [.launch]) // Advancing time alone sends nothing.
        await reporter.foregroundUsed()
        await reporter.foregroundUsed()
        #expect(triggers == [.launch, .dayRollover])
        let relaunched = makeReporter()
        await relaunched.foregroundUsed()
        await relaunched.foregroundUsed()
        #expect(triggers == [.launch, .dayRollover, .launch])
        moment = date("2026-11-11T12:00:00Z")
        await relaunched.foregroundUsed()
        await reporter.foregroundUsed() // The persisted claim suppresses another reporter on that date.
        #expect(triggers == [.launch, .dayRollover, .launch, .dayRollover])
        moment = date("2026-11-10T12:00:00Z") // Clock reversal never reclaims an old day.
        await relaunched.foregroundUsed()
        #expect(triggers.count == 4)
    }

    @Test("Opt-out does not consume a future day; opt-in resumes on foreground use")
    func optOutAndResume() async {
        let defaults = isolatedDefaults()
        defer { defaults.removeObject(forKey: DesktopAppOpenedReporter.lastDateKey) }
        var enabled = true
        var moment = date("2026-11-09T12:00:00Z")
        var triggers: [DesktopAppOpenedReporter.Trigger] = []
        let reporter = DesktopAppOpenedReporter(
            defaults: defaults, enabled: { enabled }, now: { moment },
            send: { triggers.append($0) }
        )
        await reporter.foregroundUsed()
        enabled = false
        moment = date("2026-11-10T12:00:00Z")
        await reporter.foregroundUsed()
        #expect(defaults.string(forKey: DesktopAppOpenedReporter.lastDateKey) == "2026-11-09")
        enabled = true
        await reporter.foregroundUsed()
        #expect(triggers == [.launch, .dayRollover])
    }

    @Test("Claim precedes suspension, preventing a reentrant foreground duplicate")
    func claimBeforeDelivery() async {
        let defaults = isolatedDefaults()
        defer { defaults.removeObject(forKey: DesktopAppOpenedReporter.lastDateKey) }
        var continuation: CheckedContinuation<Void, Never>?
        var triggers: [DesktopAppOpenedReporter.Trigger] = []
        let reporter = DesktopAppOpenedReporter(
            defaults: defaults, enabled: { true },
            send: { trigger in
                triggers.append(trigger)
                await withCheckedContinuation { continuation = $0 }
            }
        )
        let first = Task { await reporter.foregroundUsed() }
        while continuation == nil { await Task.yield() }
        await reporter.foregroundUsed()
        #expect(triggers == [.launch])
        continuation?.resume()
        await first.value
    }

    @Test("The helper keeps opt-outs and Desktop attribution, excluding ambient secrets")
    func helperEnvironment() {
        let environment = DesktopAppOpenedReporter.engineEnvironment([
            "HOME": "/tmp/isolated-home", "DO_NOT_TRACK": "1",
            "RAPID_MLX_TELEMETRY": "0", "CI": "1", "ANTHROPIC_API_KEY": "secret"
        ])
        #expect(environment["HOME"] == "/tmp/isolated-home")
        #expect(environment["DO_NOT_TRACK"] == "1")
        #expect(environment["RAPID_MLX_TELEMETRY"] == "0")
        #expect(environment["CI"] == "1")
        #expect(environment[EngineProcessEnvironment.roleKey] == EngineProcessEnvironment.sidecarRole)
        #expect(environment[EngineProcessEnvironment.appVersionKey] == TelemetryClient.currentVersion())
        #expect(environment["ANTHROPIC_API_KEY"] == nil)
    }

}
