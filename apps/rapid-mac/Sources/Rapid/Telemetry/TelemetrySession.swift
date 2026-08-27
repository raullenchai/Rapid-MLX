import Foundation

/// Process-lifetime lifecycle reporter shared by the launch task, the
/// post-value consent invitation, and Settings. The enabled check happens
/// before the once latch so opting in after launch can still emit the
/// current session exactly once.
@MainActor
enum TelemetrySession {
    private static var startFired = false

    static func sendStartIfNeeded() async {
        guard TelemetryConfig.isEnabled, !startFired else { return }
        startFired = true
        await TelemetryClient().send(.sessionStart(
            version: TelemetryClient.currentVersion(),
            platform: TelemetryClient.currentPlatform()
        ))
    }

    static func resetForTesting() {
        startFired = false
    }
}
