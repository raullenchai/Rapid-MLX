import AppKit
import Foundation

/// Process-lifetime lifecycle reporter shared by the launch task, the
/// launch disclosure, and Settings. The enabled check happens
/// before the once latch so opting in after launch can still emit the
/// current session exactly once.
@MainActor
enum TelemetrySession {
    private static var startFired = false
    private static var activityMonitor: Any?

    /// Real input catches rollover while the app stays active across midnight.
    /// Activation alone also counts; no background timer emits daily events.
    static func foregroundUsed() async {
        await DesktopAppOpenedReporter.shared.foregroundUsed()
    }

    static func sendStartIfNeeded() async {
        // Install even while opted out so a later Settings opt-in can start
        // reporting on input without requiring a window activation/relaunch.
        // The reporter gates every callback before reading or writing state.
        if activityMonitor == nil {
            activityMonitor = NSEvent.addLocalMonitorForEvents(
                matching: [.keyDown, .leftMouseDown, .rightMouseDown, .scrollWheel]
            ) { event in
                Task { @MainActor in await foregroundUsed() }
                return event
            }
        }
        guard TelemetryConfig.isEnabled else { return }
        await foregroundUsed()
        guard !startFired else { return }
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
