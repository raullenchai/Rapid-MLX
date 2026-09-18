import AppKit
import Foundation

/// One Desktop per user session.
///
/// A second instance of the app (`open -n`, exec'ing `Contents/MacOS/Rapid`,
/// a second copy of the bundle) started its own sidecar on the same port,
/// won it, and the first instance's chat dropped to "Couldn't start <model>
/// — check the model files" while the model was in fact loaded next door
/// (0.14.3 dogfood, 2026-09-18, reproduced with `open -n`).
/// ``LSMultipleInstancesProhibited`` in Info.plist covers Finder and
/// `open -a`; this covers everything else. ``RapidApp.init`` consults it
/// before any other side effect — in particular before
/// ``PortSweep.startLaunchSweep``, which trusts the on-disk sidecar
/// ownership record and would reap the running instance's engine as an
/// orphan.
enum SingleInstanceGuard {
    /// What the decision knows about one process.
    struct Instance: Equatable {
        var pid: pid_t
        var bundleID: String?
        /// ``NSRunningApplication.launchDate``; nil when unknown.
        var launched: Date?
    }

    /// The running Desktop this launch must defer to, or `nil` when this
    /// process is the one that stays.
    static func runningInstanceToYieldTo() -> NSRunningApplication? {
        guard let bundleID = Bundle.main.bundleIdentifier else { return nil }
        let apps = NSWorkspace.shared.runningApplications
        let own = Instance(
            pid: getpid(),
            bundleID: bundleID,
            launched: NSRunningApplication.current.launchDate
        )
        let running = apps.map {
            Instance(pid: $0.processIdentifier, bundleID: $0.bundleIdentifier, launched: $0.launchDate)
        }
        guard let pid = pidToYieldTo(own: own, running: running) else { return nil }
        return apps.first { $0.processIdentifier == pid }
    }

    /// Pure decision: the PID of the most senior OTHER process with our
    /// bundle identifier when it outranks us, else `nil`.
    ///
    /// Seniority is launch order (``outranks``): the instance that has been
    /// running longer stays, which is what the user expects when they
    /// double-click an app that is already open. Two instances of a
    /// concurrent cold launch each see the other and rank the pair from the
    /// same facts, so exactly one yields: the later-launched (or, on a tie,
    /// higher-PID) process. Our own PID never counts, and a process with no
    /// bundle identifier is never "us".
    static func pidToYieldTo(own: Instance, running: [Instance]) -> pid_t? {
        let others = running.filter {
            $0.pid != own.pid && $0.bundleID != nil && $0.bundleID == own.bundleID
        }
        guard let senior = others.min(by: outranks) else { return nil }
        return outranks(senior, own) ? senior.pid : nil
    }

    /// Strict ordering used for seniority: `a` outranks `b` when it launched
    /// earlier; with equal or unknown launch dates the lower PID outranks.
    static func outranks(_ a: Instance, _ b: Instance) -> Bool {
        if let la = a.launched, let lb = b.launched, la != lb {
            return la < lb
        }
        return a.pid < b.pid
    }

    /// Bring the survivor forward, window included.
    ///
    /// ``NSRunningApplication.activate`` only raises windows that exist; a
    /// Desktop whose last window was closed with ⌘W stays alive as a Dock
    /// icon with no window (``applicationShouldTerminateAfterLastWindowClosed``
    /// is false). So ask LaunchServices to open the survivor's bundle — the
    /// same request `open -a` and a Dock click deliver — which lands in
    /// ``AppDelegate.applicationShouldHandleReopen`` and opens the main
    /// window. The wait is bounded: a stalled LaunchServices must not keep
    /// this doomed process alive, and a plain activate is the fallback.
    static func handOff(to survivor: NSRunningApplication) {
        guard let url = survivor.bundleURL else {
            survivor.activate()
            return
        }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        configuration.createsNewApplicationInstance = false
        let done = DispatchSemaphore(value: 0)
        NSWorkspace.shared.openApplication(at: url, configuration: configuration) { _, _ in
            done.signal()
        }
        if done.wait(timeout: .now() + 2) == .timedOut {
            survivor.activate()
        }
    }
}
