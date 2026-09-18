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

    /// Strict total ordering used for seniority: `a` outranks `b` when it
    /// launched earlier; an unknown launch date ranks after every known one
    /// (a process LaunchServices cannot date is assumed to be the newest);
    /// equal or both-unknown dates fall back to the lower PID. Total, so a
    /// roster with mixed known/unknown dates still ranks the same way from
    /// every instance's point of view (codex r2: a date-then-PID rule that
    /// switched keys per pair could cycle and let three launches all stay).
    static func outranks(_ a: Instance, _ b: Instance) -> Bool {
        let la = a.launched ?? .distantFuture
        let lb = b.launched ?? .distantFuture
        if la != lb { return la < lb }
        return a.pid < b.pid
    }

    /// Bring the survivor forward, window included. Returns `false` when
    /// the survivor is gone, in which case the caller should carry on
    /// launching instead of leaving the user with no app at all.
    ///
    /// ``NSRunningApplication.activate`` only raises windows that exist; a
    /// Desktop whose last window was closed with ⌘W stays alive as a Dock
    /// icon with no window (``applicationShouldTerminateAfterLastWindowClosed``
    /// is false). So ask LaunchServices to open the survivor's bundle — the
    /// same request `open -a` and a Dock click deliver — which lands in
    /// ``AppDelegate.applicationShouldHandleReopen`` and opens the main
    /// window. The wait is bounded: a stalled LaunchServices must not keep
    /// this doomed process alive, and a plain activate is the fallback for
    /// a timeout or an error while the survivor is still running. If the
    /// survivor quit in the meantime (user quit and relaunched at once),
    /// there is nobody to hand off to.
    static func handOff(to survivor: NSRunningApplication) -> Bool {
        guard !survivor.isTerminated else { return false }
        guard let url = survivor.bundleURL else {
            return survivor.activate()
        }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        configuration.createsNewApplicationInstance = false
        let done = DispatchSemaphore(value: 0)
        var failure: Error?
        NSWorkspace.shared.openApplication(at: url, configuration: configuration) { _, error in
            failure = error
            done.signal()
        }
        let timedOut = done.wait(timeout: .now() + 2) == .timedOut
        if survivor.isTerminated { return false }
        if timedOut || failure != nil {
            survivor.activate()
        }
        return true
    }
}
