import AppKit
import Foundation
import os

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
        /// ``NSRunningApplication.isFinishedLaunching``: an established
        /// Desktop rather than a process still coming up.
        var finishedLaunching: Bool = false
    }

    /// What ``RapidApp.init`` should do about other instances.
    enum Decision {
        /// This process is the Desktop: carry on launching.
        case proceed
        /// Another instance is running: hand off to it and exit.
        case yield(to: NSRunningApplication)
        /// Another instance holds the lock but LaunchServices has not
        /// registered it yet (it is mid-launch): ask LaunchServices to open
        /// the bundle — which reaches it once registered — and exit.
        case yieldToUnregistered
    }

    /// The whole launch-time protocol, called once at the top of
    /// ``RapidApp.init``. `true`: a Desktop is already running, the launch
    /// has been handed to it, and this process must exit. `false`: this
    /// process is the Desktop and holds the instance lock (when the lock
    /// works at all).
    ///
    /// The survivor may be on its way out. Quit and relaunch within a second
    /// — ⌘Q then a Dock click, a `killall`-free restart script — and the new
    /// process finds the old one still registered, hands off to it, exits,
    /// and the old one finishes quitting: no app at all (pr_validate codex;
    /// reproduced on the 0.14.3 dogfood mini with `quit` + exec 0.2 s
    /// later). So after handing off, wait a bounded few seconds for the
    /// lock: the kernel releases it the instant the holder dies, and taking
    /// it makes this launch the Desktop. A survivor that stays is the normal
    /// case; the wait then costs a hidden process a few seconds before it
    /// exits. Without a usable lock (``.unavailable``) there is nothing to
    /// wait for, so the hand-off result decides.
    static func yieldsLaunch() -> Bool {
        let handedOff: Bool
        switch decide() {
        case .proceed:
            return false
        case .yield(let survivor):
            handedOff = handOff(to: survivor)
        case .yieldToUnregistered:
            handOffToUnregisteredHolder()
            handedOff = true
        }
        let clock = ContinuousClock()
        let deadline = clock.now + .seconds(handedOff ? survivorGraceSeconds : 0.5)
        repeat {
            switch instanceLock.acquire() {
            case .acquired:
                return false
            case .unavailable:
                return handedOff
            case .busy:
                Thread.sleep(forTimeInterval: 0.1)
            }
        } while clock.now < deadline
        return true
    }

    /// How long a yielding launch waits for a quitting survivor to release
    /// the lock. A graceful quit with a 27B model resident took 0.8 s on an
    /// M2 Pro; 5 s leaves room for a slow engine shutdown.
    static let survivorGraceSeconds: Double = 5

    /// Decide once, at the top of ``RapidApp.init``.
    ///
    /// Two layers. The advisory ``instanceLock`` is authoritative when it is
    /// available: whoever holds it is the Desktop, and it is taken before
    /// LaunchServices has registered the process, so two cold launches a few
    /// milliseconds apart cannot both miss each other in a
    /// `runningApplications` snapshot (pr_validate codex). The snapshot rule
    /// (``pidToYieldTo``) then only identifies WHICH process to hand off to —
    /// or decides on its own when the lock file cannot be created at all.
    static func decide() -> Decision {
        switch instanceLock.acquire() {
        case .acquired:
            return .proceed
        case .busy:
            // The holder may still be registering with LaunchServices.
            let clock = ContinuousClock()
            let deadline = clock.now + .seconds(1.5)
            repeat {
                if let holder = runningInstanceToYieldTo() { return .yield(to: holder) }
                Thread.sleep(forTimeInterval: 0.05)
            } while clock.now < deadline
            return .yieldToUnregistered
        case .unavailable:
            if let other = runningInstanceToYieldTo() { return .yield(to: other) }
            return .proceed
        }
    }

    /// Process-lifetime advisory lock under Application Support.
    static let instanceLock = InstanceLock(
        url: ApplicationSupportLocator.applicationSupportRoot()
            .appendingPathComponent("desktop-instance.lock")
    )

    /// `flock(2)` on a file, held until the process exits. `flock` locks are
    /// per open file description, so a second `open` + `flock` conflicts even
    /// inside one process (which is what the unit test relies on), and the
    /// kernel drops the lock when the holder dies — no stale-lock cleanup.
    final class InstanceLock: @unchecked Sendable {
        enum Outcome { case acquired, busy, unavailable }

        let url: URL
        private var descriptor: Int32 = -1
        private let queue = DispatchQueue(label: "rapid.instance-lock")

        init(url: URL) { self.url = url }

        /// Idempotent: a lock this object already holds reports `.acquired`.
        func acquire() -> Outcome {
            queue.sync {
                if descriptor >= 0 { return .acquired }
                try? FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(), withIntermediateDirectories: true
                )
                let fd = open(url.path, O_CREAT | O_RDWR | O_CLOEXEC, 0o644)
                guard fd >= 0 else { return .unavailable }
                if flock(fd, LOCK_EX | LOCK_NB) == 0 {
                    descriptor = fd
                    return .acquired
                }
                let reason = errno
                close(fd)
                return (reason == EWOULDBLOCK || reason == EAGAIN) ? .busy : .unavailable
            }
        }

        /// Tests only; the app holds the lock until the process exits.
        func release() {
            queue.sync {
                guard descriptor >= 0 else { return }
                flock(descriptor, LOCK_UN)
                close(descriptor)
                descriptor = -1
            }
        }
    }

    /// The running Desktop this launch must defer to, or `nil` when this
    /// process is the one that stays (snapshot rule).
    static func runningInstanceToYieldTo() -> NSRunningApplication? {
        guard let bundleID = Bundle.main.bundleIdentifier else { return nil }
        let apps = NSWorkspace.shared.runningApplications
        let own = Instance(
            pid: getpid(),
            bundleID: bundleID,
            launched: NSRunningApplication.current.launchDate
        )
        let running = apps.map {
            Instance(
                pid: $0.processIdentifier,
                bundleID: $0.bundleIdentifier,
                launched: $0.launchDate,
                finishedLaunching: $0.isFinishedLaunching
            )
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
        // An instance that has finished launching is the Desktop the user is
        // looking at; it wins whatever the dates say. Ranking is only for
        // processes still coming up together (concurrent cold launch), and
        // it keeps a Desktop whose launch date LaunchServices cannot report
        // from being mistaken for the newest process (codex r3).
        if let established = others.filter(\.finishedLaunching).min(by: outranks) {
            return established.pid
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

    /// Hand off to a holder LaunchServices has not registered yet: opening
    /// our own bundle reaches whichever instance of it ends up registered
    /// (``LSMultipleInstancesProhibited`` makes that a reopen, not a launch).
    /// The wait is bounded so a stalled LaunchServices cannot keep this
    /// doomed process alive. The outcome is deliberately not consulted: the
    /// holder is mid-launch and will show its own main window regardless,
    /// so this request is only a courtesy activation, and exiting is right
    /// whether or not it landed.
    static func handOffToUnregisteredHolder() {
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        configuration.createsNewApplicationInstance = false
        let done = DispatchSemaphore(value: 0)
        NSWorkspace.shared.openApplication(at: Bundle.main.bundleURL, configuration: configuration) { _, _ in
            done.signal()
        }
        _ = done.wait(timeout: .now() + 2)
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
    /// a timeout or an error while the survivor is still running. Whether
    /// that activation lands is cosmetic — the survivor is alive and is the
    /// Desktop either way — so only the survivor's liveness decides.
    static func handOff(to survivor: NSRunningApplication) -> Bool {
        guard !survivor.isTerminated else { return false }
        guard let url = survivor.bundleURL else {
            return survivor.activate()
        }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        configuration.createsNewApplicationInstance = false
        let done = DispatchSemaphore(value: 0)
        // Written on NSWorkspace's completion queue, read here after the
        // semaphore: the lock makes that hand-over explicit to the compiler.
        let failure = OSAllocatedUnfairLock<Error?>(initialState: nil)
        NSWorkspace.shared.openApplication(at: url, configuration: configuration) { _, error in
            failure.withLock { $0 = error }
            done.signal()
        }
        let timedOut = done.wait(timeout: .now() + 2) == .timedOut
        if survivor.isTerminated { return false }
        if timedOut || failure.withLock({ $0 }) != nil {
            survivor.activate()
        }
        return true
    }
}
