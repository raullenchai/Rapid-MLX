import Foundation
import Testing
@testable import Rapid

/// Pins the canonical termination ordering called by
/// ``AppDelegate.applicationWillTerminate``. Audit P1
/// `AppDelegate.swift:651-686` — the in-flight chat stream task
/// must be cancelled BEFORE the session envelope is normalised /
/// flushed and BEFORE the server child is torn down, otherwise:
///
///   * the URLSessionDataTask FIN reaches rapid-mlx after the
///     child has already been SIGTERM'd by `shutdownServer`,
///   * `finalizeStreamingForTermination` races a late-arriving
///     token chunk that could flip a placeholder back to
///     `.streaming` after the normalisation pass.
///
/// Codex r1 NIT on PR #54: the wiring change was three lines and
/// nothing pinned the stop-first invariant against a future
/// reorder. This suite is the pin.
@MainActor
@Suite("AppDelegate termination ordering")
struct TerminationOrderingTests {

    /// The audit P1 invariant in one assertion: stopStream MUST
    /// be the first call, and every child must be SIGNALLED before
    /// anything BLOCKS reaping one — that overlap is what keeps the
    /// quit path inside AppKit's terminate budget (the grace windows
    /// used to sum to ~7.5 s).
    @Test("runTerminationSequence signals both children before reaping either")
    func sequence_pins_stop_first_then_signal_then_reap() {
        var calls: [String] = []
        AppDelegate.runTerminationSequence(
            stopStream: { calls.append("stop") },
            signalServer: { calls.append("signalServer") },
            signalDownloads: { calls.append("signalDownloads") },
            reapServer: { calls.append("reapServer") },
            reapDownloads: { calls.append("reapDownloads") }
        )
        #expect(calls == [
            "stop", "signalServer", "signalDownloads", "reapServer", "reapDownloads"
        ])
    }

    /// The load-bearing half of the overlap fix, pinned independently
    /// of exact ordering: NO blocking reap may start until BOTH
    /// children have been signalled. If a future refactor collapses
    /// this back into two sequential `shutdownSync()` calls, the
    /// download SIGTERM slides after the server's 5 s grace and the
    /// windows serialise again — this fires.
    @Test("both signals strictly precede both reaps")
    func signals_strictly_precede_reaps() {
        var calls: [String] = []
        AppDelegate.runTerminationSequence(
            stopStream: { _ = calls },
            signalServer: { calls.append("signalServer") },
            signalDownloads: { calls.append("signalDownloads") },
            reapServer: { calls.append("reapServer") },
            reapDownloads: { calls.append("reapDownloads") }
        )
        let lastSignal = [
            calls.firstIndex(of: "signalServer"),
            calls.firstIndex(of: "signalDownloads")
        ].compactMap { $0 }.max()
        let firstReap = [
            calls.firstIndex(of: "reapServer"),
            calls.firstIndex(of: "reapDownloads")
        ].compactMap { $0 }.min()
        #expect(lastSignal != nil)
        #expect(firstReap != nil)
        if let s = lastSignal, let r = firstReap {
            #expect(s < r, "every child must be signalled before any blocking reap begins")
        }
    }

    /// A second, narrower assertion that survives if someone adds
    /// a NEW termination step in the middle — the audit invariant
    /// is "stop first", not "exactly these steps in this order".
    /// Future-you adds a step → this pin survives; future-you
    /// moves stop after the server signal → this pin fires.
    @Test("stopStream is strictly before the server signal under any future extension")
    func stop_is_strictly_before_server_signal() {
        var calls: [String] = []
        AppDelegate.runTerminationSequence(
            stopStream: { calls.append("stop") },
            signalServer: { calls.append("signalServer") },
            signalDownloads: { _ = calls },
            reapServer: { _ = calls },
            reapDownloads: { _ = calls }
        )
        let stopIndex = calls.firstIndex(of: "stop")
        let signalIndex = calls.firstIndex(of: "signalServer")
        #expect(stopIndex != nil)
        #expect(signalIndex != nil)
        if let s = stopIndex, let f = signalIndex {
            #expect(s < f, "stopStream must run before the server SIGTERM — audit P1 invariant")
        }
    }

    /// Symmetrical pin for the OTHER end of the audit invariant:
    /// stopStream MUST run before the server is reaped so the SSE FIN
    /// reaches rapid-mlx before SIGTERM. Same reasoning as the
    /// signal pin — if a future refactor moves the server
    /// teardown ahead of the stream cancel, this fires.
    @Test("stopStream is strictly before the server reap")
    func stop_is_strictly_before_server_teardown() {
        var calls: [String] = []
        AppDelegate.runTerminationSequence(
            stopStream: { calls.append("stop") },
            signalServer: { _ = calls },
            signalDownloads: { _ = calls },
            reapServer: { calls.append("server") },
            reapDownloads: { _ = calls }
        )
        let stopIndex = calls.firstIndex(of: "stop")
        let serverIndex = calls.firstIndex(of: "server")
        #expect(stopIndex != nil)
        #expect(serverIndex != nil)
        if let s = stopIndex, let v = serverIndex {
            #expect(s < v, "stopStream must run before the server teardown — audit P1 invariant (FIN before SIGTERM)")
        }
    }

    /// Real-world spot-check: pass live ChatViewModel.stop into the
    /// helper and verify the closure invocation actually delivers
    /// the cancel. If `ChatViewModel.stop()` ever becomes async (or
    /// throws), this test fails to compile / errors at runtime —
    /// the wiring contract on the helper is "synchronous call,
    /// non-throwing".
    @Test("runTerminationSequence accepts a real ChatViewModel.stop without async/throws")
    func sequence_accepts_real_chat_viewmodel_stop() {
        let store = SessionStore(customStoreURL: TerminationOrderingTests.scratchURL())
        let vm = ChatViewModel(store: store)
        var stopFired = false
        AppDelegate.runTerminationSequence(
            stopStream: {
                vm.stop()
                stopFired = true
            },
            signalServer: {},
            signalDownloads: {},
            reapServer: {},
            reapDownloads: {}
        )
        #expect(stopFired,
                "stopStream closure must execute synchronously inside runTerminationSequence")
    }

    private static func scratchURL() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("sessions.json", isDirectory: false)
    }
}
