import Foundation
import Testing
@testable import Rapid

/// One Desktop per session (0.14.3 dogfood, 2026-09-18): a second instance
/// launched with `open -n` or by exec'ing the binary started its own sidecar,
/// took the port, and left the first instance reporting "Couldn't start
/// <model> — check the model files" while the model was loaded next door.
/// ``RapidApp.init`` now yields to a running instance before any side effect;
/// this pins the pure decision it is built on.
@Suite("Single-instance guard decision")
struct SingleInstanceGuardTests {
    private let me = "com.rapidmlx.rapid"
    private let t0 = Date(timeIntervalSince1970: 1_800_000_000)

    private func instance(_ pid: pid_t, _ bundle: String?, at offset: TimeInterval? = 0) -> SingleInstanceGuard.Instance {
        SingleInstanceGuard.Instance(pid: pid, bundleID: bundle, launched: offset.map { t0.addingTimeInterval($0) })
    }

    @Test("Alone → nothing to yield to")
    func aloneIsNil() {
        let own = instance(100, me)
        let running = [own, instance(200, "com.apple.finder"), instance(300, nil)]
        #expect(SingleInstanceGuard.pidToYieldTo(own: own, running: running) == nil)
    }

    @Test("A long-running instance outranks a fresh launch, whatever the PIDs")
    func olderInstanceWins() {
        // The survivor's PID is HIGHER — PIDs wrap; launch order is the truth.
        let own = instance(4100, me, at: 3600)
        let running = [instance(4200, me, at: 0), own]
        #expect(SingleInstanceGuard.pidToYieldTo(own: own, running: running) == 4200)
    }

    @Test("Concurrent cold launch: the later process yields, the earlier one stays")
    func concurrentLaunchAgreesOnOneSurvivor() {
        let a = instance(700, me, at: 0)
        let b = instance(800, me, at: 0.005)
        #expect(SingleInstanceGuard.pidToYieldTo(own: a, running: [a, b]) == nil)
        #expect(SingleInstanceGuard.pidToYieldTo(own: b, running: [a, b]) == 700)
    }

    @Test("Identical launch dates: the lower PID stays — from both points of view")
    func tieBreaksOnPID() {
        let a = instance(700, me, at: 0)
        let b = instance(800, me, at: 0)
        #expect(SingleInstanceGuard.pidToYieldTo(own: a, running: [a, b]) == nil)
        #expect(SingleInstanceGuard.pidToYieldTo(own: b, running: [a, b]) == 700)
    }

    @Test("Unknown launch dates fall back to PID order")
    func unknownDatesUsePID() {
        let a = instance(700, me, at: nil)
        let b = instance(800, me, at: nil)
        #expect(SingleInstanceGuard.pidToYieldTo(own: a, running: [b, a]) == nil)
        #expect(SingleInstanceGuard.pidToYieldTo(own: b, running: [b, a]) == 700)
    }

    @Test("Several other instances → the most senior one")
    func mostSeniorOfSeveral() {
        let own = instance(800, me, at: 30)
        let running = [instance(900, me, at: 10), instance(700, me, at: 20), own]
        #expect(SingleInstanceGuard.pidToYieldTo(own: own, running: running) == 900)
    }

    @Test("Processes without a bundle identifier are never us")
    func nilBundleIsIgnored() {
        let own = instance(3, me)
        let running = [instance(1, nil), instance(2, nil), own]
        #expect(SingleInstanceGuard.pidToYieldTo(own: own, running: running) == nil)
    }

    @Test("The guard runs before any side effect in RapidApp.init")
    func guardIsFirstInInit() throws {
        let source = try String(contentsOf: Self.sourceFile("Sources/Rapid/RapidApp.swift"), encoding: .utf8)
        let initStart = try #require(source.range(of: "\n    init() {\n"))
        let body = source[initStart.upperBound...]
        let guardAt = try #require(body.range(of: "SingleInstanceGuard.runningInstanceToYieldTo()"))
        for sideEffect in ["CrashReporter.install()", "PortSweep.startLaunchSweep", "ServerManager()"] {
            let at = try #require(body.range(of: sideEffect), "\(sideEffect) moved — update this test")
            #expect(guardAt.lowerBound < at.lowerBound, "\(sideEffect) runs before the single-instance guard")
        }
    }

    @Test("Info.plist also asks LaunchServices to refuse a second instance")
    func infoPlistProhibitsMultipleInstances() throws {
        let data = try Data(contentsOf: Self.sourceFile("Resources/Info.plist"))
        let dict = try PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any]
        #expect(dict?["LSMultipleInstancesProhibited"] as? Bool == true)
    }

    private static func sourceFile(_ relative: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relative)
    }
}
