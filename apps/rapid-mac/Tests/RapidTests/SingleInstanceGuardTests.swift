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

    @Test("Mixed known/unknown launch dates still rank the same way from every viewpoint")
    func mixedDatesAreATotalOrder() {
        // codex r2: date-vs-PID switching per pair could cycle (A<B, B<C, C<A) and let all three stay.
        let a = instance(100, me, at: 10)
        let b = instance(200, me, at: nil)
        let c = instance(300, me, at: 0)
        for roster in [[a, b, c], [c, b, a], [b, a, c]] {
            #expect(SingleInstanceGuard.pidToYieldTo(own: c, running: roster) == nil, "c launched first and stays")
            #expect(SingleInstanceGuard.pidToYieldTo(own: a, running: roster) == 300)
            #expect(SingleInstanceGuard.pidToYieldTo(own: b, running: roster) == 300, "an undated process yields to any dated one")
        }
    }

    @Test("An instance that finished launching wins even with no launch date")
    func establishedInstanceWins() {
        // codex r3: with the lock unavailable, the date rule alone would rank
        // a Desktop whose launch date LaunchServices cannot report as the
        // newest process, and the fresh launch would carry on beside it.
        var established = instance(200, me, at: nil)
        established.finishedLaunching = true
        #expect(SingleInstanceGuard.pidToYieldTo(own: instance(900, me, at: 0), running: [established]) == 200)
        // Still true when the newcomer would win on dates alone.
        var late = instance(950, me, at: 30)
        late.finishedLaunching = true
        #expect(SingleInstanceGuard.pidToYieldTo(own: instance(900, me, at: 0), running: [late]) == 950)
        // Between two established instances the senior one is the target.
        var early = instance(100, me, at: -60)
        early.finishedLaunching = true
        #expect(SingleInstanceGuard.pidToYieldTo(own: instance(900, me, at: 0), running: [late, early]) == 100)
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

    @Test("Instance lock: second acquisition is busy while held, free after release")
    func instanceLockExcludes() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-instance-lock-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("desktop-instance.lock")
        let first = SingleInstanceGuard.InstanceLock(url: url)
        let second = SingleInstanceGuard.InstanceLock(url: url)
        #expect(first.acquire() == .acquired)
        #expect(first.acquire() == .acquired, "re-acquire by the holder is idempotent")
        #expect(second.acquire() == .busy, "flock is per open file description, so the second opener is excluded")
        first.release()
        #expect(second.acquire() == .acquired)
        second.release()
    }

    @Test("Instance lock: an unwritable location is unavailable, not busy")
    func instanceLockUnavailable() {
        let lock = SingleInstanceGuard.InstanceLock(url: URL(fileURLWithPath: "/dev/null/impossible/desktop-instance.lock"))
        #expect(lock.acquire() == .unavailable)
    }

    @Test("The guard runs before any side effect in RapidApp.init")
    func guardIsFirstInInit() throws {
        let source = try String(contentsOf: Self.sourceFile("Sources/Rapid/RapidApp.swift"), encoding: .utf8)
        let initStart = try #require(source.range(of: "\n    init() {\n"))
        let body = source[initStart.upperBound...]
        let guardAt = try #require(body.range(of: "SingleInstanceGuard.yieldsLaunch()"))
        for sideEffect in ["CrashReporter.install()", "PortSweep.startLaunchSweep", "ServerManager()"] {
            let at = try #require(body.range(of: sideEffect), "\(sideEffect) moved — update this test")
            #expect(guardAt.lowerBound < at.lowerBound, "\(sideEffect) runs before the single-instance guard")
        }
    }

    @Test("Settings stays out of window restoration and heals a Settings-only launch")
    func settingsSceneDoesNotHijackLaunch() throws {
        // A quit with only Settings open persisted "settings" as the whole
        // session; the next launch restored just that window, and the
        // engine — started by ContentView — never came up (0.14.3 dogfood).
        let source = try String(contentsOf: Self.sourceFile("Sources/Rapid/RapidApp.swift"), encoding: .utf8)
        let settingsScene = try #require(source.range(of: "Window(\"Settings\", id: \"settings\")"))
        let scene = source[settingsScene.upperBound...]
        let sceneEnd = try #require(scene.range(of: ".defaultSize("))
        let body = scene[..<sceneEnd.lowerBound]
        #expect(body.contains("window.isRestorable = false"))
        #expect(body.contains("if !AppDelegate.shared.hasAttachedMainWindow {"))
        #expect(body.contains("openWindow(id: \"main\")"))
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
