import Foundation
import Testing
@testable import RapidDesktopTestWatchdog

/// Verifies the real-host sampler (`/usr/bin/sample` + `ps`/`vm_stat`) writes
/// a non-trivial `.txt` artifact for a live process. Uses a short-lived
/// `sleep` child as the sample target — a real process the OS sampler can
/// capture — so no test hangs even if `sample` stalls.
@Suite("SampleInvocation — live capture")
struct SampleInvocationTests {

    @Test("descendant capture excludes unrelated processes and bounds output")
    func descendantProcessTreeIsScoped() {
        let table = """
        100 1 S 00:15 swift-test
        101 100 S 00:14 swiftpm-testing
        102 101 S 00:13 RapidTests
        103 102 S 00:12 helper
        200 1 S 00:15 unrelated
        201 200 S 00:14 xctest
        malformed row
        """
        let descendants = SampleInvocation.descendantProcesses(of: 100, processTable: table)
        #expect(descendants.map(\.pid) == [101, 102, 103])
        #expect(descendants.map(\.command) == ["swiftpm-testing", "RapidTests", "helper"])
        #expect(SampleInvocation.descendantProcesses(of: 100, processTable: table, limit: 2)
            .map(\.pid) == [101, 102])
        #expect(SampleInvocation.descendantProcesses(of: 999, processTable: table).isEmpty)
    }

    @Test("capture writes a non-empty artifact with sample + ps + vm_stat sections")
    func capturesLiveProcess() throws {
        // Spawn a real child so there is a genuine PID to sample.
        let sleeper = Process()
        sleeper.executableURL = URL(fileURLWithPath: "/bin/sleep")
        sleeper.arguments = ["10"]
        try sleeper.run()
        defer { kill(sleeper.processIdentifier, SIGKILL) }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("wd-test-\(UUID().uuidString)")
        let config = WatchdogConfig(
            deadline: .seconds(60),
            sampleDurationSeconds: 1,
            artifactDir: dir,
            artifactName: "desktop-hang"
        )
        let url = dir.appendingPathComponent("desktop-hang-\(sleeper.processIdentifier)-test.txt")

        try SampleInvocation.capture(pid: sleeper.processIdentifier, config: config, artifactURL: url)

        let text = try String(contentsOf: url, encoding: .utf8)
        #expect(text.contains("Rapid Desktop test-suite hang capture"))
        #expect(text.contains("===== ps -p"))
        #expect(text.contains("===== descendant process tree"))
        #expect(text.contains("===== vm_stat ====="))
        #expect(text.contains("===== /usr/bin/sample"))
        #expect(text.contains("sleep"))   // the sampled stack names the process
        #expect(text.count > 200)         // genuinely captured content, not a stub
    }
}
