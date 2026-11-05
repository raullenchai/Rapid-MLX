import Testing

@Suite("CIHangWatchdog heartbeat decisions")
struct CIHangWatchdogTests {
    @Test("healthy heartbeats need no explicit suite checkpoints")
    func healthyHeartbeats() {
        #expect(CIHangWatchdog.stalledLane(
            now: 1000, lastMain: 996, lastCooperative: 995, limit: 90
        ) == nil)
        #expect(CIHangWatchdog.stalledLane(
            now: 1000, lastMain: 911, lastCooperative: 911, limit: 90
        ) == nil)
    }

    @Test("either lane stalls at the quiet limit even when the other is healthy")
    func stalledHeartbeat() {
        #expect(CIHangWatchdog.stalledLane(
            now: 1000, lastMain: 910, lastCooperative: 1000, limit: 90
        ) == "main")
        #expect(CIHangWatchdog.stalledLane(
            now: 1000, lastMain: 1000, lastCooperative: 910, limit: 90
        ) == "cooperative")
        #expect(CIHangWatchdog.stalledLane(
            now: 1000, lastMain: 800, lastCooperative: 800, limit: 180
        ) == "main")
        #expect(CIHangWatchdog.stalledLane(
            now: 1000, lastMain: 910, lastCooperative: 910, limit: 180
        ) == nil)
    }
}
