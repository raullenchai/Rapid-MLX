import Testing
@testable import Rapid

@Suite("Community benchmark runtime state")
struct CommunityBenchmarkRuntimeStateTests {
    @MainActor
    @Test("Run content prioritizes results and ready models over runtime status")
    func runContent() {
        let states: [CommunityBenchmarkRuntimeState] = [.checking, .available, .unavailable]
        for runtime in states {
            #expect(CommunityBenchmarkView.runContent(hasResult: true, hasReadyModel: true, runtime: runtime) == .result)
            #expect(CommunityBenchmarkView.runContent(hasResult: true, hasReadyModel: false, runtime: runtime) == .result)
            #expect(CommunityBenchmarkView.runContent(hasResult: false, hasReadyModel: true, runtime: runtime) == .ready)
        }
        #expect(CommunityBenchmarkView.runContent(hasResult: false, hasReadyModel: false, runtime: .checking) == .checkingRuntime)
        #expect(CommunityBenchmarkView.runContent(hasResult: false, hasReadyModel: false, runtime: .available) == .unavailable)
        #expect(CommunityBenchmarkView.runContent(hasResult: false, hasReadyModel: false, runtime: .unavailable) == .unavailable)
    }
}
