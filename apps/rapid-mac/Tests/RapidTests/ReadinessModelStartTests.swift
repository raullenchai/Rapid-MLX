import Testing

@testable import Rapid

@Suite("Readiness model start")
struct ReadinessModelStartTests {
    /// A stand-in for ``ServerManager`` that records how it was asked to bring a
    /// model up. Conforming to ``ReadinessServing`` — not subclassing the
    /// `final` ``ServerManager`` — is why that protocol exists.
    @MainActor
    final class SpyServing: ReadinessServing {
        private(set) var calls: [(alias: String, hfPath: String?)] = []

        func ensureServing(alias: String, hfPath: String?) async -> Bool {
            calls.append((alias, hfPath))
            return true
        }
    }

    @Test("a readiness action starts its target through ensureServing")
    @MainActor
    func routesThroughEnsureServing() async {
        let spy = SpyServing()

        await ReadinessModelStart.perform(
            spy, alias: "qwen3.5-9b-4bit", hfPath: "mlx-community/Qwen3.5-9B-4bit"
        )

        // ensureServing switches away from a DIFFERENT resident model (e.g. an
        // Images checkpoint). `start` is cold-start only and would silently
        // no-op there — the #1739 regression this pins.
        #expect(spy.calls.count == 1)
        #expect(spy.calls.first?.alias == "qwen3.5-9b-4bit")
        #expect(spy.calls.first?.hfPath == "mlx-community/Qwen3.5-9B-4bit")
    }

    @Test("a nil hfPath (uncached alias) is forwarded unchanged")
    @MainActor
    func forwardsNilHFPath() async {
        let spy = SpyServing()
        await ReadinessModelStart.perform(spy, alias: "local-alias", hfPath: nil)
        #expect(spy.calls.count == 1)
        #expect(spy.calls.first?.hfPath == nil)
    }
}
