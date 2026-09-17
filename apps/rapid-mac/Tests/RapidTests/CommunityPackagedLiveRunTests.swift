import Foundation
import Testing
@testable import Rapid

/// The packaged sidecar, run for real, observed live.
///
/// The shell fixtures in `CommunityLiveRunningStateTests` prove the delivery
/// path is live against a *scripted* child. They cannot prove the real
/// `rapid-mlx` binary writes what the reducer expects, when it expects it —
/// and the failure that started all of this was exactly that gap: every test
/// passed while a real packaged run sat on "Getting ready" for 34 seconds.
///
/// So this drives the actual binary inside the built `.app`, with the exact
/// production arguments, and asserts on the main actor *while the benchmark is
/// still running*. It is opt-in because it needs a built bundle and a cached
/// model, and it takes as long as a real benchmark:
///
/// ```
/// RAPID_PACKAGED_LIVE_RUN="/path/to/Rapid-MLX Desktop.app" \
///   swift test --filter CommunityPackagedLiveRunTests
/// ```
@Suite("Packaged live run", .serialized)
struct CommunityPackagedLiveRunTests {
    private static var packagedCLI: URL? {
        guard let app = ProcessInfo.processInfo.environment["RAPID_PACKAGED_LIVE_RUN"],
              !app.isEmpty
        else { return nil }
        let cli = URL(fileURLWithPath: app)
            .appendingPathComponent("Contents/Resources/rapid-mlx/bin/rapid-mlx")
        return FileManager.default.isExecutableFile(atPath: cli.path) ? cli : nil
    }

    /// Stands in for the view's `@State`, recording *when* each update landed.
    @MainActor
    private final class Screen {
        struct Landing {
            let at: Date
            let stage: CommunityRunStage
            let passes: Int
            let total: Int?
            let caption: String?
            let measurement: String?
            let status: String?
            let fraction: Double?
        }

        private(set) var landings: [Landing] = []
        var didReachResult = false

        func apply(_ state: CommunityRunProgress) {
            landings.append(
                Landing(
                    at: Date(),
                    stage: state.stage,
                    passes: state.passesComplete,
                    total: state.totalPasses,
                    caption: state.passCaption,
                    measurement: state.latestMeasurement?.value,
                    status: state.statusLine,
                    fraction: state.fraction
                )
            )
        }

        var latest: Landing? { landings.last }
    }

    @MainActor
    private static func waitUntil(
        timeout: TimeInterval,
        _ condition: () -> Bool
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return condition()
    }

    @MainActor
    @Test("The real packaged sidecar drives the screen while it is still running")
    func packagedRunIsLive() async throws {
        guard let cli = Self.packagedCLI else {
            print("SKIP: set RAPID_PACKAGED_LIVE_RUN to a built .app to run this")
            return
        }

        let screen = Screen()
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        let (stream, feed) = AsyncStream<CommunityRunProgress>
            .makeStream(bufferingPolicy: .unbounded)
        let delivery = Task { @MainActor in
            for await state in stream { screen.apply(state) }
        }

        let started = Date()
        let run = Task { @MainActor in
            let output = try await CommunityBenchmarkCommand.run(
                binary: cli,
                arguments: CommunityBenchmarkCommand.benchmarkRunArguments(
                    alias: "bonsai-1.7b-2bit"
                ),
                onStandardErrorLine: { line in
                    guard let state = reducer.apply(line: line, at: Date()) else { return }
                    feed.yield(state)
                }
            )
            feed.finish()
            await delivery.value
            screen.didReachResult = true
            return output
        }

        // --- 1. The screen must say something true long before the run ends ---
        let sawStatus = await Self.waitUntil(timeout: 120) {
            screen.latest != nil
        }
        #expect(sawStatus, "nothing at all reached the screen in 120s")
        let firstAt = try #require(screen.landings.first).at
        print(
            String(
                format: "first update reached the main actor at %.3fs",
                firstAt.timeIntervalSince(started)
            )
        )
        #expect(!screen.didReachResult, "the run ended before the first assertion")

        // --- 2. A real *measured* pass must land while the child is alive ---
        //
        // Not merely the first completed pass: pass 1 is a warmup, which
        // carries no tok/s by design, so waiting on `passes >= 1` would assert
        // that a measurement is missing at the one moment it is meant to be.
        let sawMeasuredPass = await Self.waitUntil(timeout: 240) {
            (screen.latest?.passes ?? 0) >= 1 && screen.latest?.measurement != nil
        }
        #expect(sawMeasuredPass, "no measured pass reached the screen while running")
        #expect(
            !screen.didReachResult,
            "passes only appeared after the process exited — this is the original bug"
        )

        let live = try #require(screen.latest)
        print(
            String(
                format: "live at %.3fs: stage=%@ passes=%d/%@ fraction=%@ measurement=%@",
                live.at.timeIntervalSince(started),
                String(describing: live.stage),
                live.passes,
                live.total.map(String.init) ?? "nil",
                live.fraction.map { String(format: "%.3f", $0) } ?? "nil",
                live.measurement ?? "nil"
            )
        )

        // The bar is determinate, so the mascot has a real position to ride.
        #expect(live.stage > .gettingReady)
        #expect(live.total != nil, "the run's own plan event never set a denominator")
        #expect(live.fraction != nil)
        #expect(live.measurement != nil, "no measured value reached the screen")

        // --- 3. Let it finish, then check the whole sequence ---
        let output = try await run.value
        #expect(screen.didReachResult)
        #expect(CommunityBenchmarkCommand.runID(from: output) != nil)

        print("--- observed live sequence (main actor) ---")
        var lastStage: CommunityRunStage?
        for landing in screen.landings where landing.stage != lastStage {
            print(
                String(
                    format: "  %6.2fs  %-13@ %-26@ mascot@%@  %@",
                    landing.at.timeIntervalSince(started),
                    String(describing: landing.stage),
                    landing.caption ?? landing.status ?? "",
                    landing.fraction.map { String(format: "%.2f", $0) } ?? "indeterminate",
                    landing.measurement ?? ""
                )
            )
            lastStage = landing.stage
        }

        // Every stage the run actually passed through was rendered, in order.
        let stages = screen.landings.map(\.stage)
        #expect(stages == stages.sorted(), "stages went backwards on screen")
        #expect(stages.contains(.gettingReady))
        #expect(stages.last == .saving)
        let counts = screen.landings.map(\.passes)
        #expect(counts == counts.sorted(), "the pass count went backwards")
        #expect(counts.last ?? 0 > 0)
    }
}
