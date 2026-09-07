import Testing
@testable import Rapid

/// Runs every test in the suite with ``TelemetryConfig.environment`` pinned
/// to an empty environment. CI exports `RAPID_MLX_TELEMETRY=0` for every job
/// so a product launch on a build machine can never reach production
/// telemetry; the opt-in assertions in these suites are about consent, not
/// about the machine running them. The pin is a task-local scoped to the
/// test's own task, so parallel tests never observe each other's value and
/// nothing is left behind afterwards.
struct PinnedTelemetryEnvironmentTrait: SuiteTrait, TestTrait, TestScoping {
    func provideScope(
        for test: Test,
        testCase: Test.Case?,
        performing function: @Sendable () async throws -> Void
    ) async throws {
        try await TelemetryConfig.$environmentOverride.withValue([:]) {
            try await function()
        }
    }
}

extension Trait where Self == PinnedTelemetryEnvironmentTrait {
    static var pinnedTelemetryEnvironment: Self { PinnedTelemetryEnvironmentTrait() }
}
