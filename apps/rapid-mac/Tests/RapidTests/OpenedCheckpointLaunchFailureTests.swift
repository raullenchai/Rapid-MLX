import Foundation
import Testing
@testable import Rapid

@Suite("Checkpoint inspection launch failures", .serialized, TestTimeouts.hangProne)
struct OpenedCheckpointLaunchFailureTests {
    @Test func parallelMissingBinaryInspectionsReturnUnavailable() async {
        let missing = URL(fileURLWithPath: "/private/tmp/harbor-desk-issue-5650-missing-\(UUID())/rapid-mlx")
        let results = await withTaskGroup(of: Bool.self, returning: [Bool].self) { group in
            for _ in 0..<8 {
                group.addTask {
                    do {
                        _ = try await ModelCatalog.inspectLocalCheckpoints(paths: ["/owned-checkpoint"], binary: missing)
                        return false
                    } catch LocalCheckpointError.unavailable { return true }
                    catch { return false }
                }
            }
            var values: [Bool] = []
            for await result in group { values.append(result) }
            return values
        }
        #expect(results.count == 8 && results.allSatisfy { $0 })
    }

    @Test func failedLaunchUnregistersTheAdmittedProcess() async {
        let registry = CatalogProcessRegistry()
        let missing = URL(fileURLWithPath: "/private/tmp/harbor-desk-issue-5650-missing-\(UUID())/rapid-mlx")
        let result = await ModelCatalog._testingRunRapidMlx(binary: missing, args: ["models"],
            processRegistry: registry, beforeLaunch: {})
        #expect(result.isEmpty)
        #expect(registry.activeCount == 0)
    }

    @Test func shutdownBeforeLaunchReturnsWithoutStartingAnyProbe() async {
        let registry = CatalogProcessRegistry()
        registry.beginShutdown()
        let results = await withTaskGroup(of: String.self, returning: [String].self) { group in
            for _ in 0..<8 {
                group.addTask {
                    await ModelCatalog._testingRunRapidMlx(binary: URL(fileURLWithPath: "/bin/echo"),
                        args: ["must-not-launch"], processRegistry: registry, beforeLaunch: {})
                }
            }
            var values: [String] = []
            for await result in group { values.append(result) }
            return values
        }
        #expect(results.count == 8 && results.allSatisfy(\.isEmpty))
        #expect(registry.activeCount == 0)
    }

    @Test func alreadyCancelledInspectionReturnsCancellationAfterFailedLaunch() async {
        let missing = URL(fileURLWithPath: "/private/tmp/harbor-desk-issue-5650-missing-\(UUID())/rapid-mlx")
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                _ = try await ModelCatalog.inspectLocalCheckpoints(paths: ["/owned-checkpoint"], binary: missing)
                return false
            } catch is CancellationError { return true }
            catch { return false }
        }
        #expect(await cancelled.value)
    }
}
