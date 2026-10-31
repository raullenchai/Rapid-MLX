import Foundation
import SwiftUI
import Testing
@testable import Rapid

@Suite("Community benchmark loading render")
struct CommunityBenchmarkLoadingRenderTests {
    @MainActor
    @Test("Run tab shows checking while the runtime catalog is pending")
    func checkingRuntime() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("community-benchmark-loading-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let binary = directory.appendingPathComponent("rapid-mlx")
        try "#!/bin/sh\nexec sleep 20\n".write(to: binary, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)

        let stage = GoldenStage(
            CommunityBenchmarkView(
                catalog: [],
                preferredAliases: [],
                binary: binary,
                prepareServer: { UUID() },
                releaseServer: { _ in },
                retainServerDuringDeferredReap: { _ in }
            )
            .frame(width: 1240, height: 820),
            size: CGSize(width: 1240, height: 820)
        )
        try await stage.waitForIdentifier("CommunityBenchmark.Run.CheckingRuntime")
        #expect(!stage.treeText().contains("No benchmark models available"))
    }
}
