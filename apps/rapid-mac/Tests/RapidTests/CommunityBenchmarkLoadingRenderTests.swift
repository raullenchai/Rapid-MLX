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

@Suite("Community benchmark saved result render", .serialized)
struct CommunityBenchmarkSavedResultRenderTests {
    @MainActor
    @Test("Expanded saved result shows acceleration")
    func expandedAcceleration() async throws {
        let json = #"""
        {"completed_at":"2026-09-06T04:37:42Z","execution":{"config_digest":"sha256:abc","runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"},"task":{"language":{"speculative_decoding":{"method":"mtp","max_draft_tokens":3}}}},"measurements":[],"model":{"components":[{"source":{"repo_id":"example/model"}}]},"outcome":{"status":"completed"},"run_id":"run-text","workload":{"task_type":"text_generation"}}
        """#
        let result = try JSONDecoder().decode(CommunityBenchmarkResult.self, from: Data(json.utf8))
        let stage = GoldenStage(
            CommunityBenchmarkMyResultsView(
                results: [result], receipts: [:], aliasForRepo: { $0 },
                workloadForResult: { _ in .llm }, contributor: nil,
                publishedTotals: .loading, localOnlyCount: 1, sharingRunID: nil,
                onPublish: { _ in }, onRunFirstBenchmark: {}
            ).frame(width: 1240, height: 820),
            size: CGSize(width: 1240, height: 820)
        )
        try await stage.waitForIdentifier("CommunityBenchmark.MyResults.Expand.run-text")
        try stage.press("CommunityBenchmark.MyResults.Expand.run-text")
        try await stage.waitForIdentifier("CommunityBenchmark.Acceleration.run-text")
        try await stage.waitForText("MTP k=3")
    }

    @MainActor
    @Test("Catalog uses the host memory budget")
    func catalogArguments() {
        #expect(CommunityBenchmarkView.catalogArguments == ["benchmark", "catalog", "--json"])
        #expect(!CommunityBenchmarkView.catalogArguments.contains("--memory-gib"))
    }
}
