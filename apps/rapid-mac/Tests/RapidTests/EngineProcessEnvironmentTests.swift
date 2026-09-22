import Foundation
import Testing
@testable import Rapid

@Suite("Engine subprocess role environment")
struct EngineProcessEnvironmentTests {
    @Test("Sidecar role injection overrides an ambient process role")
    func sidecarRoleCannotBeSpoofed() {
        let environment = EngineProcessEnvironment.sidecar([
            "KEEP": "value",
            "RAPID_MLX_PROCESS_ROLE": "interactive",
        ])
        #expect(environment["KEEP"] == "value")
        #expect(environment["RAPID_MLX_PROCESS_ROLE"] == "desktop-sidecar")
    }

    @Test("Every app-owned rapid-mlx spawn site uses the central role injector")
    func everyEngineSpawnSiteIsAudited() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let expectedSites: [String: String] = [
            "Server/ServerManager.swift": "process = try ProcessGroupChild.spawn(",
            "Server/ServerRuntimeCapabilities.swift": "child = try ProcessGroupChild.spawn(",
            "ShareCompute/ShareComputeManager.swift": "let spawned = try ProcessGroupChild.spawn(",
            "UI/CommunityBenchmarkView.swift": "let spawned = try ProcessGroupChild.spawn(",
            "Server/DownloadManager.swift": "process.executableURL = binary",
            "Server/ModelCatalog.swift": "task.executableURL = binary",
            "Server/IntegrationCatalog.swift": "process.executableURL = binary",
        ]

        for (relativePath, spawnNeedle) in expectedSites {
            let source = try String(
                contentsOf: root.appendingPathComponent("Sources/Rapid/\(relativePath)"),
                encoding: .utf8
            )
            #expect(source.contains(spawnNeedle), "missing audited spawn: \(relativePath)")
            #expect(
                source.contains("EngineProcessEnvironment.sidecar("),
                "\(relativePath) bypasses the central sidecar-role injector"
            )
        }

        let rapidSources = root.appendingPathComponent("Sources/Rapid")
        let enumerator = try #require(
            FileManager.default.enumerator(
                at: rapidSources,
                includingPropertiesForKeys: nil
            )
        )
        var discovered: Set<String> = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let source = try String(contentsOf: url, encoding: .utf8)
            let isEngineProcess = source.contains("executableURL = binary")
                || source.contains("executableURL: binary")
            guard isEngineProcess,
                  source.contains("Process()") || source.contains("ProcessGroupChild.spawn(")
            else { continue }
            discovered.insert(String(url.path.dropFirst(rapidSources.path.count + 1)))
        }
        #expect(discovered == Set(expectedSites.keys))
    }
}
