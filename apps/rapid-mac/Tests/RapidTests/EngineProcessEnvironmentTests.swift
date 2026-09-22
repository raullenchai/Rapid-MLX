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

    @Test("ProcessGroupChild injects the sidecar role into the actual envp")
    func processGroupChildInjectsRole() throws {
        let stdout = Pipe()
        let stderr = Pipe()
        let child = try ProcessGroupChild.spawn(
            executableURL: URL(fileURLWithPath: "/usr/bin/env"),
            arguments: [],
            standardInput: .nullDevice,
            standardOutput: stdout,
            standardError: stderr,
            environmentAdditions: [
                "PATH": "/usr/bin:/bin",
                EngineProcessEnvironment.roleKey: "interactive",
            ],
            replaceEnvironment: true,
            startMonitorImmediately: false
        )
        defer {
            if child.isProcessGroupAlive { child.signalProcessGroup(SIGKILL) }
        }

        let output = String(
            decoding: stdout.fileHandleForReading.readDataToEndOfFile(),
            as: UTF8.self
        )
        #expect(output.contains("RAPID_MLX_PROCESS_ROLE=desktop-sidecar\n"))
        #expect(!output.contains("RAPID_MLX_PROCESS_ROLE=interactive"))
        #expect(!child.isProcessGroupAlive)
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
        var discovered: [String: Int] = [:]
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let source = try String(contentsOf: url, encoding: .utf8)
            let count = source.components(separatedBy: "Process()").count - 1
                + source.components(separatedBy: "posix_spawn(").count - 1
            if count > 0 {
                discovered[String(url.path.dropFirst(rapidSources.path.count + 1))] = count
            }
        }

        // Every low-level spawn in Sources is classified. These are the only
        // non-engine sites: PortSweep's netstat/ps probes, re-onboarding's
        // /bin/sh cleanup, the `gh` star prompt, sandbox-exec, and the
        // DownloadManager test-only completion helper.
        let allowedNonEngineSites: [String: Int] = [
            "Server/PortSweep.swift": 2,
            "Services/ReonboardingReset.swift": 1,
            "UI/GitHubStarPromptCoordinator.swift": 1,
            "Tools/LocalWorkspaceTools.swift": 1,
            "Server/DownloadManager.swift": 1,
        ]
        let auditedEngineLowLevelSites: [String: Int] = [
            "Server/ServerManager.swift": 1,
            "Server/DownloadManager.swift": 1,
            "Server/ModelCatalog.swift": 1,
            "Server/IntegrationCatalog.swift": 1,
        ]
        var classified = allowedNonEngineSites
        for (path, count) in auditedEngineLowLevelSites {
            classified[path, default: 0] += count
        }
        #expect(discovered == classified)

        let allowlistAnchors: [String: String] = [
            "Server/PortSweep.swift": "/usr/sbin/netstat",
            "Services/ReonboardingReset.swift": "/bin/sh",
            "UI/GitHubStarPromptCoordinator.swift": "GitHubStarChild.spawn(",
            "Tools/LocalWorkspaceTools.swift": "/usr/bin/sandbox-exec",
            "Server/DownloadManager.swift": "internal func _testingFinish(",
        ]
        for (relativePath, anchor) in allowlistAnchors {
            let source = try String(
                contentsOf: rapidSources.appendingPathComponent(relativePath),
                encoding: .utf8
            )
            #expect(source.contains(anchor), "stale non-engine allowlist: \(relativePath)")
        }
    }
}
