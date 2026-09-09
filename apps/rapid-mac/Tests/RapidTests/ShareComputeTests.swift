import Foundation
import Testing
@testable import Rapid

@Suite("Share Compute")
struct ShareComputeTests {
    @Test("Feature is opt-in")
    func featureGate() {
        let suite = "ShareComputeTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        #expect(!ShareComputeFeatureConfig.isEnabled(in: defaults))
        defaults.set(true, forKey: ShareComputeFeatureConfig.enabledKey)
        #expect(ShareComputeFeatureConfig.isEnabled(in: defaults))
    }

    @Test("Worker labels match the provider's safe grammar")
    func workerSanitizing() {
        #expect(ShareComputeModel.sanitizedWorker("  Mini / west 🚀 ") == "Miniwest")
        #expect(ShareComputeModel.sanitizedWorker("///") == "node")
        #expect(ShareComputeModel.sanitizedWorker(String(repeating: "a", count: 80)).count == 64)
    }

    @Test("Desktop status maps without exposing a credential field")
    @MainActor
    func statusMapping() throws {
        let data = Data(#"{"schema_version":1,"session":"0123456789abcdef0123456789abcdef","phase":"online","catalog_id":"qwen3.8-27b","alias":"qwen3.8-27b-4bit","worker":"Mini","node_id":"node-1","heartbeat_interval_s":10,"inflight":2,"connected_at":1,"updated_at":2}"#.utf8)
        let snapshot = try JSONDecoder().decode(ShareComputeStatusSnapshot.self, from: data)
        #expect(ShareComputeManager.state(for: snapshot) == .online)
        #expect(snapshot.inflight == 2)
    }

    @Test("Disabling the gate routes the hidden destination to Chat")
    @MainActor
    func routeRecovery() {
        #expect(ContentView.sectionAfterShareComputeGateChange(current: .shareCompute, enabled: false) == .chat)
        #expect(ContentView.sectionAfterShareComputeGateChange(current: .shareCompute, enabled: true) == .shareCompute)
    }

    @Test("Provider key uses stdin and never argv")
    func providerKeyTransport() {
        let secret = "qsppk-never-in-argv"
        let request = ShareComputeManager.spawnRequest(
            model: ShareComputeModel.supported[0],
            worker: "Mini / west",
            session: String(repeating: "a", count: 32),
            providerKey: secret,
            environment: ["PATH": "/usr/bin"]
        )
        #expect(request.arguments.contains("--provider-key-stdin"))
        #expect(request.arguments.contains("--reregister"))
        #expect(!request.arguments.contains(where: { $0.contains(secret) }))
        #expect(!request.arguments.contains("--provider-key"))
        #expect(!request.environment.contains(where: { key, value in
            key.contains(secret) || value.contains(secret)
        }))
        #expect(request.standardInput == Data((secret + "\n").utf8))
    }

    @Test("A stale join cannot erase the next join's state")
    @MainActor
    func staleJoinOwnership() async throws {
        let executable = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-compute-runner-\(UUID().uuidString).sh")
        defer { try? FileManager.default.removeItem(at: executable) }
        try Data("#!/bin/sh\ntrap 'exit 0' TERM INT\nwhile :; do sleep 60; done\n".utf8)
            .write(to: executable)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: executable.path
        )

        let server = ServerManager(testingState: .idle, binaryPath: executable)
        let blocker = try await server.prepareForCommunityBenchmark()
        let manager = ShareComputeManager(server: server)
        let firstModel = ShareComputeModel.supported[0]
        let secondModel = ShareComputeModel.supported[1]

        let staleJoin = Task { @MainActor in
            await manager.join(model: firstModel, worker: "first", providerKey: "first-key")
        }
        for _ in 0..<10 { await Task.yield() }
        manager.leave()

        let currentJoin = Task { @MainActor in
            await manager.join(model: secondModel, worker: "second", providerKey: "second-key")
        }
        for _ in 0..<10 { await Task.yield() }
        server.finishCommunityBenchmark(blocker)

        await staleJoin.value
        await currentJoin.value
        #expect(manager.activeModel == secondModel)
        #expect(manager.state == .starting)
        manager.finishShutdown()
    }

    @Test("A replacement join inherits the original restore model")
    func replacementInheritsRestoreAlias() {
        #expect(ShareComputeManager.restoreAliasForJoin(
            pending: "previous-model",
            currentlyServing: nil
        ) == "previous-model")
        #expect(ShareComputeManager.restoreAliasForJoin(
            pending: nil,
            currentlyServing: "current-model"
        ) == "current-model")
    }

    @Test("Cache paths follow the HOME inherited by the provider child")
    func runtimeHome() {
        let fallback = URL(fileURLWithPath: "/fallback", isDirectory: true)
        #expect(ShareComputeManager.runtimeHomeURL(
            environment: ["HOME": "/private/tmp/provider-home"],
            fallback: fallback
        ).path == "/private/tmp/provider-home")
        #expect(ShareComputeManager.runtimeHomeURL(
            environment: ["HOME": "relative"],
            fallback: fallback
        ) == fallback)
    }

    @Test("Status reads are bounded")
    func boundedStatusRead() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-compute-status-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 0x61, count: 65_537).write(to: url)
        #expect(ShareComputeManager.loadStatus(at: url) == nil)
    }
}
