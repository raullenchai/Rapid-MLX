import Foundation
import Testing
@testable import Rapid

@Suite("Opened checkpoint launch admission", .serialized, TestTimeouts.hangProne) @MainActor
struct OpenedCheckpointLaunchTests {
    private func directory() throws -> URL {
        let root = URL(fileURLWithPath: "/private/tmp/harbor-desk-issue-5650-launch/\(UUID().uuidString.prefix(8))")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @Test(arguments: ["with spaces", "模型", String(repeating: "x", count: 140), "trailing "])
    func pickedPathReachesActualServeAsOneUnchangedArgument(name: String) async throws {
        let fixture = OpenedCheckpointTests()
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, domain) = fixture.defaults(); defer { prefs.removePersistentDomain(forName: domain) }
        let binary = try fixture.inspector(root)
        let picked = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: picked, withIntermediateDirectories: true)
        let canonical = try #require(OpenedCheckpoints.canonicalPath(picked.path))
        let checked = try await ModelCatalog.inspectLocalCheckpoints(paths: [canonical], binary: binary)
        try OpenedCheckpoints.register(checked[0], defaults: prefs)
        let entries = await ModelCatalog.load(binary: binary, openedPaths: OpenedCheckpoints.paths(defaults: prefs))
        let selected = try #require(entries.first { $0.alias == canonical })
        let server = ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs)
        await server.start(alias: selected.alias, bypassMemoryGuard: true,
            catalogEntryHint: .init(entry: selected, generation: 0))
        // No throwing operation after a start: cleanup must also run on failure.
        let marker = try? Data(contentsOf: root.appendingPathComponent("served-model.json"))
        let served = marker.flatMap { try? JSONDecoder().decode(String.self, from: $0) }
        #expect(served == canonical)
        #expect(server.servingAlias == canonical)
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("pull-called").path))
        await server.stop()
    }

    @Test func freshCertificateRefusalPreventsSpawnDespiteCachedHint() async throws {
        let fixture = OpenedCheckpointTests()
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, domain) = fixture.defaults(); defer { prefs.removePersistentDomain(forName: domain) }
        let binary = try fixture.inspector(root)
        let picked = root.appendingPathComponent("checkpoint")
        try FileManager.default.createDirectory(at: picked, withIntermediateDirectories: true)
        let canonical = try #require(OpenedCheckpoints.canonicalPath(picked.path))
        try OpenedCheckpoints.register(fixture.check(canonical), defaults: prefs)
        let cached = try #require(OpenedCheckpoints.merging([], checks: [fixture.check(canonical)]).first)
        try Data("reject-all".utf8).write(to: root.appendingPathComponent("mode.txt"))
        let server = ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs)
        await server.start(alias: canonical, bypassMemoryGuard: true,
            catalogEntryHint: .init(entry: cached, generation: 0))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("served-model.json").path))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("pull-called").path))
        await server.stop()
    }

    @Test func registeredCanonicalReferenceCannotFollowAReplacementSymlink() async throws {
        let fixture = OpenedCheckpointTests()
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, domain) = fixture.defaults(); defer { prefs.removePersistentDomain(forName: domain) }
        let binary = try fixture.inspector(root)
        let first = root.appendingPathComponent("original")
        let other = root.appendingPathComponent("different")
        try FileManager.default.createDirectory(at: first, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        let canonical = try #require(OpenedCheckpoints.canonicalPath(first.path))
        try OpenedCheckpoints.register(fixture.check(canonical), defaults: prefs)
        try FileManager.default.removeItem(at: first)
        try FileManager.default.createSymbolicLink(at: first, withDestinationURL: other)
        #expect(OpenedCheckpoints.paths(defaults: prefs) == [canonical])
        let changed = try await ModelCatalog.inspectLocalCheckpoints(paths: [canonical], binary: binary)
        #expect(!changed[0].ready && changed[0].reason == "identity_changed")
        let server = ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs)
        await server.start(alias: canonical, bypassMemoryGuard: true)
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("served-model.json").path))
        await server.stop()
    }

    @Test func customManagedInventoryIsAlsoUsedByInspection() async throws {
        let fixture = OpenedCheckpointTests()
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let binary = try fixture.inspector(root)
        let custom = root.appendingPathComponent("custom-cache")
        let selected = custom.appendingPathComponent("snapshot")
        try FileManager.default.createDirectory(at: selected, withIntermediateDirectories: true)
        let customPath = try #require(OpenedCheckpoints.canonicalPath(custom.path))
        let canonical = try #require(OpenedCheckpoints.canonicalPath(selected.path))
        try Data(customPath.utf8).write(to: root.appendingPathComponent("expected-cache.txt"))
        let entries = await ModelCatalog.load(binary: binary,
            hubCacheOverride: URL(fileURLWithPath: customPath), openedPaths: [canonical])
        let observed = try String(contentsOf: root.appendingPathComponent("inspect-cache.txt"), encoding: .utf8)
        #expect(observed == customPath)
        #expect(entries.filter { $0.alias == canonical }.isEmpty)
        #expect(entries.first { $0.alias == "qwen3.5-4b-4bit" }?.isExternal == false)
    }

    @Test func caseDistinctPathsDoNotCollapseOrBorrowOtherHints() throws {
        let fixture = OpenedCheckpointTests()
        let first = try #require(OpenedCheckpoints.merging([], checks: [fixture.check("/A/model")]).first)
        let second = try #require(OpenedCheckpoints.merging([], checks: [fixture.check("/a/model")]).first)
        #expect(ModelPickerBar.uniqueSectionEntries([first, second], excluding: []).count == 2)
        #expect(ServerManager.validatedCatalogHint(alias: first.alias,
            hint: .init(entry: second, generation: 0), generation: 0) == nil)
    }
    @Test func restoreKeepsExactLocalIdentityAndLegacyAliasMatching() throws {
        let fixture = OpenedCheckpointTests()
        let lower = try #require(OpenedCheckpoints.merging([], checks: [fixture.check("/A/model ")]).first)
        let other = try #require(OpenedCheckpoints.merging([], checks: [fixture.check("/a/model ")]).first)
        #expect(SessionModelRestore.resolve(legacyLastAlias: lower.alias, dictationAlias: nil,
            speechAlias: nil, catalog: [lower, other]).chatAlias == lower.alias)
        #expect(SessionModelRestore.resolve(legacyLastAlias: "/A/model", dictationAlias: nil,
            speechAlias: nil, catalog: [lower]).chatAlias == nil)
        #expect(SessionModelRestore.resolve(legacyLastAlias: other.alias, dictationAlias: nil,
            speechAlias: nil, catalog: [lower]).chatAlias == nil)
        #expect(ModelReference.normalized("  QWEN  ") == "QWEN")
        #expect(ModelReference.matches("QWEN", "qwen"))
    }

    @Test(arguments: [false, true])
    func stoppedInspectionCannotPublishFailureOverNewLifecycle(startReplacement: Bool) async throws {
        let fixture = OpenedCheckpointTests()
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, domain) = fixture.defaults(); defer { prefs.removePersistentDomain(forName: domain) }
        let binary = try fixture.inspector(root)
        let selected = root.appendingPathComponent("checkpoint")
        try FileManager.default.createDirectory(at: selected, withIntermediateDirectories: true)
        let canonical = try #require(OpenedCheckpoints.canonicalPath(selected.path))
        try OpenedCheckpoints.register(fixture.check(canonical), defaults: prefs)
        try Data("held-reject".utf8).write(to: root.appendingPathComponent("mode.txt"))
        let server = ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs)
        server.runtimeCapabilitiesProvider = { _ in .conservative }
        let oldStart = Task { await server.start(alias: canonical, bypassMemoryGuard: true) }
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !FileManager.default.fileExists(atPath: root.appendingPathComponent("held-started").path),
              ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(10))
        }
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("held-started").path))
        await server.stop()
        // Stop revokes the old probe, even if its subprocess replies afterward.
        if startReplacement { await server.start(alias: "qwen3.5-4b-4bit", bypassMemoryGuard: true) }
        try? Data().write(to: root.appendingPathComponent("release-inspection"))
        await oldStart.value
        if startReplacement {
            #expect(server.servingAlias == "qwen3.5-4b-4bit")
            if case .ready = server.state {} else { Issue.record("Old inspection replaced the new ready state") }
        } else {
            #expect(server.state == .stopped)
        }
        await server.stop()
    }

    @Test(arguments: [false, true])
    func heldCapabilitiesCannotInspectOrOverwriteReplacementAfterStop(cancelCaller: Bool) async throws {
        let fixture = OpenedCheckpointTests()
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, domain) = fixture.defaults(); defer { prefs.removePersistentDomain(forName: domain) }
        let binary = try fixture.inspector(root)
        let selected = root.appendingPathComponent("checkpoint")
        try FileManager.default.createDirectory(at: selected, withIntermediateDirectories: true)
        let canonical = try #require(OpenedCheckpoints.canonicalPath(selected.path))
        try OpenedCheckpoints.register(fixture.check(canonical), defaults: prefs)
        let server = ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs)
        let gate = CheckpointCapabilitiesGate()
        server.runtimeCapabilitiesProvider = { _ in await gate.value() }
        let oldStart = Task { await server.start(alias: canonical, bypassMemoryGuard: true) }
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !gate.started, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(10)) }
        #expect(gate.started)
        if cancelCaller { oldStart.cancel() }
        await server.stop()
        server.runtimeCapabilitiesProvider = { _ in .conservative }
        await server.start(alias: "qwen3.5-4b-4bit", bypassMemoryGuard: true)
        gate.release()
        await oldStart.value
        #expect(server.servingAlias == "qwen3.5-4b-4bit")
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("inspect-called").path))
        await server.stop()
    }

    @Test func readinessStartPreservesPickedTrailingSpaceAndNeverDownloads() async throws {
        let fixture = OpenedCheckpointTests()
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, domain) = fixture.defaults(); defer { prefs.removePersistentDomain(forName: domain) }
        let binary = try fixture.inspector(root)
        let selected = root.appendingPathComponent("picked 模型 ")
        try FileManager.default.createDirectory(at: selected, withIntermediateDirectories: true)
        let canonical = try #require(OpenedCheckpoints.canonicalPath(selected.path))
        let checks = try await ModelCatalog.inspectLocalCheckpoints(paths: [canonical], binary: binary)
        try OpenedCheckpoints.register(checks[0], defaults: prefs)
        let server = ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs)
        server.memorySnapshotProvider = { MemoryProbe.Snapshot(totalBytes: 64 << 30, usedBytes: 0) }
        // This is the production launch adapter used by the current composer
        // readiness action; the old picker Start button is no longer mounted.
        await ReadinessModelStart.perform(server, alias: canonical, hfPath: nil)
        let served = (try? Data(contentsOf: root.appendingPathComponent("served-model.json")))
            .flatMap { try? JSONDecoder().decode(String.self, from: $0) }
        #expect(served == canonical)
        #expect(server.servingAlias == canonical)
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("pull-called").path))
        await server.stop()
    }

    @Test(arguments: [false, true])
    func freshRefusalPreventsResidentLoadAndPerformanceReload(reload: Bool) async throws {
        let fixture = OpenedCheckpointTests()
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, domain) = fixture.defaults(); defer { prefs.removePersistentDomain(forName: domain) }
        let binary = try fixture.inspector(root)
        let selected = root.appendingPathComponent("checkpoint")
        try FileManager.default.createDirectory(at: selected, withIntermediateDirectories: true)
        let canonical = try #require(OpenedCheckpoints.canonicalPath(selected.path))
        try OpenedCheckpoints.register(fixture.check(canonical), defaults: prefs)
        try Data("reject-all".utf8).write(to: root.appendingPathComponent("mode.txt"))
        let server = ServerManager(testingState: .ready(alias: reload ? canonical : "qwen3.5-4b-4bit"),
            binaryPath: binary, sessionDefaults: prefs)
        server._testInstallChild(ProcessGroupChild.testStub())
        server.memorySnapshotProvider = { MemoryProbe.Snapshot(totalBytes: 64 << 30, usedBytes: 0) }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CheckpointNoLoadProtocol.self]
        var client = ServerResidencyClient(); client.session = URLSession(configuration: configuration)
        server._testSetResidencyClient(client)
        CheckpointNoLoadProtocol.reset()
        if reload {
            let result = await server.reloadResidentPerformance(alias: canonical)
            if case .rejected = result {} else { Issue.record("Stale checkpoint reloaded") }
        } else {
            let result = await server.ensureServing(alias: canonical, hfPath: nil, estimatedMemoryGB: 0.01)
            #expect(!result)
        }
        #expect(CheckpointNoLoadProtocol.count == 0)
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("served-model.json").path))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("pull-called").path))
        await server.stop()
    }

    @Test(arguments: [false, true])
    func freshValidReferenceReachesResidentLoaderAndReload(reload: Bool) async throws {
        let fixture = OpenedCheckpointTests()
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, domain) = fixture.defaults(); defer { prefs.removePersistentDomain(forName: domain) }
        let binary = try fixture.inspector(root)
        let selected = root.appendingPathComponent("resident 模型 ")
        try FileManager.default.createDirectory(at: selected, withIntermediateDirectories: true)
        let canonical = try #require(OpenedCheckpoints.canonicalPath(selected.path))
        try OpenedCheckpoints.register(fixture.check(canonical), defaults: prefs)
        let server = ServerManager(testingState: .ready(alias: reload ? canonical : "qwen3.5-4b-4bit"),
            binaryPath: binary, sessionDefaults: prefs)
        server._testInstallChild(ProcessGroupChild.testStub())
        server.memorySnapshotProvider = { MemoryProbe.Snapshot(totalBytes: 64 << 30, usedBytes: 0) }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CheckpointNoLoadProtocol.self]
        var client = ServerResidencyClient(); client.session = URLSession(configuration: configuration)
        server._testSetResidencyClient(client)
        CheckpointNoLoadProtocol.reset(alias: canonical)
        defer { CheckpointNoLoadProtocol.reset() }
        if reload {
            let result = await server.reloadResidentPerformance(alias: canonical)
            if case .loaded = result {} else { Issue.record("Fresh checkpoint did not reload") }
        } else {
            let result = await server.ensureServing(alias: canonical, hfPath: nil, estimatedMemoryGB: 0.01)
            #expect(result)
        }
        #expect(CheckpointNoLoadProtocol.count >= 1)
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("inspect-called").path))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("served-model.json").path))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("pull-called").path))
        await server.stop()
    }

}


private final class CheckpointNoLoadProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var requests = 0
    static var count: Int { lock.withLock { requests } }
    nonisolated(unsafe) private static var responseAlias: String?
    static func reset(alias: String? = nil) { lock.withLock { requests = 0; responseAlias = alias } }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let alias = Self.lock.withLock { Self.requests += 1; return Self.responseAlias }
        var body = Data("{}".utf8)
        if let alias {
            let model: [String: Any] = ["id": alias, "model_path": alias, "aliases": [], "modality": "text",
                "state": "resident", "pinned": false, "primary": false, "active_requests": 0,
                "estimated_bytes": 42, "measured_bytes": 42, "idle_seconds": 0.0]
            let payload: [String: Any] = request.url!.path.hasSuffix("/residency")
                ? ["memory_limit_bytes": 64 << 30, "memory_used_bytes": 42, "memory_available_bytes": (64 << 30) - 42,
                   "idle_ttl_seconds": 1800, "loads_total": 1, "evictions_total": 0, "models": [model]] : model
            body = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data()
        }
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: alias == nil ? 422 : 200,
            httpVersion: nil, headerFields: nil)!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}


@MainActor private final class CheckpointCapabilitiesGate {
    private(set) var started = false
    private var released = false
    private var continuation: CheckedContinuation<ServerRuntimeCapabilities, Never>?
    func value() async -> ServerRuntimeCapabilities {
        started = true
        if released { return .conservative }
        return await withCheckedContinuation { continuation = $0 }
    }
    func release() {
        released = true
        continuation?.resume(returning: .conservative)
        continuation = nil
    }
}
