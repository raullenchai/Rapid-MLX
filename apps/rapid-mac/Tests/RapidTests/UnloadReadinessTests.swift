import Foundation
import Testing
@testable import Rapid

@Suite("Unload readiness")
@MainActor
struct UnloadReadinessTests {
    private let alias = "lfm2.5-2.6b-4bit"
    private let repo = "mlx-community/LFM2.5-2.6B-Instruct-4bit"

    @Test("Unload refreshes a stale inventory from the configured models folder")
    func customFolderAfterUnload() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let models = root.appendingPathComponent("custom-models")
        let snapshot = models.appendingPathComponent("models--mlx-community--LFM2.5-2.6B-Instruct-4bit/snapshots/test")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
        try Data([1]).write(to: snapshot.appendingPathComponent("model.safetensors"))
        let suite = "UnloadReadiness.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        ModelsFolderPreference.setStoredPath(models.path, defaults: defaults)
        let override = try #require(CachedModelDiscovery.requestedOverrideURL(defaults: defaults))
        let io = DiscoveryBlockingIO(limit: 2)
        let alias = self.alias
        let repo = self.repo
        let cache = ModelCatalogCache(
            loader: { _, validated in
                #expect(validated == override)
                let cached = try await DiscoveryDeadline.run(blockingIO: io) {
                    let hub = BundledModel.userHFCacheURL(environment: [:], preferredOverride: validated)
                    return hub?.resolvingSymlinksInPath().standardizedFileURL.path
                        == models.resolvingSymlinksInPath().standardizedFileURL.path && FileManager.default.fileExists(
                        atPath: snapshot.appendingPathComponent("model.safetensors").path
                    )
                }
                return [ModelEntry(alias: alias, hfRepo: repo, sizeOnDisk: nil, cached: cached)]
            },
            blockingIO: io, requestedOverride: { override }
        )
        let stale = [ModelEntry(alias: alias, hfRepo: repo, sizeOnDisk: nil, cached: false)]
        #expect(ModelReadiness.resolve(
            serverState: .stopped, alias: alias,
            cacheState: ContentView.readinessCacheState(for: alias, entries: stale, loaded: true)
        ) == .needsDownload(alias: alias, sizeText: nil))
        let beforeRefresh = ContentView.readinessCacheState(for: alias, entries: stale, loaded: false)
        #expect(beforeRefresh == .catalogPending)
        #expect(ModelReadiness.resolve(serverState: .stopped, alias: alias, cacheState: beforeRefresh)
            == .needsStart(alias: alias))

        let refreshed = await cache.discoveryResult(binary: root.appendingPathComponent("rapid-mlx"), generation: 1)
        #expect(refreshed.failure == nil)
        let cacheState = ContentView.readinessCacheState(for: alias, entries: refreshed.entries, loaded: true)
        #expect(cacheState == .onDisk)
        #expect(ModelReadiness.resolve(serverState: .stopped, alias: alias, cacheState: cacheState)
            == .needsStart(alias: alias))
    }

    @Test("After manual unload Start & Send requests the selected model through normal startup")
    func sendAfterUnload() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [UnloadResidencyProtocol.self]
        var client = ServerResidencyClient()
        client.session = URLSession(configuration: configuration)
        let server = ServerManager(testingState: .ready(alias: alias))
        server._testSetResidencyClient(client)
        server._testInstallChild(ProcessGroupChild.testStub())
        #expect(await server.unloadResidentModelsIfIdle() == .stopped)
        #expect(server.state == .stopped)

        let entry = ModelEntry(alias: alias, hfRepo: repo, sizeOnDisk: nil, cached: true)
        let readiness = ModelReadiness.resolve(
            serverState: server.readinessState(for: alias), alias: alias,
            cacheState: ContentView.readinessCacheState(for: alias, entries: [entry], loaded: true)
        )
        let target = try #require(readiness.startAndSendAlias)
        var intent = StartAndSendIntent()
        #expect(intent.begin(conversationID: UUID(), alias: target) != nil)
        let spy = ReadinessModelStartTests.SpyServing()
        await ReadinessModelStart.perform(spy, alias: target, hfPath: entry.hfRepo)
        #expect(spy.calls.count == 1)
        #expect(spy.calls.first?.alias == alias)
        #expect(spy.calls.first?.hfPath == repo)
    }

    @Test("The stop observer invalidates inventory and readiness rejects older generations")
    func stopRefreshWiring() throws {
        let package = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: package.appendingPathComponent("Sources/Rapid/UI/ContentView.swift"), encoding: .utf8)
        let uncommented = source.split(separator: "\n").map {
            $0.components(separatedBy: "//")[0]
        }.joined(separator: "\n")
        let compact = uncommented.components(separatedBy: .whitespacesAndNewlines).joined()
        #expect(compact.contains("ifcase.stopped=newState{downloads.markCacheChanged()}"))
        #expect(compact.contains("loaded:catalogLoaded&&catalogGeneration==downloads.cacheGeneration"))
    }
}

private final class UnloadResidencyProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: try! JSONEncoder().encode(ModelResidencySnapshot.empty))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
