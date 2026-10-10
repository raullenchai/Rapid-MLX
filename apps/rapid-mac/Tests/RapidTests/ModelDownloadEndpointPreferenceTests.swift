import Foundation
import Testing
@testable import Rapid

@Suite("Model download endpoint — issue #4336")
struct ModelDownloadEndpointPreferenceTests {
    @Test(arguments: ["https://hf-mirror.com/", "  https://hf-mirror.com///  "])
    func normalize(_ raw: String) {
        #expect(ModelDownloadEndpointPreference.normalized(raw) == "https://hf-mirror.com")
    }

    @Test(arguments: ["mirror.example", "file:///tmp/model", "ftp://example.com", "https://", "https://user:pass@example.com", "https://example.com?token=secret", "https://example.com#fragment", "https://exa mple.com", "https://example.com:0", "https://example.com:65536"])
    func rejectInvalid(_ raw: String) {
        #expect(ModelDownloadEndpointPreference.normalized(raw) == nil)
    }

    @Test func basePathAndLocalMirror() {
        #expect(ModelDownloadEndpointPreference.normalized("http://localhost:8080/hub/") == "http://localhost:8080/hub")
        #expect(ModelDownloadEndpointPreference.normalized(" \n ") == "")
    }

    @Test func persistenceInvalidEditAndReset() {
        let name = "rapid.tests.endpoint.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(ModelDownloadEndpointPreference.storedEndpoint(defaults: defaults) == nil)
        #expect(ModelDownloadEndpointPreference.save("https://hf-mirror.com/", defaults: defaults))
        #expect(ModelDownloadEndpointPreference.storedEndpoint(defaults: UserDefaults(suiteName: name)!) == "https://hf-mirror.com")
        #expect(!ModelDownloadEndpointPreference.save("bad", defaults: defaults))
        #expect(ModelDownloadEndpointPreference.storedEndpoint(defaults: defaults) == "https://hf-mirror.com")
        #expect(ModelDownloadEndpointPreference.save("", defaults: defaults))
        #expect(ModelDownloadEndpointPreference.storedEndpoint(defaults: defaults) == nil)
    }

    @Test func pullRoutingAndExplicitRetry() {
        var env = ["HF_ENDPOINT": "https://ambient.example", "RAPID_MLX_MODEL_MIRROR": "https://cdn.example"]
        ModelDownloadEndpointPreference.apply("https://hf-mirror.com/", env: &env)
        let source = DownloadManager.effectiveDownloadSource(.mirror, env: env)
        #expect(source == .huggingFace)
        DownloadManager.applyDownloadSource(source, env: &env)
        #expect(env["HF_ENDPOINT"] == "https://hf-mirror.com")
        #expect(env["RAPID_MLX_MODEL_MIRROR"] == "")
        DownloadManager.applyDownloadSource(.mirror, env: &env, forceMirror: true)
        #expect(env["RAPID_MLX_MODEL_MIRROR"] == "https://models.rapidmlx.com")
        #expect(env["HF_ENDPOINT"] == "https://hf-mirror.com")
    }

    @Test func serveUsesSameEndpoint() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "", ambient: ["HF_ENDPOINT": "https://ambient.example"],
            modelDownloadEndpoint: "https://hf-mirror.com/"
        )
        #expect(env["HF_ENDPOINT"] == "https://hf-mirror.com")
        #expect(env["RAPID_MLX_MODEL_MIRROR"] == "")
    }

    @Test func unsetPreservesAmbientBehavior() {
        var env = ["HF_ENDPOINT": "https://ambient.example", "RAPID_MLX_MODEL_MIRROR": "https://cdn.example"]
        let original = env
        ModelDownloadEndpointPreference.apply(nil, env: &env)
        #expect(env == original)
        ModelDownloadEndpointPreference.apply("bad", env: &env)
        #expect(env == original)
        let serve = ServerManager.serveEnvironmentAdditions(bearer: "", ambient: ["HF_ENDPOINT": "https://ambient.example"])
        #expect(serve["HF_ENDPOINT"] == "https://ambient.example")
    }
    @MainActor
    @Test(arguments: ["https://original.example", ""], ["https://new-mirror.example", ""])
    func automaticReconnectRetainsEndpoint(_ initial: String, _ edited: String) async throws {
        let name = "rapid.tests.endpoint.retry.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        ModelDownloadEndpointPreference.save(initial, defaults: defaults)
        let manager = DownloadManager(endpointPreference: {
            ModelDownloadEndpointPreference.storedEndpoint(defaults: defaults)
        })
        let original = manager._testingSeedJob(alias: "qwen3.6-27b", source: .huggingFace)
        manager._testingIngestStderr(alias: original.alias, line: "ConnectionResetError: peer reset")
        manager._testingFinish(alias: original.alias, status: 1, reason: .exit)
        #expect(original.retryDelaySeconds == 2)
        ModelDownloadEndpointPreference.save(edited, defaults: defaults)
        for _ in 0..<80 {
            if manager.job(for: original.alias)?.instanceID != original.instanceID { break }
            try await Task.sleep(for: .milliseconds(50))
        }
        let reconnected = try #require(manager.job(for: original.alias))
        #expect(reconnected.instanceID != original.instanceID)
        #expect(reconnected.downloadEndpoint == (initial.isEmpty ? nil : initial))
        // With no test binary, reconnect ends in a synthetic failure. A user
        // retry is a new attempt and should use the newly saved setting.
        _ = manager.retryDownload(alias: original.alias)
        #expect(manager.job(for: original.alias)?.downloadEndpoint == (edited.isEmpty ? nil : edited))
    }

}
