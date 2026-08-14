import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Multi-model residency")
struct ModelResidencyTests {
    @Test("Residency status decodes the server wire format")
    func decodesSnapshot() throws {
        let data = Data(
            #"""
            {
              "memory_limit_bytes": 34359738368,
              "memory_used_bytes": 10737418240,
              "memory_available_bytes": 23622320128,
              "idle_ttl_seconds": 1800,
              "loads_total": 2,
              "evictions_total": 1,
              "models": [{
                "id": "flux2-klein-4b",
                "model_path": "Runware/FLUX.2-klein-4B",
                "aliases": ["flux-klein"],
                "modality": "image-gen",
                "state": "resident",
                "pinned": false,
                "primary": false,
                "active_requests": 0,
                "estimated_bytes": 6335076761,
                "measured_bytes": 5905580032,
                "idle_seconds": 12.5,
                "performance": {
                  "kv_cache_turboquant": "k8v4",
                  "prefix_cache_enabled": true,
                  "cache_memory_mb": 4096
                }
              }]
            }
            """#.utf8
        )

        let snapshot = try JSONDecoder().decode(ModelResidencySnapshot.self, from: data)

        #expect(snapshot.memoryLimitBytes == 34_359_738_368)
        #expect(snapshot.memoryUsedBytes == 10_737_418_240)
        #expect(snapshot.loadsTotal == 2)
        #expect(snapshot.evictionsTotal == 1)
        #expect(snapshot.models.first?.modality == "image-gen")
        #expect(snapshot.models.first?.displayBytes == 6_335_076_761)
        #expect(snapshot.contains("flux2-klein-4b"))
        #expect(snapshot.contains("flux-klein"))
        #expect(snapshot.contains("Runware/FLUX.2-klein-4B"))
        #expect(snapshot.models.first?.performance == ResidentPerformanceStatus(
            config: ModelPerfConfig(
                kvCacheMode: .turboquantK8V4,
                prefixCacheEnabled: true,
                cacheMemoryMB: 4096
            )
        ))
    }

    @Test("Resident rows prefer the catalog alias over the HF path")
    func residentDisplayName() {
        let status = ResidentModelStatus(
            id: "mlx-community/Qwen3.5-4B-MLX-4bit",
            modelPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
            aliases: ["qwen3.5-4b-4bit"],
            modality: "text",
            state: "resident",
            pinned: true,
            primary: true,
            activeRequests: 0,
            estimatedBytes: 1,
            measuredBytes: nil,
            idleSeconds: 0
        )

        #expect(status.displayName() == "qwen3.5-4b-4bit")
        #expect(status.displayName(preferredAlias: "qwen3.5-4b-4bit") == "qwen3.5-4b-4bit")
    }

    @Test("Connector restart prefers a resident text model over the process-owning audio alias")
    func connectorRestartTextAlias() {
        let text = ResidentModelStatus(
            id: "qwen3.5-4b-4bit",
            modelPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
            aliases: [],
            modality: "text",
            state: "resident",
            pinned: false,
            primary: false,
            activeRequests: 0,
            estimatedBytes: 1,
            measuredBytes: nil,
            idleSeconds: 0
        )
        let snapshot = ModelResidencySnapshot(
            memoryLimitBytes: 1,
            memoryUsedBytes: 1,
            memoryAvailableBytes: 0,
            idleTTLSeconds: 1,
            loadsTotal: 1,
            evictionsTotal: 0,
            models: [text]
        )

        #expect(snapshot.preferredTextAlias(fallback: "qwen3-tts-4bit") == "qwen3.5-4b-4bit")
        #expect(ModelResidencySnapshot.empty.preferredTextAlias(fallback: "legacy-chat") == "legacy-chat")
    }

    @Test("Server readiness is resolved for every resident alias")
    func aliasSpecificReadiness() {
        let image = ResidentModelStatus(
            id: "flux2-klein-4b",
            modelPath: "Runware/FLUX.2-klein-4B",
            aliases: ["flux-klein"],
            modality: "image-gen",
            state: "resident",
            pinned: false,
            primary: false,
            activeRequests: 0,
            estimatedBytes: 6_335_076_761,
            measuredBytes: nil,
            idleSeconds: 0
        )
        let snapshot = ModelResidencySnapshot(
            memoryLimitBytes: 25 * 1_073_741_824,
            memoryUsedBytes: 10 * 1_073_741_824,
            memoryAvailableBytes: 15 * 1_073_741_824,
            idleTTLSeconds: 1800,
            loadsTotal: 1,
            evictionsTotal: 0,
            models: [image]
        )
        let server = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            residency: snapshot
        )

        #expect(server.isModelResident("qwen3.5-4b-4bit"))
        #expect(server.isModelResident("flux2-klein-4b"))
        #expect(server.isModelResident("flux-klein"))
        #expect(!server.isModelResident("z-image-turbo"))

        guard case .ready(let alias) = server.readinessState(for: "flux-klein") else {
            Issue.record("Expected resident image alias to be ready")
            return
        }
        #expect(alias == "flux-klein")
    }

    @Test("Resident ceiling reuses the Mac usable-RAM bucket")
    func residentMemoryCeiling() {
        #expect(ModelSizing.residentMemoryCeilingGB(on: mockMac(ramGB: 32)) == 25)
        #expect(ModelSizing.residentMemoryCeilingGB(on: mockMac(ramGB: 8)) == 6)
        #expect(ModelSizing.residentMemoryCeilingGB(on: mockMac(ramGB: 4)) == 4)
    }

    @Test("Residency load sends typed performance config and reload intent")
    func loadRequestCarriesPerformance() async throws {
        ResidencyLoadCaptureProtocol.capturedBody = nil
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ResidencyLoadCaptureProtocol.self]
        var client = ServerResidencyClient()
        client.session = URLSession(configuration: configuration)

        let result = await client.load(
            alias: "qwen3.5-4b-4bit",
            hfPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
            estimatedSizeGB: 4,
            imageMode: .editing,
            performance: ModelPerfConfig(
                kvCacheMode: .turboquantK8V4,
                prefixCacheEnabled: false,
                cacheMemoryMB: 4096
            ),
            reloadIfChanged: true,
            port: 8000,
            bearer: "secret"
        )
        guard case .loaded = result else {
            Issue.record("Expected the stubbed residency load to succeed")
            return
        }
        let body = try #require(ResidencyLoadCaptureProtocol.capturedBody)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let performance = try #require(json["performance"] as? [String: Any])
        #expect(json["reload_if_changed"] as? Bool == true)
        #expect(json["image_mode"] as? String == "editing")
        #expect(performance["kv_cache_dtype"] == nil)
        #expect(performance["kv_cache_turboquant"] as? String == "k8v4")
        #expect(performance["prefix_cache_enabled"] as? Bool == false)
        #expect(performance["cache_memory_mb"] as? Int == 4096)
    }

    @Test("Image residency estimate uses catalog bytes plus runtime margin")
    func imageEstimateUsesDownloadSize() {
        let estimate = ModelSizing.residentEstimateGB(
            alias: "z-image-turbo",
            sizeText: "5.5 GiB"
        )
        #expect(abs(estimate - 7.375) < 0.001)
        #expect(estimate > ModelSizing.residentEstimateGB(alias: "z-image-turbo"))
    }

    @Test("Selecting a chat model does not retain the legacy server restart flow")
    func selectionDoesNotRestartServer() throws {
        let rapidMacRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = rapidMacRoot
            .appendingPathComponent("Sources/Rapid/UI/ContentView.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        #expect(!source.contains("pendingReloadAlias"))
        #expect(!source.contains("Switch and reload"))
        #expect(!source.contains("Stops the current model and loads"))
    }

    @Test("Selecting a cached chat model activates it immediately")
    func cachedChatSelectionActivates() {
        #expect(ContentView.activatesChatModelOnSelection(
            isResident: false,
            isCached: true
        ))
        #expect(ContentView.activatesChatModelOnSelection(
            isResident: true,
            isCached: false
        ))
        #expect(!ContentView.activatesChatModelOnSelection(
            isResident: false,
            isCached: false
        ))
    }

    private func mockMac(ramGB: Int) -> MacHardware {
        MacHardware(
            brandString: "Apple M3 Pro",
            family: .m3,
            tier: .pro,
            physicalRAMBytes: UInt64(ramGB) * UInt64(1 << 30),
            memoryBandwidthGBs: 150
        )
    }
}

private final class ResidencyLoadCaptureProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var capturedBody: Data?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.capturedBody = request.httpBody ?? Self.readBodyStream(request.httpBodyStream)
        let payload = #"{"id":"qwen3.5-4b-4bit","model_path":"mlx-community/Qwen3.5-4B-MLX-4bit","aliases":[],"modality":"text","state":"resident","pinned":true,"primary":true,"active_requests":0,"estimated_bytes":1,"measured_bytes":null,"idle_seconds":0,"performance":{"kv_cache_turboquant":"k8v4","prefix_cache_enabled":false,"cache_memory_mb":4096}}"#.data(using: .utf8)!
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func readBodyStream(_ stream: InputStream?) -> Data? {
        guard let stream else { return nil }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let count = buffer.withUnsafeMutableBufferPointer { pointer in
                stream.read(pointer.baseAddress!, maxLength: pointer.count)
            }
            if count > 0 { data.append(buffer, count: count) }
            if count == 0 { return data }
            if count < 0 { return nil }
        }
    }
}
