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
                "idle_seconds": 12.5
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
