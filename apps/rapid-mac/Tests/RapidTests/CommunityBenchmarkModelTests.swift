import Testing
@testable import Rapid

@MainActor
@Suite("Community Benchmark model-first projection")
struct CommunityBenchmarkModelTests {
    @Test("Atomic tasks select a protocol without modality tabs")
    func atomicTaskProjection() throws {
        let image = ModelEntry(
            alias: "flux2-klein-4b",
            hfRepo: "mlx-community/flux",
            sizeOnDisk: nil,
            cached: true,
            taskTypes: [.imageGeneration],
            operationModes: [.textToImage]
        )
        let video = ModelEntry(
            alias: "wan2.2-ti2v-5b-q8",
            hfRepo: "mlx-community/wan",
            sizeOnDisk: nil,
            cached: false,
            taskTypes: [.videoGeneration],
            operationModes: [.textToVideo]
        )
        let audio = ModelEntry(
            alias: "qwen3-asr",
            hfRepo: "mlx-community/asr",
            sizeOnDisk: nil,
            cached: true,
            taskTypes: [.speechRecognition]
        )

        let models = CommunityBenchmarkModel.models(from: [audio, video, image])
        #expect(models.map(\.entry.alias) == [image.alias, video.alias])
        #expect(models[0].protocolName == "Rapid Image Speed v1")
        #expect(models[1].protocolName == "Rapid Video Speed v1")
        #expect(models.allSatisfy { $0.isFocus })
    }

    @Test("Legacy catalog rows remain usable during the atomic migration")
    func legacyFallback() throws {
        let text = ModelEntry(
            alias: "custom-local-text",
            hfRepo: nil,
            sizeOnDisk: "2 GB",
            cached: true
        )
        let model = try #require(CommunityBenchmarkModel.models(from: [text]).first)
        #expect(model.task == .textGeneration)
        #expect(model.protocolName == "Rapid Community Speed v1")
    }

    @Test("CLI planning metadata is the shared memory-fit authority")
    func planningMetadata() throws {
        let image = ModelEntry(
            alias: "qwen-image",
            hfRepo: "mflux-community/qwen-image",
            sizeOnDisk: nil,
            cached: false,
            taskTypes: [.imageGeneration],
            operationModes: [.textToImage]
        )
        let metadata = CommunityBenchmarkCatalogModel(
            alias: image.alias,
            focus: true,
            estimatedMemoryGib: 64,
            memoryFit: "does_not_fit"
        )
        let model = try #require(
            CommunityBenchmarkModel.models(
                from: [image], metadata: [image.alias: metadata]
            ).first
        )
        #expect(model.estimatedMemoryGib == 64)
        #expect(model.memoryFit == "does_not_fit")
    }
}
