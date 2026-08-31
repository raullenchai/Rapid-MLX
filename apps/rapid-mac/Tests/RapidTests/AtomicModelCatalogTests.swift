import Foundation
import Testing
@testable import Rapid

@Suite("Atomic product model catalog")
struct AtomicModelCatalogTests {
    private static let payload = #"""
    {
      "text": [{"alias":"chat","supports_spec_decode":true}],
      "atomic": {
        "snapshot": {
          "schema_version": 1,
          "models": [
            {"registry_model_id":"legacy/hf/chat","source":{"provider":"huggingface","repo_id":"org/chat"},"estimated_download_size_bytes":1073741824},
            {"registry_model_id":"legacy/hf/image","source":{"provider":"huggingface","repo_id":"org/image"}},
            {"registry_model_id":"legacy/hf/video","source":{"provider":"huggingface","repo_id":"org/video"}},
            {"registry_model_id":"legacy/hf/tts","source":{"provider":"huggingface","repo_id":"org/tts"}},
            {"registry_model_id":"legacy/hf/stt","source":{"provider":"huggingface","repo_id":"org/stt"}},
            {"registry_model_id":"legacy/hf/hidden","source":{"provider":"huggingface","repo_id":"org/hidden"}}
          ],
          "aliases": [
            {"alias":"chat","target":{"registry_model_id":"legacy/hf/chat"},"capabilities":{"task_types":["text_generation"],"operation_modes":["chat"],"runtime_adapter":"mlx_lm"},"availability":{"desktop":true}},
            {"alias":"image","target":{"registry_model_id":"legacy/hf/image"},"capabilities":{"task_types":["image_generation"],"operation_modes":["text_to_image","image_to_image"],"runtime_adapter":"mflux"},"availability":{"desktop":true}},
            {"alias":"video","target":{"registry_model_id":"legacy/hf/video"},"capabilities":{"task_types":["video_generation"],"operation_modes":["text_to_video"],"runtime_adapter":"rapid_mlx/video"},"availability":{"desktop":true}},
            {"alias":"tts","target":{"registry_model_id":"legacy/hf/tts"},"capabilities":{"task_types":["speech_synthesis"],"operation_modes":["preset_voice"],"runtime_adapter":"mlx_audio/qwen3_tts"},"availability":{"desktop":true}},
            {"alias":"stt","target":{"registry_model_id":"legacy/hf/stt"},"capabilities":{"task_types":["speech_recognition"],"operation_modes":["transcription"],"runtime_adapter":"mlx_audio/whisper"},"availability":{"desktop":true}},
            {"alias":"hidden","target":{"registry_model_id":"legacy/hf/hidden"},"capabilities":{"task_types":["speech_recognition"],"operation_modes":["transcription"],"runtime_adapter":"mlx_audio/whisper"},"availability":{"desktop":false}}
          ]
        }
      }
    }
    """#

    @Test("one graph maps every modality into the correct product kind")
    func mapsTasksToProductKinds() throws {
        let entries = try #require(ModelCatalog.parseAtomicModelEntriesJSON(Self.payload))
        #expect(entries.map(\.alias) == ["chat", "image", "video", "tts", "stt"])
        #expect(entries.first { $0.alias == "chat" }?.kind == .chat)
        #expect(entries.first { $0.alias == "image" }?.kind == .image)
        #expect(entries.first { $0.alias == "video" }?.kind == .video)
        #expect(entries.first { $0.alias == "tts" }?.kind == .audio)
        #expect(entries.first { $0.alias == "stt" }?.kind == .audio)
        #expect(entries.first { $0.alias == "hidden" } == nil)
    }

    @Test("atomic operations drive picker eligibility without alias heuristics")
    func operationsDriveSelection() throws {
        let entries = try #require(ModelCatalog.parseAtomicModelEntriesJSON(Self.payload))
        let image = try #require(entries.first { $0.alias == "image" })
        let tts = try #require(entries.first { $0.alias == "tts" })
        let stt = try #require(entries.first { $0.alias == "stt" })
        #expect(ModelSelectionPurpose.imageGeneration.accepts(image))
        #expect(ModelSelectionPurpose.imageEditing.accepts(image))
        #expect(ModelSelectionPurpose.textToSpeech.accepts(tts))
        #expect(ModelSelectionPurpose.speechToText.accepts(stt))
        #expect(!ModelSelectionPurpose.chat.accepts(stt))
    }

    @Test("chat projection keeps legacy speculative behavior during shadow mode")
    func chatProjectionUsesAtomicPlacement() throws {
        let parsed = try #require(ModelCatalog.parseAvailableJSON(Self.payload))
        #expect(parsed.entries.map(\.0) == ["chat"])
        #expect(parsed.excluded == ["image", "video", "tts", "stt", "hidden"])
        #expect(parsed.speculative["chat"]?.method == .suffix)
        #expect(parsed.profiles["chat"]?.isTextOnly == true)
    }

    @Test("unknown atomic tasks fail closed into the legacy downgrade path")
    func unknownTaskRejectsAtomicEnvelope() {
        let future = Self.payload.replacingOccurrences(
            of: "\"text_generation\"", with: "\"future_generation\""
        )
        #expect(ModelCatalog.parseAtomicModelEntriesJSON(future) == nil)
    }

    @Test("atomic Settings merge preserves custom and external cached models")
    func cacheMergePreservesUserModels() throws {
        let atomic = try #require(ModelCatalog.parseAtomicModelEntriesJSON(Self.payload))
        let cached: [(String, String?, String?)] = [
            ("chat", "org/chat", "1 GiB"),
            ("custom", "org/custom", "2 GiB"),
            ("hidden", "org/hidden", "3 GiB"),
            ("(external)", "org/external", "4 GiB"),
        ]
        let merged = ModelCatalog.mergeAtomicAndCached(
            atomic: atomic,
            cached: cached,
            excluded: ["image", "video", "tts", "stt", "hidden"]
        )
        #expect(merged.first { $0.alias == "chat" }?.cached == true)
        #expect(merged.first { $0.alias == "custom" }?.kind == .chat)
        #expect(merged.first { $0.alias == "hidden" } == nil)
        #expect(merged.first { $0.alias == "org/external" }?.isExternal == true)
    }
}
