import Foundation
import Testing
@testable import Rapid

@Suite("Atomic product model catalog")
struct AtomicModelCatalogTests {
    private static let unsignedPayload = #"""
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

    private static let payload = makePayload()

    private static func makePayload() -> String {
        let data = Data(unsignedPayload.utf8)
        guard var root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              var atomic = root["atomic"] as? [String: Any],
              var snapshot = atomic["snapshot"] as? [String: Any],
              var models = snapshot["models"] as? [[String: Any]],
              var aliases = snapshot["aliases"] as? [[String: Any]]
        else { fatalError("invalid unsigned fixture") }
        for index in models.indices {
            models[index]["schema_version"] = 1
            models[index]["resolution_status"] = "unresolved"
        }
        for index in aliases.indices {
            aliases[index]["schema_version"] = 1
            aliases[index]["origin"] = "builtin"
            var target = aliases[index]["target"] as! [String: Any]
            target["resolution_status"] = "unresolved"
            aliases[index]["target"] = target
            var availability = aliases[index]["availability"] as! [String: Any]
            availability["cli"] = true
            availability["server"] = true
            availability["website"] = true
            aliases[index]["availability"] = availability
            aliases[index]["default_execution_preset_id"] = NSNull()
            aliases[index]["execution_presets"] = []
        }
        snapshot["models"] = models
        snapshot["aliases"] = aliases
        atomic["snapshot"] = snapshot
        root["atomic"] = atomic
        let output = try! JSONSerialization.data(withJSONObject: root)
        return signed(String(decoding: output, as: UTF8.self))
    }

    private static func signed(_ input: String) -> String {
        guard let data = input.data(using: .utf8),
              var root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              var atomic = root["atomic"] as? [String: Any],
              var snapshot = atomic["snapshot"] as? [String: Any]
        else { fatalError("invalid atomic test fixture") }
        snapshot["recommendation_policy_digests"] = []
        guard let digest = ModelCatalog.atomicCatalogDigest(snapshot) else {
            fatalError("test fixture cannot be canonicalized")
        }
        snapshot["catalog_digest"] = digest
        atomic["snapshot"] = snapshot
        atomic["shadow_report"] = ["equivalent": true, "catalog_digest": digest]
        root["atomic"] = atomic
        guard let output = try? JSONSerialization.data(
            withJSONObject: root, options: [.sortedKeys, .withoutEscapingSlashes]
        ) else { fatalError("test fixture cannot be serialized") }
        return String(decoding: output, as: UTF8.self)
    }

    private static func mutated(
        _ update: (inout [String: Any]) -> Void,
        resign: Bool = true
    ) -> String {
        let data = Data(payload.utf8)
        guard var root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { fatalError("invalid signed fixture") }
        update(&root)
        guard let output = try? JSONSerialization.data(withJSONObject: root) else {
            fatalError("mutated fixture cannot be serialized")
        }
        let value = String(decoding: output, as: UTF8.self)
        return resign ? signed(value) : value
    }

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
        let future = Self.signed(Self.payload.replacingOccurrences(
            of: "\"text_generation\"", with: "\"future_generation\""
        ))
        #expect(ModelCatalog.parseAtomicModelEntriesJSON(future) == nil)
    }

    @Test("Swift RCJ digest agrees with the Python golden vector")
    func digestGoldenVector() {
        let snapshot: [String: Any] = [
            "schema_version": 1,
            "models": [],
            "aliases": [],
            "recommendation_policy_digests": [],
        ]
        #expect(ModelCatalog.atomicCatalogDigest(snapshot) ==
            "sha256:1710f73da67f8d5ca58ce8343929534485262218d4c1d9ebc95ad9e9e3ee599f")
    }

    @Test("broken references, digests, and shadow reports fail closed")
    func corruptSnapshotsRejectAtomicEnvelope() {
        let missingTarget = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var aliases = snapshot["aliases"] as! [[String: Any]]
            var alias = aliases[0]
            var target = alias["target"] as! [String: Any]
            target["registry_model_id"] = "legacy/hf/missing"
            alias["target"] = target
            aliases[0] = alias
            snapshot["aliases"] = aliases
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        #expect(ModelCatalog.parseAtomicModelEntriesJSON(missingTarget) == nil)

        let missingAvailability = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var aliases = snapshot["aliases"] as! [[String: Any]]
            aliases[0].removeValue(forKey: "availability")
            snapshot["aliases"] = aliases
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        #expect(ModelCatalog.parseAtomicModelEntriesJSON(missingAvailability) == nil)

        let duplicatePreset = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var aliases = snapshot["aliases"] as! [[String: Any]]
            aliases[0]["default_execution_preset_id"] = "balanced"
            aliases[0]["execution_presets"] = [
                ["preset_id": "balanced"], ["preset_id": "balanced"],
            ]
            snapshot["aliases"] = aliases
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        #expect(ModelCatalog.parseAtomicModelEntriesJSON(duplicatePreset) == nil)

        let digestMismatch = Self.mutated({ root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var models = snapshot["models"] as! [[String: Any]]
            var source = models[0]["source"] as! [String: Any]
            source["repo_id"] = "org/tampered"
            models[0]["source"] = source
            snapshot["models"] = models
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }, resign: false)
        #expect(ModelCatalog.parseAtomicModelEntriesJSON(digestMismatch) == nil)

        let failedShadow = Self.mutated({ root in
            var atomic = root["atomic"] as! [String: Any]
            var shadow = atomic["shadow_report"] as! [String: Any]
            shadow["equivalent"] = false
            atomic["shadow_report"] = shadow
            root["atomic"] = atomic
        }, resign: false)
        #expect(ModelCatalog.parseAtomicModelEntriesJSON(failedShadow) == nil)
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

    @Test("alias origin and repository siblings preserve legacy safety semantics")
    func originAndSiblingCacheArePreserved() throws {
        let userPayload = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var aliases = snapshot["aliases"] as! [[String: Any]]
            aliases[0]["origin"] = "user"
            var sibling = aliases[0]
            sibling["alias"] = "chat-sibling"
            sibling["origin"] = "builtin"
            aliases.append(sibling)
            snapshot["aliases"] = aliases
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        let userEntries = try #require(
            ModelCatalog.parseAtomicModelEntriesJSON(userPayload)
        )
        #expect(userEntries.first { $0.alias == "chat" }?.isBuiltinProfile == false)
        #expect(ModelCatalog.parseAvailableJSON(userPayload)?.profiles["chat"]?.isBuiltin == false)

        let chat = try #require(userEntries.first { $0.alias == "chat" })
        let sibling = try #require(userEntries.first { $0.alias == "chat-sibling" })
        let merged = ModelCatalog.mergeAtomicAndCached(
            atomic: [chat, sibling],
            cached: [("chat", "org/chat", "1 GiB")],
            excluded: []
        )
        #expect(merged.first { $0.alias == "chat-sibling" }?.cached == true)
        #expect(merged.first { $0.alias == "chat-sibling" }?.sizeOnDisk == "1 GiB")
    }
}
