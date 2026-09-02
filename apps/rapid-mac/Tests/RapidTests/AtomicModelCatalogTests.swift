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
          "schema_version": 2,
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
            aliases[index]["schema_version"] = 2
            aliases[index]["origin"] = "builtin"
            var capabilities = aliases[index]["capabilities"] as! [String: Any]
            capabilities["is_text_only"] = index == 0
            aliases[index]["capabilities"] = capabilities
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
        if snapshot["recommendation_policy_digests"] == nil {
            snapshot["recommendation_policy_digests"] = []
        }
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

    private static func mutateCapabilities(
        in root: inout [String: Any],
        aliasIndex: Int,
        _ update: (inout [String: Any]) -> Void
    ) {
        var atomic = root["atomic"] as! [String: Any]
        var snapshot = atomic["snapshot"] as! [String: Any]
        var aliases = snapshot["aliases"] as! [[String: Any]]
        var capabilities = aliases[aliasIndex]["capabilities"] as! [String: Any]
        update(&capabilities)
        aliases[aliasIndex]["capabilities"] = capabilities
        snapshot["aliases"] = aliases
        atomic["snapshot"] = snapshot
        root["atomic"] = atomic
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
        #expect(entries.allSatisfy { $0.recommendationPolicyDigests.isEmpty })
        #expect(entries.allSatisfy { !$0.allowsLegacyRecommendationPolicy })
    }

    @Test("legacy built-in rows preserve recommendation compatibility without weakening atomic catalogs")
    func legacyRecommendationCompatibilityIsFormatScoped() throws {
        let legacy = #"{"text":[{"alias":"chat","hf_path":"org/chat","is_builtin":true,"is_text_only":true},{"alias":"custom","hf_path":"org/custom","is_builtin":false,"is_text_only":true}]}"#
        let projection = try #require(ModelCatalog.parseAvailableJSON(legacy))
        #expect(projection.profiles["chat"]?.allowsLegacyRecommendationPolicy == true)
        #expect(projection.profiles["custom"]?.allowsLegacyRecommendationPolicy == false)

        let atomic = try #require(ModelCatalog.parseAvailableJSON(Self.payload))
        #expect(atomic.profiles["chat"]?.allowsLegacyRecommendationPolicy == false)

        let corruptAtomic = Self.mutated({ root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            snapshot["catalog_digest"] = "sha256:" + String(repeating: "0", count: 64)
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
            root["text"] = [[
                "alias": "chat", "hf_path": "org/chat",
                "is_builtin": true, "is_text_only": true,
            ]]
        }, resign: false)
        let downgraded = try #require(ModelCatalog.parseAvailableJSON(corruptAtomic))
        #expect(downgraded.profiles["chat"]?.allowsLegacyRecommendationPolicy == false)
    }

    @Test("atomic catalog preserves authenticated recommendation policy addresses")
    func preservesRecommendationPolicyDigests() throws {
        let digest = "sha256:" + String(repeating: "a", count: 64)
        let advertised = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            snapshot["recommendation_policy_digests"] = [digest]
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        let entries = try #require(ModelCatalog.parseAtomicModelEntriesJSON(advertised))
        #expect(entries.allSatisfy { $0.recommendationPolicyDigests == [digest] })
        let projection = try #require(ModelCatalog.parseAvailableJSON(advertised))
        #expect(projection.profiles["chat"]?.recommendationPolicyDigests == [digest])

        for invalid in [
            ["not-a-content-address"],
            [digest, digest],
            ["sha256:" + String(repeating: "A", count: 64)],
            ["sha256:" + String(repeating: "١", count: 64)],
        ] {
            let payload = Self.mutated { root in
                var atomic = root["atomic"] as! [String: Any]
                var snapshot = atomic["snapshot"] as! [String: Any]
                snapshot["recommendation_policy_digests"] = invalid
                atomic["snapshot"] = snapshot
                root["atomic"] = atomic
            }
            #expect(ModelCatalog.parseAtomicModelEntriesJSON(payload) == nil)
        }
    }

    @Test("multi-task aliases project into every supported product surface")
    func multiTaskAliasesAreNotExclusivelyClassified() throws {
        let multiTask = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var aliases = snapshot["aliases"] as! [[String: Any]]
            var capabilities = aliases[1]["capabilities"] as! [String: Any]
            capabilities["task_types"] = ["text_generation", "image_generation"]
            capabilities["operation_modes"] = [
                "chat", "text_to_image", "image_to_image",
            ]
            aliases[1]["capabilities"] = capabilities
            snapshot["aliases"] = aliases
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        let entries = try #require(
            ModelCatalog.parseAtomicModelEntriesJSON(multiTask)
        )
        let image = try #require(entries.first { $0.alias == "image" })
        #expect(image.kind == .image)
        #expect(image.supports(.chat))
        #expect(image.supports(.image))

        let projection = try #require(ModelCatalog.parseAvailableJSON(multiTask))
        #expect(projection.entries.map(\.0) == ["chat", "image"])
        #expect(!projection.excluded.contains("image"))
    }

    @Test("atomic operations drive picker eligibility without alias heuristics")
    func operationsDriveSelection() throws {
        let entries = try #require(ModelCatalog.parseAtomicModelEntriesJSON(Self.payload))
        let image = try #require(entries.first { $0.alias == "image" })
        let video = try #require(entries.first { $0.alias == "video" })
        let tts = try #require(entries.first { $0.alias == "tts" })
        let stt = try #require(entries.first { $0.alias == "stt" })
        #expect(ModelSelectionPurpose.imageGeneration.accepts(image))
        let completionOnly = ModelEntry(
            alias: "completion-only", hfRepo: "org/completion", sizeOnDisk: nil,
            cached: false, taskTypes: [.textGeneration], operationModes: []
        )
        #expect(!completionOnly.supports(.chat))
        #expect(!ModelSelectionPurpose.chat.accepts(completionOnly))
        #expect(ModelSelectionPurpose.imageEditing.accepts(image))
        #expect(ModelSelectionPurpose.textToSpeech.accepts(tts))
        #expect(ModelSelectionPurpose.speechToText.accepts(stt))
        #expect(!ModelSelectionPurpose.chat.accepts(stt))
        #expect(ModelSelectionPurpose.textToVideo.accepts(video))
        #expect(!ModelSelectionPurpose.imageToVideo.accepts(video))

        let atomicImageVideo = ModelEntry(
            alias: "future-i2v", hfRepo: "org/future-i2v", sizeOnDisk: nil,
            cached: false, kind: .video, taskTypes: [.videoGeneration],
            operationModes: [.imageToVideo]
        )
        #expect(ModelSelectionPurpose.imageToVideo.accepts(atomicImageVideo))
        #expect(!ModelSelectionPurpose.textToVideo.accepts(atomicImageVideo))

        let genericAtomicTTS = ModelEntry(
            alias: "future-tts", hfRepo: "org/future-tts", sizeOnDisk: nil,
            cached: false, kind: .audio, audioCapability: .speech,
            audioFamily: "future_tts", taskTypes: [.speechSynthesis],
            operationModes: [.presetVoice]
        )
        #expect(!ModelSelectionPurpose.textToSpeech.accepts(genericAtomicTTS))
        let legacyTTS = ModelEntry(
            alias: "legacy-future-tts", hfRepo: "org/future-tts", sizeOnDisk: nil,
            cached: false, kind: .audio, audioCapability: .speech,
            audioFamily: "future_tts"
        )
        #expect(!ModelSelectionPurpose.textToSpeech.accepts(legacyTTS))
    }

    @Test("image-understanding-only VLM remains atomic without entering Chat")
    func imageUnderstandingOnlyVLMDoesNotEnterChat() throws {
        let vlmPayload = Self.mutated { root in
            Self.mutateCapabilities(in: &root, aliasIndex: 0) {
                $0["task_types"] = ["vision_language"]
                $0["operation_modes"] = ["image_understanding"]
                $0["is_text_only"] = false
            }
        }
        let entries = try #require(ModelCatalog.parseAtomicModelEntriesJSON(vlmPayload))
        let vlm = try #require(entries.first { $0.alias == "chat" })
        #expect(vlm.taskTypes == [.visionLanguage])
        #expect(vlm.operationModes == [.imageUnderstanding])
        #expect(!ModelSelectionPurpose.chat.accepts(vlm))
    }

    @Test("atomic model sources reject unsafe subfolders before projection")
    func atomicSourceSubfoldersFailClosed() throws {
        for invalid in ["../escape", "/absolute", "quant/", "line\nbreak"] {
            let payload = Self.mutated { root in
                var atomic = root["atomic"] as! [String: Any]
                var snapshot = atomic["snapshot"] as! [String: Any]
                var models = snapshot["models"] as! [[String: Any]]
                var source = models[0]["source"] as! [String: Any]
                source["subfolder"] = invalid
                models[0]["source"] = source
                snapshot["models"] = models
                atomic["snapshot"] = snapshot
                root["atomic"] = atomic
            }
            #expect(ModelCatalog.parseAtomicModelEntriesJSON(payload) == nil)
        }

        let valid = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var models = snapshot["models"] as! [[String: Any]]
            var source = models[0]["source"] as! [String: Any]
            source["subfolder"] = "quant/4bit"
            models[0]["source"] = source
            snapshot["models"] = models
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        let entries = try #require(ModelCatalog.parseAtomicModelEntriesJSON(valid))
        #expect(entries.first { $0.alias == "chat" }?.sourceSubfolder == "quant/4bit")

        let booleanSize = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var models = snapshot["models"] as! [[String: Any]]
            models[0]["estimated_download_size_bytes"] = true
            snapshot["models"] = models
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        #expect(ModelCatalog.parseAtomicModelEntriesJSON(booleanSize) == nil)
    }

    @Test("text-to-image plus inpainting exposes both image operations")
    func inpaintingAndGenerationCapability() throws {
        let inpainting = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var aliases = snapshot["aliases"] as! [[String: Any]]
            var capabilities = aliases[1]["capabilities"] as! [String: Any]
            capabilities["operation_modes"] = ["text_to_image", "inpainting"]
            aliases[1]["capabilities"] = capabilities
            snapshot["aliases"] = aliases
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        let entries = try #require(
            ModelCatalog.parseAtomicModelEntriesJSON(inpainting)
        )
        let image = try #require(entries.first { $0.alias == "image" })
        #expect(image.imageCapability == .generationAndEditing)
        #expect(ModelSelectionPurpose.imageGeneration.accepts(image))
        #expect(ModelSelectionPurpose.imageEditing.accepts(image))
    }

    @Test("v1 generation_modes remains a bounded image migration fallback")
    func generationModesMigrationFallback() throws {
        let legacySpelling = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var aliases = snapshot["aliases"] as! [[String: Any]]
            var capabilities = aliases[1]["capabilities"] as! [String: Any]
            capabilities["generation_modes"] = capabilities.removeValue(
                forKey: "operation_modes"
            )
            aliases[1]["capabilities"] = capabilities
            snapshot["aliases"] = aliases
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        let entries = try #require(
            ModelCatalog.parseAtomicModelEntriesJSON(legacySpelling)
        )
        let image = try #require(entries.first { $0.alias == "image" })
        #expect(ModelSelectionPurpose.imageGeneration.accepts(image))
        #expect(ModelSelectionPurpose.imageEditing.accepts(image))
    }

    @Test("atomic operation contract rejects absent, empty, conflicting, and mismatched modes")
    func invalidAtomicOperationContractsFailClosed() {
        let invalidPayloads = [
            Self.mutated { root in
                Self.mutateCapabilities(in: &root, aliasIndex: 0) {
                    $0.removeValue(forKey: "operation_modes")
                }
            },
            Self.mutated { root in
                Self.mutateCapabilities(in: &root, aliasIndex: 0) {
                    $0["operation_modes"] = []
                }
            },
            Self.mutated { root in
                Self.mutateCapabilities(in: &root, aliasIndex: 1) {
                    $0["generation_modes"] = ["text_to_image"]
                }
            },
            Self.mutated { root in
                Self.mutateCapabilities(in: &root, aliasIndex: 0) {
                    $0["operation_modes"] = ["text_to_image"]
                }
            },
        ]
        for payload in invalidPayloads {
            #expect(ModelCatalog.parseAtomicModelEntriesJSON(payload) == nil)
        }
    }

    @Test("chat projection keeps legacy speculative behavior during shadow mode")
    func chatProjectionUsesAtomicPlacement() throws {
        let parsed = try #require(ModelCatalog.parseAvailableJSON(Self.payload))
        #expect(parsed.entries.map(\.0) == ["chat"])
        #expect(parsed.excluded == ["image", "video", "tts", "stt", "hidden"])
        #expect(parsed.speculative["chat"]?.method == .suffix)
        #expect(parsed.profiles["chat"]?.isTextOnly == true)
    }

    @Test("atomic shadow preserves verified MTP default policy")
    func atomicShadowPreservesMTPDefaultPolicy() throws {
        let mtpPayload = Self.mutated { root in
            root["text"] = [[
                "alias": "chat",
                "mtp_draft_model": "org/chat-mtp",
                "mtp_speculative_tokens": 3,
                "mtp_continuous_batching_tier": "verified",
            ]]
        }
        let preset = try #require(
            ModelCatalog.parseAvailableJSON(mtpPayload)?.speculative["chat"]
        )
        #expect(preset.method == .mtp)
        #expect(preset.model == "org/chat-mtp")
        #expect(preset.tokens == 3)
        #expect(preset.isDefaultEnabled)
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

        let mismatchedResolution = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            var aliases = snapshot["aliases"] as! [[String: Any]]
            var target = aliases[0]["target"] as! [String: Any]
            target["resolution_status"] = "resolved"
            target["model_identity_digest"] = "sha256:" + String(repeating: "a", count: 64)
            aliases[0]["target"] = target
            snapshot["aliases"] = aliases
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        #expect(ModelCatalog.parseAtomicModelEntriesJSON(mismatchedResolution) == nil)

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
        let digest = "sha256:" + String(repeating: "b", count: 64)
        let advertised = Self.mutated { root in
            var atomic = root["atomic"] as! [String: Any]
            var snapshot = atomic["snapshot"] as! [String: Any]
            snapshot["recommendation_policy_digests"] = [digest]
            atomic["snapshot"] = snapshot
            root["atomic"] = atomic
        }
        let atomic = try #require(ModelCatalog.parseAtomicModelEntriesJSON(advertised))
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
        #expect(merged.first {
            $0.alias == "chat"
        }?.recommendationPolicyDigests == [digest])
        #expect(merged.first { $0.alias == "custom" }?.kind == .chat)
        #expect(merged.first { $0.alias == "hidden" } == nil)
        #expect(merged.first { $0.alias == "org/external" }?.isExternal == true)

        let remarked = ModelCatalog.remarkCachedByRepo(
            [
                ModelEntry(
                    alias: "chat", hfRepo: "org/chat", sizeOnDisk: nil,
                    cached: false, recommendationPolicyDigests: [digest]
                ),
                ModelEntry(
                    alias: "chat-4bit", hfRepo: "org/chat", sizeOnDisk: "1 GiB",
                    cached: true
                ),
            ],
            resolvedRepos: ["chat": "org/chat"]
        )
        #expect(remarked.first {
            $0.alias == "chat"
        }?.recommendationPolicyDigests == [digest])
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
            excluded: [],
            speculative: [
                "chat-sibling": SpeculativeDecodingPreset(
                    method: .suffix, model: nil, tokens: nil
                ),
            ]
        )
        #expect(merged.first { $0.alias == "chat-sibling" }?.cached == true)
        #expect(merged.first { $0.alias == "chat-sibling" }?.sizeOnDisk == "1 GiB")
        #expect(merged.first {
            $0.alias == "chat-sibling"
        }?.speculativeDecodingPreset?.method == .suffix)
    }

    @Test("cache identity includes subfolder and preserves external ownership")
    func subfolderAndExternalCacheIdentity() {
        let subA = ModelEntry(
            alias: "sub-a", hfRepo: "org/shared", sizeOnDisk: nil, cached: false,
            sourceSubfolder: "a"
        )
        let subASibling = ModelEntry(
            alias: "sub-a-sibling", hfRepo: "org/shared", sizeOnDisk: nil,
            cached: false,
            sourceSubfolder: "a"
        )
        let subB = ModelEntry(
            alias: "sub-b", hfRepo: "org/shared", sizeOnDisk: nil, cached: false,
            sourceSubfolder: "b"
        )
        let root = ModelEntry(
            alias: "root", hfRepo: "org/root", sizeOnDisk: nil, cached: false
        )

        let managed = ModelCatalog.mergeAtomicAndCached(
            atomic: [subA, subASibling, subB],
            cached: [("sub-a", "org/shared", "1 GiB")],
            excluded: []
        )
        #expect(managed.first { $0.alias == "sub-a" }?.cached == false)
        #expect(managed.first { $0.alias == "sub-a-sibling" }?.cached == false)
        #expect(managed.first { $0.alias == "sub-b" }?.cached == false)

        let structured = ModelCatalog.mergeAtomicAndCached(
            atomic: [subA, subASibling, subB],
            cached: [("sub-a", "org/shared", "a", "1 GiB")],
            excluded: []
        )
        #expect(structured.first { $0.alias == "sub-a" }?.cached == true)
        #expect(structured.first { $0.alias == "sub-a-sibling" }?.cached == true)
        #expect(structured.first { $0.alias == "sub-b" }?.cached == false)

        let retargeted = ModelCatalog.mergeAtomicAndCached(
            atomic: [ModelEntry(
                alias: "root", hfRepo: "org/new", sizeOnDisk: nil, cached: false
            )],
            cached: [("root", "org/old", "9 GiB")],
            excluded: []
        )
        #expect(retargeted.first?.cached == false)
        #expect(retargeted.first?.hfRepo == "org/new")

        let external = ModelCatalog.mergeAtomicAndCached(
            atomic: [root, subA],
            cached: [("(external)", "org/root", nil), ("(external)", "org/shared", nil)],
            excluded: []
        )
        #expect(external.first { $0.alias == "root" }?.isExternal == true)
        #expect(external.first { $0.alias == "sub-a" }?.cached == false)
    }

    @Test("structured cache inventory preserves exact subfolder identity")
    func structuredCacheInventory() throws {
        let output = #"""
        {"cached":[
          {"alias":"nested-4bit","repo":"org/multi-quant","subfolder":"4bit","size_bytes":1073741824,"state":"ok","external":false},
          {"alias":null,"repo":"org/external","subfolder":null,"size_bytes":2048,"state":"external","external":true},
          {"alias":null,"repo":"org/partial","subfolder":null,"size_bytes":1024,"state":"incomplete","external":false}
        ]}
        """#
        let rows = try #require(ModelCatalog.parseCachedJSON(output))
        #expect(rows.count == 2)
        #expect(rows[0].alias == "nested-4bit")
        #expect(rows[0].hfRepo == "org/multi-quant")
        #expect(rows[0].subfolder == "4bit")
        #expect(rows[1].alias == "(external)")
        #expect(rows[1].subfolder == nil)
        #expect(ModelCatalog.parseCachedJSON(
            #"{"cached":[{"alias":"nested-4bit","repo":"org/multi-quant","subfolder":"4bit","size_bytes":true,"state":"ok","external":false}]}"#
        ) == nil)
    }
}
