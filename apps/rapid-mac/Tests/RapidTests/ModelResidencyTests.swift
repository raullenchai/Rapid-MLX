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
                },
                "replacement_projection": {
                  "strategy": "evict_first_if_needed",
                  "models_to_free": [{"id": "old-chat", "estimated_bytes": 6335076761}],
                  "current_bytes": 10737418240,
                  "requested_bytes": 6335076761,
                  "projected_bytes": 10737418240,
                  "limit_bytes": 34359738368,
                  "reason": "role_capacity_evict_first_required"
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
        #expect(snapshot.models.first?.replacementProjection?.modelsToFree.first?.id == "old-chat")
        #expect(snapshot.models.first?.replacementProjection?.reason == "role_capacity_evict_first_required")
        #expect(snapshot.audioLanes.isEmpty, "older snapshots remain compatible")
    }

    @Test("Audio-lane residency decodes and matches only the exact catalog model path")
    func decodesAudioLaneTruth() throws {
        let data = Data(
            #"""
            {
              "memory_limit_bytes": 34359738368,
              "memory_used_bytes": 10737418240,
              "memory_available_bytes": 23622320128,
              "idle_ttl_seconds": 1800,
              "loads_total": 0,
              "evictions_total": 0,
              "models": [],
              "audio_lanes": [
                {
                  "lane": "stt",
                  "model": "mlx-community/whisper-small-mlx",
                  "state": "resident",
                  "active_requests": 0,
                  "loaded_at": 123.0,
                  "idle_seconds": 4.0,
                  "last_error": null
                },
                {
                  "lane": "tts",
                  "model": null,
                  "state": "registered",
                  "active_requests": 0,
                  "loaded_at": null,
                  "idle_seconds": 0.0,
                  "last_error": null
                }
              ]
            }
            """#.utf8
        )

        let snapshot = try JSONDecoder().decode(ModelResidencySnapshot.self, from: data)

        #expect(snapshot.audioLanes == [
            ResidentAudioLaneStatus(
                lane: "stt",
                model: "mlx-community/whisper-small-mlx",
                state: "resident"
            ),
            ResidentAudioLaneStatus(lane: "tts", model: nil, state: "registered")
        ])
        #expect(snapshot.containsResidentAudioLane(
            modelPath: "mlx-community/whisper-small-mlx"
        ))
        #expect(!snapshot.containsResidentAudioLane(modelPath: "whisper-small"))
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

    @Test("Model switch guard prompts only for a different model with active requests")
    func modelSwitchRiskUsesActiveRequestContract() {
        func snapshot(activeRequests: Int) -> ModelResidencySnapshot {
            let current = ResidentModelStatus(
                id: "mlx-community/Qwen3.5-4B-MLX-4bit",
                modelPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
                aliases: ["qwen3.5-4b-4bit"],
                modality: "text",
                state: "resident",
                pinned: true,
                primary: true,
                activeRequests: activeRequests,
                estimatedBytes: 1,
                measuredBytes: nil,
                idleSeconds: 0
            )
            return ModelResidencySnapshot(
                memoryLimitBytes: 1,
                memoryUsedBytes: 1,
                memoryAvailableBytes: 0,
                idleTTLSeconds: 1,
                loadsTotal: 1,
                evictionsTotal: 0,
                models: [current]
            )
        }

        let busy = ModelSwitchRisk.evaluate(
            currentAlias: "qwen3.5-4b-4bit",
            targetAlias: "gemma-4-12b-4bit",
            residency: snapshot(activeRequests: 2)
        )
        #expect(busy?.activeRequests == 2)
        #expect(
            busy?.title
                == "Model qwen3.5-4b-4bit is serving 2 active requests. Switch anyway?"
        )

        #expect(ModelSwitchRisk.evaluate(
            currentAlias: "qwen3.5-4b-4bit",
            targetAlias: "gemma-4-12b-4bit",
            residency: snapshot(activeRequests: 0)
        ) == nil)
        #expect(ModelSwitchRisk.evaluate(
            currentAlias: "qwen3.5-4b-4bit",
            targetAlias: "gemma-4-12b-4bit",
            residency: nil
        ) == nil)
        #expect(ModelSwitchRisk.evaluate(
            currentAlias: "qwen3.5-4b-4bit",
            targetAlias: "qwen3.5-4b-4bit",
            residency: snapshot(activeRequests: 2)
        ) == nil)
        #expect(ModelSwitchDecision.approved.requiresProcessRestart)
        #expect(!ModelSwitchDecision.notNeeded.requiresProcessRestart)
        #expect(!ModelSwitchDecision.cancelled.requiresProcessRestart)
        #expect(!ModelSwitchDecision.requiresRevalidation(
            validatedAlias: "qwen3.5-4b-4bit",
            liveAlias: "qwen3.5-4b-4bit"
        ))
        #expect(ModelSwitchDecision.requiresRevalidation(
            validatedAlias: "qwen3.5-4b-4bit",
            liveAlias: "gemma-4-12b-4bit"
        ))
        #expect(ModelSwitchDecision.requiresRevalidation(
            validatedAlias: nil,
            liveAlias: "qwen3.5-4b-4bit"
        ))
        #expect(!ModelSwitchDecision.requiresStop(
            liveAlias: "gemma-4-12b-4bit",
            targetAlias: "gemma-4-12b-4bit"
        ))
        #expect(ModelSwitchDecision.requiresStop(
            liveAlias: "qwen3.5-4b-4bit",
            targetAlias: "gemma-4-12b-4bit"
        ))
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

    @Test("Server accepts only the exact live alias profile for photo capability")
    func liveProfileCannotLagAcrossModelSwitch() {
        let server = ServerManager(testingState: .ready(alias: "current-model"))
        server.applyActiveModelProfile(
            ServerModelProfile(
                id: "old-model",
                capabilities: ["text", "vision"],
                servingLane: "vision",
                servingLaneReason: "vision_supported"
            ),
            forAlias: "old-model"
        )
        #expect(server.activeModelProfile == nil)

        server.applyActiveModelProfile(
            ServerModelProfile(
                id: "current-model",
                capabilities: ["text"],
                servingLane: "text",
                servingLaneReason: "text_lane_forced"
            ),
            forAlias: "current-model"
        )
        #expect(server.activeModelProfile?.id == "current-model")
        #expect(!server.imageInputAvailability(
            forAlias: "current-model",
            catalogSupportsImageInput: true
        ).isAvailable)

        server.clearActiveModelProfile()
        #expect(server.activeModelProfile == nil)
    }

    @Test("Secondary resident chat aliases accept their own live photo profile")
    func secondaryResidentProfileIsAuthoritative() {
        let secondary = ResidentModelStatus(
            id: "secondary-model",
            modelPath: "repo/secondary-model",
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
        let residency = ModelResidencySnapshot(
            memoryLimitBytes: 10,
            memoryUsedBytes: 1,
            memoryAvailableBytes: 9,
            idleTTLSeconds: 60,
            loadsTotal: 1,
            evictionsTotal: 0,
            models: [secondary]
        )
        let server = ServerManager(
            testingState: .ready(alias: "primary-model"),
            residency: residency
        )
        server.applyActiveModelProfile(
            ServerModelProfile(
                id: "secondary-model",
                capabilities: ["text"],
                servingLane: "text",
                servingLaneReason: "text_lane_forced"
            ),
            forAlias: "secondary-model"
        )

        #expect(server.activeModelProfile?.id == "secondary-model")
        #expect(!server.imageInputAvailability(
            forAlias: "secondary-model",
            catalogSupportsImageInput: true
        ).isAvailable)
    }

    @Test("Resident ceiling reuses the Mac usable-RAM bucket")
    func residentMemoryCeiling() {
        #expect(ModelSizing.residentMemoryCeilingGB(on: mockMac(ramGB: 32)) == 25)
        #expect(ModelSizing.residentMemoryCeilingGB(on: mockMac(ramGB: 8)) == 6)
        #expect(ModelSizing.residentMemoryCeilingGB(on: mockMac(ramGB: 4)) == 4)
    }

    @Test("18 GB smart-to-fast replacement credits the measured resident chat model")
    func smartToFastReplacementDoesNotWarn() throws {
        let picks = RAMBucketedDefault.picks(forPhysicalRAMGB: 18)
        let smart = try #require(picks.first)
        let fast = try #require(picks.last)
        #expect(smart.alias == "qwen3.5-9b-4bit")
        #expect(fast.alias == "qwen3.5-4b-4bit")

        let admission = try #require(ServerManager.memoryAdmissionForTransition(
            host: memorySnapshot(totalGB: 18, usedGB: 14.6),
            residency: residency(
                alias: smart.alias,
                measuredGB: 6.3,
                modality: "text"
            ),
            plan: .releaseResidentModels
        ))
        let safety = ModelSizing.memorySafety(
            footprintGB: ModelSizing.estimate(alias: fast.alias).totalGB,
            usedBytes: admission.snapshot.usedBytes,
            totalBytes: admission.snapshot.totalBytes
        )

        #expect(admission.plannedReleaseBytes == UInt64(6.3 * Double(UInt64(1) << 30)))
        #expect(!ModelSizing.requiresMemoryConfirmation(safety))
    }

    @Test("Cached picker replacement over physical RAM waits for confirmation before loading")
    func cachedOverCapacityReplacementWaitsForConfirmation() async throws {
        let gib = UInt64(1) << 30
        let currentAlias = "qwen3.5-4b-4bit"
        let targetAlias = "qwen3.5-35b-8bit"
        let currentResidency = residency(
            alias: currentAlias,
            measuredGB: 4,
            modality: "text"
        )
        let server = ServerManager(
            testingState: .ready(alias: currentAlias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            residency: currentResidency
        )
        server._testInstallChild(ProcessGroupChild.testStub())
        defer { server._testClearChild() }
        server.memorySnapshotProvider = {
            MemoryProbe.Snapshot(totalBytes: 18 * gib, usedBytes: 8 * gib)
        }

        let load = Task {
            await server.ensureServing(
                alias: targetAlias,
                hfPath: nil,
                estimatedMemoryGB: 44,
                replacementGroup: .assistant
            )
        }
        for _ in 0 ..< 300 where server.pendingMemoryWarning == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        let warning = try #require(server.pendingMemoryWarning)

        #expect(warning.alias == targetAlias)
        #expect(warning.severity == .unsafe)
        #expect(warning.plannedReleaseGB == 4)
        #expect(server.state == .ready(alias: currentAlias))
        server.cancelPendingMemoryLoad(warning)
        #expect(await load.value == false)
        #expect(server.state == .ready(alias: currentAlias))
        #expect(!server.isModelResident(targetAlias))
    }

    @Test("A stale over-capacity confirmation cannot tear down a newer live model")
    func staleOverCapacityConfirmationIsCancelled() async throws {
        let gib = UInt64(1) << 30
        let originalAlias = "qwen3.5-4b-4bit"
        let newerAlias = "qwen3.5-9b-4bit"
        let targetAlias = "qwen3.5-35b-8bit"
        let server = ServerManager(
            testingState: .ready(alias: originalAlias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            residency: residency(alias: originalAlias, measuredGB: 4, modality: "text")
        )
        server._testInstallChild(ProcessGroupChild.testStub())
        defer { server._testClearChild() }
        server.memorySnapshotProvider = {
            MemoryProbe.Snapshot(totalBytes: 18 * gib, usedBytes: 8 * gib)
        }

        let load = Task {
            await server.ensureServing(
                alias: targetAlias,
                hfPath: nil,
                estimatedMemoryGB: 44,
                replacementGroup: .assistant
            )
        }
        for _ in 0 ..< 300 where server.pendingMemoryWarning == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        let warning = try #require(server.pendingMemoryWarning)
        #expect(warning.plannedReleaseAlias == originalAlias)

        server._testSetState(.ready(alias: newerAlias))
        server.confirmPendingMemoryLoad(warning)

        #expect(await load.value == false)
        #expect(server.state == .ready(alias: newerAlias))
        #expect(server.pendingMemoryWarning == nil)
    }

    @Test("Replacing a smaller chat model with 27B still warns")
    func smallerToLargerReplacementStillWarns() throws {
        let admission = try #require(ServerManager.memoryAdmissionForTransition(
            host: memorySnapshot(totalGB: 32, usedGB: 20),
            residency: residency(
                alias: "qwen3.5-4b-4bit",
                measuredGB: 4,
                modality: "text"
            ),
            plan: .releaseResidentModels
        ))
        let safety = ModelSizing.memorySafety(
            footprintGB: ModelSizing.estimate(alias: "qwen3.8-27b-4bit").totalGB,
            usedBytes: admission.snapshot.usedBytes,
            totalBytes: admission.snapshot.totalBytes
        )

        #expect(ModelSizing.requiresMemoryConfirmation(safety))
    }

    @Test("Chat-to-image replacement credits the outgoing chat model")
    func chatToImageReplacementDoesNotWarn() throws {
        let admission = try #require(ServerManager.memoryAdmissionForTransition(
            host: memorySnapshot(totalGB: 18, usedGB: 14.6),
            residency: residency(
                alias: "qwen3.5-9b-4bit",
                measuredGB: 6.3,
                modality: "text"
            ),
            plan: .releaseResidentModels
        ))
        let safety = ModelSizing.memorySafety(
            footprintGB: ModelSizing.estimate(alias: "flux2-klein-4b").totalGB,
            usedBytes: admission.snapshot.usedBytes,
            totalBytes: admission.snapshot.totalBytes
        )

        #expect(!ModelSizing.requiresMemoryConfirmation(safety))
    }

    @Test("Image-to-chat replacement credits the outgoing image model")
    func imageToChatReplacementDoesNotWarn() throws {
        let admission = try #require(ServerManager.memoryAdmissionForTransition(
            host: memorySnapshot(totalGB: 18, usedGB: 14),
            residency: residency(
                alias: "flux2-klein-4b",
                measuredGB: 5.9,
                modality: "image-gen"
            ),
            plan: .releaseResidentModels
        ))
        let safety = ModelSizing.memorySafety(
            footprintGB: ModelSizing.estimate(alias: "qwen3.5-4b-4bit").totalGB,
            usedBytes: admission.snapshot.usedBytes,
            totalBytes: admission.snapshot.totalBytes
        )

        #expect(!ModelSizing.requiresMemoryConfirmation(safety))
    }

    @Test("Process replacement also credits an outgoing audio-only lane")
    func processReplacementCreditsAudioLane() throws {
        let host = memorySnapshot(totalGB: 18, usedGB: 14.6)
        let admission = try #require(ServerManager.memoryAdmissionForTransition(
            host: host,
            residency: residency(alias: "qwen3-asr", measuredGB: 6, modality: "audio"),
            plan: .releaseResidentModels
        ))

        #expect(admission.plannedReleaseBytes == 6 * UInt64(1 << 30))
        #expect(admission.snapshot.usedBytes < host.usedBytes)
    }

    @Test("A zero resident measurement falls back to its admission estimate")
    func processReplacementCreditsEstimatedBytesWhenMeasurementIsZero() throws {
        let admission = try #require(ServerManager.memoryAdmissionForTransition(
            host: memorySnapshot(totalGB: 18, usedGB: 14.6),
            residency: residency(
                alias: "qwen3.5-9b-4bit",
                measuredGB: 0,
                estimatedGB: 6.3,
                modality: "text"
            ),
            plan: .releaseResidentModels
        ))

        #expect(admission.plannedReleaseBytes == UInt64(6.3 * Double(UInt64(1) << 30)))
    }

    @Test("No resident evidence preserves the ordinary live admission probe")
    func noResidentLeavesAdmissionUnchanged() {
        #expect(ServerManager.memoryAdmissionForTransition(
            host: memorySnapshot(totalGB: 18, usedGB: 14.6),
            residency: .empty,
            plan: .releaseResidentModels
        ) == nil)
    }

    @Test("Post-stop host growth wins over the earlier replacement projection")
    func postStopGrowthUsesConservativeAdmission() throws {
        let planned = try #require(ServerManager.memoryAdmissionForTransition(
            host: memorySnapshot(totalGB: 18, usedGB: 14),
            residency: residency(alias: "old-chat", measuredGB: 6, modality: "text"),
            plan: .releaseResidentModels
        ))
        let resolved = try #require(ServerManager.memorySnapshotForAdmission(
            planned: planned,
            live: memorySnapshot(totalGB: 18, usedGB: 10)
        ))

        #expect(resolved.usedBytes == 10 * UInt64(1 << 30))
    }

    @Test("A slower post-stop probe cannot discard the measured release credit")
    func stalePostStopProbeUsesConservativeProjection() throws {
        let planned = try #require(ServerManager.memoryAdmissionForTransition(
            host: memorySnapshot(totalGB: 18, usedGB: 14),
            residency: residency(alias: "old-chat", measuredGB: 6, modality: "text"),
            plan: .releaseResidentModels
        ))
        let resolved = try #require(ServerManager.memorySnapshotForAdmission(
            planned: planned,
            live: memorySnapshot(totalGB: 18, usedGB: 6)
        ))

        #expect(resolved.usedBytes == 8 * UInt64(1 << 30))
    }

    @Test("Engine replacement projection produces a truthful insufficient-budget reason")
    func replacementProjectionRejectionCopy() {
        let gib = UInt64(1) << 30
        let projection = ResidentReplacementProjection(
            strategy: "evict_first_if_needed",
            modelsToFree: [.init(id: "old-chat", estimatedBytes: 6 * gib)],
            currentBytes: 12 * gib,
            requestedBytes: 20 * gib,
            projectedBytes: 26 * gib,
            limitBytes: 24 * gib,
            reason: "role_capacity_insufficient_after_eviction"
        )

        let message = projection.rejectionMessage(alias: "qwen3.8-27b-4bit")
        #expect(message?.contains("release about 6 GB") == true)
        #expect(message?.contains("26 GB") == true)
        #expect(message?.contains("24 GB model-memory budget") == true)
        #expect(message?.contains("close some apps") == false)
    }

    @Test("A keep-resident plan receives no release credit when both models fit")
    func keepBothFitsWithoutEvictionCredit() throws {
        let host = memorySnapshot(totalGB: 32, usedGB: 10)
        let admission = try #require(ServerManager.memoryAdmissionForTransition(
            host: host,
            residency: residency(
                alias: "qwen3.5-4b-4bit",
                measuredGB: 4,
                modality: "text"
            ),
            plan: .keepResidentModels
        ))
        #expect(admission.snapshot == host)
        #expect(admission.plannedReleaseBytes == 0)
        #expect(ModelSizing.memorySafety(
            footprintGB: 4,
            usedBytes: admission.snapshot.usedBytes,
            totalBytes: admission.snapshot.totalBytes
        ) == .safe)
    }

    private func memorySnapshot(totalGB: Double, usedGB: Double) -> MemoryProbe.Snapshot {
        let gib = Double(UInt64(1) << 30)
        return MemoryProbe.Snapshot(
            totalBytes: UInt64(totalGB * gib),
            usedBytes: UInt64(usedGB * gib)
        )
    }

    private func residency(
        alias: String,
        measuredGB: Double,
        estimatedGB: Double? = nil,
        modality: String
    ) -> ModelResidencySnapshot {
        let gib = Double(UInt64(1) << 30)
        let measuredBytes = UInt64(measuredGB * gib)
        let estimatedBytes = UInt64((estimatedGB ?? measuredGB) * gib)
        let status = ResidentModelStatus(
            id: alias,
            modelPath: "repo/\(alias)",
            aliases: [alias],
            modality: modality,
            state: "resident",
            pinned: false,
            primary: true,
            activeRequests: 0,
            estimatedBytes: estimatedBytes,
            measuredBytes: measuredBytes,
            idleSeconds: 0
        )
        return ModelResidencySnapshot(
            memoryLimitBytes: UInt64(18 * gib),
            memoryUsedBytes: measuredBytes,
            memoryAvailableBytes: nil,
            idleTTLSeconds: 0,
            loadsTotal: 1,
            evictionsTotal: 0,
            models: [status]
        )
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
            replaceGroup: .assistant,
            memoryPolicy: .evictFirstIfNeeded,
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
        #expect(json["replace_group"] as? String == "assistant")
        #expect(json["memory_policy"] as? String == "evict_first_if_needed")
        #expect(json["image_mode"] as? String == "editing")
        #expect(performance["kv_cache_dtype"] == nil)
        #expect(performance["kv_cache_turboquant"] as? String == "k8v4")
        #expect(performance["prefix_cache_enabled"] as? Bool == false)
        #expect(performance["cache_memory_mb"] as? Int == 4096)
    }

    @Test(
        "A typed 507 replacement projection reaches the user-facing rejection",
        arguments: [false, true]
    )
    func loadDecodesReplacementProjectionRejection(legacyEnvelope: Bool) async {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ResidencyLoadProjectionRejectProtocol.self]
        var client = ServerResidencyClient()
        client.session = URLSession(configuration: configuration)

        let result = await client.load(
            alias: "qwen3.8-27b-4bit",
            hfPath: nil,
            estimatedSizeGB: 22,
            replaceGroup: .assistant,
            memoryPolicy: .evictFirstIfNeeded,
            port: 8000,
            bearer: legacyEnvelope ? "legacy-envelope" : nil
        )

        guard case .rejected(let message) = result else {
            Issue.record("Expected the typed 507 to remain a rejected resident load")
            return
        }
        #expect(message.contains("release about 6 GB"))
        #expect(message.contains("26 GB"))
        #expect(message.contains("24 GB model-memory budget"))
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

private final class ResidencyLoadProjectionRejectProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let structured = #"{"error":{"message":"insufficient capacity","type":"insufficient_capacity_error","code":"insufficient_capacity_error","param":"estimated_size_gb"},"replacement_projection":{"strategy":"evict_first_if_needed","reason":"role_capacity_insufficient_after_eviction","models_to_free":[{"id":"old-chat","estimated_bytes":6442450944}],"current_bytes":12884901888,"requested_bytes":21474836480,"projected_bytes":27917287424,"limit_bytes":25769803776}}"#
        let payload = if request.value(forHTTPHeaderField: "Authorization") == "Bearer legacy-envelope" {
            Data(#"{"detail":\#(structured)}"#.utf8)
        } else {
            Data(structured.utf8)
        }
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 507, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
