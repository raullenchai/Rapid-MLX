import Foundation
import Testing
@testable import Rapid

/// Table-driven invariants for the Desktop's memory projection across tier
/// picks and lane transitions (#2497).
///
/// ## Why a single formula with invariant tests
///
/// The 0.13.1 memory-projection chain — #2439 (false 114 % warning on a safe
/// smart→fast downgrade) → #2443 (project memory after planned replacement) →
/// #2444 (evict replaced assistant when required) → #2472 → #2478 (the
/// ">100 % confirmation bypassed" regression) — took ~5 hours across four PRs
/// because the projection lived in a handful of call sites without a single
/// formula whose invariants were pinned. A fix under pressure removed a safety
/// guard. This suite restates the projection as four invariants, each named
/// after the bug it locks, over a table of (tier pick, transition) cases.
///
/// ## The engine is the ONLY projection source
///
/// The Desktop never forks its own "replacement projection". The authoritative
/// account of what a resident-assistant load kept or released is the engine's
/// `replacement_projection` payload (`#2444`), decoded as
/// ``ResidentReplacementProjection`` and surfaced as the user-facing rejection
/// when its ``reason == "role_capacity_insufficient_after_eviction"``. Desktop
/// sends only the *policy* (`memory_policy: evict_first_if_needed`) and the
/// estimated footprint; it consumes the engine's typed projection -
/// ``ServerResidencyClient.load`` maps it to ``.rejected(detail)`` with the
/// engine's own byte-accurate ``rejectionMessage``.
///
/// The Desktop's *own* projection is strictly the pre-flight admission guard:
/// ``ServerManager.memoryAdmissionForTransition`` subtracts exactly the bytes
/// the transition's eviction plan frees from the live host sample, then
/// ``ModelSizing.memorySafety`` buckets the target onto the result. Anything
/// that adds, subtracts, or fails to subtract memory there is a projection bug
/// this suite catches. Future code that needs a projection must consume these
/// two shapes (``memoryAdmissionForTransition`` + the engine's
/// `replacement_projection`) — never a second, private reimplementation.
@MainActor
@Suite("Memory-projection invariants", .serialized)
struct MemoryProjectionInvariantTests {
    private let gib = Double(UInt64(1) << 30)

    // MARK: - Invariant 1: smart→fast never warns (#2439/#2443)

    /// A table over every curated RAM tier: (floor, smart pick, fast pick).
    /// The smart pick is live (its measured footprint occupies the tier); we
    /// switch DOWN to the fast pick and ask whether the transition would park
    /// on a memory confirmation.
    ///
    /// #2439 was the false 114 % warning: switching `qwen3.5-9b-4bit` →
    /// `qwen3.5-4b-4bit` (a *safe* downgrade) showed the memory-pressure
    /// confirmation because the projection stacked both models as resident
    /// instead of crediting the measured footprint the replacement releases.
    /// #2443 then generalised "project memory after planned replacement".
    ///
    /// The invariant: a transition DOWN the recommendation table must never
    /// require confirmation once the outgoing (smart) footprint is credited.
    /// The table is deliberately built from the recommendation SSOT's measured
    /// ``footprint_gb`` (the number the picker and the resident admission both
    /// surface) - not the params-based ``ModelSizing.estimate(:)`` heuristic,
    /// which is a fallback and can disagree with measurement on the multi-
    /// billion-param 35B. This pins the guard on ground truth.
    ///
    /// Each row is load-bearing: on the same host WITHOUT crediting the
    /// outgoing smart model the fast reload projects >100 % and would
    /// (falsely) demand confirmation - exactly the #2439 false warning. The
    /// row passes only because the release credit is applied, so deleting
    /// that credit (as an earlier regression did) fails the row.
    @Test("2439+2443: smart→fast never warns on any curated tier", .timeLimit(.minutes(1)))
    func test2439_and_2443_smartToFastNeverWarns() async throws {
        struct Row {
            let floorGB: Double
            let smart: RAMBucketedDefault.Pick
            let fast: RAMBucketedDefault.Pick
        }
        let rows: [Row] = try RAMBucketedDefault.tiers.compactMap { tier in
            let smart = try #require(tier.picks.first)
            let fast = try #require(tier.alt, "tier \(tier.floorGB) has no fast alt")
            return Row(floorGB: tier.floorGB, smart: smart, fast: fast)
        }
        #expect(rows.count >= 8, "expected every recommendation tier in the table")

        for row in rows {
            // Structural downgrade guard: the fast pick must never be heavier
            // than its own smart pick, or a downgrade stops being a downgrade
            // and #2439's false warning becomes real.
            #expect(
                row.fast.footprintGB <= row.smart.footprintGB + 0.01,
                "tier \(row.floorGB): fast \(row.fast.alias) (\(row.fast.footprintGB) GB) must not exceed smart \(row.smart.alias) (\(row.smart.footprintGB) GB)"
            )

            // Host is near-full with the smart model live (1 GB headroom), so
            // the fast reload without credit genuinely projects >100 %.
            let totalGB = row.floorGB
            let usedWithSmart = totalGB - 1.0
            let baseline = usedWithSmart - row.smart.footprintGB
            #expect(baseline >= 0, "tier \(row.floorGB): smart model must fit its tier")

            let admission = try #require(ServerManager.memoryAdmissionForTransition(
                host: memorySnapshot(totalGB: totalGB, usedGB: usedWithSmart),
                residency: residency(
                    alias: row.smart.alias,
                    measuredGB: row.smart.footprintGB,
                    modality: "text"
                ),
                // The chat-switch path credits exactly the replaced assistant
                // (#2439/#2443's mechanism); with the smart model as the only
                // resident this equals releasing it wholesale.
                plan: .releaseModel(alias: row.smart.alias)
            ))
            let safety = ModelSizing.memorySafety(
                footprintGB: row.fast.footprintGB,
                usedBytes: admission.snapshot.usedBytes,
                totalBytes: admission.snapshot.totalBytes
            )
            #expect(
                !ModelSizing.requiresMemoryConfirmation(safety),
                "tier \(row.floorGB): smart→fast (\(row.smart.alias)→\(row.fast.alias)) must NOT confirm, got \(safety)"
            )

            // Load-bearing check: without the release credit, the same fast
            // reload WOULD trip the 114%-style false warning. Confirms the row
            // is a real regression guard, not a tautology.
            let noCreditSafety = ModelSizing.memorySafety(
                footprintGB: row.fast.footprintGB,
                usedBytes: UInt64(usedWithSmart * gib),
                totalBytes: UInt64(totalGB * gib)
            )
            #expect(
                ModelSizing.requiresMemoryConfirmation(noCreditSafety),
                "tier \(row.floorGB): fast reload without release credit must be unsafe (else this invariant is not load-bearing)"
            )
            #expect(admission.plannedReleaseBytes == UInt64(row.smart.footprintGB * gib))

            // Exercise the production wiring too: ensureServing must choose
            // releaseModel(currentAlias), apply the admission snapshot, avoid
            // parking on a false warning, and reach the resident-load client.
            let child = ProcessGroupChild.testStub()
            let server = residencyServer(
                currentAlias: row.smart.alias,
                currentFootprintGB: row.smart.footprintGB,
                host: memorySnapshot(totalGB: totalGB, usedGB: usedWithSmart),
                session: MemoryInvariantLoadSuccessProtocol.session(initialSnapshot: residency(
                    alias: row.smart.alias,
                    measuredGB: row.smart.footprintGB,
                    modality: "text"
                )),
                child: child
            )
            let loaded = await server.ensureServing(
                alias: row.fast.alias,
                hfPath: nil,
                estimatedMemoryGB: row.fast.footprintGB,
                replacementGroup: .assistant
            )
            #expect(loaded, "tier \(row.floorGB): production resident load must proceed")
            #expect(server.pendingMemoryWarning == nil)
            #expect(server.state == .ready(alias: row.fast.alias))
            #expect(child.isRunning, "in-process replacement must keep the sidecar alive")
            #expect(!server.isModelResident(row.smart.alias),
                    "post-load residency must evict the replaced assistant")
            #expect(server.isModelResident(row.fast.alias),
                    "post-load residency must contain only the replacement assistant")
            server._testClearChild()
        }
    }

    // MARK: - Invariant 2: >100 % always confirms (#2478)

    /// #2478 was the ">100 % confirmation bypassed" regression: on an 18 GB
    /// projection, selecting an oversized cached model from a resident 4B
    /// model bypassed the Desktop memory warning entirely because the picker's
    /// admission saw the much larger test host. The confirmation was silently
    /// dropped and the oversized model loaded.
    ///
    /// The invariant: whenever the projection — after crediting what the
    /// transition's eviction plan frees — still exceeds 100 % of physical
    /// RAM, the confirmation MUST fire. A release credit makes the projection
    /// smaller; it must never be allowed to remove a confirmation that should
    /// still stand when the target remains over physical memory.
    @Test("2478: any >100% projection always requires confirmation (never bypassed)")
    func test2478_over100AlwaysConfirms() throws {
        struct Row {
            let label: String
            let totalGB: Double
            let usedAfterReleaseGB: Double
            let targetGB: Double
        }
        // Each row: a host, the post-release used bytes, and a target that
        // still projects over 100 %. Spread across tiers and lane shapes.
        let rows: [Row] = [
            // The exact #2478 reproduction: 4B resident (4 GB) released on an
            // 18 GB host, target a ~44 GB cached model.
            Row(label: "18GB 4B→44GB cached", totalGB: 18, usedAfterReleaseGB: 4, targetGB: 44),
            // 4B resident on 32 GB, switching up to the 27B smart pick.
            Row(label: "32GB 4B→27B", totalGB: 32, usedAfterReleaseGB: 16, targetGB: 20),
            // Minimal 8 GB tier, 1B resident, jumping to a 9B.
            Row(label: "8GB 1B→9B", totalGB: 8, usedAfterReleaseGB: 4.1, targetGB: 8.7),
            // Largest tier already holding the 27B, reaching for a 122B.
            Row(label: "96GB 27B→122B", totalGB: 96, usedAfterReleaseGB: 40, targetGB: 66),
        ]

        for row in rows {
            let projected = (row.usedAfterReleaseGB + row.targetGB) / row.totalGB
            let safety = ModelSizing.memorySafety(
                footprintGB: row.targetGB,
                usedBytes: UInt64(row.usedAfterReleaseGB * gib),
                totalBytes: UInt64(row.totalGB * gib)
            )
            #expect(
                projected > 1.0,
                "\(row.label): row must actually project > 100 % (got \(projected))"
            )
            #expect(
                safety == .unsafe && ModelSizing.requiresMemoryConfirmation(safety),
                "\(row.label): a \(projected * 100)%-of-RAM projection must confirm, got \(safety)"
            )
        }

        // Boundary guard next to it: at exactly 100 % the projection is
        // advisory (.tight), not a blocking confirmation - so the danger line
        // is crisp and the >100% rule above is not a coarse ≥threshold.
        let boundary = ModelSizing.memorySafety(
            footprintGB: 6, usedBytes: UInt64(94 * gib), totalBytes: UInt64(100 * gib)
        )
        #expect(boundary == .tight)
        #expect(!ModelSizing.requiresMemoryConfirmation(boundary))
    }

    /// Integration proof that the production guard parks, not just that the
    /// pure predicate is true: on an 18 GB host with a 4B resident, an
    /// oversized cached target must land on a `.unsafe` pending warning and
    /// must NOT load before the user confirms (Cancel leaves the 4B untouched).
    /// This is the end-to-end form of the #2478 guard that a prior regression
    /// bypassed.
    @Test("2478: oversized cached replacement parks awaiting confirmation")
    func test2478_integratedGuardParks() async throws {
        let gib = UInt64(1) << 30
        let currentAlias = "qwen3.5-4b-4bit"
        let targetAlias = "qwen3.5-35b-8bit"
        let server = ServerManager(
            testingState: .ready(alias: currentAlias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            residency: residency(alias: currentAlias, measuredGB: 4, modality: "text")
        )
        server._testInstallChild(ProcessGroupChild.testStub())
        defer { server._testClearChild() }
        server.memorySnapshotProvider = {
            MemoryProbe.Snapshot(totalBytes: 18 * gib, usedBytes: 8 * gib)
        }

        let load = Task { @MainActor in
            await server.ensureServing(
                alias: targetAlias,
                hfPath: nil,
                estimatedMemoryGB: 44,
                replacementGroup: .assistant
            )
        }
        var warning: ModelSizing.MemoryWarning?
        for _ in 0 ..< 300 where warning == nil {
            try await Task.sleep(for: .milliseconds(10))
            warning = server.pendingMemoryWarning
        }
        let parked = try #require(warning, "guard must park an oversized replacement")
        #expect(parked.severity == .unsafe)
        #expect(parked.alias == targetAlias)
        #expect(server.state == .ready(alias: currentAlias), "must not load before confirmation")

        server.cancelPendingMemoryLoad(parked)
        #expect(await load.value == false)
        #expect(server.state == .ready(alias: currentAlias))
    }

    // MARK: - Invariant 3: credit only what the eviction plan frees (#2472)

    /// The projection credits EXACTLY the memory the transition's eviction
    /// plan frees - nothing speculative, nothing general. #2472's mechanism:
    /// an assistant replacement plans an in-process eviction of the exact
    /// outgoing assistant, so it may credit only that model's bytes; sibling
    /// residents (an audio lane, a second text engine) that stay resident
    /// remain charged. A process replacement releases every resident. A keep-
    /// resident plan releases nothing. This table pins each plan's credit to
    /// the byte-exact residents it frees, so a future change that over- or
    /// under-credits fails.
    @Test("2472: credit exactly what the transition's eviction plan frees")
    func test2472_creditOnlyWhatEvictionPlanFrees() throws {
        let host = memorySnapshot(totalGB: 18, usedGB: 18)
        // Host full: the release credit is not capped by available headroom, so
        // the plan-vs-credit mapping below is exact (sums, not min-with-used).
        // Three residents: the outgoing chat model, a sibling audio lane that
        // an assistant replacement does NOT evict, and a second text engine.
        let chat = residentStatus(alias: "qwen3.5-9b-4bit", measuredGB: 6.3, modality: "text")
        let audio = residentStatus(alias: "qwen3-asr", measuredGB: 6.0, modality: "audio")
        let sibling = residentStatus(alias: "qwen3.5-4b-4bit", measuredGB: 4.0, modality: "text")
        let full = ModelResidencySnapshot(
            memoryLimitBytes: UInt64(18 * gib),
            memoryUsedBytes: UInt64(18 * gib),
            memoryAvailableBytes: nil,
            idleTTLSeconds: 0,
            loadsTotal: 3,
            evictionsTotal: 0,
            models: [chat, audio, sibling],
            audioLanes: [ResidentAudioLaneStatus(lane: "speech", model: "qwen3-asr", state: "resident")]
        )

        // Assistant replacement: ONLY the outgoing chat model is freed. The
        // audio and sibling text residents stay charged.
        let assistantAdmission = try #require(ServerManager.memoryAdmissionForTransition(
            host: host,
            residency: full,
            plan: .releaseModel(alias: "qwen3.5-9b-4bit")
        ))
        #expect(
            assistantAdmission.plannedReleaseBytes == UInt64(6.3 * gib),
            "assistant replacement must free exactly the outgoing chat model, not the sibling \(audio.id)/\(sibling.id) residents"
        )
        #expect(
            assistantAdmission.snapshot.usedBytes
                == host.usedBytes - assistantAdmission.plannedReleaseBytes,
            "sibling residents must remain charged after an assistant replacement"
        )
        #expect(assistantAdmission.snapshot.usedBytes > UInt64(11 * gib),
                "only the outgoing chat model is released; siblings stay charged")

        // Process replacement: every resident model is freed.
        let processAdmission = try #require(ServerManager.memoryAdmissionForTransition(
            host: host,
            residency: full,
            plan: .releaseResidentModels
        ))
        #expect(
            processAdmission.plannedReleaseBytes == UInt64((6.3 + 6.0 + 4.0) * gib),
            "process replacement frees every resident model (6.3 + 6.0 + 4.0)"
        )

        // The credit can never exceed what the host is actually using - a
        // safety ceiling independent of plan. Over-crediting headroom that does
        // not exist would hide a real warning the same way #2478's bypass did.
        let nearEmptyHost = memorySnapshot(totalGB: 18, usedGB: 4)
        let capped = try #require(ServerManager.memoryAdmissionForTransition(
            host: nearEmptyHost,
            residency: full,
            plan: .releaseResidentModels
        ))
        #expect(
            capped.plannedReleaseBytes == UInt64(4 * gib),
            "credit must be capped at the live used bytes (4 GB), not the 16.3 GB plan sum"
        )

        // Keep-both: no release credit.
        let keepAdmission = try #require(ServerManager.memoryAdmissionForTransition(
            host: host,
            residency: full,
            plan: .keepResidentModels
        ))
        #expect(keepAdmission.plannedReleaseBytes == 0)
        #expect(keepAdmission.snapshot == host, "keep-resident must not change the probe")

        // A release plan for a model that is not resident provides no
        // trustworthy evidence → nil → ordinary live probe.
        #expect(ServerManager.memoryAdmissionForTransition(
            host: host,
            residency: full,
            plan: .releaseModel(alias: "not-resident")
        ) == nil)
    }

    /// Production-wiring proof for the sibling-residency part of the invariant.
    /// On an 18 GB host, releasing only the 6.3 GB assistant leaves the 6 GB
    /// audio lane charged, so a 7 GB replacement still projects to 18.7 GB and
    /// MUST park. If ``ensureServing`` ever passes ``releaseResidentModels``
    /// instead, it would over-credit the audio lane and incorrectly load.
    @Test("2472: ensureServing keeps sibling residency charged", .timeLimit(.minutes(1)))
    func test2472_ensureServingKeepsSiblingResidencyCharged() async throws {
        let currentAlias = "qwen3.5-9b-4bit"
        let targetAlias = "replacement-7gb"
        let audioAlias = "qwen3-asr"
        let host = memorySnapshot(totalGB: 18, usedGB: 18)
        let current = residentStatus(
            alias: currentAlias,
            measuredGB: 6.3,
            modality: "text"
        )
        let audio = residentStatus(
            alias: audioAlias,
            measuredGB: 6,
            modality: "audio",
            primary: false
        )
        let snapshot = ModelResidencySnapshot(
            memoryLimitBytes: UInt64(18 * gib),
            memoryUsedBytes: UInt64(12.3 * gib),
            memoryAvailableBytes: nil,
            idleTTLSeconds: 0,
            loadsTotal: 2,
            evictionsTotal: 0,
            models: [current, audio],
            audioLanes: [
                ResidentAudioLaneStatus(
                    lane: "speech",
                    model: audioAlias,
                    state: "resident"
                ),
            ]
        )
        let child = ProcessGroupChild.testStub()
        var client = ServerResidencyClient()
        client.session = MemoryInvariantLoadSuccessProtocol.session(initialSnapshot: snapshot)
        let server = ServerManager(
            testingState: .ready(alias: currentAlias),
            residency: snapshot,
            activeBearer: "memory-invariant-test"
        )
        server._testSetResidencyClient(client)
        server._testInstallChild(child)
        server.memorySnapshotProvider = { host }
        defer { server._testClearChild() }

        let load = Task { @MainActor in
            await server.ensureServing(
                alias: targetAlias,
                hfPath: nil,
                estimatedMemoryGB: 7,
                replacementGroup: .assistant
            )
        }
        var warning: ModelSizing.MemoryWarning?
        for _ in 0 ..< 300 where warning == nil {
            try await Task.sleep(for: .milliseconds(10))
            warning = server.pendingMemoryWarning
        }
        let parked = try #require(
            warning,
            "assistant replacement must not over-credit the resident audio lane"
        )
        #expect(parked.severity == .unsafe)
        #expect(parked.alias == targetAlias)
        #expect(parked.plannedReleaseAlias == currentAlias)
        #expect(abs(parked.plannedReleaseGB - 6.3) < 0.01)
        #expect(server.state == .ready(alias: currentAlias))
        #expect(server.isModelResident(audioAlias), "sibling audio lane must remain charged")

        server.cancelPendingMemoryLoad(parked)
        #expect(await load.value == false)
        #expect(server.state == .ready(alias: currentAlias))
        #expect(server.isModelResident(currentAlias))
        #expect(server.isModelResident(audioAlias))
        #expect(child.isRunning, "Cancel must preserve the resident sidecar")
    }

    /// An engine reported as ``evicting`` is already being torn down - it is
    /// not memory we can further free, so it contributes exactly zero to the
    /// release plan. Over-crediting an evicting model would overstate the
    /// headroom and hide a real warning.
    @Test("2472: an evicting resident contributes zero release credit")
    func test2472_evictingResidentCreditsNothing() throws {
        let host = memorySnapshot(totalGB: 18, usedGB: 14.6)
        let evicting = ResidentModelStatus(
            id: "qwen3.5-9b-4bit",
            modelPath: "repo/qwen3.5-9b-4bit",
            aliases: ["qwen3.5-9b-4bit"],
            modality: "text",
            state: "evicting",
            pinned: false,
            primary: false,
            activeRequests: 0,
            estimatedBytes: UInt64(6.3 * gib),
            measuredBytes: UInt64(6.3 * gib),
            idleSeconds: 0
        )
        let snapshot = ModelResidencySnapshot(
            memoryLimitBytes: UInt64(18 * gib),
            memoryUsedBytes: UInt64(14.6 * gib),
            memoryAvailableBytes: nil,
            idleTTLSeconds: 0,
            loadsTotal: 1,
            evictionsTotal: 0,
            models: [evicting],
            audioLanes: []
        )
        #expect(ServerManager.memoryAdmissionForTransition(
            host: host, residency: snapshot, plan: .releaseModel(alias: "qwen3.5-9b-4bit")
        ) == nil, "an evicting model is not releaseable headroom")
    }

    // MARK: - Invariant 4: typed 507 before any eviction (#2444)

    /// #2444 introduced the engine's `replacement_projection`: when even an
    /// eviction cannot fit the requested assistant, the engine rejects with a
    /// typed "insufficient memory even after eviction" account BEFORE Desktop
    /// attempts anything further - Desktop performs no eviction of its own off
    /// that failure (it surfaces the engine's typed reason verbatim). The full
    /// wire-format handling is pinned in ``loadDecodesReplacementProjectionRejection``;
    /// this table asserts the Desktop-side message contract stays typed across
    /// the projections the engine emits.
    @Test("2444: typed engine projection surfaces an actionable insufficient-memory reason")
    func test2444_typedProjectionSurfacesInsufficientMemory() {
        func projection(freedGB: Double, projectedGB: Double, limitGB: Double)
            -> ResidentReplacementProjection {
            ResidentReplacementProjection(
                strategy: "evict_first_if_needed",
                modelsToFree: [.init(id: "old-chat", estimatedBytes: UInt64(freedGB * gib))],
                currentBytes: UInt64(12 * gib),
                requestedBytes: UInt64(20 * gib),
                projectedBytes: UInt64(projectedGB * gib),
                limitBytes: UInt64(limitGB * gib),
                reason: "role_capacity_insufficient_after_eviction"
            )
        }

        let cases: [(freedGB: Double, projectedGB: Double, limitGB: Double, alias: String, expectRelease: Bool)] = [
            (6.3, 26, 24, "qwen3.8-27b-4bit", true),
            (0, 22, 24, "qwen3.5-4b-4bit", false),
        ]
        for c in cases {
            let message = projection(
                freedGB: c.freedGB, projectedGB: c.projectedGB, limitGB: c.limitGB
            ).rejectionMessage(alias: c.alias)
            #expect(message != nil, "typed insufficient-memory projection must produce a message")
            #expect(message?.contains("\(Int(c.projectedGB)) GB") == true)
            #expect(message?.contains("\(Int(c.limitGB)) GB model-memory budget") == true)
            if c.expectRelease {
                #expect(message?.contains("release about \(max(1, Int(c.freedGB.rounded()))) GB") == true)
            }
        }

        // A non-insufficient reason (e.g. an ordinary capacity refusal) has no
        // typed insufficient-memory copy; Desktop must not fabricate one.
        let otherReason = ResidentReplacementProjection(
            strategy: "keep_then_commit",
            modelsToFree: [],
            currentBytes: UInt64(2 * gib),
            requestedBytes: UInt64(20 * gib),
            projectedBytes: UInt64(22 * gib),
            limitBytes: UInt64(24 * gib),
            reason: "role_capacity_evict_first_required"
        )
        #expect(otherReason.rejectionMessage(alias: "qwen3.8-27b-4bit") == nil)
    }

    @Test("2444: typed 507 rejection preserves the live resident process", .timeLimit(.minutes(1)))
    func test2444_typed507PreservesResident() async {
        let currentAlias = "qwen3.5-4b-4bit"
        let targetAlias = "qwen3.8-27b-4bit"
        let child = ProcessGroupChild.testStub()
        let initialResidency = residency(
            alias: currentAlias,
            measuredGB: 4,
            modality: "text"
        )
        let server = residencyServer(
            currentAlias: currentAlias,
            currentFootprintGB: 4,
            host: memorySnapshot(totalGB: 64, usedGB: 8),
            session: MemoryInvariantProjectionRejectProtocol.session(
                initialSnapshot: initialResidency
            ),
            child: child
        )
        defer { server._testClearChild() }

        let loaded = await server.ensureServing(
            alias: targetAlias,
            hfPath: nil,
            estimatedMemoryGB: 20,
            replacementGroup: .assistant
        )

        #expect(!loaded)
        #expect(
            await server.refreshResidency(),
            "post-rejection residency must remain observable from the engine"
        )
        #expect(server.pendingMemoryWarning == nil,
                "Desktop preflight is safe; this rejection must come from the typed 507")
        #expect(server.state == .ready(alias: currentAlias))
        #expect(server.servingAlias == currentAlias)
        #expect(server.isModelResident(currentAlias))
        #expect(child.isRunning, "typed rejection must not stop the resident sidecar")
        let failure = server.residentLoadFailure(for: targetAlias)
        #expect(failure?.message.contains("release about 6 GB") == true)
        #expect(failure?.message.contains("26 GB") == true)
        #expect(failure?.message.contains("24 GB model-memory budget") == true)
    }

    // MARK: - Helpers

    private func memorySnapshot(totalGB: Double, usedGB: Double) -> MemoryProbe.Snapshot {
        MemoryProbe.Snapshot(
            totalBytes: UInt64(totalGB * gib),
            usedBytes: UInt64(usedGB * gib)
        )
    }

    private func residentStatus(
        alias: String,
        measuredGB: Double,
        modality: String,
        primary: Bool = true
    ) -> ResidentModelStatus {
        let bytes = UInt64(measuredGB * gib)
        return ResidentModelStatus(
            id: alias,
            modelPath: "repo/\(alias)",
            aliases: [alias],
            modality: modality,
            state: "resident",
            pinned: false,
            primary: primary,
            activeRequests: 0,
            estimatedBytes: bytes,
            measuredBytes: bytes,
            idleSeconds: 0
        )
    }

    private func residency(
        alias: String,
        measuredGB: Double,
        modality: String
    ) -> ModelResidencySnapshot {
        let bytes = UInt64(measuredGB * gib)
        let status = residentStatus(alias: alias, measuredGB: measuredGB, modality: modality)
        return ModelResidencySnapshot(
            memoryLimitBytes: UInt64(18 * gib),
            memoryUsedBytes: bytes,
            memoryAvailableBytes: nil,
            idleTTLSeconds: 0,
            loadsTotal: 1,
            evictionsTotal: 0,
            models: [status],
            audioLanes: []
        )
    }

    private func residencyServer(
        currentAlias: String,
        currentFootprintGB: Double,
        host: MemoryProbe.Snapshot,
        session: URLSession,
        child: ProcessGroupChild = .testStub()
    ) -> ServerManager {
        var client = ServerResidencyClient()
        client.session = session
        let server = ServerManager(
            testingState: .ready(alias: currentAlias),
            residency: residency(
                alias: currentAlias,
                measuredGB: currentFootprintGB,
                modality: "text"
            ),
            activeBearer: "memory-invariant-test"
        )
        server._testSetResidencyClient(client)
        server._testInstallChild(child)
        server.memorySnapshotProvider = { host }
        return server
    }
}

private final class MemoryInvariantLoadSuccessProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var residentSnapshot: ModelResidencySnapshot?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    static func session(initialSnapshot: ModelResidencySnapshot) -> URLSession {
        lock.lock()
        residentSnapshot = initialSnapshot
        lock.unlock()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MemoryInvariantLoadSuccessProtocol.self]
        return URLSession(configuration: configuration)
    }

    override func startLoading() {
        let path = request.url?.path
        let payload: Data
        let statusCode: Int
        if request.httpMethod == "POST", path == "/v1/models/load" {
            let body = request.httpBody ?? Self.readBodyStream(request.httpBodyStream) ?? Data()
            let object = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
            guard let alias = object?["model"] as? String,
                  object?["replace_group"] as? String == "assistant",
                  object?["memory_policy"] as? String == "evict_first_if_needed"
            else {
                respond(
                    statusCode: 400,
                    payload: Data(#"{"error":{"message":"invalid replacement policy"}}"#.utf8)
                )
                return
            }
            let estimatedGB = object?["estimated_size_gb"] as? Double ?? 0
            let estimatedBytes = UInt64(estimatedGB * Double(UInt64(1) << 30))
            let status = Self.status(alias: alias, estimatedBytes: estimatedBytes)
            Self.lock.lock()
            let previous = Self.residentSnapshot
            Self.residentSnapshot = ModelResidencySnapshot(
                memoryLimitBytes: previous?.memoryLimitBytes ?? UInt64(64) << 30,
                memoryUsedBytes: status.displayBytes,
                memoryAvailableBytes: previous?.memoryAvailableBytes,
                idleTTLSeconds: previous?.idleTTLSeconds ?? 0,
                loadsTotal: (previous?.loadsTotal ?? 0) + 1,
                evictionsTotal: (previous?.evictionsTotal ?? 0) + 1,
                models: [status]
            )
            Self.lock.unlock()
            payload = try! JSONEncoder().encode(status)
            statusCode = 200
        } else if request.httpMethod == "GET", path == "/v1/models/residency" {
            Self.lock.lock()
            let snapshot = Self.residentSnapshot
            Self.lock.unlock()
            payload = try! JSONEncoder().encode(snapshot ?? .empty)
            statusCode = 200
        } else {
            respond(
                statusCode: 404,
                payload: Data(#"{"error":{"message":"unexpected request"}}"#.utf8)
            )
            return
        }
        respond(statusCode: statusCode, payload: payload)
    }

    private func respond(statusCode: Int, payload: Data) {
        let response = HTTPURLResponse(
            url: request.url!, statusCode: statusCode, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    fileprivate static func readBodyStream(_ stream: InputStream?) -> Data? {
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

    private static func status(alias: String, estimatedBytes: UInt64) -> ResidentModelStatus {
        ResidentModelStatus(
            id: alias,
            modelPath: "repo/\(alias)",
            aliases: [alias],
            modality: "text",
            state: "resident",
            pinned: false,
            primary: true,
            activeRequests: 0,
            estimatedBytes: estimatedBytes,
            measuredBytes: estimatedBytes,
            idleSeconds: 0
        )
    }
}

private final class MemoryInvariantProjectionRejectProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var residentSnapshot: ModelResidencySnapshot?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    static func session(initialSnapshot: ModelResidencySnapshot) -> URLSession {
        lock.lock()
        residentSnapshot = initialSnapshot
        lock.unlock()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MemoryInvariantProjectionRejectProtocol.self]
        return URLSession(configuration: configuration)
    }

    override func startLoading() {
        if request.httpMethod == "GET", request.url?.path == "/v1/models/residency" {
            Self.lock.lock()
            let snapshot = Self.residentSnapshot
            Self.lock.unlock()
            respond(
                statusCode: 200,
                payload: try! JSONEncoder().encode(snapshot ?? .empty)
            )
            return
        }
        guard request.httpMethod == "POST", request.url?.path == "/v1/models/load" else {
            respond(
                statusCode: 404,
                payload: Data(#"{"error":{"message":"unexpected request"}}"#.utf8)
            )
            return
        }
        let body = request.httpBody
            ?? MemoryInvariantLoadSuccessProtocol.readBodyStream(request.httpBodyStream)
            ?? Data()
        let object = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
        guard object?["replace_group"] as? String == "assistant",
              object?["memory_policy"] as? String == "evict_first_if_needed"
        else {
            respond(
                statusCode: 400,
                payload: Data(#"{"error":{"message":"invalid replacement policy"}}"#.utf8)
            )
            return
        }
        let payload = Data(#"{"error":{"message":"insufficient capacity"},"replacement_projection":{"strategy":"evict_first_if_needed","reason":"role_capacity_insufficient_after_eviction","models_to_free":[{"id":"old-chat","estimated_bytes":6442450944}],"current_bytes":12884901888,"requested_bytes":21474836480,"projected_bytes":27917287424,"limit_bytes":25769803776}}"#.utf8)
        respond(statusCode: 507, payload: payload)
    }

    private func respond(statusCode: Int, payload: Data) {
        let response = HTTPURLResponse(
            url: request.url!, statusCode: statusCode, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
