import Foundation
import Testing
@testable import Rapid

/// Pin the two v0.5.3 contracts:
///
///   1. ``ServerManager.lastServedAlias`` round-trips through
///      UserDefaults so the next launch can auto-resume the model
///      (LM Studio shape).
///   2. ``ServerManager.ensureServing`` short-circuits when the
///      server is already on the requested alias — that's the
///      noop branch the regen-with-other-alias chevron relies on
///      to NOT spawn a redundant restart while a stream is mid-
///      flight.
///
/// We do NOT exercise the full ``start(alias:)`` path here — that
/// requires a real ``rapid-mlx`` binary, an HF cache, and ~30 s of
/// model load. The headless ``TestDriver`` chat smoke covers the
/// happy-path lifecycle; these tests pin the bookkeeping around it.
@MainActor
@Suite("v0.5.3 — auto-restart + ensureServing contracts")
struct AutoRestartAndRegenRaceTests {
    @Test("lastServedAlias returns nil on fresh install")
    func freshInstallNoResume() {
        // The ServerManager API reads from .standard; verify the
        // fresh-install case explicitly: if no key is set, nil.
        UserDefaults.standard.removeObject(forKey: "rapid.serve.lastAlias")
        #expect(ServerManager.lastServedAlias() == nil)
    }

    @Test("lastServedAlias trims whitespace and rejects empty strings")
    func emptyStringRejected() {
        UserDefaults.standard.set("   ", forKey: "rapid.serve.lastAlias")
        defer { UserDefaults.standard.removeObject(forKey: "rapid.serve.lastAlias") }
        #expect(ServerManager.lastServedAlias() == nil)

        UserDefaults.standard.set("", forKey: "rapid.serve.lastAlias")
        #expect(ServerManager.lastServedAlias() == nil)
    }

    @Test("lastServedAlias returns the trimmed value when set")
    func roundTripAlias() {
        UserDefaults.standard.set("  qwen3.6-27b-4bit  ", forKey: "rapid.serve.lastAlias")
        defer { UserDefaults.standard.removeObject(forKey: "rapid.serve.lastAlias") }
        #expect(ServerManager.lastServedAlias() == "qwen3.6-27b-4bit")
    }

    @Test("ensureServing is a noop when the server is already on the requested alias")
    func ensureServingIsNoopWhenAlreadyServing() async {
        // The whole point of the noop branch: the chevron-driven
        // regen path uses this to avoid wasting 5-30s on a needless
        // restart when the user picks the SAME model they're
        // already using.
        let server = ServerManager(testingState: .ready(alias: "qwen3.6-27b-4bit"))
        let ok = await server.ensureServing(alias: "qwen3.6-27b-4bit")
        #expect(ok == true)
        // The state must NOT have transitioned through .stopped /
        // .starting — verify by checking it's still .ready against
        // the same alias.
        if case .ready(let a) = server.state {
            #expect(a == "qwen3.6-27b-4bit")
        } else {
            Issue.record("Expected state to remain .ready, got \(server.state)")
        }
    }

    @Test("replacement group is also a noop for the already-serving alias")
    func replacementGroupDoesNotReloadCurrentAlias() async {
        let server = ServerManager(testingState: .ready(alias: "qwen3.6-27b-4bit"))
        let ok = await server.ensureServing(
            alias: "qwen3.6-27b-4bit",
            hfPath: nil,
            estimatedMemoryGB: nil,
            replacementGroup: .assistant
        )

        #expect(ok == true)
        guard case .ready(let alias) = server.state else {
            Issue.record("Expected the current model to remain ready")
            return
        }
        #expect(alias == "qwen3.6-27b-4bit")
    }

    @Test("ensureServing returns false when the binary is missing")
    func ensureServingFailsWithoutBinary() async {
        // The ``.missing`` test seam simulates rapid-mlx not being
        // on PATH. ensureServing's stop+start dance will fall
        // through to ``state = .missing`` and the function returns
        // false. The chevron path uses this to surface a friendly
        // "couldn't switch" error instead of streaming to the void.
        let server = ServerManager(testingState: .missing, binaryPath: nil)
        let ok = await server.ensureServing(alias: "qwen3.6-27b-4bit")
        #expect(ok == false)
    }

    @Test("ensureServing rejects empty / whitespace alias")
    func ensureServingRejectsEmpty() async {
        let server = ServerManager(testingState: .idle)
        #expect(await server.ensureServing(alias: "") == false)
        #expect(await server.ensureServing(alias: "   ") == false)
    }

    @Test("internal model replacement preserves the resume alias")
    func modelReplacementPreservesLastServedAlias() {
        #expect(!ServerManager.shouldClearLastServedAlias(
            expectedStop: true,
            preservingLastServedAlias: true
        ))
        #expect(ServerManager.shouldClearLastServedAlias(
            expectedStop: true,
            preservingLastServedAlias: false
        ))
    }

    @Test("audio aliases skip the in-process residency load and use stop/start")
    func audioBypassesResidencyLoad() {
        // The engine's residency loader raises a 500 for the audio (and
        // video-gen) modality, which is NOT the 404/405 that triggers
        // ensureServing's stop/start fallback. Audio callers therefore pass
        // residencyEligible: false and must skip the residency path even when
        // a non-audio model is already resident (ready + child present).
        #expect(!ServerManager.residencyLoadApplies(
            residencyEligible: false,
            readyWithChild: true
        ))
        // Residency-eligible modalities (chat/VLM, image-gen, text-diffusion)
        // still admit a second engine in-process when one is already running.
        #expect(ServerManager.residencyLoadApplies(
            residencyEligible: true,
            readyWithChild: true
        ))
        // No resident process yet: even an eligible model has nothing to load
        // into, so ensureServing goes straight to start().
        #expect(!ServerManager.residencyLoadApplies(
            residencyEligible: true,
            readyWithChild: false
        ))
    }

}
