import Foundation
import Testing
@testable import Rapid

/// Voice co-loading: speech (STT/TTS) should run side-by-side with the
/// primary chat LLM/VLM instead of replacing it. The desktop mounts an
/// ``/v1/audio/*`` lane on every server (unconditional ``--enable-audio``) and
/// lazy-loads the chosen voice engine on demand, so when a model is already
/// serving we can reuse that process for audio. These tests pin the two
/// decisions that make "voice + LLM/VLM coexist" true:
///
/// * ``ServerManager.voiceCoLoadsOnPrimary`` — is there a live primary server
///   (ready + authed) whose ``/v1/audio/*`` lane audio can target?
/// * ``ServerManager.ensureVoiceLane`` — reuse that primary process when
///   possible, otherwise (nothing running) fall back to serving the voice
///   model as its own audio-only process.
@Suite("Voice co-loader (voice + LLM/VLM together)")
@MainActor
struct VoiceCoLoadTests {

    // MARK: - voiceCoLoadsOnPrimary

    @Test("no server -> voice does not co-load")
    func notCoLoadedWhenIdle() {
        let server = ServerManager(testingState: .idle)
        #expect(server.voiceCoLoadsOnPrimary == false, "idle server cannot host the voice lane")
    }

    @Test("ready model without a bearer -> not co-loaded (needs auth to target the lane)")
    func notCoLoadedWithoutBearer() {
        let server = ServerManager(testingState: .ready(alias: "qwen3.6-27b-4bit"))
        #expect(server.activeBearer == nil)
        #expect(server.voiceCoLoadsOnPrimary == false, "targeting the lane requires the minted bearer")
    }

    @Test("ready model with a bearer -> voice co-loads on the primary server")
    func coLoadedWhenReadyAndAuthed() {
        let server = ServerManager(
            testingState: .ready(alias: "qwen3.6-27b-4bit"),
            activeBearer: "test-bearer"
        )
        #expect(server.servingAlias == "qwen3.6-27b-4bit")
        #expect(server.voiceCoLoadsOnPrimary == true)
    }

    @Test("mid-start is not co-loaded (no live serving alias yet)")
    func notCoLoadedWhileStarting() {
        let server = ServerManager(
            testingState: .starting(alias: "qwen3.6-27b-4bit"),
            activeBearer: "test-bearer"
        )
        #expect(server.voiceCoLoadsOnPrimary == false)
    }

    // MARK: - ensureVoiceLane

    @Test("co-loaded primary is reused without spawning a replacement")
    func ensureVoiceLaneReusesPrimary() async {
        let server = ServerManager(
            testingState: .ready(alias: "qwen3.6-27b-4bit"),
            activeBearer: "test-bearer"
        )
        // Must succeed immediately (reuse) even though no voice model binary
        // could ever be spawned in this test harness — proving we are NOT
        // swapping the process.
        let result = await server.ensureVoiceLane(alias: "whisper-tiny", hfPath: nil)
        #expect(result == true)
        // The primary chat model is untouched by the voice request.
        #expect(server.servingAlias == "qwen3.6-27b-4bit")
    }

    @Test("with nothing running, voice falls back to its own server")
    func ensureVoiceLaneFallsBackWhenNothingRunning() async {
        // No binary → the fallback start short-circuits to .missing and fails
        // closed, which is the truthful "not available" answer. The point of
        // this test is that a voice request does NOT tear down a primary model
        // (there is none) — it just reports it cannot start.
        let server = ServerManager(testingState: .idle)
        let result = await server.ensureVoiceLane(alias: "whisper-tiny", hfPath: nil)
        #expect(result == false)
    }

    @Test("crashed / stopped server is never co-loaded (no live serving alias)")
    func notCoLoadedWhenCrashedOrStopped() {
        let crashed = ServerManager(
            testingState: .crashed(alias: "qwen3.6-27b-4bit", message: "boom"),
            activeBearer: "test-bearer"
        )
        #expect(crashed.voiceCoLoadsOnPrimary == false)
        let stopped = ServerManager(testingState: .stopped, activeBearer: "test-bearer")
        #expect(stopped.voiceCoLoadsOnPrimary == false)
    }

    @Test("voice co-load requires BOTH a live model AND a bearer")
    @MainActor
    func coLoadRequiresBothReadyAndAuthed() {
        // Bearer but no model.
        let bearerOnly = ServerManager(testingState: .idle, activeBearer: "b")
        #expect(bearerOnly.voiceCoLoadsOnPrimary == false)
        // Model but no bearer.
        let readyNoAuth = ServerManager(testingState: .ready(alias: "qwen3.6-27b-4bit"))
        #expect(readyNoAuth.voiceCoLoadsOnPrimary == false)
    }
}
