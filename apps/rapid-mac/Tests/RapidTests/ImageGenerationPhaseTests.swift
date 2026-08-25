import Testing
@testable import Rapid

@Suite("Image generation phase semantics")
@MainActor
struct ImageGenerationPhaseTests {
    @Test("ETA waits for a step-start transition and stays frozen within a step")
    func etaSamplesStepStartsOnly() {
        var eta = ImageDenoiseETA()
        eta.observe(step: 0, total: 4, elapsed: 12)
        #expect(eta.secondsRemaining == nil)

        eta.observe(step: 1, total: 4, elapsed: 20)
        #expect(eta.secondsRemaining == nil)

        eta.observe(step: 1, total: 4, elapsed: 28)
        #expect(eta.secondsRemaining == nil)

        eta.observe(step: 2, total: 4, elapsed: 30)
        #expect(eta.secondsRemaining == 30)

        eta.observe(step: 2, total: 4, elapsed: 39)
        #expect(eta.secondsRemaining == 30)
    }

    @Test("ETA reacts to a genuinely slower later step but not polling time")
    func etaAdaptsOnlyAfterSlowStepCompletes() {
        var eta = ImageDenoiseETA()
        eta.observe(step: 1, total: 6, elapsed: 1)
        eta.observe(step: 2, total: 6, elapsed: 2)
        #expect(eta.secondsRemaining == 5)

        // Five seconds of polling within the same reported step must
        // not manufacture five seconds of additional remaining time.
        eta.observe(step: 2, total: 6, elapsed: 7)
        #expect(eta.secondsRemaining == 5)

        // The next step-start transition really took ten seconds. The EMA moves
        // from 1.0 to 3.25 seconds/step; current step 3 plus steps 4...6 remain.
        eta.observe(step: 3, total: 6, elapsed: 12)
        #expect(eta.secondsRemaining == 13)
    }

    @Test("ETA handles skipped progress samples and changing totals")
    func etaHandlesStepJumpsAndNewTotals() {
        var eta = ImageDenoiseETA()
        eta.observe(step: 1, total: 8, elapsed: 40)
        eta.observe(step: 3, total: 8, elapsed: 52)
        #expect(eta.secondsRemaining == 36)

        eta.observe(step: 3, total: 10, elapsed: 60)
        #expect(eta.secondsRemaining == 48)
    }

    @Test("ETA includes the running final step and reset discards the previous run")
    func etaIncludesFinalStepAndResets() {
        var eta = ImageDenoiseETA()
        eta.observe(step: 1, total: 4, elapsed: 10)
        eta.observe(step: 2, total: 4, elapsed: 15)
        #expect(eta.secondsRemaining == 15)

        eta.observe(step: 4, total: 4, elapsed: 25)
        #expect(eta.secondsRemaining == 5)

        eta.reset()
        eta.observe(step: 1, total: 4, elapsed: 55)
        #expect(eta.secondsRemaining == nil)
    }

    @Test("The final denoise step becomes finalizing until the response lands")
    func completedDenoiseFinalizes() {
        let final = ImageClient.ImageProgress(
            running: false, step: 4, total: 4, elapsedMs: 1_000
        )
        #expect(ImageGenViewModel.nextPhase(from: .denoising, progress: final) == .finalizing)
        #expect(ImageGenViewModel.nextPhase(from: .finalizing, progress: final) == .finalizing)
        #expect(ImageGenViewModel.nextPhase(from: .preparing, progress: final) == .finalizing)
        let finalStillRunning = ImageClient.ImageProgress(
            running: true, step: 4, total: 4, elapsedMs: 1_000
        )
        #expect(
            ImageGenViewModel.nextPhase(from: .denoising, progress: finalStillRunning)
                == .denoising
        )
    }

    @Test("Idle progress before sampling remains preparation")
    func idleProgressPrepares() {
        let idle = ImageClient.ImageProgress(
            running: false, step: 0, total: 0, elapsedMs: 0
        )
        #expect(ImageGenViewModel.nextPhase(from: .preparing, progress: idle) == .preparing)
    }

    @Test("Progress-bar seed steps match each family's engine default")
    func seedStepsMatchEngineDefaults() {
        // The bar is scaled from these before the server reports a live
        // total; a turbo-sized seed on a 20-step model makes the bar slam to
        // full and sit there. Values mirror _DEFAULT_STEPS_BY_FAMILY in
        // vllm_mlx/image/engine.py.
        #expect(ImageGenViewModel.seedSteps(for: "qwen-image") == 20)
        #expect(ImageGenViewModel.seedSteps(for: "qwen-image-edit") == 20)
        #expect(ImageGenViewModel.seedSteps(for: "z-image-turbo") == 8)
        #expect(ImageGenViewModel.seedSteps(for: "flux2-klein-4b") == 4)
    }
}
