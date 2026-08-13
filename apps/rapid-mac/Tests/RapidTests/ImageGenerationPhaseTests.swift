import Testing
@testable import Rapid

@Suite("Image generation phase semantics")
@MainActor
struct ImageGenerationPhaseTests {
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
                == .finalizing
        )
    }

    @Test("Idle progress before sampling remains preparation")
    func idleProgressPrepares() {
        let idle = ImageClient.ImageProgress(
            running: false, step: 0, total: 0, elapsedMs: 0
        )
        #expect(ImageGenViewModel.nextPhase(from: .preparing, progress: idle) == .preparing)
    }
}
