import Foundation
import Testing
@testable import Rapid

@Suite("Memory-load confirmation request isolation (#1463)")
struct MemoryLoadConfirmationQueueTests {
    private func warning(_ alias: String) -> ModelSizing.MemoryWarning {
        ModelSizing.MemoryWarning(
            alias: alias,
            hfPath: nil,
            isAutoRespawn: false,
            severity: .unsafe,
            footprintGB: 24,
            freeGB: 4,
            totalGB: 32
        )
    }

    @Test("overlapping loads receive only their own decisions")
    func decisionsAreRequestScoped() {
        var queue = MemoryLoadConfirmationQueue()
        let requestA = UUID()
        let requestB = UUID()
        let warningA = warning("model-a")
        let warningB = warning("model-b")

        queue.enqueue(warning: warningA, requestID: requestA)
        queue.enqueue(warning: warningB, requestID: requestB)

        #expect(queue.currentWarning?.id == warningA.id)
        let staleResolution = queue.resolveCurrent(warning: warningB, decision: .confirmed(sequence: 99))
        let cancelledA = queue.resolveCurrent(warning: warningA, decision: .cancelled)
        let decisionA = queue.takeDecision(for: requestA)
        let prematureDecisionB = queue.takeDecision(for: requestB)
        #expect(staleResolution == false)
        #expect(cancelledA == true)
        #expect(decisionA == .cancelled)
        #expect(prematureDecisionB == nil)

        #expect(queue.currentWarning?.id == warningB.id)
        let confirmedB = queue.resolveCurrent(warning: warningB, decision: .confirmed(sequence: 7))
        let decisionB = queue.takeDecision(for: requestB)
        #expect(confirmedB == true)
        #expect(decisionB == .confirmed(sequence: 7))
        #expect(queue.currentWarning == nil)
        queue.completeConfirmedLaunch(warningID: warningB.id)
        #expect(queue.currentWarning == nil)
    }

    @Test("same-alias loads remain distinct requests")
    func duplicateAliasesRemainDistinct() {
        var queue = MemoryLoadConfirmationQueue()
        let requestA = UUID()
        let requestB = UUID()
        let warningA = warning("same-model")
        let warningB = warning("same-model")

        queue.enqueue(warning: warningA, requestID: requestA)
        queue.enqueue(warning: warningB, requestID: requestB)
        #expect(warningA.id != warningB.id)

        let confirmedA = queue.resolveCurrent(warning: warningA, decision: .confirmed(sequence: 1))
        let decisionA = queue.takeDecision(for: requestA)
        let decisionB = queue.takeDecision(for: requestB)
        #expect(confirmedA)
        #expect(decisionA == .confirmed(sequence: 1))
        #expect(decisionB == nil)
        #expect(queue.currentWarning == nil)
        queue.completeConfirmedLaunch(warningID: warningA.id)
        #expect(queue.currentWarning?.id == warningB.id)
    }

    @Test("direct starts do not retain an unconsumed decision")
    func directStartDecisionIsNotRetained() {
        var queue = MemoryLoadConfirmationQueue()
        let warning = warning("picker-start")
        queue.enqueue(warning: warning, requestID: nil)

        let confirmed = queue.resolveCurrent(warning: warning, decision: .confirmed(sequence: 3))
        #expect(confirmed)
        #expect(queue.currentWarning == nil)
        queue.completeConfirmedLaunch(warningID: warning.id)
        #expect(queue.currentWarning == nil)
    }

    @Test("a cancelled waiter leaves the visible prompt usable without leaking a result")
    func abandonedWaiterDoesNotLeakDecision() {
        var queue = MemoryLoadConfirmationQueue()
        let request = UUID()
        let warning = warning("cancelled-chat")
        queue.enqueue(warning: warning, requestID: request)

        queue.abandonWaiter(request)
        #expect(queue.currentWarning?.id == warning.id)
        let confirmed = queue.resolveCurrent(warning: warning, decision: .confirmed(sequence: 4))
        #expect(confirmed)
        queue.completeConfirmedLaunch(warningID: warning.id)
        let decision = queue.takeDecision(for: request)
        #expect(decision == nil)
    }

    @Test("next prompt waits for both confirmed launch completion and result consumption")
    func confirmedLaunchSerializesNextPrompt() {
        var queue = MemoryLoadConfirmationQueue()
        let requestA = UUID()
        let requestB = UUID()
        let warningA = warning("model-a")
        let warningB = warning("model-b")
        queue.enqueue(warning: warningA, requestID: requestA)
        queue.enqueue(warning: warningB, requestID: requestB)

        let confirmedA = queue.resolveCurrent(warning: warningA, decision: .confirmed(sequence: 1))
        #expect(confirmedA)
        #expect(queue.currentWarning == nil)
        let delayedCancel = queue.resolveCurrent(warning: warningA, decision: .cancelled)
        #expect(delayedCancel == false)
        queue.completeConfirmedLaunch(warningID: warningA.id)
        #expect(queue.currentWarning == nil)
        let decisionA = queue.takeDecision(for: requestA)
        #expect(decisionA == .confirmed(sequence: 1))
        #expect(queue.currentWarning?.id == warningB.id)
    }

    @Test("cancellation after confirmation drains the retained decision")
    func confirmedThenAbandonedDoesNotLeakDecision() {
        var queue = MemoryLoadConfirmationQueue()
        let request = UUID()
        let warning = warning("confirmed-then-cancelled")
        queue.enqueue(warning: warning, requestID: request)

        let confirmed = queue.resolveCurrent(warning: warning, decision: .confirmed(sequence: 8))
        #expect(confirmed)
        queue.abandonWaiter(request)
        let decision = queue.takeDecision(for: request)
        #expect(decision == nil)
        queue.completeConfirmedLaunch(warningID: warning.id)
        #expect(queue.currentWarning == nil)
    }

    @Test("live memory refresh updates facts without replacing the parked decision")
    func liveRefreshPreservesDecisionIdentity() throws {
        let gib = UInt64(1 << 30)
        var queue = MemoryLoadConfirmationQueue()
        let request = UUID()
        let original = ModelSizing.MemoryWarning(
            alias: "qwen3.5-9b-4bit",
            hfPath: nil,
            isAutoRespawn: false,
            severity: .unsafe,
            footprintGB: ModelSizing.estimate(alias: "qwen3.5-9b-4bit").totalGB,
            freeGB: 2,
            totalGB: 32
        )
        queue.enqueue(warning: original, requestID: request)

        let refreshed = queue.refreshCurrentWarning(
            snapshot: .init(totalBytes: 32 * gib, usedBytes: 2 * gib)
        )
        let transition = try #require(refreshed)
        #expect(transition.old.severity == .unsafe)
        #expect(transition.new.severity == .safe)
        #expect(transition.new.id == original.id)
        #expect(transition.new.freeGB == 30)
        #expect(queue.currentWarning == transition.new)

        let unsafeAgain = queue.refreshCurrentWarning(
            snapshot: .init(totalBytes: 32 * gib, usedBytes: 30 * gib)
        )
        #expect(unsafeAgain?.old.severity == .safe)
        #expect(unsafeAgain?.new.severity == .unsafe)
        #expect(unsafeAgain?.new.id == original.id)

        // Refreshing is not a decision: the original waiter remains parked
        // until the user activates the newly-safe Load model action.
        #expect(queue.takeDecision(for: request) == nil)
        let current = queue.resolveCurrent(
            warningID: original.id,
            decision: .confirmed(sequence: 12)
        )
        #expect(current == unsafeAgain?.new)
        #expect(queue.takeDecision(for: request) == .confirmed(sequence: 12))
    }

    @Test("live refresh reuses the originally captured footprint")
    func liveRefreshPreservesOriginalFootprint() throws {
        let gib = UInt64(1 << 30)
        var queue = MemoryLoadConfirmationQueue()
        let original = ModelSizing.MemoryWarning(
            alias: "custom-local-model",
            hfPath: "/models/custom-local-model",
            isAutoRespawn: false,
            severity: .unsafe,
            footprintGB: 24,
            freeGB: 2,
            totalGB: 32
        )
        queue.enqueue(warning: original, requestID: nil)

        let result = queue.refreshCurrentWarning(
            snapshot: .init(totalBytes: 32 * gib, usedBytes: 7 * gib)
        )
        let refreshed = try #require(result)

        #expect(refreshed.new.severity == .tight)
        #expect(refreshed.new.footprintGB == 24)
        #expect(refreshed.new.hfPath == original.hfPath)
    }

    @Test("live refresh preserves a parked video's artifact contract")
    func liveRefreshPreservesVideoOutputDirectory() throws {
        let gib = UInt64(1 << 30)
        var queue = MemoryLoadConfirmationQueue()
        let original = ModelSizing.MemoryWarning(
            alias: "ltx-2.3-mlx-q4",
            hfPath: "org/ltx-video",
            videoOutputDirectory: "/tmp/Rapid/VideoArtifacts",
            isAutoRespawn: false,
            severity: .unsafe,
            footprintGB: 24,
            freeGB: 2,
            totalGB: 32
        )
        queue.enqueue(warning: original, requestID: nil)

        let result = queue.refreshCurrentWarning(
            snapshot: .init(totalBytes: 64 * gib, usedBytes: 8 * gib)
        )
        let refreshed = try #require(result)

        #expect(refreshed.new.id == original.id)
        #expect(refreshed.new.videoOutputDirectory == original.videoOutputDirectory)
        #expect(refreshed.new.footprintGB == original.footprintGB)
    }

    @Test("post-stop refresh does not credit the released model twice")
    func liveRefreshUsesPostStopHostTruth() throws {
        let gib = UInt64(1 << 30)
        var queue = MemoryLoadConfirmationQueue()
        let original = ModelSizing.MemoryWarning(
            alias: "qwen3.8-27b-4bit",
            hfPath: nil,
            isAutoRespawn: false,
            severity: .unsafe,
            footprintGB: 20,
            freeGB: 18,
            totalGB: 32,
            plannedReleaseGB: 6
        )
        queue.enqueue(warning: original, requestID: nil)

        let result = queue.refreshCurrentWarning(
            snapshot: .init(totalBytes: 32 * gib, usedBytes: 14 * gib)
        )
        let refreshed = try #require(result)

        // The fresh 14 GB sample is post-stop truth. Crediting the old 6 GB a
        // second time would incorrectly classify 8 + 20 GB as safe.
        #expect(refreshed.new.severity == .unsafe)
        #expect(refreshed.new.plannedReleaseGB == 6)
    }

    @Test("pre-stop replacement refresh keeps its pending release credit")
    func liveRefreshCreditsPendingReplacement() throws {
        let gib = UInt64(1 << 30)
        let queue = MemoryLoadConfirmationQueue()
        let original = ModelSizing.MemoryWarning(
            alias: "next-chat",
            hfPath: nil,
            isAutoRespawn: false,
            severity: .unsafe,
            footprintGB: 7,
            freeGB: 2,
            totalGB: 18,
            plannedReleaseGB: 4,
            plannedReleaseIsPending: true
        )
        queue.enqueue(warning: original, requestID: nil)

        let result = queue.refreshCurrentWarning(
            snapshot: .init(totalBytes: 18 * gib, usedBytes: 12 * gib)
        )
        let refreshed = try #require(result)

        #expect(refreshed.new.severity == .safe)
        #expect(refreshed.new.freeGB == 10)
        #expect(refreshed.new.plannedReleaseIsPending)
    }

    @Test("refresh is ignored after the visible decision starts launching")
    func liveRefreshCannotRewriteLaunchingDecision() {
        let gib = UInt64(1 << 30)
        var queue = MemoryLoadConfirmationQueue()
        let original = warning("qwen3.5-9b-4bit")
        queue.enqueue(warning: original, requestID: nil)
        let confirmed = queue.resolveCurrent(warning: original, decision: .confirmed(sequence: 2))
        #expect(confirmed)
        let refreshed = queue.refreshCurrentWarning(
            snapshot: .init(totalBytes: 32 * gib, usedBytes: 2 * gib)
        )
        #expect(refreshed == nil)
    }

    @Test("activation check owns the warning before alert dismissal")
    func activationCheckCannotBeCancelledByAlertDismissal() throws {
        let gib = UInt64(1 << 30)
        let request = UUID()
        var queue = MemoryLoadConfirmationQueue()
        let original = warning("qwen3.5-9b-4bit")
        queue.enqueue(warning: original, requestID: request)

        let beganChecking = queue.beginChecking(warningID: original.id)
        #expect(beganChecking)
        #expect(queue.currentWarning == nil)
        #expect(queue.isPending(request))
        let dismissed = queue.resolveCurrent(warning: original, decision: .cancelled)
        #expect(!dismissed)
        #expect(queue.isPending(request))

        let checkedWarning = queue.checkingWarning(
            warningID: original.id,
            snapshot: .init(totalBytes: 32 * gib, usedBytes: 30 * gib)
        )
        let checked = try #require(checkedWarning)
        #expect(checked.id == original.id)
        #expect(checked.severity == .unsafe)

        queue.restoreAwaiting(warningID: original.id)
        #expect(queue.currentWarning == checked)
        #expect(queue.takeDecision(for: request) == nil)
    }
}
