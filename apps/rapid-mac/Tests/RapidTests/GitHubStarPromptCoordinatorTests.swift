import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("GitHub star value-moment policy")
struct GitHubStarPromptCoordinatorTests {
    @Test("The first card follows 35 successful outcomes, not app launches")
    func initialWorkloadGate() {
        let defaults = isolatedDefaults()
        let prompt = GitHubStarPromptCoordinator(
            defaults: defaults,
            quietWindow: .zero,
            presentationActive: true
        )

        for _ in 0..<(GitHubStarPromptCoordinator.initialWorkloadThreshold - 1) {
            prompt.productValueDelivered(.chatReply)
        }
        #expect(!prompt.isPresented)

        prompt.productValueDelivered(.generatedImage)
        #expect(prompt.isPresented)
        #expect(defaults.integer(forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions) == 35)
    }

    @Test("Busy work and competing surfaces preserve the invitation until quiet")
    func waitsForCalmWorkspace() {
        let defaults = isolatedDefaults()
        defaults.set(34, forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions)
        let prompt = GitHubStarPromptCoordinator(
            defaults: defaults,
            quietWindow: .zero,
            presentationActive: true
        )

        prompt.updatePresentationContext(.init(isBusy: true, hasBlockingSurface: false))
        prompt.productValueDelivered(.dictationTranscript)
        #expect(!prompt.isPresented)

        prompt.updatePresentationContext(.init(isBusy: false, hasBlockingSurface: true))
        #expect(!prompt.isPresented)

        prompt.updatePresentationContext(.ready)
        #expect(prompt.isPresented)

        prompt.updatePresentationContext(.init(isBusy: false, hasBlockingSurface: true))
        #expect(!prompt.isPresented, "a new dialog temporarily tucks away an already-visible card")
        prompt.updatePresentationContext(.ready)
        #expect(prompt.isPresented)
    }

    @Test("Later applies a three-day floor and doubles subsequent workload")
    func exponentialDeferral() {
        var currentDate = Date(timeIntervalSince1970: 1_800_000_000)
        let defaults = isolatedDefaults()
        defaults.set(34, forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions)
        let prompt = GitHubStarPromptCoordinator(
            defaults: defaults,
            now: { currentDate },
            quietWindow: .zero,
            presentationActive: true
        )

        prompt.productValueDelivered(.chatReply)
        #expect(prompt.isPresented)
        prompt.deferPrompt()
        #expect(!prompt.isPresented)
        #expect(defaults.integer(forKey: GitHubStarPromptCoordinator.Keys.baselineSuccessfulActions) == 35)
        #expect(defaults.integer(forKey: GitHubStarPromptCoordinator.Keys.nextWorkloadThreshold) == 70)

        for _ in 0..<69 { prompt.productValueDelivered(.chatReply) }
        currentDate.addTimeInterval(GitHubStarPromptCoordinator.deferralInterval - 1)
        prompt.productValueDelivered(.chatReply)
        #expect(!prompt.isPresented, "workload alone cannot bypass the cooldown")

        currentDate.addTimeInterval(1)
        prompt.productValueDelivered(.chatReply)
        #expect(prompt.isPresented)
        prompt.deferPrompt()
        #expect(defaults.integer(forKey: GitHubStarPromptCoordinator.Keys.nextWorkloadThreshold) == 140)
    }

    @Test("Opening GitHub completes the invitation without account probing")
    func completionIsDurable() {
        let defaults = isolatedDefaults()
        defaults.set(34, forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions)
        let prompt = GitHubStarPromptCoordinator(
            defaults: defaults,
            quietWindow: .zero,
            presentationActive: true
        )

        prompt.productValueDelivered(.chatReply)
        prompt.repositoryOpened()
        #expect(defaults.bool(forKey: GitHubStarPromptCoordinator.Keys.completed))
        #expect(!prompt.isPresented)

        for _ in 0..<500 { prompt.productValueDelivered(.generatedImage) }
        #expect(!prompt.isPresented)
        #expect(defaults.integer(forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions) == 35)
    }

    @Test("Closing the window suspends presentation and reopening earns a new quiet window")
    func windowLifecycleRestartsQuietWindow() async {
        let defaults = isolatedDefaults()
        let quietWindow = QuietWindowGate()
        defaults.set(34, forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions)
        let prompt = GitHubStarPromptCoordinator(
            defaults: defaults,
            quietWindow: .seconds(1),
            presentationActive: true,
            waitForQuietWindow: { _ in await quietWindow.wait() }
        )

        prompt.productValueDelivered(.dictationTranscript)
        await Task.yield()
        #expect(quietWindow.pendingCount == 1)
        prompt.stopMonitoringUserActivity()
        quietWindow.releaseNext()
        await Task.yield()
        #expect(!prompt.isPresented, "an absent window cannot consume the quiet window offscreen")

        prompt.startMonitoringUserActivity()
        await Task.yield()
        #expect(quietWindow.pendingCount == 1)
        #expect(!prompt.isPresented, "reopening must start a fresh quiet window")
        quietWindow.releaseNext()
        await Task.yield()
        #expect(prompt.isPresented)
        prompt.stopMonitoringUserActivity()
    }

    private func isolatedDefaults() -> UserDefaults {
        let name = "GitHubStarPromptCoordinatorTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }
}

@MainActor
private final class QuietWindowGate {
    private var waiters: [CheckedContinuation<Void, Never>] = []

    var pendingCount: Int { waiters.count }

    func wait() async {
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func releaseNext() {
        waiters.removeFirst().resume()
    }
}
