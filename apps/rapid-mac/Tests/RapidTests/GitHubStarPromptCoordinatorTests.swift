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

    @Test("A successful gh CLI star completes the invitation without a browser handoff")
    func directStarCompletesLocally() async {
        let defaults = isolatedDefaults()
        defaults.set(34, forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions)
        let recorder = DirectStarRecorder()
        let prompt = GitHubStarPromptCoordinator(
            defaults: defaults,
            quietWindow: .zero,
            presentationActive: true,
            starExecutor: { url in
                await recorder.record(url)
            }
        )

        prompt.productValueDelivered(.chatReply)
        #expect(prompt.isPresented)

        #expect(await prompt.attemptDirectStar())
        #expect(await recorder.recordedURL() == GitHubCommunity.repositoryURL)
        #expect(defaults.bool(forKey: GitHubStarPromptCoordinator.Keys.completed))
        #expect(!prompt.isPresented)
        #expect(!prompt.isStarring)
    }

    @Test("A failed direct star preserves the browser invitation")
    func directStarFailureFallsBackLater() async {
        let defaults = isolatedDefaults()
        defaults.set(34, forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions)
        let prompt = GitHubStarPromptCoordinator(
            defaults: defaults,
            quietWindow: .zero,
            presentationActive: true,
            starExecutor: { _ in
                throw GitHubStarCLIError.commandFailed
            }
        )

        prompt.productValueDelivered(.chatReply)
        #expect(prompt.isPresented)

        let directStarSucceeded = await prompt.attemptDirectStar()
        #expect(!directStarSucceeded)
        #expect(!defaults.bool(forKey: GitHubStarPromptCoordinator.Keys.completed))
        #expect(prompt.isPresented)
        #expect(!prompt.isStarring)
    }

    @Test("A direct star finishing after deferral does not override cadence")
    func deferredDirectStarDoesNotCompleteInvitation() async {
        let defaults = isolatedDefaults()
        defaults.set(34, forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions)
        let promptReference = PromptReference()
        let prompt = GitHubStarPromptCoordinator(
            defaults: defaults,
            quietWindow: .zero,
            presentationActive: true,
            starExecutor: { _ in
                await promptReference.deferPrompt()
            }
        )
        promptReference.prompt = prompt

        prompt.productValueDelivered(.chatReply)
        #expect(prompt.isPresented)

        let directStarSucceeded = await prompt.attemptDirectStar()
        #expect(!directStarSucceeded)
        #expect(!defaults.bool(forKey: GitHubStarPromptCoordinator.Keys.completed))
        #expect(!prompt.isPresented)
    }

    @Test("A successful direct star completes after transient hiding")
    func transientlyHiddenDirectStarCompletesLocally() async {
        let defaults = isolatedDefaults()
        defaults.set(34, forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions)
        let promptReference = PromptReference()
        let prompt = GitHubStarPromptCoordinator(
            defaults: defaults,
            quietWindow: .zero,
            presentationActive: true,
            starExecutor: { _ in
                await promptReference.hidePrompt()
            }
        )
        promptReference.prompt = prompt

        prompt.productValueDelivered(.chatReply)
        #expect(prompt.isPresented)

        let directStarSucceeded = await prompt.attemptDirectStar()
        #expect(directStarSucceeded)
        #expect(defaults.bool(forKey: GitHubStarPromptCoordinator.Keys.completed))
        #expect(!prompt.isPresented)
    }

    @Test("The gh star request pins the canonical host and authenticated-user route")
    func ghStarArgumentsAreCanonical() {
        #expect(GitHubStarCLI.apiArguments() == [
            "api", "--method", "PUT", "--silent",
            "--hostname", "github.com",
            "user/starred/raullenchai/Rapid-MLX"
        ])
    }

    @Test("The default gh executor launches, waits, and reports command status")
    func ghStarExecutorRunsTheSubprocess() async {
        await #expect(throws: Never.self) {
            try await GitHubStarCLI.star(
                GitHubCommunity.repositoryURL,
                executableURL: URL(fileURLWithPath: "/usr/bin/true")
            )
        }

        await #expect(throws: GitHubStarCLIError.commandFailed) {
            try await GitHubStarCLI.star(
                GitHubCommunity.repositoryURL,
                executableURL: URL(fileURLWithPath: "/usr/bin/false")
            )
        }
    }

    @Test("A hung gh child is force-stopped at the configured timeout")
    func ghStarTimeoutKillsHungChild() async throws {
        let executable = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-star-hung-gh-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: executable) }
        try Data("#!/bin/sh\nsleep 30\n".utf8).write(to: executable)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: executable.path
        )

        let clock = ContinuousClock()
        let start = clock.now
        await #expect(throws: GitHubStarCLIError.timedOut) {
            try await GitHubStarCLI.star(
                GitHubCommunity.repositoryURL,
                executableURL: executable,
                timeout: .milliseconds(100)
            )
        }
        #expect(clock.now - start < .seconds(3))
    }

    @Test("A second direct star cannot start while the first is in flight")
    func directStarReentryIsRejected() async {
        let defaults = isolatedDefaults()
        defaults.set(34, forKey: GitHubStarPromptCoordinator.Keys.totalSuccessfulActions)
        let gate = DirectStarGate()
        let prompt = GitHubStarPromptCoordinator(
            defaults: defaults,
            quietWindow: .zero,
            presentationActive: true,
            starExecutor: { _ in
                await gate.wait()
            }
        )

        prompt.productValueDelivered(.chatReply)
        #expect(prompt.isPresented)

        let firstAttempt = Task { await prompt.attemptDirectStar() }
        await gate.waitUntilEntered()
        #expect(await prompt.attemptDirectStar() == false)
        await gate.release()
        let firstAttemptSucceeded = await firstAttempt.value
        #expect(firstAttemptSucceeded)
        #expect(!prompt.isStarring)
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

private actor DirectStarRecorder {
    private var url: URL?

    func record(_ newValue: URL) {
        url = newValue
    }

    func recordedURL() -> URL? {
        url
    }
}

private final class PromptReference: @unchecked Sendable {
    var prompt: GitHubStarPromptCoordinator?

    func deferPrompt() async {
        await MainActor.run { prompt?.deferPrompt() }
    }

    func hidePrompt() async {
        await MainActor.run {
            prompt?.updatePresentationContext(.init(isBusy: true, hasBlockingSurface: false))
        }
    }
}

private actor DirectStarGate {
    private var entered = false
    private var continuations: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        entered = true
        await withCheckedContinuation { continuation in
            continuations.append(continuation)
        }
    }

    func waitUntilEntered() async {
        while !entered {
            await Task.yield()
        }
    }

    func release() {
        continuations.forEach { $0.resume() }
        continuations.removeAll()
    }
}
