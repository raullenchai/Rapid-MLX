import Foundation
import Testing
@testable import Rapid

/// Re-onboarding erases real, unrecoverable state, so the two things worth
/// pinning are the copy the user reads before saying yes, and that each
/// checkbox erases only what it names.
@Suite("Re-onboarding reset")
@MainActor
struct ReonboardingResetTests {

    // MARK: - Copy

    /// The dialog has to name the worst thing in the scope. A generic
    /// "this cannot be undone" reads the same whether it is about a flag or
    /// about every conversation on the Mac, which is exactly when a reader
    /// stops reading.
    @Test("The message names each selected loss, worst first")
    func messageEnumeratesLosses() {
        let all = ReonboardingReset.confirmation(
            for: [.onboarding, .preferences, .conversations, .telemetry]
        )
        #expect(all.message.contains("every conversation"))
        #expect(all.message.contains("every setting"))
        #expect(all.message.contains("telemetry decision"))
        #expect(all.message.contains("Quickstart"))

        // Conversations lead: losing them is the only irreversible item.
        let conversationsIndex = all.message.range(of: "every conversation")?.lowerBound
        let settingsIndex = all.message.range(of: "every setting")?.lowerBound
        #expect(conversationsIndex != nil && settingsIndex != nil)
        if let c = conversationsIndex, let s = settingsIndex { #expect(c < s) }
    }

    /// Deleting conversations earns a different verb from restarting into a
    /// wizard, because it is a different promise.
    @Test("The confirm button escalates when conversations are included")
    func confirmTitleEscalates() {
        #expect(ReonboardingReset.confirmation(for: .onboarding).confirmTitle
            == "Restart into onboarding")
        #expect(ReonboardingReset.confirmation(for: [.onboarding, .conversations]).confirmTitle
            == "Erase and restart")
    }

    /// The CLI shares this decision, and a user who wipes it from the desktop
    /// will be asked again in a terminal with no idea why.
    @Test("The telemetry line says the CLI is affected")
    func telemetryMentionsTheCLI() {
        let copy = ReonboardingReset.confirmation(for: [.onboarding, .telemetry])
        #expect(copy.message.contains("rapid-mlx CLI"))
    }

    @Test("An empty scope promises nothing")
    func emptyScopeSaysSo() {
        #expect(ReonboardingReset.confirmation(for: []).message.contains("Nothing is selected"))
    }

    // MARK: - Erasure

    private func scratch() throws -> (URL, URL) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("reonboard-\(UUID().uuidString)", isDirectory: true)
        let telemetry = root.appendingPathComponent(".rapid-mlx", isDirectory: true)
        try FileManager.default.createDirectory(at: telemetry, withIntermediateDirectories: true)
        let conversations = root.appendingPathComponent("conversations.json")
        try Data("[]".utf8).write(to: conversations)
        try Data("consent: true".utf8)
            .write(to: telemetry.appendingPathComponent("telemetry-consent.yaml"))
        return (conversations, telemetry)
    }

    /// The narrow scope is the common one — rehearsing the wizard should not
    /// cost the conversations sitting in the sidebar.
    @Test("Onboarding-only leaves the conversations and the consent file alone")
    func onboardingOnlyTouchesNothingElse() throws {
        let (conversations, telemetry) = try scratch()
        defer { try? FileManager.default.removeItem(at: conversations.deletingLastPathComponent()) }
        let suite = TestDefaultsScope.mintSuiteName(prefix: "reonboard")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let defaults = UserDefaults(suiteName: suite)!
        defaults.set(true, forKey: TelemetryConfig.enabledKey)

        let quickstart = QuickstartCoordinator(defaults: defaults)
        quickstart.markDone()

        ReonboardingReset.eraseState(
            scope: .onboarding,
            quickstart: quickstart,
            conversationsURL: conversations,
            telemetryDirectory: telemetry,
            defaults: defaults,
            bundleIdentifier: suite
        )

        #expect(quickstart.done == false, "the wizard flag is the one thing this scope must clear")
        #expect(FileManager.default.fileExists(atPath: conversations.path))
        #expect(defaults.object(forKey: TelemetryConfig.enabledKey) != nil)
    }

    @Test("Each optional scope erases only its own destination")
    func scopesAreIndependent() throws {
        let (conversations, telemetry) = try scratch()
        defer { try? FileManager.default.removeItem(at: conversations.deletingLastPathComponent()) }
        let consent = telemetry.appendingPathComponent("telemetry-consent.yaml")
        let suite = TestDefaultsScope.mintSuiteName(prefix: "reonboard")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let defaults = UserDefaults(suiteName: suite)!
        defaults.set(true, forKey: TelemetryConfig.enabledKey)

        ReonboardingReset.eraseState(
            scope: [.onboarding, .conversations],
            quickstart: QuickstartCoordinator(defaults: defaults),
            conversationsURL: conversations,
            telemetryDirectory: telemetry,
            defaults: defaults,
            bundleIdentifier: suite
        )
        #expect(!FileManager.default.fileExists(atPath: conversations.path))
        #expect(FileManager.default.fileExists(atPath: consent.path),
                "conversations scope reached the telemetry file")

        ReonboardingReset.eraseState(
            scope: [.onboarding, .telemetry],
            quickstart: QuickstartCoordinator(defaults: defaults),
            conversationsURL: conversations,
            telemetryDirectory: telemetry,
            defaults: defaults,
            bundleIdentifier: suite
        )
        #expect(!FileManager.default.fileExists(atPath: consent.path))
        #expect(defaults.object(forKey: TelemetryConfig.enabledKey) == nil,
                "the consent sheet is gated on this key being absent")
    }

    /// The whole-domain wipe also removes Quickstart's own keys, so the
    /// in-memory coordinator has to be reset after it — otherwise a
    /// coordinator still holding `done = true` would write the flag straight
    /// back on the next save.
    @Test("Wiping preferences still leaves the coordinator reset")
    func preferencesWipeOrderingLeavesTheWizardArmed() throws {
        let (conversations, telemetry) = try scratch()
        defer { try? FileManager.default.removeItem(at: conversations.deletingLastPathComponent()) }
        let suite = TestDefaultsScope.mintSuiteName(prefix: "reonboard")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let defaults = UserDefaults(suiteName: suite)!
        defaults.set("something", forKey: "rapid.appearance.v1")

        let quickstart = QuickstartCoordinator(defaults: defaults)
        quickstart.markDone()

        ReonboardingReset.eraseState(
            scope: [.onboarding, .preferences],
            quickstart: quickstart,
            conversationsURL: conversations,
            telemetryDirectory: telemetry,
            defaults: defaults,
            bundleIdentifier: suite
        )

        #expect(defaults.object(forKey: "rapid.appearance.v1") == nil)
        #expect(quickstart.done == false)
    }

    /// `stop()` returns early when no child is running, so the alias it
    /// would have cleared survives — and the wizard is gated on that alias
    /// as well as on its own flag. An idle app would restart straight past
    /// Quickstart, which is the whole point of the button.
    @Test("Onboarding scope forgets the last-served alias even with nothing running")
    func onboardingForgetsTheServedAlias() throws {
        let (conversations, telemetry) = try scratch()
        defer { try? FileManager.default.removeItem(at: conversations.deletingLastPathComponent()) }
        let suite = TestDefaultsScope.mintSuiteName(prefix: "reonboard")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let defaults = UserDefaults(suiteName: suite)!
        defaults.set("qwen3.5-9b-4bit", forKey: "rapid.serve.lastAlias")

        ReonboardingReset.eraseState(
            scope: .onboarding,
            quickstart: QuickstartCoordinator(defaults: defaults),
            conversationsURL: conversations,
            telemetryDirectory: telemetry,
            defaults: defaults,
            bundleIdentifier: suite
        )

        #expect(defaults.object(forKey: "rapid.serve.lastAlias") == nil)
    }

    // MARK: - Relaunch

    /// The relaunch must outlive this process, so the pid and the bundle path
    /// both have to reach the detached helper — and as arguments, not spliced
    /// into the script, because every install path contains a space.
    @Test("Relaunch hands the pid and path to a detached waiter, then quits")
    func relaunchSpawnsAWaiterBeforeTerminating() {
        var spawned: (String, [String])?
        var terminated = false
        ReonboardingReset.relaunch(
            bundleURL: URL(fileURLWithPath: "/Applications/Rapid-MLX Desktop.app"),
            processIdentifier: 4242,
            spawn: { path, args in spawned = (path, args) },
            terminate: { terminated = true }
        )

        #expect(terminated, "the old instance has to go away or the waiter never fires")
        guard let (launchPath, arguments) = spawned else {
            Issue.record("nothing was spawned, so nothing will restart the app")
            return
        }
        #expect(launchPath == "/bin/sh")
        #expect(arguments.contains("4242"))
        #expect(arguments.contains("/Applications/Rapid-MLX Desktop.app"))
        // The script must reference them positionally — an interpolated path
        // would split on the space in "Rapid-MLX Desktop.app".
        let script = arguments.first(where: { $0.contains("kill -0") }) ?? ""
        #expect(script.contains("$1") && script.contains("$2"))
        #expect(!script.contains("Rapid-MLX Desktop.app"))
    }

    @Test("An empty scope erases nothing")
    func emptyScopeIsANoOp() throws {
        let (conversations, telemetry) = try scratch()
        defer { try? FileManager.default.removeItem(at: conversations.deletingLastPathComponent()) }
        let suite = TestDefaultsScope.mintSuiteName(prefix: "reonboard")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let defaults = UserDefaults(suiteName: suite)!
        let quickstart = QuickstartCoordinator(defaults: defaults)
        quickstart.markDone()

        ReonboardingReset.eraseState(
            scope: [],
            quickstart: quickstart,
            conversationsURL: conversations,
            telemetryDirectory: telemetry,
            defaults: defaults,
            bundleIdentifier: nil
        )

        #expect(quickstart.done == true)
        #expect(FileManager.default.fileExists(atPath: conversations.path))
    }
}
