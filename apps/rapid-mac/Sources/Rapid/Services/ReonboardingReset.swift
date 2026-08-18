import AppKit
import Foundation

/// What a re-onboarding run erases.
///
/// The set is a choice rather than a constant because the flows worth
/// rehearsing differ: tuning the Quickstart wizard wants the flags gone and
/// nothing else, while checking what a genuinely new Mac sees wants the
/// conversations and the consent prompt back too. A fixed scope would make the
/// cheap case as destructive as the expensive one.
struct ReonboardingScope: OptionSet, Sendable {
    let rawValue: Int

    /// Quickstart's completion flags, plus the last-served alias that gates
    /// the wizard independently of them.
    static let onboarding = ReonboardingScope(rawValue: 1 << 0)
    /// Every preference this app has written — theme, sampling, tool
    /// toggles, model favourites, window state.
    static let preferences = ReonboardingScope(rawValue: 1 << 1)
    /// `conversations.json`. Hard delete, no undo.
    static let conversations = ReonboardingScope(rawValue: 1 << 2)
    /// The telemetry decision, so the consent sheet runs again. Shared with
    /// the `rapid-mlx` CLI — see ``ReonboardingReset/confirmation(for:)``.
    static let telemetry = ReonboardingScope(rawValue: 1 << 3)
}

/// Clears the state that makes this Mac "already onboarded", then relaunches.
///
/// Kept out of any SwiftUI ``View`` so the confirmation copy is a pure
/// function a test can call — the same split ``ModelCacheActions`` uses for
/// the model-deletion dialog.
enum ReonboardingReset {

    // MARK: - Copy

    struct Confirmation: Equatable {
        let title: String
        let message: String
        /// The destructive button's label. Names the worst thing in the
        /// scope, because that is what the reader needs to weigh.
        let confirmTitle: String
    }

    /// One sentence per thing that will be gone, in descending order of how
    /// much it would hurt to lose. Generated rather than written out per
    /// combination: with four independent toggles there are sixteen, and a
    /// hand-written table would drift from the toggles the first time one
    /// is added.
    static func confirmation(for scope: ReonboardingScope) -> Confirmation {
        var losses: [String] = []
        if scope.contains(.conversations) {
            losses.append("every conversation in the sidebar, permanently")
        }
        if scope.contains(.preferences) {
            losses.append("every setting, back to first-install defaults")
        }
        if scope.contains(.telemetry) {
            losses.append(
                "the telemetry decision — this one is shared with the "
                    + "rapid-mlx CLI, so the command line will ask again too"
            )
        }
        if scope.contains(.onboarding) {
            losses.append("the record that Quickstart has run")
        }

        let message: String
        if losses.isEmpty {
            message = "Nothing is selected, so nothing will be erased."
        } else {
            message = "This erases " + Self.sentenceList(losses)
                + ". Rapid restarts immediately afterwards."
        }

        return Confirmation(
            title: "Erase this Mac's Rapid state and restart?",
            message: message,
            confirmTitle: scope.contains(.conversations)
                ? "Erase and restart"
                : "Restart into onboarding"
        )
    }

    /// "a", "a and b", "a, b, and c" — the serial comma is deliberate; two of
    /// the clauses above contain their own commas.
    private static func sentenceList(_ items: [String]) -> String {
        switch items.count {
        case 0: return ""
        case 1: return items[0]
        case 2: return "\(items[0]) and \(items[1])"
        default:
            return items.dropLast().joined(separator: ", ")
                + ", and " + items[items.count - 1]
        }
    }

    // MARK: - Execution

    /// Erase, then hand back. The caller relaunches — separated so a test can
    /// exercise the erasure without the process going away underneath it.
    ///
    /// The server is stopped first for two reasons: its shutdown path is what
    /// clears `rapid.serve.lastAlias` (the gate Quickstart checks alongside
    /// its own flag), and a sidecar still holding the port would keep the
    /// relaunched instance from binding it.
    @MainActor
    static func perform(
        scope: ReonboardingScope,
        quickstart: QuickstartCoordinator
    ) async {
        guard !scope.isEmpty else { return }
        // Persist the last chat edit and reap the server + download children
        // BEFORE erasing, so ``eraseState``'s conversation delete is the LAST
        // write. The other order lets the termination flush — and a streaming
        // ``stopAndPersist`` — resurrect the conversations we just erased
        // (#1973). ``runStandardTermination`` reaps the server too, so the
        // file handles on the state we're about to delete are already gone.
        AppDelegate.runStandardTermination()
        eraseState(scope: scope, quickstart: quickstart)
    }

    /// The erasure itself, with every destination injectable.
    ///
    /// Split from ``perform(scope:quickstart:server:)`` so it can be tested
    /// against temporary paths and a scratch defaults suite: a test that had
    /// to build a real ``ServerManager`` to check which file got deleted
    /// would be starting a subprocess to assert on `unlink`.
    @MainActor
    static func eraseState(
        scope: ReonboardingScope,
        quickstart: QuickstartCoordinator,
        conversationsURL: URL = ConversationStore.fileURL(),
        telemetryDirectory: URL = TelemetryIdentity.sharedTelemetryDirectory(),
        defaults: UserDefaults = .standard,
        bundleIdentifier: String? = Bundle.main.bundleIdentifier
    ) {
        guard !scope.isEmpty else { return }

        if scope.contains(.conversations) {
            try? FileManager.default.removeItem(at: conversationsURL)
        }

        if scope.contains(.preferences), let bundleIdentifier {
            // Wipes the whole domain, which includes Quickstart's own flags —
            // so this runs BEFORE the onboarding reset, letting that step put
            // the in-memory coordinator back in step with the erased disk.
            defaults.removePersistentDomain(forName: bundleIdentifier)
        }

        if scope.contains(.telemetry) {
            defaults.removeObject(forKey: TelemetryConfig.enabledKey)
            defaults.removeObject(forKey: TelemetryConfig.sharedConsentMigrationKey)
            try? FileManager.default.removeItem(
                at: telemetryDirectory
                    .appendingPathComponent("telemetry-consent.yaml", isDirectory: false)
            )
        }

        if scope.contains(.onboarding) {
            quickstart.resetForReonboarding()
            // The wizard is gated on this alias as well as on its own flags,
            // and `stop()` only clears it when there was a child to stop —
            // so an idle app would otherwise restart straight past Quickstart.
            ServerManager.forgetLastServedAlias(defaults: defaults)
        }
    }

    // MARK: - Relaunch

    /// Quit, and have something outside this process start us again.
    ///
    /// Two things about this app make the obvious spellings fail.
    ///
    /// `NSWorkspace.openApplication` with `createsNewApplicationInstance`,
    /// terminating from its completion, was tried first: measured on a local
    /// ad-hoc signed build, no second instance appeared and the completion
    /// never fired. Hence a detached shell that waits for this PID to vanish
    /// and only then opens the app — by the time it runs there is no second
    /// instance for LaunchServices to arbitrate.
    ///
    /// `NSApp.terminate` then turned out not to quit this app at all.
    /// ``MainWindowCloseInterceptor/windowShouldClose(_:)`` runs the
    /// hide-the-Dock-icon prompt on any window close and returns `false` when
    /// the answer is "hide" — and AppKit routes terminate through exactly
    /// that path, so one `false` cancels the whole sequence. Measured:
    /// `osascript … to quit` returns `-128 userCancelled`, and ⌘Q behaves the
    /// same. That is a defect in its own right and is being reported
    /// separately; this button cannot wait on it.
    ///
    /// So: run the termination work explicitly, then `exit`. Everything
    /// ``AppDelegate/applicationWillTerminate(_:)`` does that still matters
    /// here is done by the caller — ``perform(scope:quickstart:server:)``
    /// already awaited `server.stop()` — plus the clean-shutdown marker,
    /// without which the next launch reports a crash that did not happen.
    ///
    /// pid and path go in as `$1` / `$2` rather than interpolated into the
    /// script: the bundle path contains a space in every install
    /// ("Rapid-MLX Desktop.app"), and quoting it into a shell string is a
    /// bug waiting for the first user with a space in their home directory.
    @MainActor
    static func relaunch(
        bundleURL: URL = Bundle.main.bundleURL,
        processIdentifier: Int32 = ProcessInfo.processInfo.processIdentifier,
        spawn: (String, [String]) -> Void = Self.spawnDetached,
        terminate: @MainActor () -> Void = Self.exitAfterCleanShutdown
    ) {
        spawn("/bin/sh", [
            "-c",
            "while kill -0 \"$1\" 2>/dev/null; do sleep 0.1; done; open \"$2\"",
            "sh",
            String(processIdentifier),
            bundleURL.path,
        ])
        terminate()
    }

    /// Record the clean shutdown ``applicationWillTerminate`` would have
    /// recorded, then leave. `exit` rather than `NSApp.terminate` for the
    /// reason spelled out on ``relaunch(bundleURL:processIdentifier:spawn:terminate:)``.
    @MainActor
    static func exitAfterCleanShutdown() {
        // ``perform`` already ran ``AppDelegate.runStandardTermination`` (chat
        // persisted, server + downloads reaped) BEFORE erasing state — doing
        // it here instead would re-persist the conversations ``eraseState``
        // just deleted (#1973). So this only records the clean-shutdown marker
        // and leaves.
        CrashReporter.recordCleanShutdown()
        exit(0)
    }

    /// Launch and forget. The child outlives us on purpose — it exists to
    /// notice that we are gone.
    static func spawnDetached(_ launchPath: String, _ arguments: [String]) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments
        try? process.run()
    }
}
