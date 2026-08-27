import Foundation
import Testing

@testable import Rapid

/// Guard for #1623: the Settings → Privacy consent switch must re-render when
/// it is pressed.
///
/// `telemetryEnabledBinding`'s getter used to read `TelemetryConfig.isEnabled`
/// directly — a plain `static var` over `UserDefaults.standard`. Reading it
/// records no SwiftUI dependency, so the setter wrote the preference and the
/// control kept rendering its old value. The consent *was* stored (the
/// preference flipped, a client ID was minted), but the switch appeared to snap
/// back to off — a privacy control that looks like it refused the user's
/// choice. It seemed to fix itself only because navigating away and back
/// rebuilds the panel for unrelated reasons.
///
/// ViewInspector is not in this target (#1492), so — like
/// ``AccessibilityIdentifierInventoryTests`` — the wiring is pinned by a
/// source-grep guard over the comment- and whitespace-stripped view file.
/// A behavioural test is not possible here; what *is* checkable is the exact
/// shape that caused the bug, and that it has not come back.
@Suite("Telemetry consent toggle re-renders")
struct TelemetryToggleRerenderTests {

    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
    }

    private func strippedSettingsSource() throws -> String {
        let url = Self.sourceRoot.appendingPathComponent(
            "Sources/Rapid/UI/SettingsView.swift")
        let body = try String(contentsOf: url, encoding: .utf8)
        return CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(body)
    }

    /// Without this, every other assertion here is decorative: the `Toggle`
    /// could be repointed at a freshly broken binding while the now-unused
    /// `telemetryEnabledBinding` sits there satisfying the greps.
    @Test("The consent Toggle is the thing wired to this binding")
    func toggleUsesTheAuditedBinding() throws {
        let stripped = try strippedSettingsSource()
        #expect(
            stripped.contains("Toggle(isOn:telemetryEnabledBinding)"),
            """
            The Privacy consent Toggle must be driven by telemetryEnabledBinding \
            — the binding the rest of this suite audits. Pointing it at another \
            binding would move the control outside every check here.
            """
        )
    }

    @Test("The binding's getter reads view state, not the UserDefaults static")
    func getterDoesNotReadTheStaticDirectly() throws {
        let stripped = try strippedSettingsSource()

        #expect(
            !stripped.contains("get:{TelemetryConfig.isEnabled}"),
            """
            telemetryEnabledBinding's getter reads TelemetryConfig.isEnabled \
            directly again. That is a static var over UserDefaults.standard, so \
            SwiftUI records no dependency and the switch will not re-render when \
            pressed — the user sees a consent control snap back to off while \
            they are in fact opted in (#1623). Read @State instead and let the \
            setter update it.
            """
        )
        #expect(
            stripped.contains("get:{telemetryEnabled}"),
            "telemetryEnabledBinding must read the @State mirror."
        )
    }

    @Test("The setter updates the mirror before recording consent")
    func setterUpdatesTheMirror() throws {
        let stripped = try strippedSettingsSource()
        #expect(
            stripped.contains("telemetryEnabled=enabled"),
            """
            The setter must drive the view from the value the user just chose. \
            Without it the @State mirror never moves and the toggle is inert.
            """
        )
        #expect(
            stripped.contains("deferredTelemetryConsent.settingsChanged(enabled:enabled)"),
            "The setter must route the durable choice through the app-owned consent coordinator."
        )
    }

    @Test("The panel re-reads stored consent on appear")
    func panelResyncsOnAppear() throws {
        let stripped = try strippedSettingsSource()
        #expect(
            stripped.contains(".onAppear{telemetryEnabled=TelemetryConfig.isEnabled}"),
            """
            The panel must re-read the stored value when it appears. The \
            post-value invitation writes the same preference, so a value \
            seeded at init can be stale by the time Settings is first opened — \
            which would show the opposite of the truth.
            """
        )
    }

    /// `onAppear` alone leaves a real hole: Settings can be opened *over* the
    /// still-attached post-value invitation, and answering "Share" there would leave
    /// this already-visible switch reading off while telemetry is running.
    @Test("The panel also re-reads consent written while it is visible")
    func panelResyncsOnDefaultsChange() throws {
        let stripped = try strippedSettingsSource()
        // Pinned as ONE expression, publisher through body. Asserting the
        // pieces separately is not equivalent: a bare
        // "{telemetryEnabled=TelemetryConfig.isEnabled}" is already satisfied
        // by the onAppear body above, so the observer could be emptied out and
        // this would stay green.
        #expect(
            stripped.contains(
                ".onReceive(NotificationCenter.default"
                    + ".publisher(for:UserDefaults.didChangeNotification)"
                    + ".receive(on:RunLoop.main)"
                    + "){_intelemetryEnabled=TelemetryConfig.isEnabled}"),
            """
            The Privacy panel must observe UserDefaults.didChangeNotification \
            ON THE MAIN RUN LOOP and resync the mirror in the handler. The \
            post-value consent invitation writes the same key and can \
            be answered while this panel is already on screen — onAppear will \
            not fire again for that. The hop to main is not ceremony: the \
            notification is delivered on the thread that made the write, so a \
            background write to any key would otherwise mutate SwiftUI @State \
            off the main thread.
            """
        )
    }

    /// The store is what the setter delegates to; if this stopped round-tripping
    /// the display fix above would be showing a value nothing agrees with.
    @Test("Recording consent round-trips through the config")
    func recordingConsentRoundTrips() {
        let suiteName = "com.rapidmlx.rapid.tests.telemetry-toggle-\(UUID().uuidString)"
        guard let defaults = UserDefaults(suiteName: suiteName) else {
            Issue.record("could not create an isolated defaults suite")
            return
        }
        defer { defaults.removePersistentDomain(forName: suiteName) }

        #expect(TelemetryConfig.isEnabled(defaults: defaults) == false)

        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        #expect(TelemetryConfig.isEnabled(defaults: defaults) == true)

        defaults.set(false, forKey: TelemetryConfig.enabledKey)
        #expect(TelemetryConfig.isEnabled(defaults: defaults) == false)
    }
}
