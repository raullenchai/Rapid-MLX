#if DEBUG
import SwiftUI

/// Settings → Developer. Present only in debug builds.
///
/// Rehearsing the first-run experience used to mean either `defaults delete`
/// against a list of keys nobody keeps current, or `scripts/dogfood-isolate.sh`
/// and a fake `$HOME`. Both work; neither is available to somebody who is
/// already looking at the app and wants to see what a new user sees.
///
/// The whole file is inside `#if DEBUG`, so none of it — not the panel, not
/// the category, not the copy — reaches a release binary. That is also why
/// `scripts/build.sh` needs `RAPID_BUILD_CONFIG=debug` before this appears.
struct SettingsDeveloperPanel: View {
    @Environment(QuickstartCoordinator.self) private var quickstart
    @Environment(ServerManager.self) private var server

    /// Always on and not offered as a choice: without it this button does
    /// nothing that deserves a restart.
    private let onboardingAlwaysIncluded = true

    @State private var erasePreferences = false
    @State private var eraseConversations = false
    @State private var eraseTelemetry = false
    @State private var confirming = false
    @State private var isWorking = false

    private var scope: ReonboardingScope {
        var scope: ReonboardingScope = .onboarding
        if erasePreferences { scope.insert(.preferences) }
        if eraseConversations { scope.insert(.conversations) }
        if eraseTelemetry { scope.insert(.telemetry) }
        return scope
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                SectionHeader(
                    "Developer",
                    subtitle: "Debug builds only. These actions erase real state on this Mac — they are not sandboxed and there is no undo.",
                    emphasis: .page
                )
                reonboardingSection
            }
            .padding(RapidTheme.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .accessibilityIdentifier("Settings.Developer.Panel")
    }

    private var reonboardingSection: some View {
        SettingsSection(
            "Re-onboarding",
            subtitle: "Erase what makes this Mac count as already set up, then restart into the Quickstart wizard. Everything below is off by default; the Quickstart flags themselves are always cleared."
        ) {
            Toggle(isOn: $erasePreferences) {
                SettingsRowLabel(
                    title: "Also erase all settings",
                    description: "Theme, sampling, tool toggles, model favourites, window state — back to first-install defaults."
                )
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            .accessibilityIdentifier("Settings.Developer.Scope.Preferences")

            SettingsRowDivider()

            Toggle(isOn: $eraseConversations) {
                SettingsRowLabel(
                    title: "Also erase every conversation",
                    description: "Deletes conversations.json. Permanent — the sidebar comes back empty."
                )
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            .accessibilityIdentifier("Settings.Developer.Scope.Conversations")

            SettingsRowDivider()

            Toggle(isOn: $eraseTelemetry) {
                SettingsRowLabel(
                    title: "Also erase the telemetry decision",
                    description: "Brings back the consent sheet that runs before Quickstart. This decision is shared with the rapid-mlx CLI, so the command line will ask again too."
                )
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            .accessibilityIdentifier("Settings.Developer.Scope.Telemetry")

            SettingsRowDivider()

            HStack {
                Spacer(minLength: 0)
                Button("Erase and restart") { confirming = true }
                    .buttonStyle(.rapidDestructiveCompact)
                    .disabled(isWorking)
                    .accessibilityIdentifier("Settings.Developer.Reonboard")
            }
        }
        // ``confirmationDialog`` over ``alert`` so the cancel-role button is
        // Return-bound — the same reasoning as the cached-model delete dialog
        // in ``SettingsModelManagementPanel``, and it matters more here: this
        // one can take the conversations with it.
        .confirmationDialog(
            ReonboardingReset.confirmation(for: scope).title,
            isPresented: $confirming,
            titleVisibility: .visible
        ) {
            Button(ReonboardingReset.confirmation(for: scope).confirmTitle, role: .destructive) {
                erase()
            }
            .accessibilityIdentifier("Settings.Developer.ConfirmReonboard")
            Button("Cancel", role: .cancel) { confirming = false }
                .accessibilityIdentifier("Settings.Developer.CancelReonboard")
        } message: {
            Text(ReonboardingReset.confirmation(for: scope).message)
        }
    }

    private func erase() {
        isWorking = true
        Task { @MainActor in
            await ReonboardingReset.perform(
                scope: scope, quickstart: quickstart
            )
            ReonboardingReset.relaunch()
        }
    }
}
#endif
