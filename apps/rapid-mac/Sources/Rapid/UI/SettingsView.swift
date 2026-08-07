import Carbon.HIToolbox
import SwiftUI
import UniformTypeIdentifiers

/// macOS-native Settings window (Cmd+,). Modelled on ChatGPT
/// Desktop's settings shape — a left sidebar with category
/// titles, a right detail panel scoped to the active category.
///
/// v0.4.1 ships a single "Tools" category. The structure is
/// already a sidebar so v0.5 can extend it with categories like
/// "Appearance", "Sampling", "Privacy" without redesigning the
/// shell.
///
/// The category list is hard-coded rather than driven from
/// ``ToolDefinition`` registries so a future category that
/// isn't tool-shaped (e.g. "Default model alias") slots in
/// without inverting the dependency.
struct SettingsView: View {
    // #547 §4/§14: category switches and the save-status banner animate via
    // the shared spring and drop to instant under Reduce Motion.
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(AppearanceConfig.self) private var appearance
    @Environment(SettingsRouter.self) private var router
    /// Read-only here — only needed so the Phase 3b toggle can
    /// trigger ``refreshBinary()`` immediately when the server is
    /// idle (no child running). On `.starting/.ready` we leave the
    /// running child alone and let the explicit restart land the
    /// new binary; the toggle copy already calls that out.
    @Environment(ServerManager.self) private var server
    /// #191: Settings → App panel binds the desktop self-update
    /// poller. ``RapidApp`` injects it into the Settings scene's
    /// environment chain so the panel can render the same
    /// "available update" state the MenuBarExtra already drives.
    @Environment(UpdateChecker.self) private var appUpdater
    /// #191 companion: the in-app installer is what powers the
    /// "Install and Restart" button in ``UpdateInstallView``; we
    /// also bind it here so the panel can show "Updating…" /
    /// "Failed: …" inline when the dedicated update window is
    /// closed.
    @Environment(Installer.self) private var appInstaller
    /// #260: persisted "hide Dock icon on close" choice. Settings →
    /// App surfaces a toggle so the user can change their mind
    /// without re-triggering the one-time prompt, plus a "Reset
    /// onboarding alerts" affordance that brings the prompt back.
    @Environment(DockVisibilityPromptStore.self) private var dockPromptStore
    /// #191: the App panel's "Update Rapid-MLX Desktop" CTA
    /// opens the existing ``update-install`` scene — the same
    /// one the MenuBarExtra menu drives. We need
    /// ``\Environment.openWindow`` for the open call.
    @Environment(\.openWindow) private var openWindow

    /// Stable reference shared by the sidebar and detail canvas. Keeping the
    /// frequently-mutated category outside this large view's value state means
    /// a selection change only invalidates the two children that read it,
    /// rather than rebuilding the entire Settings shell and all environment
    /// lookups on every click.
    @State private var categorySelection = CategorySelection()
    // v0.6.7's NavigationSplitView-with-locked-Binding shape kept the
    // sidebar visible but couldn't kill the title-bar sidebar-toggle
    // pictogram on macOS 14 — `.toolbar(removing: .sidebarToggle)`
    // doesn't reliably strip the system-added NSToolbarItem. v0.6.8
    // drops NavigationSplitView in favour of a plain HStack since
    // the sidebar is permanently visible anyway; no system chrome →
    // no orphan toggle button.

    enum Category: String, CaseIterable, Identifiable {
        /// Issue #210 — file-manager-style cache inspector (what's on
        /// disk, what to delete or download) plus the model-behaviour
        /// preferences. The single home for everything about models;
        /// the older stand-alone "Models" tab was folded in here so
        /// users don't face two competing model surfaces.
        case modelManagement
        /// Built-in tools the model can call: on/off per tool, the
        /// web-search backend + key, and the browse approval mode.
        case tools
        case appearance
        case privacy
        /// Rapid-MLX Desktop app updates. The .app self-update is the
        /// only correct way to bump the bundled engine.
        case app

        var id: String { rawValue }
        var title: String {
            switch self {
            case .modelManagement: return "Model Management"
            case .tools: return "Tools"
            case .appearance: return "Appearance"
            case .privacy: return "Privacy"
            case .app: return "App"
            }
        }
        var iconName: String {
            switch self {
            case .modelManagement: return "externaldrive.fill"
            case .tools: return "wrench.and.screwdriver.fill"
            case .appearance: return "paintpalette.fill"
            case .privacy: return "lock.shield.fill"
            case .app: return "app.badge.fill"
            }
        }
    }

    @MainActor
    @Observable
    final class CategorySelection {
        var selected: Category

        init(selected: Category = .modelManagement) {
            self.selected = selected
        }
    }

    private struct CategoryRail: View {
        let selection: CategorySelection
        @State private var hoveredCategory: Category?

        var body: some View {
            List {
                ForEach(Category.allCases) { cat in
                    Button {
                        selection.selected = cat
                    } label: {
                        Label(cat.title, systemImage: cat.iconName)
                            .frame(maxWidth: .infinity, minHeight: 30, alignment: .leading)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.pressable)
                    .listRowBackground(
                        categoryRowBackground(
                            isSelected: selection.selected == cat,
                            isHovered: hoveredCategory == cat
                        )
                    )
                    .onHover { hovering in
                        if hovering {
                            hoveredCategory = cat
                        } else if hoveredCategory == cat {
                            hoveredCategory = nil
                        }
                    }
                    .rapidAnimation(RapidMotion.quick, value: hoveredCategory)
                    .accessibilityLabel(cat.title)
                    .accessibilityAddTraits(selection.selected == cat ? .isSelected : [])
                    .accessibilityIdentifier("Settings.Category.\(cat.rawValue)")
                }
            }
            .listStyle(.sidebar)
            .frame(width: 200)
            .focusable()
            // #579: keep the rail keyboard-focusable for ↑/↓ nav but drop
            // the system focus ring that painted a blue box around the whole
            // sidebar the moment the settings window opened and auto-focused
            // it. Selection is already shown by the brand-tint row fill, so
            // the ring is redundant chrome here.
            .focusEffectDisabled()
            .onKeyPress(.upArrow) { moveCategorySelection(by: -1); return .handled }
            .onKeyPress(.downArrow) { moveCategorySelection(by: 1); return .handled }
            .accessibilityLabel("Settings categories")
        }

        private func moveCategorySelection(by delta: Int) {
            if let next = SettingsView.category(selection.selected, movedBy: delta) {
                selection.selected = next
            }
        }

        @ViewBuilder
        private func categoryRowBackground(isSelected: Bool, isHovered: Bool) -> some View {
            if isSelected {
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(RapidTheme.brandTint)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
            } else if isHovered {
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(Color.primary.opacity(0.055))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
            } else {
                Color.clear
            }
        }
    }

    private struct DetailCanvas<Content: View>: View {
        let selection: CategorySelection
        let content: (Category) -> Content

        init(
            selection: CategorySelection,
            @ViewBuilder content: @escaping (Category) -> Content
        ) {
            self.selection = selection
            self.content = content
        }

        var body: some View {
            let selected = selection.selected
            ScrollView {
                content(selected)
                    .frame(maxWidth: 600, alignment: .leading)
                    .padding(28)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .background(RapidTheme.canvas)
        }
    }

    enum WebSearchKeyCommit: Equatable {
        case unchanged
        case clear
        case save(String)
    }

    static func webSearchKeyCommitAction(draft: String, wasEdited: Bool) -> WebSearchKeyCommit {
        guard wasEdited else { return .unchanged }
        let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? .clear : .save(trimmed)
    }

    /// Pure helper that decides whether the draft + dirty flag
    /// should be reset after a commit. v0.6.7 codex r1 P2: a failed
    /// Keychain write surfaces a "try again" banner, so the draft
    /// must survive the failure or the retry advice is impossible
    /// (the user would have to re-paste the secret with no fallback).
    /// Pulled out as a static helper so the contract can be pinned
    /// by a unit test without standing up a SwiftUI host.
    static func shouldResetWebSearchKeyDraftAfterCommit(
        keychainWriteSucceeded: Bool
    ) -> Bool {
        keychainWriteSucceeded
    }

    /// v0.6.7 Save-button feedback. Surfaced inline below the key
    /// field for ~2.5 s after a Save / Return commit so the user sees
    /// "yes, it landed in Keychain" — closes the silent-write loop
    /// the green-checkmark-only shape had. ``generation`` is a
    /// monotonic counter minted by ``presentSaveFeedback`` so a
    /// second identical Save still equates as a value change for
    /// SwiftUI's ``onChange`` task; without it the auto-dismiss
    /// would never reschedule.
    enum WebSearchKeySaveFeedback: Equatable {
        case saved(generation: Int)
        case cleared(generation: Int)
        case writeFailed(generation: Int)
    }

    var body: some View {
        // v0.6.7 sidebar lock — the binding's setter snaps any
        // user-driven collapse back to ``.all`` so dragging the
        // column divider, hitting the View menu's "Hide Sidebar", or
        // any programmatic collapse path cannot hide the category
        // list. Paired with ``.toolbar(removing: .sidebarToggle)``
        // below for belt-and-braces: that line strips the title-bar
        // affordance, this binding enforces the invariant if macOS
        // ever surfaces the toggle through another route.
        return HStack(spacing: 0) {
            // v0.5: restrained selection — no native `selection:` binding
            // (which paints a saturated system-blue block and forces white
            // text). CategoryRail keeps that styling while isolating its
            // frequent selection updates from this large parent view.
            // #550: keyboard navigation + focus ring. Tab focuses the
            // category rail, then ↑/↓ move the selection through the
            // categories — the native macOS settings-sidebar behaviour
            // the tap-only implementation lacked, added without adopting
            // the native selection paint.
            CategoryRail(selection: categorySelection)

            Divider()

            // Keep one stable ScrollView and replace only its detail content.
            // Animating this conditional subtree keeps the outgoing and
            // incoming panels alive together, which makes model loading appear
            // on top of the previous category and makes a valid click look
            // ignored until the cross-fade catches up.
            DetailCanvas(selection: categorySelection) { category in
                detailPanel(for: category)
            }
        }
        .frame(minWidth: 720, minHeight: 480)
        .onAppear {
            // v0.4.37: consume any pending deep-link request. Fires
            // when the Settings scene is created (typical for the
            // FIRST open this app session) — covers the case where
            // a deep-link click set ``requestedCategory`` before
            // ever opening Settings, so the window opens already on
            // the target tab instead of the default Tools tab. The
            // ``.onChange`` below handles the SECOND case (Settings
            // already open and being re-focused by another deep-link
            // click).
            consumeRouterRequest()
        }
        .onChange(of: router.requestedCategory) { _, _ in
            consumeRouterRequest()
        }
    }

    /// Pure navigation step for ``CategoryRail.moveCategorySelection(by:)``: the
    /// category `delta` rows from `current` in `Category.allCases`, or
    /// nil at the ends. No wrap-around — matches the native sidebar,
    /// where arrowing past the last row is a no-op. Static + pure so the
    /// clamping contract is unit-testable without the SwiftUI view.
    static func category(_ current: Category, movedBy delta: Int) -> Category? {
        let all = Category.allCases
        guard let idx = all.firstIndex(of: current) else { return nil }
        let next = idx + delta
        guard next >= 0, next < all.count else { return nil }
        return all[next]
    }

    /// Pop the pending deep-link target off the router and apply it.
    /// Clears the field back to nil so a subsequent
    /// `openWindow(id: "settings")` without a request lands on whatever
    /// tab the user was last on.
    private func consumeRouterRequest() {
        if let target = router.requestedCategory {
            categorySelection.selected = target
            router.requestedCategory = nil
        }
    }

    @ViewBuilder
    private func detailPanel(for category: Category) -> some View {
        switch category {
        case .modelManagement:
            SettingsModelManagementPanel()
        case .tools:
            SettingsToolsPanel()
        case .appearance:
            appearancePanel
        case .privacy:
            privacyPanel
        case .app:
            appPanel
        }
    }

    /// v0.4.25: 3-way appearance override panel. The radio-style
    /// Picker matches macOS System Settings → Appearance so the
    /// affordance reads as familiar. Auto means "follow system";
    /// Light / Dark force the override and persist across launches.
    private var appearancePanel: some View {
        @Bindable var a = appearance
        return VStack(alignment: .leading, spacing: 16) {
            sectionHeader(
                "Appearance",
                "Override the system theme. Auto follows your macOS setting; Light and Dark force the app to stay there regardless of system changes."
            )
            settingsCard {
                Picker("Theme", selection: $a.mode) {
                    ForEach(AppearanceMode.allCases) { mode in
                        Text(mode.displayName)
                            .accessibilityLabel(mode.displayName)
                            .accessibilityIdentifier(mode.accessibilityIdentifier)
                            .tag(mode)
                    }
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()
            }
        }
    }

    @ViewBuilder
    private var privacyPanel: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Privacy")
                    .font(.title2.weight(.semibold))
                Text("Rapid-MLX is local-first. Prompts, attachments, and model responses never leave your Mac. Anonymous usage data is sent only after you opt in.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Toggle(isOn: telemetryEnabledBinding) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Send anonymous usage data")
                        .font(.callout.weight(.medium))
                    Text("Versions, Mac hardware tier, public model and feature names, coarse performance, redacted crash diagnostics, and error categories. Never prompts, responses, attachments, keys, account details, or unredacted user paths.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            .accessibilityIdentifier("Settings.Privacy.TelemetryToggle")
            // The first-run consent sheet (ContentView) writes the same
            // preference, so the seeded value can be stale by the time this
            // panel is first shown...
            .onAppear { telemetryEnabled = TelemetryConfig.isEnabled }
            // ...and it can go stale *while* the panel is open: Settings can be
            // opened over the still-attached first-run sheet, and answering
            // "Share" there would otherwise leave this switch reading off while
            // telemetry is running. Re-reading on any defaults change keeps the
            // two surfaces honest without either one knowing about the other.
            //
            // `.receive(on: RunLoop.main)` is load-bearing, not ceremony:
            // `didChangeNotification` is delivered on the thread that made the
            // write, so a background write to ANY key — not just this one —
            // would otherwise mutate SwiftUI `@State` off the main thread.
            .onReceive(
                NotificationCenter.default
                    .publisher(for: UserDefaults.didChangeNotification)
                    .receive(on: RunLoop.main)
            ) { _ in
                telemetryEnabled = TelemetryConfig.isEnabled
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Where the data goes")
                    .font(.callout.weight(.medium))
                Text("telemetry.rapidmlx.com — a Cloudflare Worker that strips client IPs before writing to storage. Source is open at github.com/raullenchai/rapidmlx.com under telemetry-worker/.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // All three point at documents that exist in the repository. Two
            // of them did not: "Privacy policy" opened rapidmlx.com/privacy,
            // which 404s (the page has never been published), and
            // "Open-source credits" opened blob/main/THIRD_PARTY.md — the
            // repository ROOT — while the file has always lived one directory
            // down, under apps/rapid-mac/. Both are now pointed at the real
            // documents; ``RepositoryLinkTargetsTests`` fails the build if
            // either path stops existing. When rapidmlx.com/privacy is
            // published, the privacy link can move back to the website.
            //
            // Each link is named for the DOCUMENT it opens, not for its
            // visible label: "License (EULA)" is the kind of string that gets
            // reworded, and ``RepositoryLinkTargetsTests`` already pins the
            // destinations, so the document is the stable half.
            HStack(spacing: 12) {
                Link("Privacy policy",
                     destination: URL(string: "https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/PRIVACY.md")!)
                    .accessibilityIdentifier("Settings.Privacy.Link.PrivacyPolicy")
                Link("License (EULA)",
                     destination: URL(string: "https://github.com/raullenchai/Rapid-MLX/blob/main/LICENSE")!)
                    .accessibilityIdentifier("Settings.Privacy.Link.License")
                Link("Open-source credits",
                     destination: URL(string: "https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/THIRD_PARTY.md")!)
                    .accessibilityIdentifier("Settings.Privacy.Link.Credits")
            }
            .font(.callout)

            Spacer(minLength: 0)
        }
    }

    /// Mirrors the stored consent so SwiftUI has something to invalidate on.
    ///
    /// The getter used to read ``TelemetryConfig.isEnabled`` directly — a plain
    /// `static var` over `UserDefaults.standard`. Reading it records no
    /// dependency, so pressing the switch wrote the preference and then left
    /// the control rendering its old value: to the user, a consent switch that
    /// snaps back to off while they are in fact opted in (#1623). It only
    /// appeared to correct itself because leaving the panel and returning
    /// rebuilds the view for unrelated reasons.
    ///
    /// Seeded once and re-read in ``onAppear`` so a change made elsewhere —
    /// the first-run consent sheet writes the same key — is still reflected.
    @State private var telemetryEnabled = TelemetryConfig.isEnabled

    private var telemetryEnabledBinding: Binding<Bool> {
        Binding(
            get: { telemetryEnabled },
            set: { enabled in
                // Drive the view from the value the user just chose, then let
                // the store confirm it. Reading the preference back would
                // reintroduce the same problem the moment a write is deferred
                // or rejected.
                telemetryEnabled = enabled
                TelemetryConsent.record(enabled: enabled)
                if enabled {
                    Task { await TelemetrySession.sendStartIfNeeded() }
                }
            }
        )
    }

    /// Settings → App panel. The visible home for the existing
    /// ``UpdateChecker`` → GitHub Releases self-update flow — both
    /// the bottom-bar version chip and the MenuBarExtra menu deep-link
    /// users here to install a newer Rapid-MLX Desktop.
    ///
    /// State table:
    ///   * ``availableUpdate`` non-nil → prominent "Update Rapid-MLX
    ///     Desktop" CTA that opens the existing in-app installer
    ///     window, mirroring the MenuBarExtra menu entry. Falls
    ///     back to the GitHub Releases page when the release
    ///     payload doesn't carry a DMG URL.
    ///   * Otherwise → calm "Up to date" check + a Recheck button.
    ///     The poller also runs on a 6 h timer from ``RapidApp``;
    ///     manual recheck is for users who saw the menubar tint
    ///     change and want to confirm.
    @ViewBuilder
    private var appPanel: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Rapid-MLX")
                    .font(.title2.weight(.semibold))
                Text("Self-update for Rapid-MLX. New releases bundle the latest models, performance improvements, and bug fixes.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Divider()
            VStack(alignment: .leading, spacing: 12) {
                versionRow(
                    label: "Installed",
                    value: "v\(appUpdater.currentVersion)",
                    monospaced: true
                )
                // Only show the manifest's version when it is actually at or
                // ahead of what's installed. A manifest BEHIND the installed
                // build means the release feed is stale (or this is a dev /
                // pre-release build) — labelling an older number "Latest
                // release" reads as if the user is somehow ahead of the
                // world, or that their install is wrong.
                //
                // ``appUpdateStatus`` already models exactly this case
                // (installed strictly newer → ``.unknown``, see the truth
                // table below), but this row rendered ``latest`` directly and
                // so bypassed that judgement. Gate it on the same predicate.
                if let release = appUpdater.latest,
                   !UpdateChecker.isNewer(appUpdater.currentVersion, than: release.version) {
                    versionRow(
                        label: "Latest release",
                        value: "v\(release.version)",
                        monospaced: true
                    )
                }
            }
            Divider()
            appUpdateActionRow
            if let release = appUpdater.availableUpdate,
               !release.notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                appReleaseNotesPanel(notes: release.notes)
            }
            if let err = appUpdater.lastError {
                HStack(spacing: 6) {
                    Image(systemName: "wifi.exclamationmark")
                        .foregroundStyle(.orange)
                    Text("Last check failed: \(err)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Divider()
            diagnosticsSection
            Divider()
            dockVisibilitySection
        }
    }

    /// One-click support bundle. A resident app pulls in non-technical
    /// users; this hands them a single button that saves everything we
    /// need to debug (version + machine + sidecar state + scrubbed log
    /// tail) so a bug report arrives actionable instead of "it broke".
    @ViewBuilder
    private var diagnosticsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(
                "Diagnostics",
                "Save a support report to share if something goes wrong. Includes your app version, Mac model, and recent logs — no prompts, files, or personal data."
            )
            Button {
                DiagnosticsBundle.exportViaSavePanel(server: server)
            } label: {
                Label("Export diagnostics…", systemImage: "stethoscope")
            }
            .accessibilityIdentifier("Settings.App.ExportDiagnostics")
        }
    }

    /// #260: Settings → App "Hide Dock icon when closing window"
    /// toggle. Mirrors the persisted ``DockVisibilityPromptStore``
    /// state so the user can change their mind without re-triggering
    /// the one-time prompt; "Reset onboarding alerts" brings the
    /// prompt back so a curious user can re-see it.
    @ViewBuilder
    private var dockVisibilitySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Window")
                    .font(.title3.weight(.semibold))
                Text("Choose what happens when you close the main window. Rapid-MLX keeps running in the menu bar either way — this only affects whether the Dock icon stays visible.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Toggle(isOn: hideDockOnCloseBinding) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Hide Dock icon when closing window")
                        .font(.callout.weight(.medium))
                    Text("Closing the window flips Rapid-MLX to the menu bar. Re-open via the menu-bar icon. Turn off to keep the Dock icon visible after closing.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            .accessibilityIdentifier("Settings.App.HideDockOnCloseToggle")
            HStack {
                Spacer()
                Button("Reset onboarding alerts") {
                    dockPromptStore.resetOnboarding()
                }
                .controlSize(.small)
                .accessibilityIdentifier("Settings.App.ResetDockOnboardingCTA")
            }
        }
    }

    /// Bridges the ``DockVisibilityPromptStore`` choice into a
    /// ``Toggle``-shaped binding. Set true → ``.hideAlways``; set
    /// false → ``.keepAlways``. Both flip the live ``NSApp``
    /// activation policy via ``applyHide`` so the user sees the
    /// effect immediately — the Dock icon comes or goes without
    /// requiring a relaunch.
    private var hideDockOnCloseBinding: Binding<Bool> {
        Binding(
            get: { dockPromptStore.choice == .hideAlways },
            set: { newValue in
                dockPromptStore.setHideOnClose(newValue)
                DockVisibilityPromptStore.applyHide(newValue)
            }
        )
    }

    /// Top action row inside ``appPanel``. Three states, all
    /// non-blocking (the actual download + install runs in the
    /// dedicated ``UpdateInstallView`` scene so this row stays
    /// responsive even if the user closed the update window
    /// mid-install).
    /// Coarse status the App panel renders. Keyed off the app
    /// self-update poller's observable surface so the "Up to date"
    /// green check only fires after a check has actually established
    /// that the local version is the latest.
    /// Codex r1 P2 (Settings → App update gating): treating every
    /// nil ``availableUpdate`` as "current" lied to users whose
    /// first check was still in flight or had failed offline.
    enum AppUpdateStatus: Equatable {
        case available(version: String)
        case upToDate(version: String)
        case checking
        /// Poll succeeded but the manifest is behind the installed build.
        /// Not an error and not "unknown" — see ``resolveAppUpdateStatus``.
        case aheadOfManifest(current: String, manifest: String)
        case unknown(reason: String?)
    }

    /// Pure derivation from the ``UpdateChecker`` observable surface
    /// to ``AppUpdateStatus``. Exposed as ``static`` + parameterised
    /// so a unit test can pin the truth table without standing up a
    /// SwiftUI host.
    ///
    /// Truth table (priority top-to-bottom):
    ///   1. ``availableUpdate`` non-nil → ``.available(release)`` —
    ///      always wins; the actionable signal.
    ///   2. ``lastCheckedAt == nil`` (no check has completed yet):
    ///      either ``.checking`` (one is in flight) or
    ///      ``.unknown(lastError)`` (none in flight, possibly with
    ///      a transport error from a prior attempt).
    ///   3. A check completed AND ``latest != nil`` AND
    ///      ``latest.version == currentVersion`` →
    ///      ``.upToDate(currentVersion)``. We require *equality*, not
    ///      "not strictly newer", so a dev / pre-release build whose
    ///      ``currentVersion`` is ahead of the manifest does NOT
    ///      collapse into the reassuring "up to date" state (v0.7.4
    ///      status-bar regression).
    ///   4. A check completed and returned a manifest OLDER than the
    ///      installed build → ``.aheadOfManifest``. Distinct from
    ///      ``.unknown``: nothing failed and nothing is missing, so
    ///      telling the user to press "Check for updates" would send
    ///      them to re-run a poll that already succeeded and will keep
    ///      returning the same answer.
    ///   5. Otherwise (check completed but ``latest == nil`` — worker
    ///      errored or payload rejected) → ``.unknown(lastError)``.
    static func resolveAppUpdateStatus(
        currentVersion: String,
        availableUpdate: UpdateChecker.Release?,
        latest: UpdateChecker.Release?,
        checking: Bool,
        lastCheckedAt: Date?,
        lastError: String?
    ) -> AppUpdateStatus {
        if let release = availableUpdate {
            return .available(version: release.version)
        }
        if lastCheckedAt == nil {
            return checking ? .checking : .unknown(reason: lastError)
        }
        if let latest = latest,
           !UpdateChecker.isNewer(currentVersion, than: latest.version) {
            // A completed poll resolved a release AND our installed
            // build is not strictly newer than it → genuine
            // up-to-date. (``availableUpdate`` would have fired above
            // if ``latest`` were strictly newer than us, so the only
            // remaining case here is equality.)
            return .upToDate(version: currentVersion)
        }
        if let latest, UpdateChecker.isNewer(currentVersion, than: latest.version) {
            // The poll worked; the feed is just behind us. A dev build, or a
            // release that shipped to GitHub before `latest.json` was
            // republished. Either way it is a statement of fact, not an
            // error, and re-checking cannot change it.
            return .aheadOfManifest(
                current: currentVersion,
                manifest: latest.version
            )
        }
        // ``lastCheckedAt`` set but EITHER ``latest`` is nil (most
        // recent attempt populated ``lastError`` instead of a
        // payload) OR our installed build is strictly newer than the
        // manifest (dev / pre-release / stale manifest). Never
        // "up to date" in either case.
        return .unknown(reason: lastError)
    }

    private var appUpdateStatus: AppUpdateStatus {
        Self.resolveAppUpdateStatus(
            currentVersion: appUpdater.currentVersion,
            availableUpdate: appUpdater.availableUpdate,
            latest: appUpdater.latest,
            checking: appUpdater.checking,
            lastCheckedAt: appUpdater.lastCheckedAt,
            lastError: appUpdater.lastError
        )
    }

    @ViewBuilder
    private var appUpdateActionRow: some View {
        HStack(spacing: 12) {
            switch appUpdateStatus {
            case .available(let version):
                HStack(spacing: 6) {
                    Image(systemName: "arrow.up.circle.fill")
                        .foregroundStyle(RapidTheme.brand)
                    Text("Update available — v\(version)")
                        .font(.callout.weight(.semibold))
                        .accessibilityIdentifier("Settings.App.UpdateHeadline")
                }
                Spacer()
                // Codex r1 P2 (CTA never disabled): the CTA only
                // opens the existing update-install scene — leaving
                // it tappable while ``appInstaller.isRunning`` lets
                // the user reopen the progress window if they
                // accidentally closed it mid-download. ``Updating…``
                // copy makes the live state obvious.
                Button {
                    appOpenUpdateWindow()
                } label: {
                    if appInstaller.isRunning {
                        Label("Updating…", systemImage: "arrow.down.circle")
                    } else {
                        Label("Update Rapid-MLX", systemImage: "arrow.down.circle.fill")
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .accessibilityIdentifier("Settings.App.UpdateCTA")
            case .upToDate(let version):
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text("Up to date — v\(version) is the latest release.")
                        .font(.callout)
                        .foregroundStyle(.primary)
                        .accessibilityIdentifier("Settings.App.UpToDate")
                }
                Spacer()
                appUpdateRecheckButton
            case .checking:
                HStack(spacing: 6) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Checking for updates…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("Settings.App.Checking")
                }
                Spacer()
                appUpdateRecheckButton
            case .aheadOfManifest(let current, _):
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle")
                        .foregroundStyle(.secondary)
                    Text("Up to date — v\(current).")
                        .font(.callout)
                        .foregroundStyle(.primary)
                        .accessibilityIdentifier("Settings.App.AheadOfManifest")
                }
                Spacer()
                // No re-check button. The poll already succeeded; the feed is
                // simply behind this build, and pressing it again returns the
                // same answer. An action that provably cannot change anything
                // is worse than no action — it invites the user to keep
                // trying and reads as if something is wrong.
            case .unknown(let reason):
                HStack(spacing: 6) {
                    Image(systemName: "questionmark.circle")
                        .foregroundStyle(.secondary)
                    Text(reason == nil
                        ? "Update status unknown — press Check for updates."
                        : "Update status unknown — last check failed.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("Settings.App.Unknown")
                }
                Spacer()
                appUpdateRecheckButton
            }
        }
    }

    /// Shared Recheck button — same shape across the "up to date",
    /// "checking", and "unknown" cases so a future "you should
    /// check again" copy nudge only needs one site.
    @ViewBuilder
    private var appUpdateRecheckButton: some View {
        Button {
            Task { await appUpdater.check() }
        } label: {
            if appUpdater.checking {
                Label("Checking…", systemImage: "arrow.clockwise")
            } else {
                Label("Check for updates", systemImage: "arrow.clockwise")
            }
        }
        .controlSize(.small)
        .disabled(appUpdater.checking)
        .accessibilityIdentifier("Settings.App.RecheckCTA")
    }

    /// Release-notes preview inside ``appPanel``. The dedicated
    /// ``UpdateInstallView`` window shows the full notes; here we
    /// render an inline scroll-bounded preview so the user can see
    /// "what's new" without opening another window.
    private func appReleaseNotesPanel(notes: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Release notes")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
            ScrollView(.vertical, showsIndicators: true) {
                Text(notes)
                    .scaledSystemFont(12)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .frame(maxHeight: 140)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color.secondary.opacity(0.08))
            )
        }
    }

    /// Open the in-app installer window — same window the
    /// MenuBarExtra menu drives. Wrapped here so the panel reads
    /// cleanly. v0.5.4 documented the runloop-tick workaround for
    /// ``openWindow`` against a never-instantiated scene; we
    /// mirror the same pattern (small sleep + activate first)
    /// for the same reason.
    private func appOpenUpdateWindow() {
        Task { @MainActor in
            NSApp.activate(ignoringOtherApps: true)
            try? await Task.sleep(nanoseconds: 50_000_000)
            openWindow(id: "update-install")
        }
    }

    private func versionRow(label: String, value: String, monospaced: Bool) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(label)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
                .frame(width: 120, alignment: .leading)
            Text(value)
                .font(monospaced
                    ? .system(size: 12, design: .monospaced)
                    : .system(size: 12))
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
        }
    }

    // MARK: - Shared layout (v0.5 card refresh)

    /// Section header: a title + an optional descriptive subtitle,
    /// rendered above a card. Keeps the heading typography consistent
    /// across every panel and softer than a bare ``.title2`` slammed
    /// onto the content.
    @ViewBuilder
    private func sectionHeader(_ title: String, _ subtitle: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.title3.weight(.semibold))
            if let subtitle {
                Text(subtitle)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// Rounded "card" container for a group of settings controls. A
    /// near-white fill on a light canvas, a hairline border, and a
    /// generous inset — the Linear / Apple-System-Settings look the
    /// v0.5 refresh targets. Every panel routes its controls through
    /// this so the Settings window reads as a set of calm cards.
    @ViewBuilder
    private func settingsCard<Content: View>(
        padding: CGFloat = 16,
        @ViewBuilder _ content: () -> Content
    ) -> some View {
        content()
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(padding)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                    .fill(RapidTheme.card)
            )
            .clipShape(RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                    .stroke(RapidTheme.hairline, lineWidth: 1)
            )
    }

    /// Same glyph mapping as the compose-bar popover. Kept here
    /// in duplicate (rather than refactored to a shared helper)
    /// because v0.4.1 is the only commit they live side by side
    /// in — the compose popover may collapse into a "quick
    /// status" indicator in v0.5 once Settings is the canonical
    /// surface, at which point the duplicate goes away.
    private func glyph(for name: String) -> String {
        switch name {
        case "read_file", "list_directory": return "folder"
        case "write_file": return "square.and.pencil"
        case "edit_file": return "pencil.and.list.clipboard"
        case "run_command": return "terminal"
        case "web_search": return "magnifyingglass"
        case "calculator": return "function"
        case "weather": return "cloud.sun"
        case "current_datetime": return "clock"
        default: return "wrench.and.screwdriver"
        }
    }
}
