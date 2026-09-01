import AppKit
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
    @Environment(CustomInstructionsConfig.self) private var customInstructions
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
    @Environment(SparkleUpdateController.self) private var sparkleUpdater
    @Environment(\.dismissWindow) private var dismissWindow
    /// #260: persisted "hide Dock icon on close" choice. Settings →
    /// App surfaces a toggle so the user can change their mind
    /// without re-triggering the one-time prompt, plus a "Reset
    /// onboarding alerts" affordance that brings the prompt back.
    @Environment(DockVisibilityPromptStore.self) private var dockPromptStore
    @Environment(QuickstartCoordinator.self) private var quickstart
    @Environment(DeferredTelemetryConsentCoordinator.self) private var deferredTelemetryConsent
    @State private var confirmingSetupRestart = false
    @State private var restartingSetup = false

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
        /// Global system-prompt default applied before any conversation-specific
        /// override. Stored locally in UserDefaults under the legacy key.
        case instructions
        /// Persistent memory entries learned from conversations.
        case memory
        /// Built-in tools the model can call: on/off per tool, the
        /// web-search backend + key, and the browse approval mode.
        case tools
        /// Issue #1716 — MCP connectors: which servers the engine runs, the
        /// tools they expose, and the per-tool consent record. Separate from
        /// ``tools`` because the two are different in kind: built-in tools
        /// ship with the app and are audited by us, connectors are programs
        /// the user installs and points us at.
        case connectors
        /// Issue #1717 — per-model engine performance: KV-cache precision,
        /// prefix caching, cache budget. Deliberately NOT folded into
        /// ``modelManagement``'s sampling controls: those shape what the model
        /// says, these shape how fast (and, for some choices, whether) it says
        /// the same thing. Mixing them would invite the "I moved a slider and
        /// quality changed" confusion the issue is written to avoid.
        case performance
        case appearance
        case privacy
        /// Rapid-MLX Desktop app updates. The .app self-update is the
        /// only correct way to bump the bundled engine.
        case app
        #if DEBUG
        /// Debug builds only — state-erasing actions for rehearsing flows
        /// that are one-shot per Mac, chiefly Quickstart. Compiled out of
        /// release entirely rather than hidden at the rail, so the copy and
        /// the erase logic cannot ship even unreachable.
        case developer
        #endif

        var id: String { rawValue }
        var title: String {
            switch self {
            case .modelManagement: return "Model Management"
            case .instructions: return "System Prompt"
            case .memory: return "Memory"
            case .tools: return "Tools"
            case .connectors: return "Connectors"
            case .performance: return "Performance"
            case .appearance: return "Appearance"
            case .privacy: return "Privacy"
            case .app: return "App"
            #if DEBUG
            case .developer: return "Developer"
            #endif
            }
        }
        var iconName: String {
            switch self {
            case .modelManagement: return "externaldrive.fill"
            case .instructions: return "text.bubble.fill"
            case .memory: return "brain"
            case .tools: return "wrench.and.screwdriver.fill"
            case .connectors: return "powerplug.fill"
            case .performance: return "speedometer"
            case .appearance: return "paintpalette.fill"
            case .privacy: return "lock.shield.fill"
            case .app: return "app.badge.fill"
            #if DEBUG
            case .developer: return "hammer.fill"
            #endif
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
                        categoryRowContent(cat, isSelected: selection.selected == cat)
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
            .frame(width: SettingsView.railWidth)
            .background(RapidTheme.surfaceSidebar)
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

        /// One rail row: glyph and title, both wearing the SAME colour.
        ///
        /// Deliberately an explicit `HStack`, not a `Label`. A `Label`
        /// inside `List(.sidebar)` hands its icon to the list style,
        /// which paints it with the sidebar's own item colour — so a
        /// `.foregroundStyle` on the row reached the title and left the
        /// glyph white. The selected row therefore read as an amber word
        /// beside a white icon, which is not a state any other list in
        /// the app produces.
        ///
        /// ``symbolRenderingMode(.monochrome)`` is the second half: some
        /// of these glyphs (`paintpalette.fill` especially) have
        /// multicolour variants that ignore a foreground style entirely
        /// unless the rendering mode is pinned.
        @ViewBuilder
        private func categoryRowContent(_ cat: Category, isSelected: Bool) -> some View {
            let tint = isSelected ? RapidTheme.brandPrimaryDeep : RapidTheme.textPrimary
            HStack(spacing: RapidTheme.Space.sm) {
                Image(systemName: cat.iconName)
                    .symbolRenderingMode(.monochrome)
                    .font(.system(size: 13, weight: .medium))
                    .frame(width: RapidTheme.Layout.iconSlot, alignment: .center)
                Text(cat.title)
                    .font(RapidFont.body)
                    .fontWeight(isSelected ? .semibold : .regular)
                    // Reserve the selected weight so the row does not
                    // reflow as the selection moves.
                    .background(
                        Text(cat.title)
                            .font(RapidFont.body)
                            .fontWeight(.semibold)
                            .hidden()
                    )
                    .lineLimit(1)
            }
            .foregroundStyle(tint)
            .frame(
                maxWidth: .infinity,
                minHeight: RapidTheme.ControlHeight.row,
                alignment: .leading
            )
            .contentShape(Rectangle())
        }

        @ViewBuilder
        private func categoryRowBackground(isSelected: Bool, isHovered: Bool) -> some View {
            if isSelected {
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .fill(RapidTheme.brandPrimaryTint)
                    .padding(.horizontal, RapidTheme.Space.xs)
                    .padding(.vertical, RapidTheme.Space.xxs)
            } else if isHovered {
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .fill(RapidTheme.hoverFill)
                    .padding(.horizontal, RapidTheme.Space.xs)
                    .padding(.vertical, RapidTheme.Space.xxs)
            } else {
                Color.clear
            }
        }
    }

    /// Width of the category rail. Fixed, like every macOS settings
    /// sidebar — at the 720pt window floor this still leaves 519pt of
    /// detail, which is more than the widest panel needs.
    // `nonisolated`: read by the nonisolated layout arithmetic below. An
    // immutable Sendable constant is isolation-free on every compiler; the
    // CI toolchain (stricter default isolation than the local one) otherwise
    // rejects the reads.
    nonisolated static let railWidth: CGFloat = 200

    /// Width below which the detail column is "compact" and panels
    /// should shed optional columns rather than clip them.
    ///
    /// Derived, not guessed: Model Management's table commits a 158pt
    /// meters column plus a 124pt size column plus ~90pt of glyphs and
    /// gaps before the model name gets a single point. Under ~520pt of
    /// content the name is squeezed to nothing, so that is where the
    /// meters have to go.
    nonisolated static let compactContentWidth: CGFloat = 520

    /// The width a panel's content column actually gets inside a window
    /// `windowWidth` points wide.
    ///
    /// Pure, and `static`, because the responsive contract is a claim
    /// about arithmetic — "nothing this window commits to is wider than
    /// the window" — and that claim should be checkable without
    /// rendering anything. ``SettingsResponsiveLayoutTests`` walks the
    /// supported sizes through this and the table's committed widths.
    nonisolated static func contentColumnWidth(forWindowWidth windowWidth: CGFloat) -> CGFloat {
        // rail + the 1pt divider between rail and detail
        let detail = windowWidth - railWidth - 1
        let available = detail - RapidTheme.Space.xl * 2
        return max(0, min(available, RapidTheme.Layout.pageMaxWidth))
    }

    /// Whether a window that wide puts panels into their compact layout.
    nonisolated static func isCompact(forWindowWidth windowWidth: CGFloat) -> Bool {
        contentColumnWidth(forWindowWidth: windowWidth) < compactContentWidth
    }

    /// The narrowest window Settings supports. Paired with
    /// ``minWindowHeight`` on the shell's `.frame(minWidth:minHeight:)`,
    /// and re-used by the tests so the floor is stated once.
    static let minWindowWidth: CGFloat = 720
    static let minWindowHeight: CGFloat = 480

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
            // The column reads its own width and publishes a compact
            // flag, so a panel adapts to the space it actually has
            // instead of to a guess about the window. Long pages scroll;
            // nothing is solved by raising the window floor.
            GeometryReader { proxy in
                let available = proxy.size.width - RapidTheme.Space.xl * 2
                let column = max(0, min(available, RapidTheme.Layout.pageMaxWidth))
                ScrollView {
                    content(selected)
                        .frame(maxWidth: column, alignment: .leading)
                        .padding(RapidTheme.Space.xl)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                }
                .environment(\.settingsContentIsCompact, column < SettingsView.compactContentWidth)
            }
            .background(RapidTheme.surfaceCanvas)
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
        .frame(minWidth: Self.minWindowWidth, minHeight: Self.minWindowHeight)
        .safeAreaInset(edge: .top, spacing: 0) {
            if router.quickstartCatalogReturnPending {
                VStack(spacing: 0) {
                    HStack {
                        QuickstartBackButton { window in
                            dismissWindow(id: "settings")
                            window.close()
                            DispatchQueue.main.async {
                                router.completeQuickstartCatalogRoundTrip()
                            }
                        }
                        .frame(width: 130, height: 28)
                        Spacer()
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    Divider()
                }
            }
        }
        .onAppear {
            // v0.4.37: consume any pending deep-link request. Fires when the
            // Settings scene is created (typical for the first open this app
            // session). The onChange below handles an already-open window.
            consumeRouterRequest()
        }
        .onChange(of: router.requestedCategory) { _, _ in
            consumeRouterRequest()
        }
        .onDisappear {
            // onDisappear fires while AppKit is still closing the window.
            // Restoring the onboarding sheet synchronously would start a new
            // modal session and trap the Settings window mid-close.
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 200_000_000)
                if router.quickstartCatalogReturnPending {
                    router.completeQuickstartCatalogRoundTrip()
                }
            }
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
        case .instructions:
            instructionsPanel
        case .memory:
            SettingsMemoryPanel()
        case .tools:
            SettingsToolsPanel()
        case .connectors:
            SettingsConnectorsPanel()
        case .performance:
            SettingsPerformancePanel()
        case .appearance:
            appearancePanel
        case .privacy:
            privacyPanel
        case .app:
            appPanel
        #if DEBUG
        case .developer:
            SettingsDeveloperPanel()
        #endif
        }
    }

    private var instructionsPanel: some View {
        @Bindable var config = customInstructions
        return VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            SectionHeader(
                "System Prompt",
                subtitle: "Sent as a system message with every conversation. Conversation prompts can override it.",
                emphasis: .page
            )
            InstructionEditorSection(
                "Global default",
                subtitle: "Used when a conversation has no conflicting prompt. Stored only on this Mac.",
                clearEnabled: CustomInstructionsConfig.normalized(config.global) != nil,
                onClear: { config.global = "" }
            ) {
                InstructionTextEditor(
                    text: $config.global,
                    placeholder: "For example: Answer concisely, use plain language, and include code examples when useful.",
                    height: 172,
                    accessibilityIdentifier: "Settings.Instructions.GlobalEditor"
                )
            }
            EffectiveSystemPromptDisclosure(
                global: config.global,
                conversation: "",
                accessibilityIdentifier: "Settings.SystemPrompt.EffectivePreview"
            )
        }
    }

    /// v0.4.25: 3-way appearance override panel. The radio-style
    /// Picker matches macOS System Settings → Appearance so the
    /// affordance reads as familiar. Auto means "follow system";
    /// Light / Dark force the override and persist across launches.
    private var appearancePanel: some View {
        @Bindable var a = appearance
        return VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            SectionHeader(
                "Appearance",
                subtitle: "Override the system theme. Auto follows your macOS setting; Light and Dark force the app to stay there regardless of system changes.",
                emphasis: .page
            )
            SettingsSection {
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
                .accessibilityIdentifier("Settings.Appearance.ThemePicker")
            }
        }
    }

    @ViewBuilder
    private var privacyPanel: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            SectionHeader(
                "Privacy",
                subtitle: "Rapid-MLX is local-first. Prompts, attachments, and model responses never leave your Mac. Anonymous usage data is sent only after you opt in.",
                emphasis: .page
            )

            SettingsSection {
            Toggle(isOn: telemetryEnabledBinding) {
                SettingsRowLabel(
                    title: "Send anonymous usage data",
                    description: "Versions, Mac hardware tier, public model and feature names, coarse performance, redacted crash diagnostics, and error categories. For each first successful text chat reply, dictation, or generated image, only the milestone name and “Desktop” are sent. This version does not send a vision-reply milestone. The collector derives a country code but never stores your IP. Never prompts, responses, attachments, keys, account details, or unredacted user paths."
                )
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            .accessibilityIdentifier("Settings.Privacy.TelemetryToggle")
            // The post-value consent invitation writes the same
            // preference, so the seeded value can be stale by the time this
            // panel is first shown...
            .onAppear { telemetryEnabled = TelemetryConfig.isEnabled }
            // ...and it can go stale *while* the panel is open: Settings can be
            // opened while the invitation is visible, and answering
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

            SettingsRowDivider()

            SettingsRowLabel(
                title: "Where the data goes",
                description: "telemetry.rapidmlx.com — a Cloudflare Worker that strips client IPs before writing to storage. Source is open at github.com/raullenchai/rapidmlx.com under telemetry-worker/."
            )
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
            HStack(spacing: RapidTheme.Space.lg) {
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
            .font(RapidFont.body)
            // `.foregroundStyle`, not `.tint`: a SwiftUI ``Link`` paints
            // itself with the system link colour and ignores the ambient
            // tint, so these three rendered in macOS blue while every
            // other link in the app used the steel-blue link token.
            .foregroundStyle(RapidTheme.linkLabel)

            Link("Powered by MTPLX",
                 destination: URL(string: "https://github.com/youssofal/mtplx")!)
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.linkLabel)
                .accessibilityIdentifier("Settings.Privacy.Link.MTPLX")

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
    /// the post-value consent invitation writes the same key — is still reflected.
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
                deferredTelemetryConsent.settingsChanged(enabled: enabled)
            }
        )
    }

    /// Settings → App panel. The visible home for the existing
    /// ``UpdateChecker`` → GitHub Releases self-update flow — both
    /// the bottom-bar version chip and the MenuBarExtra menu deep-link
    /// users here to install a newer Rapid-MLX Desktop.
    ///
    /// State table:
    ///   * ``availableUpdate`` non-nil → prominent "Update Rapid-MLX"
    ///     CTA that hands off to Sparkle's update panel, mirroring the
    ///     MenuBarExtra menu entry. Disabled on unsigned builds where
    ///     Sparkle is not configured.
    ///   * Otherwise → calm "Up to date" check + a Recheck button.
    ///     The status is refreshed once per launch from ``RapidApp``;
    ///     manual recheck is how a user refreshes it without relaunching.
    @ViewBuilder
    private var appPanel: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            SectionHeader(
                "Rapid-MLX",
                subtitle: "Self-update for Rapid-MLX. New releases bundle the latest models, performance improvements, and bug fixes.",
                emphasis: .page
            )
            SettingsSection("Version") {
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
                    SettingsRowDivider()
                    versionRow(
                        label: "Latest release",
                        value: "v\(release.version)",
                        monospaced: true
                    )
                }
            }

            SettingsSection("Updates") {
                Toggle(isOn: automaticUpdateBinding) {
                    Text("Automatically download updates")
                }
                .accessibilityIdentifier("Settings.App.AutomaticUpdatesToggle")
                    .disabled(!sparkleUpdater.isEnabled)
                    .help(sparkleUpdater.isEnabled
                          ? "Downloaded updates are installed when Rapid-MLX quits."
                          : "Automatic updates are enabled in signed release builds.")
                SettingsRowDivider()
                appUpdateActionRow
                if let release = appUpdater.availableUpdate,
                   !release.notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    SettingsRowDivider()
                    appReleaseNotesPanel(notes: release.notes)
                }
            }
            if let err = appUpdater.lastError {
                InlineNotice(
                    message: "Last check failed: \(err)",
                    tone: .warning
                )
            }

            diagnosticsSection
            setupSection
            dockVisibilitySection
        }
        .confirmationDialog(
            ReonboardingReset.confirmation(for: .onboarding).title,
            isPresented: $confirmingSetupRestart,
            titleVisibility: .visible
        ) {
            Button("Restart into setup", role: .destructive) { restartIntoSetup() }
                .accessibilityIdentifier("Settings.App.ConfirmRunSetupAgain")
            Button("Cancel", role: .cancel) { confirmingSetupRestart = false }
                .accessibilityIdentifier("Settings.App.CancelRunSetupAgain")
        } message: {
            Text(ReonboardingReset.confirmation(for: .onboarding).message)
        }
    }

    @ViewBuilder
    private var setupSection: some View {
        SettingsSection(
            "Setup",
            subtitle: "Run the guided model setup again. Your settings, conversations, downloaded models, and telemetry choice stay untouched."
        ) {
            HStack {
                Spacer(minLength: 0)
                Button("Run setup again…") { confirmingSetupRestart = true }
                    .buttonStyle(.rapidSecondaryCompact)
                    .disabled(restartingSetup)
                    .accessibilityIdentifier("Settings.App.RunSetupAgain")
            }
        }
    }

    private func restartIntoSetup() {
        restartingSetup = true
        Task { @MainActor in
            await ReonboardingReset.perform(scope: .onboarding, quickstart: quickstart)
            ReonboardingReset.relaunch()
        }
    }

    /// One-click support bundle. A resident app pulls in non-technical
    /// users; this hands them a single button that saves everything we
    /// need to debug (version + machine + sidecar state + scrubbed log
    /// tail) so a bug report arrives actionable instead of "it broke".
    @ViewBuilder
    private var diagnosticsSection: some View {
        // Copy is unchanged from before the migration: the explanation
        // moves into the section subtitle (where every other panel puts
        // it) and the button keeps its exact original label.
        SettingsSection(
            "Diagnostics",
            subtitle: "Save a support report to share if something goes wrong. Includes your app version, Mac model, and recent logs — no prompts, files, or personal data."
        ) {
            HStack {
                Button {
                    DiagnosticsBundle.exportViaSavePanel(server: server)
                } label: {
                    Label("Export diagnostics…", systemImage: "stethoscope")
                }
                .buttonStyle(.rapidSecondaryCompact)
                .accessibilityIdentifier("Settings.App.ExportDiagnostics")
                Spacer(minLength: 0)
            }
        }
    }

    /// #260: Settings → App "Hide Dock icon when closing window"
    /// toggle. Mirrors the persisted ``DockVisibilityPromptStore``
    /// state so the user can change their mind without re-triggering
    /// the one-time prompt; "Reset onboarding alerts" brings the
    /// prompt back so a curious user can re-see it.
    @ViewBuilder
    private var dockVisibilitySection: some View {
        SettingsSection(
            "Window",
            subtitle: "Choose what happens when you close the main window. Rapid-MLX keeps running in the menu bar either way — this only affects whether the Dock icon stays visible."
        ) {
            Toggle(isOn: hideDockOnCloseBinding) {
                SettingsRowLabel(
                    title: "Hide Dock icon when closing window",
                    description: "On close, Rapid-MLX stays available from the menu bar. Turn off to keep the Dock icon visible; disabling takes effect immediately."
                )
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            .accessibilityLabel("Hide Dock icon when closing window")
            .accessibilityHint("Rapid-MLX remains available from the menu bar.")
            .accessibilityIdentifier("Settings.App.HideDockOnCloseToggle")

            SettingsRowDivider()

            HStack {
                Spacer(minLength: 0)
                // Label unchanged; only the button tier and the
                // container around it moved onto the shared system.
                Button("Reset onboarding alerts") {
                    dockPromptStore.resetOnboarding()
                }
                .buttonStyle(.rapidSecondaryCompact)
                .accessibilityIdentifier("Settings.App.ResetDockOnboardingCTA")
            }
        }
    }

    /// Bridges the ``DockVisibilityPromptStore`` choice into a
    /// ``Toggle``-shaped binding. Set true → ``.hideAlways``; set
    /// false → ``.keepAlways``. Enabling applies on the next close, matching
    /// the control's label. Disabling restores the Dock icon immediately so
    /// a currently menu-bar-only app never traps the user in accessory mode.
    private var hideDockOnCloseBinding: Binding<Bool> {
        Binding(
            get: { dockPromptStore.choice == .hideAlways },
            set: { newValue in
                dockPromptStore.setHideOnClose(newValue)
                if !newValue {
                    DockVisibilityPromptStore.applyHide(false)
                }
            }
        )
    }

    /// Top action row inside ``appPanel``. Three states, all
    /// non-blocking: the CTA only hands off to Sparkle, which drives the
    /// download and install behind its own panel.
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
        // `.top` rather than centre: at the 720pt floor the status
        // sentences wrap to two lines, and a centred trailing button
        // then drifts down the row as the text grows. Anchoring to the
        // first line keeps the control still while the text reflows.
        HStack(alignment: .top, spacing: RapidTheme.Space.md) {
            switch appUpdateStatus {
            case .available(let version):
                statusLine(
                    symbol: "arrow.up.circle.fill",
                    tint: RapidTheme.statusWorking,
                    text: "Update available — v\(version)",
                    emphasised: true,
                    identifier: "Settings.App.UpdateHeadline"
                )
                Spacer(minLength: RapidTheme.Space.sm)
                // Hands off to Sparkle's own update panel, which owns the
                // download/verify/install-on-quit UI.
                //
                // Unsigned builds have no Sparkle key and, since the in-app
                // installer was removed, no install path at all — so a bare
                // disabled button would be a dead end on the one screen the
                // tray sends those users to. Offer the release page instead,
                // through the same allowlist check the missing-runtime overlay
                // uses, so a compromised manifest cannot turn this into a
                // phishing redirect.
                if sparkleUpdater.isEnabled {
                    if sparkleUpdater.canCheckForUpdates {
                        Button {
                            sparkleUpdater.checkForUpdates()
                        } label: {
                            Label("Update Rapid-MLX", systemImage: "arrow.down.circle.fill")
                        }
                        .buttonStyle(.rapidPrimary)
                        .help("Opens the updater to download and install this release.")
                        .accessibilityIdentifier("Settings.App.UpdateCTA")
                    } else {
                        HStack(spacing: RapidTheme.Space.sm) {
                            ProgressView()
                                .controlSize(.small)
                            Text("Update in progress…")
                                .font(RapidFont.bodyEmphasis)
                        }
                        .foregroundStyle(RapidTheme.textSecondary)
                        .help("Rapid-MLX is checking or downloading the update in the background.")
                        .accessibilityElement(children: .combine)
                        .accessibilityIdentifier("Settings.App.UpdateBusy")
                    }
                } else if let releaseURL = ContentView.missingOverlayDownloadURL(
                    for: appUpdater.availableUpdate
                ) {
                    Button {
                        NSWorkspace.shared.open(releaseURL)
                    } label: {
                        Label("Download from the release page", systemImage: "arrow.up.right.square")
                    }
                    .buttonStyle(.rapidPrimary)
                    .help("In-app updates need a signed release build; download this version manually.")
                    .accessibilityIdentifier("Settings.App.UpdateCTA")
                }
            case .upToDate(let version):
                statusLine(
                    symbol: "checkmark.circle.fill",
                    tint: RapidTheme.statusReady,
                    text: "Up to date — v\(version) is the latest release.",
                    identifier: "Settings.App.UpToDate"
                )
                Spacer(minLength: RapidTheme.Space.sm)
                appUpdateRecheckButton
            case .checking:
                HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.sm) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Checking for updates…")
                        .font(RapidFont.body)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("Settings.App.Checking")
                }
                Spacer(minLength: RapidTheme.Space.sm)
                appUpdateRecheckButton
            case .aheadOfManifest(let current, _):
                statusLine(
                    symbol: "checkmark.circle",
                    tint: RapidTheme.textSecondary,
                    text: "Up to date — v\(current).",
                    identifier: "Settings.App.AheadOfManifest"
                )
                Spacer(minLength: RapidTheme.Space.sm)
                // No re-check button. The poll already succeeded; the feed is
                // simply behind this build, and pressing it again returns the
                // same answer. An action that provably cannot change anything
                // is worse than no action — it invites the user to keep
                // trying and reads as if something is wrong.
            case .unknown(let reason):
                statusLine(
                    symbol: "questionmark.circle",
                    tint: RapidTheme.textSecondary,
                    text: reason == nil
                        ? "Update status unknown — press Check for updates."
                        : "Update status unknown — last check failed.",
                    secondary: true,
                    identifier: "Settings.App.Unknown"
                )
                Spacer(minLength: RapidTheme.Space.sm)
                appUpdateRecheckButton
            }
        }
    }

    /// One "glyph + sentence" status line, shared by the four update
    /// states that render one. They were four near-identical `HStack`s
    /// with four hand-picked colours (`RapidTheme.brand`, `.green`,
    /// `.secondary`, `.secondary`); folding them into one helper is what
    /// makes "status colours are semantic" enforceable rather than
    /// aspirational.
    @ViewBuilder
    private func statusLine(
        symbol: String,
        tint: Color,
        text: String,
        emphasised: Bool = false,
        secondary: Bool = false,
        identifier: String
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.sm) {
            Image(systemName: symbol)
                .foregroundStyle(tint)
                .accessibilityHidden(true)
            Text(text)
                .font(emphasised ? RapidFont.bodyEmphasis : RapidFont.body)
                .foregroundStyle(secondary ? RapidTheme.textSecondary : RapidTheme.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier(identifier)
        }
    }

    /// Shared Recheck button — same shape across the "up to date",
    /// "checking", and "unknown" cases so a future "you should
    /// check again" copy nudge only needs one site.
    @ViewBuilder
    private var appUpdateRecheckButton: some View {
        Button {
            if sparkleUpdater.isEnabled {
                sparkleUpdater.checkForUpdates()
            } else {
                Task { await appUpdater.check() }
            }
        } label: {
            if appUpdater.checking {
                Label("Checking…", systemImage: "arrow.clockwise")
            } else {
                Label("Check for updates", systemImage: "arrow.clockwise")
            }
        }
        .buttonStyle(.rapidSecondaryCompact)
        .disabled(sparkleUpdater.isEnabled
                  ? !sparkleUpdater.canCheckForUpdates
                  : appUpdater.checking)
        .accessibilityIdentifier("Settings.App.RecheckCTA")
    }

    private var automaticUpdateBinding: Binding<Bool> {
        Binding(
            get: {
                sparkleUpdater.automaticallyDownloadsUpdates
            },
            set: { enabled in
                sparkleUpdater.setAutomaticallyDownloadsUpdates(enabled)
            }
        )
    }

    /// Release-notes preview inside ``appPanel``. Sparkle's own update
    /// panel shows the notes embedded in the appcast; here we render an
    /// inline scroll-bounded preview so the user can see "what's new"
    /// without starting an update.
    private func appReleaseNotesPanel(notes: String) -> some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            SectionHeader("Release notes")
            ScrollView(.vertical, showsIndicators: true) {
                Text(notes)
                    .scaledSystemFont(12)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(RapidTheme.Space.sm)
            }
            .frame(maxHeight: 140)
            // Recessed, not a second card: notes are inset INTO the
            // Updates section, so they use the code/inset ground rather
            // than another raised surface on top of a raised surface.
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.code, style: .continuous)
                    .fill(RapidTheme.surfaceCode)
            )
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// A label/value pair inside the Version section.
    ///
    /// The label column is a shared constant rather than a literal so the
    /// two rows line up, and it is a `maxWidth` rather than a hard
    /// `width`: at the 720pt floor a fixed 120pt label column plus a
    /// long version string was the row most likely to overrun.
    private func versionRow(label: String, value: String, monospaced: Bool) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.md) {
            Text(label)
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.textSecondary)
                .frame(maxWidth: 120, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
            Text(value)
                .font(monospaced ? RapidFont.metric : RapidFont.secondary)
                .foregroundStyle(RapidTheme.textPrimary)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 0)
        }
    }

    // NOTE: three private helpers used to sit here and were removed by
    // the UI-1 migration:
    //
    //   * ``sectionHeader(_:_:)`` and ``settingsCard(padding:_:)`` — one
    //     of four copies of the same two helpers across the Settings
    //     panels. Both are now ``SectionHeader`` + ``SettingsSection``.
    //   * ``glyph(for:)`` — dead since the tools list moved to
    //     ``SettingsToolsPanel`` (which has its own, live copy). It
    //     mapped eight tool names, five of which no longer exist.
}

/// AppKit-backed because AXPress on a SwiftUI button in a secondary Window can
/// report success without invoking its closure while another app window remains
/// main. NSButton's target/action contract is deterministic for both people and
/// Accessibility-first automation.
private struct QuickstartBackButton: NSViewRepresentable {
    let action: (NSWindow) -> Void

    final class Coordinator: NSObject {
        var action: (NSWindow) -> Void

        init(action: @escaping (NSWindow) -> Void) {
            self.action = action
        }

        @MainActor @objc func invoke(_ sender: NSButton) {
            guard let window = sender.window else { return }
            action(window)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(action: action)
    }

    func makeNSView(context: Context) -> NSButton {
        let button = NSButton(title: "Back to setup", target: context.coordinator, action: #selector(Coordinator.invoke(_:)))
        button.bezelStyle = .rounded
        button.image = NSImage(systemSymbolName: "chevron.left", accessibilityDescription: nil)
        button.imagePosition = .imageLeading
        button.setAccessibilityIdentifier("Settings.BackToQuickstart")
        return button
    }

    func updateNSView(_ button: NSButton, context: Context) {
        context.coordinator.action = action
    }
}
