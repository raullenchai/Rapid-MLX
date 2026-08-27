import SwiftUI

/// Settings → Tools. Owns everything about the built-in tools the model can
/// call:
///
///   * Per-tool on/off switches (persisted in ``UserDefaults`` by
///     ``ChatViewModel.setToolEnabled``). A disabled tool is stripped from the
///     request body AND refused at dispatch, so a model that names it anyway
///     gets a clean error instead of a silent run.
///   * The ``web_search`` backend + its API key. Keys live in the Keychain
///     (``WebSearchConfig``), never in UserDefaults.
///   * The ``browse`` approval mode — per-fetch prompt (default) or
///     auto-approve for unattended use.
struct SettingsToolsPanel: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(WebSearchConfig.self) private var webSearch
    @Environment(BrowseApprovalStore.self) private var browseApproval

    /// Draft of the API key field. Committed on Return or Save so we don't
    /// write to the Keychain on every keystroke.
    @State private var keyDraft: String = ""
    @State private var keyDraftEdited: Bool = false
    @State private var saveFeedback: SettingsView.WebSearchKeySaveFeedback?
    @State private var feedbackGeneration: Int = 0
    @FocusState private var focusedKeyProvider: WebSearchProvider?
    /// Which tool rows have their technical detail open. Session state:
    /// a disclosure is a reading aid, not a preference, so it
    /// deliberately does not persist.
    @State private var expandedTools: Set<String>

    /// - Parameter initiallyExpanded: seam for the dev snapshot harness,
    ///   which has to capture the disclosure OPEN and cannot drive
    ///   private `@State` from outside. Production always uses the
    ///   default (all rows collapsed).
    init(initiallyExpanded: Set<String> = []) {
        _expandedTools = State(initialValue: initiallyExpanded)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            SectionHeader(
                "Tools",
                subtitle: "Tools the model can call during a chat. Turn one off and it is never offered — and never runs, even if the model asks for it by name.",
                emphasis: .page
            )
            toolsSection
            webSearchSection
            browseSection
        }
    }

    // MARK: - Available tools

    private var toolsSection: some View {
        SettingsSection("Available tools") {
            let definitions = chat.builtinDefinitions
            ForEach(Array(definitions.enumerated()), id: \.element.function.name) { index, def in
                if index > 0 { SettingsRowDivider() }
                toolRow(def)
            }
        }
    }

    /// One built-in tool.
    ///
    /// The row leads with a HUMAN name, not the wire identifier. Before
    /// this it was headed `web_search` in a monospaced face — an
    /// implementation detail presented as a title — followed by the
    /// model-facing prompt text, which runs to seven lines for `browse`
    /// and reads as documentation rather than as a setting.
    ///
    /// So: a display name, a one-line summary of what turning it off
    /// costs, and the full engine-facing description behind a
    /// disclosure. Nothing is deleted — the technical text is exactly
    /// the string the model receives, and the wire identifier is shown
    /// alongside it, because a user debugging a prompt needs both.
    @ViewBuilder
    private func toolRow(_ def: ToolDefinition) -> some View {
        let name = def.function.name
        let isExpanded = expandedTools.contains(name)
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            Toggle(isOn: toolBinding(name)) {
                HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                    Image(systemName: Self.glyph(for: name))
                        // Utility glyph, not a call to action: neutral,
                        // where it used to be one of the app's most
                        // repeated steel-blue accents.
                        .symbolRenderingMode(.monochrome)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(RapidTheme.utilityActionLabel)
                        .frame(width: RapidTheme.Layout.iconSlot)
                        .accessibilityHidden(true)
                    SettingsRowLabel(
                        title: Self.displayName(for: name),
                        description: Self.summary(for: name, fallback: def.function.description)
                    )
                }
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            // Keyed on the TOOL NAME (the wire identifier the
            // engine and the request body use), not on the row's
            // display text — the label is the tool's own
            // description and would drift with copy edits.
            // Upstream (#1822) gave every tool toggle a spoken name and a
            // hint; both are preserved verbatim through the UI-1
            // migration. ``voiceOverLabel`` stays its own function rather
            // than being folded into ``displayName``: they are separate
            // strings upstream, and quietly re-capitalising a VoiceOver
            // label is not this PR's call to make.
            .accessibilityLabel(Self.voiceOverLabel(for: def.function.name))
            .accessibilityHint(def.function.description)
            // Spelled with `def.function.name` rather than the local
            // `name` binding: identical at runtime, but
            // ``AccessibilityIdentifierInventoryTests`` pins the source
            // SHAPE so a future edit cannot quietly swap the per-item
            // key for display text.
            .accessibilityIdentifier("Settings.Tools.Toggle.\(def.function.name)")

            disclosure(for: def, isExpanded: isExpanded)
                // Indent to the text column so the disclosure lines up
                // under the summary it expands, not under the glyph.
                .padding(.leading, RapidTheme.Layout.iconSlot + RapidTheme.Space.sm)
        }
    }

    /// The "Details" disclosure: a compact chevron affordance, and the
    /// engine-facing text when open.
    @ViewBuilder
    private func disclosure(for def: ToolDefinition, isExpanded: Bool) -> some View {
        let name = def.function.name
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            Button {
                if isExpanded {
                    expandedTools.remove(name)
                } else {
                    expandedTools.insert(name)
                }
            } label: {
                HStack(spacing: RapidTheme.Space.xs) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                    Text("Details")
                        .font(RapidFont.caption)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(RapidTheme.utilityActionLabel)
            .rapidAnimation(RapidMotion.quick, value: isExpanded)
            .accessibilityLabel("Details for \(Self.displayName(for: name))")
            .accessibilityAddTraits(isExpanded ? [.isButton, .isSelected] : .isButton)
            .accessibilityHint(isExpanded ? "Collapse" : "Expand")
            .accessibilityIdentifier("Settings.Tools.Details.\(name)")

            if isExpanded {
                VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                    Text(def.function.description)
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    // The wire identifier, where a user debugging a
                    // prompt can still find it — just not as the title.
                    Text(name)
                        .font(RapidFont.code)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .textSelection(.enabled)
                }
                .padding(RapidTheme.Space.md)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.code, style: .continuous)
                        .fill(RapidTheme.surfaceCode)
                )
                .accessibilityIdentifier("Settings.Tools.DetailsBody.\(name)")
            }
        }
    }

    // MARK: - Presentation of tool identity
    //
    // Display names and summaries are PRESENTATION ONLY. The wire
    // identifiers (`web_search`, `browse`, `weather`) are what the
    // registry, the request body, the disabled-tool set, the dispatch
    // guard and every accessibility identifier still use — none of that
    // is touched by anything below.


    /// Human name for a built-in tool. Falls back to the raw identifier
    /// so a tool added to the registry without an entry here still shows
    /// something true rather than nothing.
    static func displayName(for toolName: String) -> String {
        switch toolName {
        case "web_search": return "Web Search"
        case "browse":     return "Browse Web Page"
        case "weather":    return "Weather"
        default:           return toolName
        }
    }

    /// One line on what the tool does, in the user's terms. The engine's
    /// own description is written FOR THE MODEL — it carries pagination
    /// offsets and calling conventions — so it stays in the disclosure.
    static func summary(for toolName: String, fallback: String) -> String {
        switch toolName {
        case "web_search":
            return "Looks up current information on the web when a question needs it."
        case "browse":
            return "Opens a web page you or the model names and reads it. You approve each page."
        case "weather":
            return "Gets the current weather for a place you name."
        default:
            return fallback
        }
    }

    private func toolBinding(_ name: String) -> Binding<Bool> {
        Binding(
            get: { !chat.disabledTools.contains(name) },
            set: { chat.setToolEnabled(name, $0) }
        )
    }

    private static func voiceOverLabel(for toolName: String) -> String {
        switch toolName {
        case "web_search": "Web search"
        case "browse": "Browse pages"
        case "weather": "Weather"
        default: toolName.replacingOccurrences(of: "_", with: " ")
        }
    }

    // MARK: - Web search

    private var webSearchSection: some View {
        @Bindable var config = webSearch
        return SettingsSection(
            "Web search",
            subtitle: "Which backend `web_search` queries. Keenable works with no account; add a free key (Parallel recommended) for the best results. Keys stay in your Keychain."
        ) {
                VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                    // Native macOS radio group, kept native: radios are
                    // exactly the control class the platform should own,
                    // and rebuilding one would cost the arrow-key group
                    // navigation SwiftUI gives for free.
                    //
                    // Three things ARE ours: the body type (it defaulted
                    // to the system size, which read a step larger than
                    // every other choice in Settings), `.controlSize`
                    // (the radios rendered at the regular size, notably
                    // bigger than the compact switches beside them), and
                    // `.tint(nil)` so the selected dot uses the system
                    // control accent instead of inheriting brand amber —
                    // amber is the SELECTION colour for rows and
                    // segments, and putting it inside a radio too made
                    // three different amber affordances on one card.
                    Picker("Backend", selection: $config.provider) {
                        ForEach(WebSearchProvider.allCases) { provider in
                            Text(provider.displayName)
                                .font(RapidFont.body)
                                .tag(provider)
                                // Per-radio identifier keyed on the provider's
                                // stable raw value (duckduckgo / brave /
                                // tavily), so a flow selects a backend by what
                                // it IS rather than by its marketing name.
                                .accessibilityIdentifier(
                                    "Settings.Tools.WebSearch.Backend.\(provider.id)"
                                )
                        }
                    }
                    .pickerStyle(.radioGroup)
                    .labelsHidden()
                    .controlSize(.small)
                    .tint(nil)
                    .accessibilityIdentifier("Settings.Tools.WebSearch.Backend")
                    .onChange(of: config.provider) { _, provider in
                        resetKeyDraft()
                        // This is a user-driven provider selection, not view
                        // appearance. Required-key backends may now inspect
                        // their one account without presenting auth UI.
                        guard provider.requiresKey else { return }
                        Task { await webSearch.prefetchAPIKey(for: provider) }
                    }

                    Text(config.provider.subtitle)
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)

                    // ``acceptsKey``, not ``requiresKey``: Keenable's
                    // key is optional (it lifts the shared keyless
                    // rate limit) but still needs the field.
                    if config.provider.acceptsKey {
                        SettingsRowDivider()
                        keyField(for: config.provider)
                    }
                }
        }
    }

    @ViewBuilder
    private func keyField(for provider: WebSearchProvider) -> some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            HStack(spacing: RapidTheme.Space.sm) {
                SecureField("API key", text: $keyDraft)
                    // Native field, shared metrics: a secure field is one
                    // of the controls macOS should keep owning, so the
                    // only thing standardised here is its height.
                    .textFieldStyle(.roundedBorder)
                    .font(RapidFont.body)
                    .frame(minHeight: RapidTheme.ControlHeight.small)
                    .focused($focusedKeyProvider, equals: provider)
                    .onChange(of: focusedKeyProvider) { _, focused in
                        guard focused == provider else { return }
                        // Optional-key providers (notably Keenable) stay truly
                        // zero-touch until the user enters their key field.
                        Task { await webSearch.prefetchAPIKey(for: provider) }
                    }
                    .onChange(of: keyDraft) { _, _ in keyDraftEdited = true }
                    .onSubmit { commitKey(for: provider) }
                    .accessibilityIdentifier(
                        "Settings.Tools.WebSearch.KeyField.\(provider.id)"
                    )
                Button("Save") { commitKey(for: provider) }
                    .buttonStyle(.rapidSecondaryCompact)
                    .disabled(!keyDraftEdited)
                    .accessibilityIdentifier(
                        "Settings.Tools.WebSearch.SaveKey.\(provider.id)"
                    )
            }
            if let url = provider.keyDashboardURL {
                Link("Get a \(provider.displayName) key", destination: url)
                    .font(RapidFont.caption)
                    // See the Privacy links: ``Link`` ignores the tint.
                    .foregroundStyle(RapidTheme.linkLabel)
                    // Keyed on the provider, not the label: "Get a Brave key"
                    // is display copy and will be reworded.
                    .accessibilityIdentifier(
                        "Settings.Tools.WebSearch.KeyDashboardLink.\(provider.id)"
                    )
            }
            if let feedback = saveFeedback {
                Text(Self.feedbackCopy(feedback))
                    .font(RapidFont.caption)
                    .foregroundStyle(
                        Self.isFailure(feedback)
                            ? RapidTheme.statusError
                            : RapidTheme.textSecondary
                    )
                    .fixedSize(horizontal: false, vertical: true)
            } else if webSearch.cachedKeyState(for: provider) == .unavailable {
                Text("The saved key can’t be accessed. Enter it again and save to replace it.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.statusError)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier(
                        "Settings.Tools.WebSearch.KeyUnavailable.\(provider.id)"
                    )
            } else if webSearch.cachedKeyState(for: provider) == .unknown {
                Text("Saved key status hasn’t been checked. Select this backend or focus the field to check it.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if webSearch.cachedKeyState(for: provider).hasKey {
                Text("A key is stored for \(provider.displayName).")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text(Self.noKeyCaption(for: provider))
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onAppear { resetKeyDraft() }
    }

    /// What an empty key slot means depends on the provider: a
    /// key-REQUIRING backend silently degrades to the keyless chain
    /// at dispatch, while Keenable just keeps using the shared pool.
    /// Static + internal so the copy is pinned by tests.
    static func noKeyCaption(for provider: WebSearchProvider) -> String {
        if provider.requiresKey {
            return "No key stored — searches fall back to Keenable until you save one."
        }
        return "No key stored — \(provider.displayName) works without one; a free key lifts the shared rate limit."
    }

    private func commitKey(for provider: WebSearchProvider) {
        feedbackGeneration += 1
        let generation = feedbackGeneration
        switch SettingsView.webSearchKeyCommitAction(draft: keyDraft, wasEdited: keyDraftEdited) {
        case .unchanged:
            saveFeedback = nil
            return
        case .clear:
            let ok = webSearch.setAPIKey(nil, for: provider)
            saveFeedback = ok ? .cleared(generation: generation) : .writeFailed(generation: generation)
            // A failed Keychain write keeps the draft so the "try again" advice
            // is actually followable — the user would otherwise have to
            // re-paste a secret they no longer have on the clipboard.
            if SettingsView.shouldResetWebSearchKeyDraftAfterCommit(keychainWriteSucceeded: ok) {
                resetKeyDraft()
            }
        case .save(let value):
            let ok = webSearch.setAPIKey(value, for: provider)
            saveFeedback = ok ? .saved(generation: generation) : .writeFailed(generation: generation)
            if SettingsView.shouldResetWebSearchKeyDraftAfterCommit(keychainWriteSucceeded: ok) {
                resetKeyDraft()
            }
        }
    }

    /// Clear the draft back to empty. The stored secret is never echoed back
    /// into the field — a SecureField pre-filled with the real key puts it one
    /// screenshot away, and the row below already says whether one is stored.
    private func resetKeyDraft() {
        keyDraft = ""
        keyDraftEdited = false
    }

    static func feedbackCopy(_ feedback: SettingsView.WebSearchKeySaveFeedback) -> String {
        switch feedback {
        case .saved: return "Saved to your Keychain."
        case .cleared: return "Key removed."
        case .writeFailed: return "Couldn't write to the Keychain. Try again."
        }
    }

    static func isFailure(_ feedback: SettingsView.WebSearchKeySaveFeedback) -> Bool {
        if case .writeFailed = feedback { return true }
        return false
    }

    // MARK: - Browsing

    private var browseSection: some View {
        SettingsSection(
            "Browsing",
            subtitle: "`browse` fetches a page and hands its text to the model. The model picks the URL, so by default you approve each destination first."
        ) {
            Toggle(isOn: browseAutoApproveBinding) {
                SettingsRowLabel(
                    title: "Approve every page automatically",
                    description: "Skips the confirmation for unattended use. Private and local addresses stay blocked either way."
                )
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            .accessibilityLabel("Approve every page automatically")
            .accessibilityHint("Skips confirmation for public pages. Private and local addresses remain blocked.")
            .accessibilityIdentifier("Settings.Tools.Browse.AutoApproveToggle")
        }
    }

    private var browseAutoApproveBinding: Binding<Bool> {
        Binding(
            get: { browseApproval.mode == .autoApproveAll },
            set: { browseApproval.mode = $0 ? .autoApproveAll : .ask }
        )
    }

    // MARK: - Shared layout

    // NOTE: private ``header(_:_:)`` and ``card(_:)`` helpers lived here
    // and were the second of four copies of the same pair. Both are now
    // ``SectionHeader`` + ``SettingsSection``.

    static func glyph(for name: String) -> String {
        switch name {
        case "web_search": return "magnifyingglass"
        case "browse": return "globe"
        case "weather": return "cloud.sun"
        default: return "wrench.and.screwdriver"
        }
    }
}
