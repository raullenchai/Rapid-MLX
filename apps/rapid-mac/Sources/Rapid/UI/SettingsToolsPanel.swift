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

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            toolsSection
            webSearchSection
            browseSection
        }
        .task {
            // Warm the Keychain cache off the main actor before the key rows
            // render — a cold `SecItemCopyMatching` crosses securityd XPC and
            // can stall the panel's first paint.
            await webSearch.prefetchAllAPIKeys()
        }
    }

    // MARK: - Available tools

    private var toolsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            header(
                "Tools",
                "Tools the model can call during a chat. Turn one off and it is never offered — and never runs, even if the model asks for it by name."
            )
            card {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(chat.builtinDefinitions, id: \.function.name) { def in
                        Toggle(isOn: toolBinding(def.function.name)) {
                            HStack(alignment: .top, spacing: 10) {
                                Image(systemName: Self.glyph(for: def.function.name))
                                    .foregroundStyle(RapidTheme.brand)
                                    .frame(width: 18)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(def.function.name)
                                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                                    Text(def.function.description)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                        }
                        .toggleStyle(TrailingSettingsToggleStyle())
                        // Keyed on the TOOL NAME (the wire identifier the
                        // engine and the request body use), not on the row's
                        // display text — the label is the tool's own
                        // description and would drift with copy edits.
                        .accessibilityIdentifier("Settings.Tools.Toggle.\(def.function.name)")
                    }
                }
            }
        }
    }

    private func toolBinding(_ name: String) -> Binding<Bool> {
        Binding(
            get: { !chat.disabledTools.contains(name) },
            set: { chat.setToolEnabled(name, $0) }
        )
    }

    // MARK: - Web search

    private var webSearchSection: some View {
        @Bindable var config = webSearch
        return VStack(alignment: .leading, spacing: 8) {
            header(
                "Web search",
                "Which backend `web_search` queries. DuckDuckGo needs no account but is rate-limited; the keyed backends are more reliable."
            )
            card {
                VStack(alignment: .leading, spacing: 14) {
                    Picker("Backend", selection: $config.provider) {
                        ForEach(WebSearchProvider.allCases) { provider in
                            Text(provider.displayName)
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
                    .accessibilityIdentifier("Settings.Tools.WebSearch.Backend")
                    .onChange(of: config.provider) { _, _ in
                        resetKeyDraft()
                    }

                    Text(config.provider.subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    if config.provider.requiresKey {
                        keyField(for: config.provider)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func keyField(for provider: WebSearchProvider) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                SecureField("API key", text: $keyDraft)
                    .textFieldStyle(.roundedBorder)
                    .onChange(of: keyDraft) { _, _ in keyDraftEdited = true }
                    .onSubmit { commitKey(for: provider) }
                    .accessibilityIdentifier(
                        "Settings.Tools.WebSearch.KeyField.\(provider.id)"
                    )
                Button("Save") { commitKey(for: provider) }
                    .disabled(!keyDraftEdited)
                    .accessibilityIdentifier(
                        "Settings.Tools.WebSearch.SaveKey.\(provider.id)"
                    )
            }
            if let url = provider.keyDashboardURL {
                Link("Get a \(provider.displayName) key", destination: url)
                    .font(.caption)
                    // Keyed on the provider, not the label: "Get a Brave key"
                    // is display copy and will be reworded.
                    .accessibilityIdentifier(
                        "Settings.Tools.WebSearch.KeyDashboardLink.\(provider.id)"
                    )
            }
            if let feedback = saveFeedback {
                Text(Self.feedbackCopy(feedback))
                    .font(.caption)
                    .foregroundStyle(Self.isFailure(feedback) ? RapidTheme.statusError : .secondary)
            } else if webSearch.cachedKeyState(for: provider).hasKey {
                Text("A key is stored for \(provider.displayName).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("No key stored — searches fall back to DuckDuckGo until you save one.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .onAppear { resetKeyDraft() }
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
        VStack(alignment: .leading, spacing: 8) {
            header(
                "Browsing",
                "`browse` fetches a page and hands its text to the model. The model picks the URL, so by default you approve each destination first."
            )
            card {
                Toggle(isOn: browseAutoApproveBinding) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Approve every page automatically")
                            .font(.callout.weight(.medium))
                        Text("Skips the confirmation for unattended use. Private and local addresses stay blocked either way.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .toggleStyle(TrailingSettingsToggleStyle())
                .accessibilityIdentifier("Settings.Tools.Browse.AutoApproveToggle")
            }
        }
    }

    private var browseAutoApproveBinding: Binding<Bool> {
        Binding(
            get: { browseApproval.mode == .autoApproveAll },
            set: { browseApproval.mode = $0 ? .autoApproveAll : .ask }
        )
    }

    // MARK: - Shared layout

    @ViewBuilder
    private func header(_ title: String, _ subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.title3.weight(.semibold))
            Text(subtitle)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        content()
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
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

    static func glyph(for name: String) -> String {
        switch name {
        case "web_search": return "magnifyingglass"
        case "browse": return "globe"
        case "weather": return "cloud.sun"
        default: return "wrench.and.screwdriver"
        }
    }
}
