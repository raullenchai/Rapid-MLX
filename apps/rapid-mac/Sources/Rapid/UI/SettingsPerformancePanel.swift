import SwiftUI

/// Settings → Performance. Issue #1717's user-facing surface.
///
/// The engine's throughput knobs were CLI-only, which made the GUI the slow
/// way to use our own engine. This panel exposes the subset that survived the
/// audit, and it is deliberately small:
///
///   * **Only audited, CI-gated flags.** `--kv-bits`, `--kv-group-size`,
///     `--draft-model` and `--num-draft-tokens` — named in the issue — are in
///     the engine's deprecated-no-op block and are parsed but never read, so a
///     switch for them would be wired to nothing. Speculative decoding's
///     canonical entry point is `--speculative-config`, a JSON blob; its
///     flag-shaped spellings are all `argparse.SUPPRESS` legacy aliases.
///     Neither is here.
///   * **Per model.** Settings attach to the alias, because the right KV
///     setting for a 4B dense model is not the right one for a 35B MoE.
///   * **One line of cost per control**, from ``KVCacheMode/tradeOff``.
///   * **Restart stated before the click**, not after: these are `serve`
///     launch flags, so a change only reaches a running model on respawn.
struct SettingsPerformancePanel: View {
    @Environment(ModelPerfConfigStore.self) private var perf
    @Environment(ServerManager.self) private var server

    /// True while the Restart button is cycling the child.
    @State private var isRestarting: Bool = false

    /// The alias this panel edits. The running model when there is one,
    /// otherwise the last one served — editing "whatever runs next" with no
    /// name attached is how a user ends up surprised about which model they
    /// changed.
    private var targetAlias: String? {
        server.servingAlias ?? server.launchedChildAlias
    }

    /// Whether the running child was launched before the current settings, so
    /// its argv predates them. Derived, never stored — the same reasoning as
    /// ``SettingsConnectorsPanel``: `@State` dies with the view while the
    /// condition it described is still true.
    private var needsRestart: Bool {
        guard let alias = targetAlias, server.launchedChildAlias != nil else { return false }
        return perf.launchFlags(forAlias: alias) != launchedFlags
    }

    /// Flags the running child was actually spawned with, for the alias in
    /// question. Nil-safe: with no child, there is nothing to compare against.
    @State private var launchedFlags: [String] = []

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                SectionHeader(
                    "Performance",
                    subtitle: "These settings change speed and memory use, and some can change what the model writes. They apply to one model at a time and take effect when that model next starts.",
                    emphasis: .page
                )
                if let alias = targetAlias {
                    if needsRestart { restartBanner(alias: alias) }
                    kvSection(alias: alias)
                    prefixSection(alias: alias)
                    footer(alias: alias)
                } else {
                    noModelNotice
                }
                if let error = perf.loadError {
                    InlineNotice(message: error, tone: .error)
                }
            }
            .padding(RapidTheme.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .accessibilityIdentifier("Settings.Performance.Panel")
        .task(id: server.launchedChildAlias) {
            // Snapshot what the child was spawned with so ``needsRestart`` can
            // compare against it rather than against "has any override", which
            // would keep the banner up forever after a restart.
            launchedFlags = targetAlias.map { perf.launchFlags(forAlias: $0) } ?? []
        }
    }

    // MARK: - Sections

    private var noModelNotice: some View {
        InlineNotice(
            message: "Start a model to configure its performance settings.",
            tone: .info
        )
        .accessibilityIdentifier("Settings.Performance.NoModel")
    }

    private func restartBanner(alias: String) -> some View {
        InlineNotice(
            message: "Restart \(alias) to apply. The running model was started with the previous settings.",
            tone: .warning,
            actionTitle: isRestarting ? "Restarting…" : "Restart",
            action: { restart(alias: alias) }
        )
        .disabled(isRestarting)
        .accessibilityIdentifier("Settings.Performance.RestartNotice")
    }

    private func kvSection(alias: String) -> some View {
        SettingsSection(
            "KV cache precision",
            subtitle: "How the model's attention cache is stored. Lower precision means less memory and faster long-context decoding."
        ) {
            VStack(alignment: .leading, spacing: 10) {
                // One picker, not two. The engine resolves --kv-cache-dtype
                // only when TurboQuant is off, so independent controls could
                // show a dtype the engine silently ignored.
                Picker("", selection: kvBinding(alias: alias)) {
                    Text("Engine default").tag(KVCacheMode?.none)
                    ForEach(KVCacheMode.allCases, id: \.self) { mode in
                        Text(mode.title).tag(KVCacheMode?.some(mode))
                    }
                }
                .labelsHidden()
                .pickerStyle(.radioGroup)
                .accessibilityLabel("KV cache precision")
                .accessibilityIdentifier("Settings.Performance.KVMode")

                if let mode = perf.config(forAlias: alias).kvCacheMode {
                    tradeOffLine(mode.tradeOff, warns: mode.canChangeOutput)
                    if mode.isSubjectToArchitectureDowngrade {
                        Text("Sliding-window (Gemma, GPT-OSS) and MLA (DeepSeek, Kimi) models fall back to full precision regardless of this setting.")
                            .font(RapidFont.caption)
                            .foregroundStyle(RapidTheme.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                } else {
                    tradeOffLine("The engine picks a precision measured for this model.", warns: false)
                }
            }
        }
    }

    private func prefixSection(alias: String) -> some View {
        SettingsSection(
            "Prefix cache",
            subtitle: "Reuses computation for a prompt prefix the model has already seen. Speeds up multi-turn chat and repeated system prompts."
        ) {
            VStack(alignment: .leading, spacing: 10) {
                RapidSegmentedControl(
                    selection: prefixBinding(alias: alias),
                    options: [
                        .init(value: Bool?.none, title: "Engine default", identifier: "Settings.Performance.Prefix.Default"),
                        .init(value: Bool?.some(true), title: "On", identifier: "Settings.Performance.Prefix.On"),
                        .init(value: Bool?.some(false), title: "Off", identifier: "Settings.Performance.Prefix.Off"),
                    ],
                    accessibilityLabel: "Prefix cache"
                )
                .accessibilityIdentifier("Settings.Performance.PrefixCache")

                tradeOffLine(
                    "Costs memory, never changes output. Turning it off is mainly useful for measuring what it buys you.",
                    warns: false
                )

                SettingsRowDivider()

                cacheBudgetRow(alias: alias)
            }
        }
    }

    private func cacheBudgetRow(alias: String) -> some View {
        let config = perf.config(forAlias: alias)
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Cache budget")
                    .font(RapidFont.bodyEmphasis)
                Spacer()
                Text(config.cacheMemoryMB.map { "\($0) MB" } ?? "Automatic")
                    .font(RapidFont.metric)
                    .foregroundStyle(RapidTheme.textSecondary)
            }
            HStack(spacing: 12) {
                Slider(
                    value: cacheBudgetBinding(alias: alias),
                    in: Double(ModelPerfConfig.cacheMemoryMBRange.lowerBound)
                        ... Double(ModelPerfConfig.cacheMemoryMBRange.upperBound),
                    step: 256
                )
                .accessibilityLabel("Cache budget")
                .accessibilityIdentifier("Settings.Performance.CacheBudget")
                if config.cacheMemoryMB != nil {
                    Button("Automatic") { update(alias: alias) { $0.cacheMemoryMB = nil } }
                        .buttonStyle(.rapidTertiary)
                        .accessibilityIdentifier("Settings.Performance.CacheBudgetAutomatic")
                }
            }
            tradeOffLine(
                "Automatic uses about 20% of RAM. A larger budget holds more prefixes; it never changes output.",
                warns: false
            )
        }
    }

    private func footer(alias: String) -> some View {
        HStack {
            Text(perf.hasOverride(forAlias: alias)
                 ? "Customized for \(alias)."
                 : "\(alias) is using measured defaults.")
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.textSecondary)
            Spacer()
            Button("Reset to measured defaults") {
                perf.resetToDefaults(forAlias: alias)
            }
            .buttonStyle(.rapidSecondaryCompact)
            .disabled(!perf.hasOverride(forAlias: alias))
            .accessibilityIdentifier("Settings.Performance.Reset")
        }
    }

    /// The issue's "state the trade-off in one line each, next to the control".
    /// `warns` marks the choices that can change output, not merely speed —
    /// the distinction the issue's "the trap" section is built around.
    private func tradeOffLine(_ text: String, warns: Bool) -> some View {
        Label {
            Text(text)
                .font(RapidFont.caption)
                .foregroundStyle(warns ? RapidTheme.statusWarning : RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: warns ? "exclamationmark.triangle.fill" : "info.circle")
                .font(RapidFont.caption)
                .foregroundStyle(warns ? RapidTheme.statusWarning : RapidTheme.textSecondary)
        }
    }

    // MARK: - Bindings

    private func kvBinding(alias: String) -> Binding<KVCacheMode?> {
        Binding(
            get: { perf.config(forAlias: alias).kvCacheMode },
            set: { newValue in update(alias: alias) { $0.kvCacheMode = newValue } }
        )
    }

    private func prefixBinding(alias: String) -> Binding<Bool?> {
        Binding(
            get: { perf.config(forAlias: alias).prefixCacheEnabled },
            set: { newValue in update(alias: alias) { $0.prefixCacheEnabled = newValue } }
        )
    }

    private func cacheBudgetBinding(alias: String) -> Binding<Double> {
        Binding(
            get: {
                Double(perf.config(forAlias: alias).cacheMemoryMB
                       ?? ModelPerfConfig.cacheMemoryMBRange.lowerBound)
            },
            set: { newValue in update(alias: alias) { $0.cacheMemoryMB = Int(newValue) } }
        )
    }

    private func update(alias: String, _ mutate: (inout ModelPerfConfig) -> Void) {
        var config = perf.config(forAlias: alias)
        mutate(&config)
        perf.setConfig(config, forAlias: alias)
    }

    private func restart(alias: String) {
        isRestarting = true
        Task {
            await server.stop()
            await server.start(alias: alias)
            // Only acknowledge the new argv after the replacement child is
            // actually ready. A failed restart must keep the notice visible;
            // otherwise Settings claims an override was applied when no
            // serving process has it (#1717).
            if Self.restartApplied(state: server.state, alias: alias) {
                launchedFlags = perf.launchFlags(forAlias: alias)
            }
            isRestarting = false
        }
    }

    /// A restart counts as applied only when the replacement child reached
    /// ready for the same model. In particular, `.crashed` and `.idle` must
    /// leave the notice up so Settings never reports argv that no process has.
    static func restartApplied(state: ServerState, alias: String) -> Bool {
        guard case .ready(let readyAlias) = state else { return false }
        return readyAlias.caseInsensitiveCompare(alias) == .orderedSame
    }
}
