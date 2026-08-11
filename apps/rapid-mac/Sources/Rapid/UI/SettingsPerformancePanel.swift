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
            VStack(alignment: .leading, spacing: 18) {
                header
                if let alias = targetAlias {
                    if needsRestart { restartBanner(alias: alias) }
                    kvSection(alias: alias)
                    prefixSection(alias: alias)
                    footer(alias: alias)
                } else {
                    noModelNotice
                }
                if let error = perf.loadError {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .font(.callout)
                        .foregroundStyle(RapidTheme.amberDeep)
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .task(id: server.launchedChildAlias) {
            // Snapshot what the child was spawned with so ``needsRestart`` can
            // compare against it rather than against "has any override", which
            // would keep the banner up forever after a restart.
            launchedFlags = targetAlias.map { perf.launchFlags(forAlias: $0) } ?? []
        }
    }

    // MARK: - Sections

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Performance")
                .font(.title3.weight(.semibold))
            Text("These settings change speed and memory use, and some can change what the model writes. They apply to one model at a time and take effect when that model next starts.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var noModelNotice: some View {
        Label(
            "Start a model to configure its performance settings.",
            systemImage: "info.circle"
        )
        .font(.callout)
        .foregroundStyle(.secondary)
    }

    private func restartBanner(alias: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: "arrow.triangle.2.circlepath")
                .foregroundStyle(RapidTheme.amberDeep)
            VStack(alignment: .leading, spacing: 2) {
                Text("Restart \(alias) to apply")
                    .font(.callout.weight(.medium))
                Text("The running model was started with the previous settings.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Button(isRestarting ? "Restarting…" : "Restart") { restart(alias: alias) }
                .disabled(isRestarting)
        }
        .padding(12)
        .background(RapidTheme.amberTint, in: RoundedRectangle(cornerRadius: RapidTheme.cardRadius))
    }

    private func kvSection(alias: String) -> some View {
        card {
            VStack(alignment: .leading, spacing: 10) {
                sectionTitle("KV cache precision", subtitle: "How the model's attention cache is stored. Lower precision means less memory and faster long-context decoding.")

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

                if let mode = perf.config(forAlias: alias).kvCacheMode {
                    tradeOffLine(mode.tradeOff, warns: mode.canChangeOutput)
                    if mode.isSubjectToArchitectureDowngrade {
                        Text("Sliding-window (Gemma, GPT-OSS) and MLA (DeepSeek, Kimi) models fall back to full precision regardless of this setting.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                } else {
                    tradeOffLine("The engine picks a precision measured for this model.", warns: false)
                }
            }
        }
    }

    private func prefixSection(alias: String) -> some View {
        card {
            VStack(alignment: .leading, spacing: 10) {
                sectionTitle("Prefix cache", subtitle: "Reuses the computation for a prompt prefix the model has already seen. Speeds up multi-turn chat and repeated system prompts.")

                Picker("", selection: prefixBinding(alias: alias)) {
                    Text("Engine default").tag(Bool?.none)
                    Text("On").tag(Bool?.some(true))
                    Text("Off").tag(Bool?.some(false))
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(maxWidth: 280)

                tradeOffLine(
                    "Costs memory, never changes output. Turning it off is mainly useful for measuring what it buys you.",
                    warns: false
                )

                Divider().padding(.vertical, 2)

                cacheBudgetRow(alias: alias)
            }
        }
    }

    private func cacheBudgetRow(alias: String) -> some View {
        let config = perf.config(forAlias: alias)
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Cache budget")
                    .font(.callout.weight(.medium))
                Spacer()
                Text(config.cacheMemoryMB.map { "\($0) MB" } ?? "Automatic")
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 12) {
                Slider(
                    value: cacheBudgetBinding(alias: alias),
                    in: Double(ModelPerfConfig.cacheMemoryMBRange.lowerBound)
                        ... Double(ModelPerfConfig.cacheMemoryMBRange.upperBound),
                    step: 256
                )
                if config.cacheMemoryMB != nil {
                    Button("Automatic") { update(alias: alias) { $0.cacheMemoryMB = nil } }
                        .buttonStyle(.link)
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
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button("Reset to measured defaults") {
                perf.resetToDefaults(forAlias: alias)
            }
            .disabled(!perf.hasOverride(forAlias: alias))
        }
    }

    // MARK: - Building blocks

    private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        content()
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RapidTheme.card, in: RoundedRectangle(cornerRadius: RapidTheme.cardRadius))
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.cardRadius)
                    .stroke(RapidTheme.hairline, lineWidth: 1)
            )
    }

    private func sectionTitle(_ title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.callout.weight(.semibold))
            Text(subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The issue's "state the trade-off in one line each, next to the control".
    /// `warns` marks the choices that can change output, not merely speed —
    /// the distinction the issue's "the trap" section is built around.
    private func tradeOffLine(_ text: String, warns: Bool) -> some View {
        Label {
            Text(text)
                .font(.caption)
                .foregroundStyle(warns ? RapidTheme.amberDeep : .secondary)
                .fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: warns ? "exclamationmark.triangle.fill" : "info.circle")
                .font(.caption)
                .foregroundStyle(warns ? RapidTheme.amberDeep : .secondary)
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
            launchedFlags = perf.launchFlags(forAlias: alias)
            isRestarting = false
        }
    }
}
