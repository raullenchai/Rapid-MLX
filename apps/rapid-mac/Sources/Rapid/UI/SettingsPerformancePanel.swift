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
///     switch for them would be wired to nothing. The speculative-decoding
///     switch emits the alias registry's audited preset;
///     aliases without one still show the control disabled with a reason.
///   * **Per model.** Settings attach to the alias, because the right KV
///     setting for a 4B dense model is not the right one for a 35B MoE.
///   * **One line of cost per control**, from ``KVCacheMode/tradeOff``.
///   * **Reload stated before the click**, not after: these are engine
///     construction settings, so a resident model must be rebuilt to apply.
struct SettingsPerformancePanel: View {
    @Environment(ModelPerfConfigStore.self) private var perf
    @Environment(ServerManager.self) private var server
    @Environment(DownloadManager.self) private var downloads

    /// True while the target resident model is being rebuilt.
    @State private var isReloading: Bool = false
    @State private var catalog: [ModelEntry] = []
    @State private var selectedAlias: String?
    @State private var applyError: String?

    /// The alias this panel edits. The running model when there is one,
    /// otherwise the last one served — editing "whatever runs next" with no
    /// name attached is how a user ends up surprised about which model they
    /// changed.
    private var targetAlias: String? {
        selectedAlias ?? server.servingAlias ?? server.launchedChildAlias
    }

    private var modelChoices: [ModelEntry] {
        var byAlias = Dictionary(uniqueKeysWithValues: catalog
            .filter { $0.kind == .chat }
            .map { ($0.alias.lowercased(), $0) })
        for alias in perf.configuredAliases where byAlias[alias.lowercased()] == nil {
            byAlias[alias.lowercased()] = ModelEntry(
                alias: alias, hfRepo: nil, sizeOnDisk: nil, cached: false
            )
        }
        return byAlias.values.sorted {
            $0.alias.localizedCaseInsensitiveCompare($1.alias) == .orderedAscending
        }
    }

    /// Whether the running child was launched before the current settings, so
    /// its argv predates them. Derived, never stored — the same reasoning as
    /// ``SettingsConnectorsPanel``: `@State` dies with the view while the
    /// condition it described is still true.
    private var needsReload: Bool {
        guard let alias = targetAlias, server.isModelResident(alias) else { return false }
        let wantsSpeculative = perf.config(forAlias: alias).speculativePreset != nil
        if wantsSpeculative != server.hasAppliedSpeculativeDecoding(forAlias: alias) {
            return true
        }
        if let resident = server.residency.models.first(where: { $0.matches(alias) }),
           let applied = resident.performance {
            return !applied.matches(effectiveConfig(for: alias))
        }
        guard server.launchedChildAlias != nil else { return false }
        return perf.launchFlags(forAlias: alias) != launchedFlags
    }

    private func effectiveConfig(for alias: String) -> ModelPerfConfig {
        ModelPerfConfig(launchFlags: ServerManager.mergedPerformanceFlags(
            recommended: RAMBucketedDefault.launchFlags(
                forAlias: alias,
                physicalRAMGB: MacHardware.detect().physicalRAMGB
            ),
            userOverrides: perf.launchFlags(forAlias: alias)
        ))
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
                modelSection
                if let alias = targetAlias {
                    if needsReload { reloadBanner(alias: alias) }
                    kvSection(alias: alias)
                    speculativeDecodingSection(alias: alias)
                    prefixSection(alias: alias)
                    footer(alias: alias)
                } else {
                    noModelNotice
                }
                if let error = perf.loadError {
                    InlineNotice(message: error, tone: .error)
                }
                if let applyError {
                    InlineNotice(message: applyError, tone: .error)
                }
            }
            .padding(RapidTheme.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .accessibilityIdentifier("Settings.Performance.Panel")
        .task(id: server.launchedChildAlias) {
            // Snapshot what the original child was spawned with so the legacy
            // primary-record fallback can compare stored opinion with reality.
            launchedFlags = targetAlias.map { perf.launchFlags(forAlias: $0) } ?? []
        }
        .task(id: downloads.cacheGeneration) {
            guard let binary = server.binaryPath else { return }
            catalog = await ModelCatalogCache.shared.entries(
                binary: binary, generation: downloads.cacheGeneration
            )
            if selectedAlias == nil {
                selectedAlias = server.servingAlias
                    ?? server.launchedChildAlias
                    ?? modelChoices.first(where: { $0.cached })?.alias
                    ?? modelChoices.first?.alias
            }
        }
        .onChange(of: selectedAlias) { _, alias in
            launchedFlags = alias.map { perf.launchFlags(forAlias: $0) } ?? []
            applyError = nil
        }
    }

    // MARK: - Sections

    private var modelSection: some View {
        SettingsSection(
            "Model",
            subtitle: "Choose which model owns these settings. It does not need to be running."
        ) {
            if modelChoices.isEmpty {
                Text("No chat models are available yet.")
                    .font(RapidFont.body)
                    .foregroundStyle(RapidTheme.textSecondary)
            } else {
                Picker("Model", selection: $selectedAlias) {
                    ForEach(modelChoices) { entry in
                        Text(entry.cached ? entry.alias : "\(entry.alias) · not downloaded")
                            .tag(Optional(entry.alias))
                    }
                }
                .accessibilityIdentifier("Settings.Performance.ModelPicker")
            }
        }
    }

    private var noModelNotice: some View {
        InlineNotice(
            message: "Start a model to configure its performance settings.",
            tone: .info
        )
        .accessibilityIdentifier("Settings.Performance.NoModel")
    }

    private func reloadBanner(alias: String) -> some View {
        let speculativeChanged = (perf.config(forAlias: alias).speculativePreset != nil)
            != server.hasAppliedSpeculativeDecoding(forAlias: alias)
        return InlineNotice(
            message: speculativeChanged
                ? "Restart \(alias) to apply speculative decoding. Other resident models will unload; downloaded weights and conversations stay available."
                : "Reload \(alias) to apply. Other resident models will stay available.",
            tone: .warning,
            actionTitle: isReloading ? "Restarting…" : (speculativeChanged ? "Restart model" : "Reload model"),
            actionIdentifier: "Settings.Performance.ReloadModel",
            action: { reload(alias: alias) }
        )
        .disabled(isReloading)
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

    private func speculativeDecodingSection(alias: String) -> some View {
        let preset = modelChoices.first {
            $0.alias.caseInsensitiveCompare(alias) == .orderedSame
        }?.speculativeDecodingPreset
        return SettingsSection(
            "Speculative decoding",
            subtitle: "Drafts candidate tokens and verifies them with the full model."
        ) {
            VStack(alignment: .leading, spacing: 10) {
                Toggle(
                    preset.map { "Enable \($0.displayName)" }
                        ?? "No verified preset for this model",
                    isOn: speculativeDecodingBinding(alias: alias, preset: preset)
                )
                    .toggleStyle(.switch)
                    .disabled(preset == nil)
                    .accessibilityIdentifier("Settings.Performance.SpeculativeDecoding.Enabled")
                tradeOffLine(
                    preset == nil
                        ? "This alias does not declare a verified speculative-decoding preset."
                        : "Off by default. It can improve generation speed on some Macs, but may be slower on others; accepted output remains token-exact.",
                    warns: false
                )
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
        let measured = !ModelPerfConfig(
            launchFlags: RAMBucketedDefault.launchFlags(
                forAlias: alias,
                physicalRAMGB: MacHardware.detect().physicalRAMGB
            )
        ).isEmpty
        return VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            HStack {
                Text(perf.hasOverride(forAlias: alias)
                     ? "Customized for \(alias)."
                     : measured
                        ? "\(alias) will use its measured defaults."
                        : "\(alias) will use the engine defaults; no measured profile is published for it.")
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.textSecondary)
                Spacer()
                Button(measured ? "Reset to measured defaults" : "Reset to engine defaults") {
                    perf.resetToDefaults(forAlias: alias)
                }
                .buttonStyle(.rapidSecondaryCompact)
                .disabled(!perf.hasOverride(forAlias: alias))
                .accessibilityIdentifier("Settings.Performance.Reset")
            }
            if !server.isModelResident(alias) {
                Text("Saved. These settings will apply the next time this model loads.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .accessibilityIdentifier("Settings.Performance.AppliesNextLoad")
            }
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

    private func speculativeDecodingBinding(
        alias: String,
        preset: SpeculativeDecodingPreset?
    ) -> Binding<Bool> {
        Binding(
            get: { perf.config(forAlias: alias).speculativePreset != nil },
            set: { enabled in
                update(alias: alias) { $0.speculativePreset = enabled ? preset : nil }
            }
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

    private func reload(alias: String) {
        isReloading = true
        applyError = nil
        Task {
            let entry = modelChoices.first { $0.alias.caseInsensitiveCompare(alias) == .orderedSame }
            let speculativeChanged = (perf.config(forAlias: alias).speculativePreset != nil)
                != server.hasAppliedSpeculativeDecoding(forAlias: alias)
            if speculativeChanged {
                let restarted = await server.restartForSpeculativePerformance(
                    alias: alias,
                    hfPath: entry?.hfRepo
                )
                if restarted {
                    launchedFlags = perf.launchFlags(forAlias: alias)
                } else {
                    applyError = "Could not restart this model with its speculative-decoding setting."
                }
                isReloading = false
                return
            }
            let result = await server.reloadResidentPerformance(
                alias: alias,
                hfPath: entry?.hfRepo
            )
            if case .loaded = result {
                launchedFlags = perf.launchFlags(forAlias: alias)
            } else if case .rejected(let message) = result {
                applyError = message
            } else {
                applyError = "This bundled model server cannot reload one resident model."
            }
            isReloading = false
        }
    }

}
