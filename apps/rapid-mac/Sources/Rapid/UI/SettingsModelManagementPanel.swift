import AppKit
import SwiftUI

/// Settings → Model Management — the file-manager-style surface
/// for cache state (issue #210).
///
/// Why this exists, given the older ``SettingsModelsPanel`` was
/// already shipping a download / delete UI: user feedback
/// (2026-06-16) called out that the picker dropdown is
/// overloaded — it conflates "switch active alias" with "manage
/// the on-disk cache", and casual users miss the right-click
/// affordances. ``SettingsModelManagementPanel`` is the dedicated
/// sidebar tab that owns cache state. The picker stays a
/// switcher; the panel is the inspector.
///
/// Layout (top to bottom):
///   * Search box + ``All / Cached / Not cached`` segmented
///     filter + sort menu — top of the panel so a user with 60
///     aliases doesn't scroll to find the one they want.
///   * One row per alias — alias name + family/quant chip + size
///     line, a status badge in the middle, and a single action
///     button on the right that morphs across
///     ``Download / Cancel / Delete``.
///   * "Total: X GB across N models" footer that aggregates the
///     cached subset. Hidden when nothing is cached.
///
/// Every shared primitive lives in ``ModelCacheActions``: the
/// confirmation copy, the status-badge derivation, the
/// filter/sort/aggregate helpers, and the delete-and-format
/// dispatch. The view is intentionally thin so all the truth
/// tables are unit-testable without a SwiftUI host.
struct SettingsModelManagementPanel: View {
    @Environment(ServerManager.self) private var server
    @Environment(DownloadManager.self) private var downloads

    @State private var catalog: [ModelEntry] = []
    @State private var loading: Bool = true
    @State private var pendingDeletion: ModelEntry?
    @State private var lastError: String?
    @State private var lastFreed: String?

    @State private var query: String = ""
    @State private var filterMode: ModelCacheActions.FilterMode = .all
    @State private var sortOrder: ModelCacheActions.SortOrder = .familyThenSize

    /// Issue #503: the user's chosen models folder (absolute path), or
    /// ``nil`` for the default location. Mirrors
    /// ``ModelsFolderPreference`` so the row + buttons re-render the
    /// moment the user picks / resets a folder without waiting on a
    /// catalog reload.
    @State private var customFolderPath: String? = ModelsFolderPreference.storedPath()

    /// Detected once. Drives the "Recommended for your N GB Mac" header
    /// and which RAM bucket's role picks surface at the top (issue #507).
    /// Cheap sysctl probe; constant for the panel's lifetime.
    @State private var hardware: MacHardware = .detect()

    /// User-pinned favorites (issue #507). Floated to the top of the
    /// "All models" table regardless of sort. Seeded from defaults;
    /// toggled in-row via the star.
    @State private var favorites: Set<String> = ModelFavorites.load()

    /// codex r1 P2 (#210): without this we'd ride the stale catalog
    /// snapshot after a background download finishes — the row
    /// flips from ``Downloading…`` straight back to ``Not cached``
    /// because ``ModelCacheActions.statusBadge`` falls through to
    /// the catalog's ``cached == false`` once the job exits
    /// ``.running``. Track each alias' last-observed job status
    /// here; when any moves to ``.completed`` we re-read the
    /// catalog so the row re-resolves to ``.cached`` + the disk
    /// footer aggregation pulls in the new bytes.
    @State private var lastObservedJobStatuses: [String: ObservedJobStatus] = [:]

    /// Coarse fingerprint of ``DownloadManager.Job.Status`` so the
    /// ``onChange`` diff against ``downloads.jobs`` doesn't churn
    /// on every tqdm tick (which only mutates
    /// ``job.progress.phase`` inside ``.running``). Coarsening to
    /// the discriminator alone lets the catalog refresh fire on
    /// running → completed / running → failed / running →
    /// cancelled transitions only.
    enum ObservedJobStatus: Equatable, Sendable {
        case running
        case completed
        case failed
        case cancelled

        init(_ status: DownloadManager.Job.Status) {
            switch status {
            case .running: self = .running
            case .completed: self = .completed
            case .failed: self = .failed
            case .cancelled: self = .cancelled
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            modelsFolderSection
            controlsRow
            if showRecommendedSection {
                recommendedSection
            }
            if loading && catalog.isEmpty {
                loadingState
            } else if catalog.isEmpty {
                emptyState
            } else {
                allModelsSection
            }
            if let lastError {
                errorBanner(lastError)
            }
            if let lastFreed {
                freedBanner(lastFreed)
            }
        }
        // Loading, empty, and catalog branches are deliberately exclusive.
        // Animating this container retains both conditional trees during the
        // transition, which can overlay the spinner on stale model rows.
        .task {
            await refreshCatalog()
        }
        // codex r1 P2 / codex r2 P2 fix: catch the running → terminal
        // transition without relying on SwiftUI's observation graph,
        // which can't see ``job.status`` mutate on the existing
        // ``Job`` reference type (``DownloadManager.handleExit``
        // doesn't reassign the dict entry). Without this watcher a
        // finished pull would stay on ``Downloading…`` until the
        // user switched tabs and back.
        //
        // The task is alive for the panel's lifetime; the inner
        // loop only spins while at least one job is ``.running``
        // (cheap 500 ms cadence — same as a tqdm tick — so a
        // terminal flip lands inside half a second). When no
        // running jobs remain the loop awaits a longer beat so the
        // idle Settings tab does no work. Tests pin the pure
        // transition predicate in ``shouldRefreshCatalog``.
        .task {
            await jobReconciliationLoop()
        }
        // ``confirmationDialog`` over ``alert`` so the cancel-role
        // button is Return-bound — same reasoning as the picker's
        // dialog in v0.6 P1, and we route the title/message
        // through ``ModelCacheActions.deletionConfirmation`` so
        // the wording matches.
        .confirmationDialog(
            ModelCacheActions.deletionConfirmation(
                for: pendingDeletion ?? ModelEntry(alias: "", hfRepo: nil, sizeOnDisk: nil, cached: false)
            ).title,
            isPresented: Binding(
                get: { pendingDeletion != nil },
                set: { if !$0 { pendingDeletion = nil } }
            ),
            titleVisibility: .visible,
            presenting: pendingDeletion
        ) { entry in
            Button("Delete from disk", role: .destructive) {
                Task { await deleteAlias(entry) }
                pendingDeletion = nil
            }
            Button("Keep on disk", role: .cancel) {
                pendingDeletion = nil
            }
        } message: { entry in
            Text(ModelCacheActions.deletionConfirmation(for: entry).message)
        }
    }

    // MARK: - Header / controls

    @ViewBuilder
    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Model Management")
                .font(.title3.weight(.semibold))
            Text("Manage the on-disk model cache. Download what you need in the background; delete what you don't to reclaim space.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Models folder (issue #503)

    /// Where Rapid keeps the models it downloads. Defaults to an
    /// internal location; the user can point it at any folder — e.g. a
    /// large shared model collection on an external drive — so downloads
    /// stop clogging the internal disk. The engine, the app's disk
    /// scanning, and deletion all follow this same folder
    /// (``ModelsFolderPreference``), so the numbers stay honest.
    @ViewBuilder
    private var modelsFolderSection: some View {
        let unavailable = ModelsFolderPreference.customFolderUnavailable()
        VStack(alignment: .leading, spacing: 8) {
            Text("Models folder")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Image(systemName: customFolderPath == nil ? "internaldrive" : "externaldrive")
                        .foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(customFolderPath == nil ? "Default location" : "Custom folder")
                            .font(.callout.weight(.medium))
                        Text(effectiveFolderDisplayPath)
                            .scaledSystemFont(11, design: .monospaced)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .lineLimit(2)
                            .truncationMode(.middle)
                            .accessibilityIdentifier("Settings.ModelManagement.FolderPath")
                    }
                    Spacer(minLength: 0)
                }

                if unavailable {
                    HStack(alignment: .top, spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text("Your chosen models folder isn't available right now — the drive may be unplugged. Rapid is using its default location until it's back.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .accessibilityIdentifier("Settings.ModelManagement.FolderUnavailable")
                }

                HStack(spacing: 10) {
                    Button("Choose…") { chooseModelsFolder() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .accessibilityIdentifier("Settings.ModelManagement.ChooseFolder")
                    if customFolderPath != nil {
                        Button("Use default") { resetModelsFolder() }
                            .buttonStyle(.borderless)
                            .controlSize(.small)
                            .accessibilityIdentifier("Settings.ModelManagement.UseDefaultFolder")
                    }
                    Spacer(minLength: 0)
                }

                Text("Point Rapid at a folder where it already keeps downloaded models — for example on an external drive. New models download here; ones you already have stay where they are. Models downloaded by other apps in other formats won't appear here. New location takes effect the next time a model loads or downloads.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                    .fill(RapidTheme.card)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                    .stroke(RapidTheme.hairline, lineWidth: 1)
            )
        }
    }

    /// The path shown in the folder row. When the user picked a custom
    /// folder we show exactly what they picked (even while unavailable,
    /// so the warning has context); otherwise we resolve + show the
    /// default location so "Default location" isn't an opaque label.
    private var effectiveFolderDisplayPath: String {
        if let custom = customFolderPath { return custom }
        let resolved = BundledModel.userHFCacheURL(
            environment: ProcessInfo.processInfo.environment
        )
        return resolved?.path ?? "~/.cache/huggingface/hub"
    }

    /// Present a folder picker and persist the choice. Directories only;
    /// the app is not sandboxed so no security-scoped bookmark is
    /// needed to read/write the picked folder (including an external
    /// volume). Re-reads the catalog so the cached/size badges reflect
    /// what's in the newly chosen folder.
    private func chooseModelsFolder() {
        let panel = NSOpenPanel()
        panel.title = "Choose a models folder"
        panel.message = "Pick the folder where Rapid should keep downloaded models."
        panel.prompt = "Use Folder"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        if let current = customFolderPath {
            panel.directoryURL = URL(fileURLWithPath: current, isDirectory: true)
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        ModelsFolderPreference.setStoredPath(url.path)
        customFolderPath = ModelsFolderPreference.storedPath()
        Task { await refreshCatalog() }
    }

    /// Clear the custom folder and fall back to the default location.
    private func resetModelsFolder() {
        ModelsFolderPreference.setStoredPath(nil)
        customFolderPath = nil
        Task { await refreshCatalog() }
    }

    @ViewBuilder
    private var controlsRow: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                    TextField("Search models", text: $query)
                        .textFieldStyle(.plain)
                        .accessibilityIdentifier("Settings.ModelManagement.Search")
                    if !query.isEmpty {
                        Button {
                            query = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Clear search")
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(Color.secondary.opacity(0.08))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .stroke(RapidTheme.hairline, lineWidth: 1)
                )
                .frame(maxWidth: .infinity)

                Menu {
                    ForEach(ModelCacheActions.SortOrder.allCases) { order in
                        Button {
                            sortOrder = order
                        } label: {
                            if sortOrder == order {
                                Label(order.displayLabel, systemImage: "checkmark")
                            } else {
                                Text(order.displayLabel)
                            }
                        }
                    }
                } label: {
                    Label("Sort", systemImage: "arrow.up.arrow.down")
                        .font(.caption.weight(.medium))
                }
                .menuStyle(.borderlessButton)
                .frame(maxWidth: 80)
                .accessibilityIdentifier("Settings.ModelManagement.SortMenu")
            }
            Picker("Filter", selection: $filterMode) {
                ForEach(ModelCacheActions.FilterMode.allCases) { mode in
                    Text(mode.displayLabel).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .accessibilityIdentifier("Settings.ModelManagement.Filter")
        }
    }

    // MARK: - Recommended (issue #507)

    /// The recommended picks for this Mac's RAM: the primary (index 0)
    /// plus an optional faster alternative (only the smallest tier carries
    /// one). One card per pick.
    private var recommendedPicks: [(pick: RAMBucketedDefault.Pick, isPrimary: Bool)] {
        hardware.recommendedPicks.enumerated().map { index, pick in
            (pick, index == 0)
        }
    }

    /// alias → the badge an "All models" row carries (RECOMMENDED for the
    /// primary, FASTER for the alt). Primary wins if an alias somehow
    /// appears twice.
    private var recommendedBadgeByAlias: [String: String] {
        var map: [String: String] = [:]
        for (index, pick) in hardware.recommendedPicks.enumerated() where map[pick.alias] == nil {
            map[pick.alias] = index == 0 ? "RECOMMENDED" : "FASTER"
        }
        return map
    }

    /// Cards only make sense on the unfiltered default view. Hide them
    /// the moment the user searches or switches the cached filter so
    /// those controls act on the whole catalog without a fixed 4-card
    /// header in the way.
    private var showRecommendedSection: Bool {
        !catalog.isEmpty
            && query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && filterMode == .all
    }

    @ViewBuilder
    private var recommendedSection: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text("Recommended for your \(hardware.shortDescription)")
                .font(.caption.weight(.semibold))
                .textCase(.uppercase)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("Settings.ModelManagement.RecommendedHeader")
            ForEach(recommendedPicks, id: \.pick.alias) { entry in
                recommendedCard(pick: entry.pick, isPrimary: entry.isPrimary)
            }
        }
    }

    @ViewBuilder
    private func recommendedCard(pick: RAMBucketedDefault.Pick, isPrimary: Bool) -> some View {
        let alias = pick.alias
        let entry = entry(forAlias: alias)
        let badge = ModelCacheActions.statusBadge(
            for: entry,
            downloadJob: downloads.jobs[entry.alias],
            servingAlias: server.servingAlias
        )
        // Meters live UNDER the name/blurb (not as a fixed right column) so
        // the card fits the narrow Settings pane at the app's 720pt minimum
        // window — a fixed brand + meters + action row overflows and clips
        // the action button there (design review B1).
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Label(isPrimary ? "Best pick" : "Faster",
                      systemImage: isPrimary ? "star.fill" : "hare.fill")
                    .font(.caption.weight(.bold))
                    .labelStyle(.titleAndIcon)
                if isPrimary {
                    Text("BEST PICK")
                        .scaledSystemFont(9, weight: .bold)
                        .foregroundStyle(RapidTheme.brand)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(RapidTheme.brand.opacity(0.12)))
                }
            }
            .frame(width: 74, alignment: .leading)

            BrandIcon(alias: alias)

            VStack(alignment: .leading, spacing: 7) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(modelSubtitle(alias))
                        .font(.body.weight(.semibold))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(Self.pickStatsLine(pick))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                // No per-axis standard-benchmark meters here: the
                // recommendation card shows only the curated capability /
                // speed stats above (its single source of truth). The
                // standard-bench bars live in the "All models" rows below,
                // so a pick never shows two conflicting sets of numbers.
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            recommendedAction(entry: entry, badge: badge)
                .frame(width: 92, alignment: .trailing)
        }
        .padding(13)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                .fill(isPrimary ? RapidTheme.brandTint : RapidTheme.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                .stroke(isPrimary ? RapidTheme.brand.opacity(0.35) : RapidTheme.hairline, lineWidth: 1)
        )
        // #552 (§12 depth): lift the recommended cards above the flush
        // All-models table below them so the tier reads as a raised, more
        // important surface. Same light shadow the onboarding wizard's
        // centred card carries (QuickstartView).
        .shadow(color: Color.black.opacity(0.10), radius: 6, x: 0, y: 3)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("Settings.ModelManagement.Recommended.\(isPrimary ? "primary" : "alt")")
    }

    /// Compact stats line under a recommended card's model name:
    /// "7.6 GB · 86% capability · ~17 tok/s" (tok/s omitted when we have
    /// no local measurement for that tier). When the pick carries a
    /// ``caveat`` (e.g. a chat specialist), that word replaces the
    /// capability % — "4.8 GB · ~117 tok/s · Chat only" — because the
    /// blended score would understate conversation and overstate the rest.
    static func pickStatsLine(_ pick: RAMBucketedDefault.Pick) -> String {
        var parts = [String(format: "%.1f GB", pick.footprintGB)]
        if let caveat = pick.caveat {
            if let tps = pick.tokensPerSec {
                parts.append("~\(Int(tps.rounded())) tok/s")
            }
            parts.append(caveat)
        } else {
            parts.append("\(pick.capabilityPct)% capability")
            if let tps = pick.tokensPerSec {
                parts.append("~\(Int(tps.rounded())) tok/s")
            }
        }
        return parts.joined(separator: " · ")
    }

    @ViewBuilder
    private func recommendedAction(entry: ModelEntry, badge: ModelCacheActions.StatusBadge) -> some View {
        switch badge {
        case .cached, .inUse:
            statusBadgeView(badge)
        default:
            HStack(spacing: 8) {
                if case .notCached = badge, let gb = downloadSizeLabel(entry.alias) {
                    Text(gb).font(.caption).foregroundStyle(.secondary)
                }
                actionButton(for: entry, badge: badge)
            }
        }
    }

    // MARK: - All models section

    @ViewBuilder
    private var allModelsSection: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 6) {
                Text("All models").font(.caption.weight(.semibold)).textCase(.uppercase)
                Text("· \(catalog.count)").font(.caption).foregroundStyle(.tertiary)
            }
            .foregroundStyle(.secondary)
            // The meter legend belongs HERE — these are the only rows that
            // render the Quality · Speed bars. The recommendation cards
            // above show the curated capability / speed stats instead, so a
            // top-of-panel legend misattributed them.
            meterLegend
            columnHeader
            listSection
            if let footer = ModelCacheActions.diskUsageFooter(
                ModelCacheActions.aggregateOnDiskBytes(catalog)
            ) {
                Text(footer)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
                    .accessibilityIdentifier("Settings.ModelManagement.Footer")
            }
        }
    }

    /// One-line meaning of the two meters + the em-dash. Rendered inside
    /// the "All models" section, immediately above the only rows that
    /// carry the Quality · Speed bars. (It used to sit at the panel top,
    /// but the recommendation cards no longer show meters — they show the
    /// curated capability / speed stats — so a top-of-panel legend
    /// misattributed those curated numbers as published benchmarks.)
    @ViewBuilder
    private var meterLegend: some View {
        Text("Quality = the author's published benchmark, labelled per row (Accuracy / Code / Tool / Instructions) · Speed = tokens/sec on this class of Mac · “—” = the author hasn't published that score.")
            .scaledSystemFont(10)
            .foregroundStyle(.tertiary)
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityIdentifier("Settings.ModelManagement.MeterLegend")
    }

    @ViewBuilder
    private var columnHeader: some View {
        HStack(spacing: 10) {
            Spacer().frame(width: 15)
            Spacer().frame(width: 30)
            Text("Model").frame(maxWidth: .infinity, alignment: .leading)
            Text("Quality · Speed").frame(width: 158, alignment: .leading)
            Text("Size").frame(width: 84, alignment: .trailing)
        }
        .scaledSystemFont(10, weight: .semibold)
        .foregroundStyle(.tertiary)
        .textCase(.uppercase)
        .padding(.horizontal, 14)
    }

    // MARK: - Shared meters

    @ViewBuilder
    private func metersView(alias: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            SegmentedBenchMeter(meter: ModelMeter.qualityMeter(for: alias))
            SegmentedBenchMeter(meter: ModelMeter.speedMeter(for: alias))
        }
    }

    // MARK: - Row helpers (issue #507)

    /// The catalog entry for an alias, or a synthesised not-cached stub
    /// when the alias isn't in the catalog snapshot (defensive — the
    /// catalog is the full alias list, so this is the empty-catalog race).
    private func entry(forAlias alias: String) -> ModelEntry {
        catalog.first { $0.alias == alias }
            ?? ModelEntry(alias: alias, hfRepo: nil, sizeOnDisk: nil, cached: false)
    }

    /// "Qwen 3.6 · 35B · 4-bit" — family + params + quant.
    private func modelSubtitle(_ alias: String) -> String {
        var parts = [ModelBrandStyle.displayFamily(forAlias: alias)]
        if let p = paramsLabel(alias) { parts.append(p) }
        parts.append("\(ModelSizing.parseBitsPerWeight(alias))-bit")
        return parts.joined(separator: " · ")
    }

    /// Table meta line — subtitle plus the context window when known.
    private func rowMeta(_ alias: String) -> String {
        var s = modelSubtitle(alias)
        if let ctx = contextLabel(alias) { s += " · \(ctx)" }
        return s
    }

    private func paramsLabel(_ alias: String) -> String? {
        guard let p = ModelSizing.estimate(alias: alias).paramsBillions else { return nil }
        if p >= 1 {
            return p.truncatingRemainder(dividingBy: 1) == 0
                ? "\(Int(p))B"
                : String(format: "%.1fB", p)
        }
        return String(format: "%.1fB", p)
    }

    private func contextLabel(_ alias: String) -> String? {
        guard let ctx = ModelInfoCatalog.familyAndContext(for: alias).contextWindow else { return nil }
        if ctx >= 1024, ctx % 1024 == 0 { return "\(ctx / 1024)k" }
        return "\(ctx)"
    }

    private func downloadSizeLabel(_ alias: String) -> String? {
        let fp = ModelSizing.estimate(alias: alias)
        guard fp.weightsGB > 0 else { return nil }
        return String(format: "%.1f GB", fp.weightsGB)
    }

    @ViewBuilder
    private func favoriteStar(_ alias: String) -> some View {
        let isFav = favorites.contains(alias)
        Button {
            toggleFavorite(alias)
        } label: {
            Image(systemName: isFav ? "star.fill" : "star")
                .font(.system(size: 13))
                .foregroundStyle(isFav ? RapidTheme.amber : Color.secondary.opacity(0.45))
        }
        .buttonStyle(.plain)
        .frame(width: 15)
        .accessibilityLabel(isFav ? "Unpin \(alias)" : "Pin \(alias)")
        .accessibilityIdentifier("Settings.ModelManagement.Favorite.\(alias)")
    }

    private func toggleFavorite(_ alias: String) {
        if ModelFavorites.toggle(alias) {
            favorites.insert(alias)
        } else {
            favorites.remove(alias)
        }
    }

    @ViewBuilder
    private func rowBadge(for alias: String) -> some View {
        if let badge = recommendedBadgeByAlias[alias] {
            badgePill(badge, color: RapidTheme.brand)
        } else if ModelBrandStyle.modelType(forAlias: alias) == .vision {
            badgePill("VISION", color: Self.visionColor)
        }
    }

    @ViewBuilder
    private func badgePill(_ text: String, color: Color) -> some View {
        Text(text)
            .scaledSystemFont(9, weight: .bold)
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 1)
            .background(Capsule().fill(color.opacity(0.14)))
    }

    private static let visionColor = Color(red: 0x8E / 255.0, green: 0x44 / 255.0, blue: 0xEF / 255.0)

    /// Right-hand "Size" column: cached → check + delete; serving →
    /// label; not-cached → size + download; downloading → cancel;
    /// failed → retry. Preserves the accessibility identifiers the
    /// original text buttons carried so existing selectors still resolve.
    @ViewBuilder
    private func sizeAction(entry: ModelEntry, badge: ModelCacheActions.StatusBadge) -> some View {
        switch badge {
        case .cached:
            HStack(spacing: 8) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(RapidTheme.green)
                    .accessibilityLabel("On disk")
                Button(role: .destructive) {
                    pendingDeletion = entry
                } label: {
                    Image(systemName: "trash").font(.system(size: 11))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .accessibilityLabel("Delete \(entry.alias) from disk")
                .accessibilityIdentifier("Settings.ModelManagement.Delete.\(entry.alias)")
            }
        case .inUse:
            Text("Serving")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
        case .notCached:
            HStack(spacing: 8) {
                if let gb = downloadSizeLabel(entry.alias) {
                    Text(gb).font(.caption).foregroundStyle(.secondary)
                }
                Button {
                    _ = downloads.startDownload(alias: entry.alias, hfPath: entry.hfRepo)
                } label: {
                    Image(systemName: "arrow.down.circle").font(.system(size: 15))
                }
                .buttonStyle(.plain)
                .foregroundStyle(RapidTheme.brand)
                .accessibilityLabel("Download \(entry.alias)")
                .accessibilityIdentifier("Settings.ModelManagement.Download.\(entry.alias)")
            }
        case .downloading(let pct):
            Button {
                downloads.cancelDownload(alias: entry.alias)
            } label: {
                Text(pct.map { "\($0)%" } ?? "Cancel").font(.caption)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Cancel download")
            .accessibilityLabel(pct.map { "Cancel download, \($0) percent" } ?? "Cancel download")
            .accessibilityIdentifier("Settings.ModelManagement.Cancel.\(entry.alias)")
        case .failed:
            Button {
                downloads.dismissJob(alias: entry.alias)
                _ = downloads.startDownload(alias: entry.alias, hfPath: entry.hfRepo)
            } label: {
                Text("Retry").font(.caption)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .accessibilityIdentifier("Settings.ModelManagement.Retry.\(entry.alias)")
        }
    }

    // MARK: - List

    private var visibleEntries: [ModelEntry] {
        let filtered = ModelCacheActions.filter(catalog, by: filterMode, query: query)
        let sorted = ModelCacheActions.sorted(filtered, order: sortOrder)
        return ModelFavorites.favoritesFirst(sorted, favorites: favorites)
    }

    @ViewBuilder
    private var listSection: some View {
        let entries = visibleEntries
        if entries.isEmpty {
            Text(noMatchesCopy)
                .font(.callout)
                .foregroundStyle(.secondary)
                .padding(.vertical, 12)
        } else {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(entries.enumerated()), id: \.element.alias) { idx, entry in
                    row(for: entry)
                    if idx < entries.count - 1 {
                        Divider().opacity(0.5)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                    .fill(RapidTheme.card)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                    .stroke(RapidTheme.hairline, lineWidth: 1)
            )
        }
    }

    private var noMatchesCopy: String {
        if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            switch filterMode {
            case .all:
                return "No models found. Restart Rapid-MLX to try again."
            case .cached:
                return "Nothing cached on disk yet. Pick a row from \"Not cached\" and hit Download."
            case .notCached:
                return "Every model in the catalog is already downloaded."
            }
        }
        return "No matches for \"\(query)\"."
    }

    @ViewBuilder
    private func row(for entry: ModelEntry) -> some View {
        let badge = ModelCacheActions.statusBadge(
            for: entry,
            downloadJob: downloads.jobs[entry.alias],
            servingAlias: server.servingAlias
        )
        HStack(spacing: 10) {
            favoriteStar(entry.alias)
            BrandIcon(alias: entry.alias)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 7) {
                    Text(entry.alias)
                        .font(.body.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    rowBadge(for: entry.alias)
                }
                Text(rowMeta(entry.alias))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            metersView(alias: entry.alias)
                .frame(width: 158)
            sizeAction(entry: entry, badge: badge)
                .frame(width: 84, alignment: .trailing)
        }
        .padding(.vertical, 8)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("Settings.ModelManagement.Row.\(entry.alias)")
    }

    @ViewBuilder
    private func statusBadgeView(_ badge: ModelCacheActions.StatusBadge) -> some View {
        switch badge {
        case .cached:
            pill(text: "On disk", color: RapidTheme.green)
        case .inUse:
            pill(text: "In use", color: RapidTheme.green)
        case .notCached:
            pill(text: "Not cached", color: .secondary)
        case .downloading(let pct):
            let label: String = {
                if let pct {
                    return "Downloading… \(pct)%"
                }
                return "Downloading…"
            }()
            pill(text: label, color: RapidTheme.brand)
        case .failed:
            pill(text: "Failed", color: .red)
        }
    }

    @ViewBuilder
    private func pill(text: String, color: Color) -> some View {
        Text(text)
            .font(.caption.weight(.medium))
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                Capsule(style: .continuous)
                    .fill(color.opacity(0.15))
            )
            .lineLimit(1)
            .accessibilityIdentifier("Settings.ModelManagement.Status.\(text)")
    }

    /// The prominent action button on a Recommended CARD. The dense
    /// table row uses ``sizeAction`` instead; this helper is card-only,
    /// so its identifiers are namespaced ``.Recommended.*`` to stay
    /// unique when the same alias also appears as a table row below
    /// (design review / correctness MINOR: duplicate identifiers).
    @ViewBuilder
    private func actionButton(for entry: ModelEntry, badge: ModelCacheActions.StatusBadge) -> some View {
        switch badge {
        case .cached:
            Button(role: .destructive) {
                pendingDeletion = entry
            } label: {
                Text("Delete")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .accessibilityIdentifier("Settings.ModelManagement.Recommended.Delete.\(entry.alias)")
        case .inUse:
            // rapid-mlx holds the weights mmap'd — a mid-serve rm would
            // either fail or corrupt inference. Mirror the picker's
            // "currently-serving rows are off-limits" rule.
            Text("Serving")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
        case .notCached:
            Button {
                _ = downloads.startDownload(alias: entry.alias, hfPath: entry.hfRepo)
            } label: {
                Text("Download")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .accessibilityIdentifier("Settings.ModelManagement.Recommended.Download.\(entry.alias)")
        case .downloading:
            Button {
                downloads.cancelDownload(alias: entry.alias)
            } label: {
                Text("Cancel")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .accessibilityIdentifier("Settings.ModelManagement.Recommended.Cancel.\(entry.alias)")
        case .failed:
            Button {
                downloads.dismissJob(alias: entry.alias)
                _ = downloads.startDownload(alias: entry.alias, hfPath: entry.hfRepo)
            } label: {
                Text("Retry")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .accessibilityIdentifier("Settings.ModelManagement.Recommended.Retry.\(entry.alias)")
        }
    }

    // MARK: - States

    @ViewBuilder
    private var loadingState: some View {
        HStack(spacing: 10) {
            ProgressView().controlSize(.small)
            Text("Loading model catalog…")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 12)
    }

    @ViewBuilder
    private var emptyState: some View {
        Text("Couldn't load the model list. Restart Rapid-MLX to try again.")
            .font(.callout)
            .foregroundStyle(.secondary)
            .padding(.vertical, 12)
    }

    @ViewBuilder
    private func errorBanner(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.callout)
                .foregroundStyle(.red)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            Button("Dismiss") { lastError = nil }
                .buttonStyle(.plain)
                .font(.caption)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.red.opacity(0.08))
        )
    }

    @ViewBuilder
    private func freedBanner(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "checkmark.seal.fill")
                .foregroundStyle(RapidTheme.green)
            Text(message)
                .font(.callout)
                .foregroundStyle(RapidTheme.green)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            Button("Dismiss") { lastFreed = nil }
                .buttonStyle(.plain)
                .font(.caption)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(RapidTheme.green.opacity(0.08))
        )
    }

    // MARK: - Job reconciliation

    /// Coarse fingerprint of every job's status. ``onChange`` /
    /// ``task(id:)`` only fires when this changes — running →
    /// completed, running → failed, running → cancelled, or new
    /// jobs appearing / dropping. tqdm phase updates inside
    /// ``.running`` are intentionally ignored so we don't refresh
    /// the catalog 30 times a second mid-pull.
    ///
    /// Computed property (not @State) so it always reflects the
    /// live ``downloads.jobs`` snapshot; the @State
    /// ``lastObservedJobStatuses`` is the previous-frame copy that
    /// ``reconcileJobs`` diffs against to detect transitions.
    private var jobStatusFingerprint: [String: ObservedJobStatus] {
        var out: [String: ObservedJobStatus] = [:]
        out.reserveCapacity(downloads.jobs.count)
        for (alias, job) in downloads.jobs {
            out[alias] = ObservedJobStatus(job.status)
        }
        return out
    }

    /// Persistent reconciliation loop. Spins at 500 ms while any
    /// job is running, settles to 5 s when idle. Re-reads the
    /// catalog whenever a running → terminal transition is
    /// detected so the row flips from ``Downloading…`` to ``On
    /// disk`` / ``Failed`` / ``Not cached`` at the same cadence
    /// the user sees on the picker's download strip.
    ///
    /// The loop honours ``Task.isCancelled`` (the panel's
    /// ``.task`` is cancelled when the Settings view is dismissed
    /// or rebuilt) so this never out-lives the surface.
    private func jobReconciliationLoop() async {
        while !Task.isCancelled {
            let current = jobStatusFingerprint
            let previous = lastObservedJobStatuses
            let shouldRefresh = Self.shouldRefreshCatalog(
                previous: previous,
                current: current
            )
            lastObservedJobStatuses = current
            if shouldRefresh {
                await refreshCatalog()
            }
            // Hot poll while a pull is mid-flight; settle to a
            // light beat when no running jobs remain so the
            // idle Settings tab isn't waking on a sub-second
            // tick forever.
            let anyRunning = current.values.contains(.running)
            let interval: UInt64 = anyRunning ? 500_000_000 : 5_000_000_000
            try? await Task.sleep(nanoseconds: interval)
        }
    }

    /// Pure predicate driving the ``reconcileJobs`` decision —
    /// exposed ``static`` so a unit test can pin every transition
    /// branch without standing up a SwiftUI host. The catalog
    /// needs a re-read whenever any alias' status transitioned
    /// FROM ``.running`` to a terminal state (``.completed`` is
    /// the on-disk flip; ``.failed`` / ``.cancelled`` are the
    /// give-up flips that also need the row to re-resolve so the
    /// action button settles into Retry / Download).
    static func shouldRefreshCatalog(
        previous: [String: ObservedJobStatus],
        current: [String: ObservedJobStatus]
    ) -> Bool {
        for (alias, newStatus) in current {
            let oldStatus = previous[alias]
            if oldStatus == .running && newStatus != .running {
                return true
            }
        }
        return false
    }

    // MARK: - Actions

    private func refreshCatalog() async {
        loading = true
        defer { loading = false }
        guard let binary = server.binaryPath else {
            catalog = []
            return
        }
        catalog = await ModelCatalog.load(binary: binary)
    }

    private func deleteAlias(_ entry: ModelEntry) async {
        lastError = nil
        lastFreed = nil
        let outcome = await ModelCacheActions.runDeletion(
            for: entry,
            binaryPath: server.binaryPath
        )
        switch outcome {
        case .success(let message, _):
            lastFreed = message
            // Other surfaces (picker dropdown, upgrade banner) hold
            // their own catalog snapshots; without this they keep
            // showing the deleted model as downloaded.
            downloads.markCacheChanged()
            await refreshCatalog()
        case .failure(let message):
            lastError = message
        }
    }
}
