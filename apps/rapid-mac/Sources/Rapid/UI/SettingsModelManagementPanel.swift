import AppKit
import SwiftUI

/// Settings → Model Management — the single surface for everything
/// about your models: the file-manager-style cache inspector (issue
/// #210) plus the model-behaviour preferences.
///
/// Why it owns all of it: user feedback (2026-06-16) called out that
/// the picker dropdown is overloaded — it conflates "switch active
/// alias" with "manage the on-disk cache", and casual users miss the
/// right-click affordances. This dedicated sidebar tab took over cache
/// state; the picker stays a switcher and this panel is the inspector.
/// A separate, older "Models" tab used to duplicate the download/delete
/// list and carry two behaviour toggles; it was folded in here so users
/// no longer face two competing model surfaces.
///
/// Layout (top to bottom):
///   * ``Models folder`` + ``Preferences`` cards — where models live,
///     and the picker-visibility / auto-start toggles.
///   * Search box + ``All / Cached / Not cached`` segmented
///     filter + sort menu — so a user with 60
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

    // Seeded from the process-wide cache rather than starting empty: an
    // empty start re-rendered the spinner on every visit regardless of
    // whether the data was already cached.
    @State private var catalog: [ModelEntry] = ModelCatalogCache.seed(generation: 0) ?? []
    @State private var loading: Bool = ModelCatalogCache.seed(generation: 0) == nil
    @State private var pendingDeletion: ModelEntry?
    @State private var lastError: String?
    @State private var lastFreed: String?
    @State private var modelsVolumeFreeBytes: Int64?

    @State private var query: String = ""
    /// Which capability tab is showing (Chat vs Image vs Audio vs future Video). Model
    /// Management manages every kind, but never mixes them in one list.
    @State private var capability: ModelKind = .chat
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

    /// Power-user override for the picker's sub-1B filter (cycle-7).
    /// Defaults OFF so first-time users don't meet `qwen3-0.6b-*` in the
    /// dropdown — those tinies hallucinate within 1-2 turns and read as
    /// broken during evaluation. Lives here, alongside the cache it
    /// governs, rather than in a second "Models" tab.
    @AppStorage(ModelPickerVisibility.showAllStorageKey) private var showAllModels: Bool = false

    /// Launch-time auto-start opt-out (FU-1). Defaults ON (the v0.7.x
    /// behaviour) so upgrades see no change; flipping OFF skips the
    /// next-launch spawn while leaving every manual start path untouched.
    @AppStorage(AutoStartPreference.storageKey) private var autoStartOnLaunch: Bool = AutoStartPreference.defaultValue

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

    @Environment(\.settingsContentIsCompact) private var isCompact

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            header
            modelsFolderSection
            storageOverviewSection
            preferencesSection
            capabilityTabs
            controlsRow
            if capability == .chat {
                if showRecommendedSection {
                    recommendedSection
                }
            }
            if loading && catalog.isEmpty {
                loadingState
            } else if catalog.isEmpty {
                emptyState
            } else {
                allModelsSection
            }
            if let lastError {
                InlineNotice(
                    message: lastError,
                    tone: .error,
                    actionTitle: "Dismiss",
                    actionIdentifier: "Settings.ModelManagement.DismissError",
                    action: { self.lastError = nil }
                )
            }
            if let lastFreed {
                InlineNotice(
                    message: lastFreed,
                    tone: .success,
                    actionTitle: "Dismiss",
                    actionIdentifier: "Settings.ModelManagement.DismissSuccess",
                    action: { self.lastFreed = nil }
                )
            }
        }
        // Loading, empty, and catalog branches are deliberately exclusive.
        // Animating this container retains both conditional trees during the
        // transition, which can overlay the spinner on stale model rows.
        .task {
            await refreshCatalog()
            refreshStorageCapacity()
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
                for: pendingDeletion ?? ModelEntry(alias: "", hfRepo: nil, sizeOnDisk: nil, cached: false),
                isServing: pendingDeletion?.alias == server.servingAlias
            ).title,
            isPresented: Binding(
                get: { pendingDeletion != nil },
                set: { if !$0 { pendingDeletion = nil } }
            ),
            titleVisibility: .visible,
            presenting: pendingDeletion
        ) { entry in
            Button(
                entry.alias == server.servingAlias ? "Stop and delete" : "Delete from disk",
                role: .destructive
            ) {
                Task { await deleteAlias(entry) }
                pendingDeletion = nil
            }
            .accessibilityIdentifier("Settings.ModelManagement.ConfirmDelete")
            Button("Keep on disk", role: .cancel) {
                pendingDeletion = nil
            }
            .accessibilityIdentifier("Settings.ModelManagement.KeepOnDisk")
        } message: { entry in
            Text(ModelCacheActions.deletionConfirmation(
                for: entry,
                isServing: entry.alias == server.servingAlias
            ).message)
        }
    }

    // MARK: - Header / controls

    @ViewBuilder
    private var header: some View {
        SectionHeader(
            "Model Management",
            subtitle: "Manage the on-disk model cache. Download what you need in the background; delete what you don't to reclaim space.",
            emphasis: .page
        )
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
        SettingsSection("Models folder") {
            VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                    Image(systemName: customFolderPath == nil ? "internaldrive" : "externaldrive")
                        .foregroundStyle(RapidTheme.utilityActionLabel)
                        .frame(width: RapidTheme.Layout.iconSlot)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                        Text(customFolderPath == nil ? "Default location" : "Custom folder")
                            .font(RapidFont.bodyEmphasis)
                            .foregroundStyle(RapidTheme.textPrimary)
                        Text(effectiveFolderDisplayPath)
                            .font(RapidFont.code)
                            .foregroundStyle(RapidTheme.textSecondary)
                            .textSelection(.enabled)
                            .lineLimit(2)
                            .truncationMode(.middle)
                            .accessibilityIdentifier("Settings.ModelManagement.FolderPath")
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                if unavailable {
                    InlineNotice(
                        message: "Your chosen models folder isn't available right now — the drive may be unplugged. Rapid is using its default location until it's back.",
                        tone: .warning
                    )
                    .accessibilityIdentifier("Settings.ModelManagement.FolderUnavailable")
                }

                HStack(spacing: RapidTheme.Space.sm) {
                    Button("Choose…") { chooseModelsFolder() }
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier("Settings.ModelManagement.ChooseFolder")
                    if customFolderPath != nil {
                        Button("Use default") { resetModelsFolder() }
                            .buttonStyle(.rapidTertiary)
                            .accessibilityIdentifier("Settings.ModelManagement.UseDefaultFolder")
                    }
                    Spacer(minLength: 0)
                }

                Text("Point Rapid at a folder where it already keeps downloaded models — for example on an external drive. New models download here; ones you already have stay where they are. Models downloaded by other apps in other formats won't appear here. New location takes effect the next time a model loads or downloads.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Preferences

    /// The two model-behaviour toggles that used to live in a separate
    /// "Models" tab. Folded in here — the surface that already owns
    /// everything about your models — as one labelled card so the app
    /// has a single place to manage models rather than two competing
    /// ones. Styled to match ``modelsFolderSection`` above: a secondary
    /// section label over a hairline card, the two toggles split by a
    /// divider so they read as a pair without two floating boxes.
    @ViewBuilder
    private var preferencesSection: some View {
        SettingsSection("Preferences") {
                Toggle(isOn: $showAllModels) {
                    SettingsRowLabel(
                        title: "Show small (<1B) models in the picker",
                        description: "Sub-1B models (qwen3-0.6b-*) are hidden from the model picker by default — they hallucinate within 1-2 turns and are intended for unit tests, not chat. Turn on to see every model, including the tiny ones."
                    )
                }
                .toggleStyle(TrailingSettingsToggleStyle())
                .accessibilityLabel("Show small models in the picker")
                .accessibilityHint("Sub-1B models are hidden by default — they hallucinate within 1-2 turns and are intended for unit tests, not chat.")
                // Identifier intentionally KEPT as `Settings.Models.*` after
                // the move from the old Models tab: it is a stable AX hook, so
                // relocating the control shouldn't rename it out from under any
                // VoiceOver/automation client that already targets it.
                .accessibilityIdentifier("Settings.Models.ShowAllModelsToggle")

                SettingsRowDivider()

                Toggle(isOn: $autoStartOnLaunch) {
                    SettingsRowLabel(
                        title: "Auto-start model on launch",
                        description: "On launch, Rapid-MLX loads your last-used model into memory so the chat is interactive immediately. Nothing loads while first-run setup is still open. Turn off if you sometimes open Rapid-MLX just to browse past conversations — you can still start a model manually by picking one in the message box and sending."
                    )
                }
                .toggleStyle(TrailingSettingsToggleStyle())
                .accessibilityLabel("Auto-start model on launch")
                .accessibilityHint("When off, opening Rapid-MLX will not load a model until you start one manually from the picker.")
                // Stable AX hook kept as `Settings.Models.*` across the move.
                .accessibilityIdentifier("Settings.Models.AutoStartOnLaunchToggle")
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

    private var effectiveModelsFolderURL: URL? {
        ModelsFolderPreference.validatedOverrideURL()
            ?? BundledModel.userHFCacheURL(environment: ProcessInfo.processInfo.environment)
    }

    private func refreshStorageCapacity() {
        modelsVolumeFreeBytes = effectiveModelsFolderURL.flatMap {
            DiskSpaceProbe.freeBytes(forPath: $0.path)
        }
    }

    /// Always-visible cache overview. Unlike the old footer this spans Chat,
    /// Image, and Audio, and it excludes read-only entries owned by another
    /// runtime because this panel cannot reclaim those bytes.
    @ViewBuilder
    private var storageOverviewSection: some View {
        let managed = catalog.filter { !$0.isExternal }
        let usage = ModelCacheActions.aggregateOnDiskBytes(managed)
        let largest = ModelCacheActions.largestManagedEntry(managed)
        // Upstream (#1822) added this card in the pre-migration idiom —
        // `.callout` type, the legacy 12pt `cardRadius`, a local 12pt
        // inset. Same content and the same identifiers, moved onto the
        // shared section so it does not ship as the one un-migrated card
        // in the window.
        SettingsSection("Disk overview") {
            HStack(spacing: RapidTheme.Space.md) {
                Label("Models", systemImage: "internaldrive")
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(RapidTheme.textPrimary)
                Spacer(minLength: RapidTheme.Space.sm)
                Text(ModelCacheActions.storageSummary(
                    usage: usage,
                    freeBytes: modelsVolumeFreeBytes
                ))
                .font(RapidFont.metric)
                .foregroundStyle(RapidTheme.textSecondary)
                .accessibilityIdentifier("Settings.ModelManagement.StorageSummary")
            }
            if let largest {
                SettingsRowDivider()
                HStack(spacing: RapidTheme.Space.sm) {
                    Text("Largest")
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                    Text(largest.alias)
                        .font(RapidFont.bodyEmphasis)
                        .foregroundStyle(RapidTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: RapidTheme.Space.sm)
                    Text(largest.sizeOnDisk ?? "")
                        .font(RapidFont.metric)
                        .foregroundStyle(RapidTheme.textSecondary)
                }
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("Settings.ModelManagement.LargestModel")
            }
        }
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
        refreshStorageCapacity()
        Task { await refreshCatalog() }
    }

    /// Clear the custom folder and fall back to the default location.
    private func resetModelsFolder() {
        ModelsFolderPreference.setStoredPath(nil)
        customFolderPath = nil
        refreshStorageCapacity()
        Task { await refreshCatalog() }
    }

    /// Capability tabs — Chat / Image / Audio (/ Video, once it has aliases). Only
    /// shown when there's more than one kind installed, so a chat-only setup
    /// looks exactly as it did before image models existed.
    @ViewBuilder
    private var capabilityTabs: some View {
        if availableKinds.count > 1 {
            RapidSegmentedControl(
                selection: $capability,
                options: availableKinds.map {
                    .init(value: $0, title: "\($0.tabLabel) models")
                },
                accessibilityLabel: "Model type"
            )
            .accessibilityIdentifier("Settings.ModelManagement.CapabilityTabs")
        }
    }

    @ViewBuilder
    private var controlsRow: some View {
        VStack(spacing: RapidTheme.Space.sm) {
            HStack(spacing: RapidTheme.Space.sm) {
                HStack(spacing: RapidTheme.Space.xs) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(RapidTheme.utilityActionLabel)
                        .accessibilityHidden(true)
                    TextField("Search models", text: $query)
                        .textFieldStyle(.plain)
                        .font(RapidFont.body)
                        .accessibilityIdentifier("Settings.ModelManagement.Search")
                    if !query.isEmpty {
                        QuietIconButton(
                            symbol: "xmark.circle.fill",
                            label: "Clear search",
                            size: RapidTheme.ControlHeight.mini
                        ) {
                            query = ""
                        }
                        .accessibilityIdentifier("Settings.ModelManagement.ClearSearch")
                    }
                }
                .padding(.horizontal, RapidTheme.Space.sm)
                .frame(height: RapidTheme.ControlHeight.medium)
                // The one input treatment: same radius and border the
                // composer uses, instead of a local 7pt capsule over a
                // hand-mixed `Color.secondary.opacity(0.08)`.
                .background(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                        .fill(RapidTheme.surfaceRaised)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                        .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
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
                        .accessibilityIdentifier("Settings.ModelManagement.Sort.\(order.rawValue)")
                    }
                } label: {
                    Label("Sort", systemImage: "arrow.up.arrow.down")
                        .font(RapidFont.body)
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                // `.tint(nil)` as well as `.foregroundStyle`: the scene
                // applies `.tint(brandAmber)` app-wide, and a borderless
                // Menu's label reads the TINT, not the foreground style —
                // so without this the Sort control rendered amber and read
                // as the page's primary action rather than a utility next
                // to the search field.
                .tint(nil)
                .foregroundStyle(RapidTheme.utilityActionLabel)
                .accessibilityIdentifier("Settings.ModelManagement.SortMenu")
            }
            RapidSegmentedControl(
                selection: $filterMode,
                options: ModelCacheActions.FilterMode.allCases.map {
                    .init(value: $0, title: $0.displayLabel)
                },
                accessibilityLabel: "Filter"
            )
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
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            SectionHeader("Recommended for your \(hardware.shortDescription)")
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
            // One marker, not two. This column used to render "Best pick"
            // as a label AND "BEST PICK" as a capsule directly beneath it
            // — the same two words, stacked, on the same card. The label
            // stays because it carries the star and reads at a glance; the
            // capsule goes because the card's own brand tint, brand border
            // and shadow already say "this is the featured one", and the
            // table below still pills the same alias as RECOMMENDED.
            Label(isPrimary ? "Best pick" : "Faster",
                  systemImage: isPrimary ? "star.fill" : "hare.fill")
                .font(RapidFont.caption)
                .foregroundStyle(isPrimary ? RapidTheme.brandPrimaryDeep : RapidTheme.textSecondary)
                .labelStyle(.titleAndIcon)
                .frame(width: RecommendedCardLayout.markerColumnWidth, alignment: .leading)

            BrandIcon(alias: alias)

            VStack(alignment: .leading, spacing: 7) {
                VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                    Text(modelSubtitle(alias))
                        .font(RapidFont.bodyEmphasis)
                        .foregroundStyle(RapidTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(Self.pickStatsLine(pick))
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
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

            // ``fixedSize`` is the actual guarantee: it hands the action
            // its intrinsic width so a label can never be compressed into
            // an ellipsis. The frame is alignment only — a floor wide
            // enough for the longest label this slot renders, so the two
            // stacked cards line up, with no ceiling to clip against.
            //
            // The bug this replaces: a hard ``.frame(width: 92)`` around a
            // caption + button cluster. The caption ate ~50pt and the
            // prominent "Download" button was left with ~40, rendering as
            // "Dow…" for every user whose best pick wasn't cached. Widening
            // the Settings window did nothing — the clamp was on the card.
            recommendedAction(entry: entry, badge: badge)
                .fixedSize(horizontal: true, vertical: false)
                .frame(minWidth: RecommendedCardLayout.actionColumnWidth, alignment: .trailing)
        }
        .padding(SettingsCardMetrics.inset(isCompact: isCompact))
        // The featured card is distinguished by the product's selection
        // pair — amber tint and an amber edge — not by a steel-blue tint
        // and a drop shadow. The shadow is gone: the tint plus the border
        // already separate this tier from the flush table below it, and a
        // shadow was the only one of its kind on the page.
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                .fill(isPrimary ? RapidTheme.brandPrimaryTint : RapidTheme.surfaceRaised)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                .strokeBorder(
                    isPrimary ? RapidTheme.brandPrimary.opacity(0.35) : RapidTheme.hairline,
                    lineWidth: 1
                )
        )
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

    /// The card's trailing slot: a status pill when there is nothing to
    /// do, otherwise the one action button.
    ///
    /// No size caption here. ``pickStatsLine`` already opens with the
    /// pick's footprint two columns to the left ("7.6 GB · 86%
    /// capability · ~17 tok/s"), so a second "7.6 GB" beside the button
    /// was the same fact twice on one card — and the two were computed
    /// from different sources (the curated ``Pick.footprintGB`` vs
    /// ``ModelSizing.estimate``), so they could disagree by a rounding
    /// step while claiming to be the same number. It was also what
    /// starved the button into "Dow…".
    @ViewBuilder
    private func recommendedAction(entry: ModelEntry, badge: ModelCacheActions.StatusBadge) -> some View {
        switch badge {
        case .cached, .inUse:
            statusBadgeView(badge)
        default:
            actionButton(for: entry, badge: badge)
        }
    }

    // MARK: - All models section

    @ViewBuilder
    private var allModelsSection: some View {
        // Resolved once and handed to both the heading and the list, so
        // the number in the heading and the rows under it can never
        // describe different sets.
        let entries = visibleEntries
        let kindEntries = catalog.filter { $0.kind == capability }
        let heading = ModelCacheActions.listHeading(
            filter: filterMode,
            query: query,
            visibleCount: entries.count,
            totalCount: kindEntries.count
        )
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            ModelsTableHeading(heading: heading)
            // The meter legend + Quality·Speed column belong to CHAT rows only
            // — image models have no tok/s benchmark, so their tab shows a
            // leaner row (name · repo · size · download). They also need
            // room: under ``compactContentWidth`` the 158pt meters column
            // squeezes the model name to nothing, so the meters stand down
            // and the name keeps the width. Nothing actionable is hidden —
            // the meters are read-only.
            if capability == .chat && showsMeters {
                meterLegend
                columnHeader
            }
            listSection(entries)
            if let footer = ModelCacheActions.diskUsageFooter(
                ModelCacheActions.aggregateOnDiskBytes(kindEntries)
            ) {
                Text(footer)
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .padding(.top, RapidTheme.Space.xxs)
                    .accessibilityIdentifier("Settings.ModelManagement.Footer")
            }
        }
    }

    /// One-line meaning of the two meters + the explicit unknown state. Rendered inside
    /// the "All models" section, immediately above the only rows that
    /// carry the Quality · Speed bars. (It used to sit at the panel top,
    /// but the recommendation cards no longer show meters — they show the
    /// curated capability / speed stats — so a top-of-panel legend
    /// misattributed those curated numbers as published benchmarks.)
    @ViewBuilder
    private var meterLegend: some View {
        Text("Quality = published benchmark, labelled per row (Accuracy / Code / Tool / Instructions) · Speed = measured tokens/sec on this class of Mac · Untested = no compatible result recorded yet.")
            .font(RapidFont.caption)
            .foregroundStyle(RapidTheme.textTertiary)
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityIdentifier("Settings.ModelManagement.MeterLegend")
    }

    @ViewBuilder
    private var columnHeader: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Spacer().frame(width: 15)
            Spacer().frame(width: 30)
            Text("Model").frame(maxWidth: .infinity, alignment: .leading)
            Text("Quality · Speed").frame(width: ModelTableLayout.metersColumnWidth, alignment: .leading)
            Text("Size").frame(width: ModelTableLayout.sizeColumnWidth, alignment: .trailing)
        }
        .font(RapidFont.groupLabel)
        .foregroundStyle(RapidTheme.textTertiary)
        .padding(.horizontal, RapidTheme.Space.lg)
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

    /// ESTIMATED download size for a model that is not on disk, derived
    /// from the alias string by ``ModelSizing`` — not a measurement of
    /// anything. The caller renders it with a leading "~" so it can't be
    /// mistaken for the measured on-disk figure cached rows in the same
    /// column now show; #1550 has a case where this estimate lands ~12%
    /// under the real download.
    nonisolated static func downloadSizeLabel(_ alias: String) -> String? {
        let fp = ModelSizing.estimate(alias: alias)
        guard fp.weightsGB > 0 else { return nil }
        return String(format: "%.1f GB", fp.weightsGB)
    }

    /// MEASURED size of a cached model, exactly as ``rapid-mlx ls``
    /// reported it (and as Settings → Models quotes it, so the two
    /// surfaces can't print different numbers for the same model).
    /// Never an estimate: if the measurement is missing, the row shows no
    /// size rather than substituting ``ModelSizing``'s guess.
    nonisolated static func onDiskSizeLabel(_ entry: ModelEntry) -> String? {
        guard let raw = entry.sizeOnDisk?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else { return nil }
        return raw
    }

    @ViewBuilder
    private func favoriteStar(_ alias: String) -> some View {
        let isFav = favorites.contains(alias)
        Button {
            toggleFavorite(alias)
        } label: {
            Image(systemName: isFav ? "star.fill" : "star")
                .font(.system(size: 13))
                .foregroundStyle(isFav ? RapidTheme.brandPrimaryDeep : RapidTheme.textTertiary)
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
            badgePill(badge, color: RapidTheme.brandPrimaryDeep)
        } else if ModelBrandStyle.modelType(forAlias: alias) == .vision {
            badgePill("VISION", color: RapidTheme.textSecondary)
        }
        ForEach(ModelCacheActions.retentionBadges(
            alias: alias,
            starterAlias: QuickstartCoordinator.defaultChoice.alias,
            lastServedAlias: ServerManager.lastServedAlias()
        ), id: \.self) { badge in
            badgePill(badge, color: badge == "STARTER" ? RapidTheme.amber : RapidTheme.green)
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

    // NOTE: a hard-coded purple ``visionColor`` used to live here. It was
    // the only violet in the product and carried no meaning the palette
    // owns — "this model takes images" is a capability, not a status — so
    // the VISION pill now uses the neutral secondary text colour and is
    // distinguished by its word, not by a unique hue.

    /// Right-hand "Size" column: cached → check + delete; serving →
    /// label; not-cached → size + download; downloading → cancel;
    /// failed → retry. Preserves the accessibility identifiers the
    /// original text buttons carried so existing selectors still resolve.
    @ViewBuilder
    private func sizeAction(entry: ModelEntry, badge: ModelCacheActions.StatusBadge) -> some View {
        switch badge {
        case .cached:
            // The size is the point of this tab. A cached row used to show
            // the check and the trash and nothing else, so the one surface
            // whose job is reclaiming space never said how much any given
            // model would reclaim — while its own "Size (largest first)"
            // sort ordered the table by exactly that number.
            HStack(spacing: ModelTableLayout.cellSpacing) {
                Image(systemName: "checkmark.circle.fill")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.statusReady)
                    .accessibilityHidden(true)
                if let size = Self.onDiskSizeLabel(entry) {
                    Text(size)
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .lineLimit(1)
                        .minimumScaleFactor(ModelTableLayout.cellMinimumScaleFactor)
                        .help(
                            entry.isExternal
                                ? "Measured size on disk. Downloaded by another app — "
                                    + "Rapid can't delete it."
                                : "Measured size on disk. Deleting frees this much."
                        )
                        .accessibilityLabel("On disk, \(size)")
                } else {
                    // No measurement from ``rapid-mlx ls``. Say "On disk"
                    // and stop — substituting the alias-derived estimate
                    // here would quote a guess as a measurement.
                    Text("On disk")
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .lineLimit(1)
                        .minimumScaleFactor(ModelTableLayout.cellMinimumScaleFactor)
                }
                // A model another MLX runtime downloaded gets no delete
                // button (#1718). Deletion rebuilds
                // ``<hub-root>/models--<repo>``, which is not where this
                // one lives, so the button would either do nothing or
                // remove an unrelated hub entry of the same name. We did
                // not download it, so it is not ours to remove — same
                // reasoning as the absent delete on a serving model below.
                if !entry.isExternal {
                    // Destructive, and now visibly so: the shared quiet
                    // icon button turns red under the pointer instead of
                    // staying the same grey as the copy glyph beside it.
                    QuietIconButton(
                        symbol: "trash",
                        label: "Delete \(entry.alias) from disk",
                        help: "Delete from disk",
                        tint: RapidTheme.statusError,
                        size: RapidTheme.ControlHeight.mini
                    ) {
                        pendingDeletion = entry
                    }
                    .accessibilityIdentifier("Settings.ModelManagement.Delete.\(entry.alias)")
                }
            }
        case .inUse:
            // A serving model is a CACHED model, so it owes the same
            // answer as any other cached row: how much disk it is using.
            // Showing only "Serving" left the one model the user is most
            // likely to be weighing as the single row with no size.
            // Deletion remains available, but its confirmed action stops the
            // server before touching weights that are currently mmap'd.
            HStack(spacing: ModelTableLayout.cellSpacing) {
                if let size = Self.onDiskSizeLabel(entry) {
                    Text(size)
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .lineLimit(1)
                        .minimumScaleFactor(ModelTableLayout.cellMinimumScaleFactor)
                        .help("Measured size on disk. Stop this model before deleting it.")
                        .accessibilityLabel("On disk, \(size)")
                }
                Text("Serving")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.statusReady)
                    .lineLimit(1)
                    .minimumScaleFactor(ModelTableLayout.cellMinimumScaleFactor)
                QuietIconButton(
                    symbol: "trash",
                    label: "Stop serving and delete \(entry.alias) from disk",
                    help: "Stop serving and delete this model from disk.",
                    tint: RapidTheme.statusError,
                    size: RapidTheme.ControlHeight.mini
                ) {
                    pendingDeletion = entry
                }
                .accessibilityIdentifier("Settings.ModelManagement.Delete.\(entry.alias)")
            }
        case .notCached:
            HStack(spacing: 8) {
                if let gb = Self.downloadSizeLabel(entry.alias) {
                    Text("~\(gb)")
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .lineLimit(1)
                        .minimumScaleFactor(ModelTableLayout.cellMinimumScaleFactor)
                        .help("Estimated download size.")
                        .accessibilityLabel("Estimated download, about \(gb)")
                }
                QuietIconButton(
                    symbol: "arrow.down.circle",
                    label: "Download \(entry.alias)",
                    help: "Download",
                    size: RapidTheme.ControlHeight.mini,
                    symbolSize: 14
                ) {
                    _ = downloads.startDownload(alias: entry.alias, hfPath: entry.hfRepo)
                }
                .accessibilityIdentifier("Settings.ModelManagement.Download.\(entry.alias)")
            }
        case .downloading(let pct):
            Button {
                downloads.cancelDownload(alias: entry.alias)
            } label: {
                Text(pct.map { "\($0)%" } ?? "Cancel")
            }
            .buttonStyle(.rapidSecondaryCompact)
            .help("Cancel download")
            .accessibilityLabel(pct.map { "Cancel download, \($0) percent" } ?? "Cancel download")
            .accessibilityIdentifier("Settings.ModelManagement.Cancel.\(entry.alias)")
        case .failed:
            Button {
                downloads.dismissJob(alias: entry.alias)
                _ = downloads.startDownload(alias: entry.alias, hfPath: entry.hfRepo)
            } label: {
                Text("Retry")
            }
            .buttonStyle(.rapidSecondaryCompact)
            .accessibilityIdentifier("Settings.ModelManagement.Retry.\(entry.alias)")
        }
    }

    // MARK: - List

    private var visibleEntries: [ModelEntry] {
        let byCapability = catalog.filter { $0.kind == capability }
        let filtered = ModelCacheActions.filter(byCapability, by: filterMode, query: query)
        let sorted = ModelCacheActions.sorted(filtered, order: sortOrder)
        return ModelFavorites.favoritesFirst(sorted, favorites: favorites)
    }

    /// Whether the Quality·Speed meters column has the room to render.
    ///
    /// Read-only decoration, so it is the correct thing to drop when the
    /// column gets narrow — every ACTION in the row (download, cancel,
    /// retry, delete) stays reachable at every supported window size.
    private var showsMeters: Bool { !isCompact }

    /// Kinds that actually have models to manage — the tab bar only offers
    /// these (Video stays hidden until the video lane surfaces aliases).
    private var availableKinds: [ModelKind] {
        ModelKind.allCases.filter { kind in catalog.contains { $0.kind == kind } }
    }

    @ViewBuilder
    private func listSection(_ entries: [ModelEntry]) -> some View {
        if entries.isEmpty {
            Text(noMatchesCopy)
                .font(RapidFont.body)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.vertical, RapidTheme.Space.md)
        } else {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(entries.enumerated()), id: \.element.alias) { idx, entry in
                    if entry.kind == .chat {
                        row(for: entry)
                    } else {
                        capabilityRow(for: entry)
                    }
                    if idx < entries.count - 1 {
                        Rectangle()
                            .fill(RapidTheme.hairline)
                            .frame(height: 1)
                            .accessibilityHidden(true)
                    }
                }
            }
            .settingsGroupedCard()
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
        HStack(spacing: RapidTheme.Space.sm) {
            favoriteStar(entry.alias)
            BrandIcon(alias: entry.alias)
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                HStack(spacing: RapidTheme.Space.xs) {
                    Text(entry.alias)
                        .font(RapidFont.bodyEmphasis)
                        .foregroundStyle(RapidTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    rowBadge(for: entry.alias)
                }
                Text(rowMeta(entry.alias))
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textTertiary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if showsMeters {
                metersView(alias: entry.alias)
                    .frame(width: ModelTableLayout.metersColumnWidth)
            }
            sizeAction(entry: entry, badge: badge)
                .frame(width: ModelTableLayout.sizeColumnWidth, alignment: .trailing)
        }
        .padding(.vertical, RapidTheme.Space.md)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("Settings.ModelManagement.Row.\(entry.alias)")
    }

    /// A leaner row for image/audio models: no chat tok/s meters, just
    /// name · capability/repo · size and the same download/delete control.
    @ViewBuilder
    private func capabilityRow(for entry: ModelEntry) -> some View {
        let badge = ModelCacheActions.statusBadge(
            for: entry,
            downloadJob: downloads.jobs[entry.alias],
            servingAlias: server.servingAlias
        )
        HStack(spacing: RapidTheme.Space.sm) {
            BrandIcon(alias: entry.alias)
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text(entry.alias)
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(RapidTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if entry.kind == .audio, let audioCapability = entry.audioCapability {
                    Text(audioCapabilityLabel(audioCapability))
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                }
                if entry.kind == .image, let imageCapability = entry.imageCapability {
                    Text(imageCapability.label)
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                }
                if let repo = entry.hfRepo {
                    Text(repo)
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            sizeAction(entry: entry, badge: badge)
                .frame(width: ModelTableLayout.sizeColumnWidth, alignment: .trailing)
        }
        .padding(.vertical, RapidTheme.Space.md)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("Settings.ModelManagement.Row.\(entry.alias)")
    }

    private func audioCapabilityLabel(_ capability: AudioModelCapability) -> String {
        switch capability {
        case .transcription: return "Speech to text"
        case .alignment: return "Forced alignment"
        case .speech: return "Text to speech"
        case .voiceCloning: return "Voice cloning"
        case .voiceDesign: return "Voice design"
        }
    }

    @ViewBuilder
    private func statusBadgeView(_ badge: ModelCacheActions.StatusBadge) -> some View {
        switch badge {
        case .cached:
            pill(text: "On disk", color: RapidTheme.statusReady)
        case .inUse:
            pill(text: "In use", color: RapidTheme.statusReady)
        case .notCached:
            pill(text: "Not cached", color: RapidTheme.statusIdle)
        case .downloading(let pct):
            let label: String = {
                if let pct {
                    return "Downloading… \(pct)%"
                }
                return "Downloading…"
            }()
            pill(text: label, color: RapidTheme.statusWorking)
        case .failed:
            pill(text: "Failed", color: RapidTheme.statusError)
        }
    }

    @ViewBuilder
    private func pill(text: String, color: Color) -> some View {
        Text(text)
            .font(RapidFont.caption)
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                Capsule(style: .continuous)
                    .fill(color.opacity(0.15))
            )
            .lineLimit(1)
            .fixedSize()
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
            if entry.isExternal {
                // Downloaded by another MLX runtime (#1718): usable, but
                // outside the hub root the delete path addresses. Say where
                // it came from instead of offering a delete that cannot
                // reach it.
                Text("External")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .help("Found outside Rapid's models folder. Rapid didn't download it, so it can't remove it.")
                    .accessibilityIdentifier(
                        "Settings.ModelManagement.Recommended.External.\(entry.alias)"
                    )
            } else {
                Button {
                    pendingDeletion = entry
                } label: {
                    Text("Delete")
                }
                .buttonStyle(.rapidDestructiveCompact)
                .accessibilityIdentifier("Settings.ModelManagement.Recommended.Delete.\(entry.alias)")
            }
        case .inUse:
            // rapid-mlx holds the weights mmap'd — a mid-serve rm would
            // either fail or corrupt inference. Mirror the picker's
            // "currently-serving rows are off-limits" rule.
            Text("Serving")
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.statusReady)
        case .notCached:
            Button {
                _ = downloads.startDownload(alias: entry.alias, hfPath: entry.hfRepo)
            } label: {
                Text("Download")
            }
            .buttonStyle(.rapidPrimaryCompact)
            .accessibilityIdentifier("Settings.ModelManagement.Recommended.Download.\(entry.alias)")
        case .downloading:
            Button {
                downloads.cancelDownload(alias: entry.alias)
            } label: {
                Text("Cancel")
            }
            .buttonStyle(.rapidSecondaryCompact)
            .accessibilityIdentifier("Settings.ModelManagement.Recommended.Cancel.\(entry.alias)")
        case .failed:
            Button {
                downloads.dismissJob(alias: entry.alias)
                _ = downloads.startDownload(alias: entry.alias, hfPath: entry.hfRepo)
            } label: {
                Text("Retry")
            }
            .buttonStyle(.rapidSecondaryCompact)
            .accessibilityIdentifier("Settings.ModelManagement.Recommended.Retry.\(entry.alias)")
        }
    }

    // MARK: - States

    @ViewBuilder
    private var loadingState: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            ProgressView().controlSize(.small)
            Text("Loading model catalog…")
                .font(RapidFont.body)
                .foregroundStyle(RapidTheme.textSecondary)
        }
        .padding(.vertical, RapidTheme.Space.md)
    }

    @ViewBuilder
    private var emptyState: some View {
        Text("Couldn't load the model list. Restart Rapid-MLX to try again.")
            .font(RapidFont.body)
            .foregroundStyle(RapidTheme.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.vertical, RapidTheme.Space.md)
    }

    // NOTE: local ``errorBanner(_:)`` and ``freedBanner(_:)`` builders
    // lived here. Each drew its own rounded rectangle out of
    // `Color.red.opacity(0.08)` / `RapidTheme.green.opacity(0.08)` at a
    // local 8pt radius, with its own Dismiss button — two more banner
    // styles on a window that already had three. Both call sites now use
    // ``InlineNotice`` with the `.error` / `.success` tones, which own
    // the same job for every surface in the app.

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
        guard let binary = server.binaryPath else {
            catalog = []
            loading = false
            return
        }
        let generation = downloads.cacheGeneration
        // Show a cached snapshot straight away and skip the spinner entirely —
        // flashing "loading" over data we already have makes every visit to
        // this panel feel like a cold start.
        if let hit = await ModelCatalogCache.shared.cached(
            binary: binary, generation: generation
        ) {
            async let image = ModelCatalog.imageEntries(binary: binary)
            async let audio = ModelCatalog.audioEntries(binary: binary)
            catalog = hit + (await image) + (await audio)
            reconcileCapability()
            loading = false
            return
        }
        loading = true
        defer { loading = false }
        // Chat catalog + image-gen aliases, managed side by side. The image
        // rows carry ``kind == .image`` so the capability tabs keep them out of
        // the chat list (and vice-versa).
        let chat = await ModelCatalogCache.shared.entries(
            binary: binary, generation: generation
        )
        async let image = ModelCatalog.imageEntries(binary: binary)
        async let audio = ModelCatalog.audioEntries(binary: binary)
        catalog = chat + (await image) + (await audio)
        reconcileCapability()
    }

    /// Keep the selected capability tab valid after the catalog changes.
    /// Deleting the last image model while the Image tab is active (the tab bar
    /// then collapses because only one kind remains) would otherwise strand the
    /// panel on an empty, un-switchable ``.image`` view even though chat models
    /// exist — so fall back to an available kind.
    private func reconcileCapability() {
        let kinds = availableKinds
        if !kinds.isEmpty, !kinds.contains(capability) {
            capability = kinds.first ?? .chat
        }
    }

    private func deleteAlias(_ entry: ModelEntry) async {
        lastError = nil
        lastFreed = nil
        if server.servingAlias == entry.alias {
            await server.stop()
            guard server.servingAlias != entry.alias else {
                lastError = "Couldn't stop \(entry.alias), so it was not deleted."
                return
            }
        }
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

/// The heading above the models table: the subset on screen and how many
/// rows that is.
///
/// Its own view so the dev snapshot harness can render every filter state
/// side by side without re-implementing markup the panel ships — the
/// heading is the thing this pass changed, so it has to be reviewable in
/// all four of its states, not just the one a running panel happens to be
/// in.
struct ModelsTableHeading: View {
    let heading: ModelCacheActions.ListHeading

    var body: some View {
        HStack(spacing: RapidTheme.Space.xs) {
            Text(heading.title).font(RapidFont.groupLabel)
            Text("· \(heading.countText)")
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.textTertiary)
                .accessibilityIdentifier("Settings.ModelManagement.VisibleCount")
        }
        .foregroundStyle(RapidTheme.textSecondary)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(heading.accessibilityLabel)
    }
}

/// Fixed geometry of a Recommended card, lifted out of the view body so
/// the widths that decide whether a label renders in full are values a
/// test can measure instead of literals buried in a modifier chain.
///
/// The card clipped its primary call to action to "Dow…" because the
/// trailing slot was pinned to a hard 92pt that its own contents did not
/// fit inside. Deriving the floor from the labels — rather than picking
/// another literal and hoping — means adding a longer label, or bumping
/// the control size, moves the column with it.
enum RecommendedCardLayout {
    /// Leading marker column ("Best pick" / "Faster").
    static let markerColumnWidth: CGFloat = 74

    /// Horizontal chrome around a ``controlSize(.small)`` push-button's
    /// title: the bezel plus its internal padding. Deliberately generous
    /// — an over-estimate only widens a right-aligned column by a couple
    /// of points, while an under-estimate is the ellipsis we are fixing.
    static let smallButtonChrome: CGFloat = 26

    /// Horizontal chrome around a status pill's text — 8pt of capsule
    /// padding on each side (see ``SettingsModelManagementPanel.pill``).
    static let pillChrome: CGFloat = 16

    /// Every button label the card's trailing slot can render. "Delete"
    /// is included because ``actionButton`` still carries that branch,
    /// even though the cached card currently resolves to a pill.
    static let actionButtonTitles = ["Download", "Delete", "Cancel", "Retry"]

    /// Every status pill the same slot can render.
    static let actionPillTitles = ["On disk", "In use", "Serving", "External"]

    /// Intrinsic width of a small push-button with this title.
    static func buttonWidth(title: String) -> CGFloat {
        textWidth(title, font: .systemFont(ofSize: NSFont.smallSystemFontSize))
            + smallButtonChrome
    }

    /// Intrinsic width of a status pill with this text.
    static func pillWidth(text: String) -> CGFloat {
        textWidth(text, font: .systemFont(ofSize: NSFont.smallSystemFontSize, weight: .medium))
            + pillChrome
    }

    /// Floor for the card's trailing slot: the widest thing it renders.
    /// The slot is free to grow past this (its content is `fixedSize`d);
    /// the floor exists so the stacked cards' buttons line up.
    static let actionColumnWidth: CGFloat = {
        let widest = actionButtonTitles.map(buttonWidth(title:))
            + actionPillTitles.map(pillWidth(text:))
        return (widest.max() ?? 92).rounded(.up)
    }()

    /// Width of a `.caption` run — the font the removed size caption used,
    /// kept so a test can show why it could not share the slot.
    static func captionWidth(_ text: String) -> CGFloat {
        textWidth(text, font: .systemFont(ofSize: 10))
    }

    /// Width of a `.caption.weight(.medium)` run — the table's "Serving"
    /// label.
    static func captionMediumWidth(_ text: String) -> CGFloat {
        textWidth(text, font: .systemFont(ofSize: 10, weight: .medium))
    }

    fileprivate static func textWidth(_ string: String, font: NSFont) -> CGFloat {
        (string as NSString).size(withAttributes: [.font: font]).width
    }
}

/// Geometry of the "All models" table's trailing Size column.
///
/// The column is a fixed width shared by the header and every row —
/// variable widths would put each row's meters at a different x and make
/// the table ragged — so the width has to be chosen against the widest
/// cell rather than the narrowest.
///
/// It was 84pt, sized when a cached cell held two glyphs and nothing
/// between them. Serving rows now carry measured size, status, and a delete
/// control, so the column is wide enough for all three without truncation.
enum ModelTableLayout {
    /// Shared width of the Size column.
    static let sizeColumnWidth: CGFloat = 124

    /// Shared width of the Quality·Speed meters column.
    ///
    /// Was a bare `158` repeated in the row and again in the column
    /// header — two literals that had to agree for the table not to go
    /// ragged, with nothing enforcing it.
    static let metersColumnWidth: CGFloat = 158

    /// Leading chrome in a chat row before the model name: the favourite
    /// star, the brand icon, and the three gaps between the four columns.
    static let rowLeadingChromeWidth: CGFloat = 15 + 30 + RapidTheme.Space.sm * 3

    /// Everything a chat row commits to before the flexible model-name
    /// column gets a single point.
    ///
    /// Pure so ``SettingsResponsiveLayoutTests`` can assert the row still
    /// leaves a readable name at every supported window size, instead of
    /// that assertion living only in a screenshot somebody has to
    /// remember to take.
    static func committedRowWidth(showsMeters: Bool) -> CGFloat {
        rowLeadingChromeWidth
            + (showsMeters ? metersColumnWidth + RapidTheme.Space.sm : 0)
            + sizeColumnWidth
            // the grouped card's own horizontal inset, both edges
            + RapidTheme.Space.lg * 2
    }

    /// The narrowest model name we are willing to render before calling
    /// the layout broken. Roughly 14 characters at 13pt — enough for
    /// `qwen3.5-9b-4bit` to read with a middle truncation.
    static let minimumNameWidth: CGFloat = 120

    /// Spacing between the glyph, the figure and the button in a cell.
    static let cellSpacing: CGFloat = 6

    /// How far a figure in the Size column may shrink before it starts
    /// truncating. The cells apply this as `.minimumScaleFactor`.
    ///
    /// Settings is NOT inside ``rapidChatDynamicTypeClamp``, so these
    /// `.caption` runs scale with the system text size, and a fixed-width
    /// column plus `lineLimit(1)` means growth eventually clips the
    /// number — which is the defect this column was widened to fix. A
    /// 20% shrink floor buys roughly 1.25x of text growth before that
    /// happens, covering the non-accessibility sizes. It is a bound, not
    /// a Dynamic Type pass: at the AX sizes this table needs to reflow,
    /// which is a change to every column and not this fix's business.
    static let cellMinimumScaleFactor: CGFloat = 0.8

    /// Does a cell of intrinsic width ``needed`` survive ``scale`` times
    /// text growth without truncating, given the shrink floor?
    ///
    /// Deliberately conservative: it grows the WHOLE cell, including the
    /// fixed glyphs and spacing that do not scale. The real cell grows by
    /// less, so a pass here is a real pass; a fail may be pessimistic.
    /// That is the right direction for a guard whose failure mode is a
    /// clipped number.
    static func fits(_ needed: CGFloat, atTextScale scale: CGFloat) -> Bool {
        needed * scale * cellMinimumScaleFactor <= sizeColumnWidth
    }

    /// A `.caption`-sized SF Symbol (state glyph) or the 11pt trash.
    static let glyphWidth: CGFloat = 15

    /// Width a cached cell needs: state glyph + measured size + delete.
    static func cachedCellWidth(size: String) -> CGFloat {
        glyphWidth
            + cellSpacing
            + RecommendedCardLayout.captionWidth(size)
            + cellSpacing
            + glyphWidth
    }

    /// Width a not-cached cell needs: estimate + download glyph. The 8pt
    /// spacing is the one that branch renders.
    static func notCachedCellWidth(size: String) -> CGFloat {
        RecommendedCardLayout.captionWidth("~" + size) + 8 + glyphWidth
    }

    /// Width a serving cell needs: measured size + status + stop-and-delete.
    static func inUseCellWidth(size: String) -> CGFloat {
        RecommendedCardLayout.captionWidth(size)
            + cellSpacing
            + RecommendedCardLayout.captionMediumWidth("Serving")
            + cellSpacing
            + glyphWidth
    }
}
