import SwiftUI

/// Settings → Models tab. The discoverable surface for "download
/// another model in the background", "delete a model to free disk
/// space", and "see what's installed at a glance".
///
/// Before #160 these actions only existed as right-click context
/// menu entries inside the model-picker dropdown — a power-user
/// trail that no first-time user found. The picker still has them
/// (muscle memory), but this panel is the canonical surface.
///
/// Sections, top to bottom:
///   * **In use** — one row for the currently-serving alias with
///     its on-disk size. Disabled "In use" chip in place of the
///     [Delete] button — rapid-mlx holds the weights mmap'd and a
///     mid-serve delete would either fail or corrupt the live
///     inference.
///   * **Downloaded (N.N GB total)** — every cached alias other
///     than the serving one. [Delete] launches a confirmation
///     alert that names the alias + reclaimable size before
///     destructively rm-rf'ing the HF cache directory.
///   * **Available to download** — uncached aliases with their
///     estimated weights footprint and fit indicator. [Download]
///     kicks ``DownloadManager.startDownload`` in the background;
///     the row swaps to a live progress strip + [Cancel] while
///     the pull is running.
///
/// The panel re-loads the catalog on appear and on every successful
/// delete / download-complete so the disk-size column reflects
/// the new state without a manual refresh.
struct SettingsModelsPanel: View {
    @Environment(ServerManager.self) private var server
    @Environment(DownloadManager.self) private var downloads

    // Seeded from the process-wide cache rather than starting empty. `@State`
    // is rebuilt every time this panel re-appears, so an empty start meant the
    // spinner rendered on the first frame of every visit — even when the data
    // was already in hand and the refresh would resolve instantly.
    //
    // Generation 0 is the right key here: `@State` initialisers can't read
    // `@Environment`, and a fresh app run starts at 0. If the real generation
    // has moved on, the `.task` below simply refetches — the seed is an
    // optimisation, never the source of truth.
    @State private var catalog: [ModelEntry] = ModelCatalogCache.seed(generation: 0) ?? []
    @State private var loading: Bool = ModelCatalogCache.seed(generation: 0) == nil
    @State private var pendingDeletion: ModelEntry?
    @State private var lastError: String?
    @State private var lastFreed: String?
    /// cycle-7: power-user override for the picker's sub-1B filter.
    /// Defaults OFF (filter active) so first-time users don't see
    /// `qwen3-0.6b-*` in the dropdown — those tinies hallucinate
    /// within 1-2 turns and read as broken if the user picks one
    /// during evaluation. Flip ON to expose every alias including
    /// the 600M test models. Persisted across launches via
    /// ``ModelPickerVisibility.showAllStorageKey``.
    @AppStorage(ModelPickerVisibility.showAllStorageKey) private var showAllModels: Bool = false
    /// FU-1: persisted opt-out for the launch-time auto-start of the
    /// bundled rapid-mlx sidecar. Defaults to ``true`` (the v0.7.x
    /// behavior — auto-start when the 3-condition gate passes) so
    /// existing users see no change on upgrade. Flipping OFF skips the
    /// next-launch spawn entirely; the user can still manually start
    /// the sidecar via the model picker's Start CTA without flipping
    /// this back on (the toggle exclusively governs the launch-time
    /// path, not the manual lifecycle).
    @AppStorage(AutoStartPreference.storageKey) private var autoStartOnLaunch: Bool = AutoStartPreference.defaultValue

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            pickerVisibilitySection
            autoStartOnLaunchSection
            if loading && catalog.isEmpty {
                loadingState
            } else if catalog.isEmpty {
                emptyState
            } else {
                inUseSection
                downloadedSection
                availableSection
            }
            if let lastError {
                errorBanner(lastError)
            }
            if let lastFreed {
                freedBanner(lastFreed)
            }
        }
        .task {
            await refreshCatalog()
        }
        .alert(item: $pendingDeletion) { entry in
            Alert(
                title: Text("Delete \(entry.alias)?"),
                message: Text(deleteAlertMessage(for: entry)),
                primaryButton: .destructive(Text("Delete from disk")) {
                    Task { await deleteAlias(entry) }
                },
                secondaryButton: .cancel()
            )
        }
    }

    // MARK: - Sections

    @ViewBuilder
    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Models")
                .font(.title3.weight(.semibold))
            Text("Download models in the background, see what's on disk, and reclaim space when you're done with a model.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// cycle-7: power-user toggle that exposes sub-1B aliases
    /// (qwen3-0.6b-*, …) in the model picker. The default OFF state
    /// hides them because 600M models hallucinate within 1-2 turns
    /// of chat — surfacing them to first-time users in a 92-row
    /// dropdown gives the app a bad first impression.
    @ViewBuilder
    private var pickerVisibilitySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(isOn: $showAllModels) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Show small (<1B) models in the picker")
                        .font(.callout.weight(.medium))
                    Text("Sub-1B models (qwen3-0.6b-*) are hidden from the model picker by default — they hallucinate within 1-2 turns and are intended for unit tests, not chat. Turn on to see every model, including the tiny ones.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            // Codex r1 NIT: explicit accessibility label so VoiceOver
            // reads the short phrase instead of concatenating the
            // long explanatory caption. The caption stays as a
            // SwiftUI hint via the .help() modifier-equivalent
            // (handled by the secondary Text inside the toggle's
            // label slot for sighted users; VO can still surface it
            // via the .accessibilityHint route).
            .accessibilityLabel("Show small models in the picker")
            .accessibilityHint("Sub-1B models are hidden by default — they hallucinate within 1-2 turns and are intended for unit tests, not chat.")
            .accessibilityIdentifier("Settings.Models.ShowAllModelsToggle")
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                .fill(RapidTheme.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                .stroke(RapidTheme.hairline, lineWidth: 1)
        )
    }

    /// FU-1: opt-out for the launch-time auto-start of the bundled
    /// rapid-mlx sidecar. Closes the v0.7.19 audit gap surfaced in
    /// PR #341 — pre-fix, opening the app just to browse chat history
    /// always paid the full GPU + RAM cost of loading a model. The
    /// toggle defaults ON to preserve current behavior; flipping OFF
    /// skips the next-launch spawn entirely while leaving every
    /// manual start path (model picker, Start CTA) untouched.
    ///
    /// Sits below ``pickerVisibilitySection`` because both are
    /// "model behavior on launch" toggles; the user reads them as a
    /// natural pair and the panel keeps its existing card rhythm.
    ///
    /// The body copy's "your last-used model" was aspirational until
    /// #1589: on a first run there is no last-used model, and auto-start
    /// picked one anyway (alphabetically, from whatever happened to be in
    /// the shared HF cache). ``AutoStartDecision`` now stands down while
    /// the first-run surfaces are still owed, so the sentence describes
    /// what actually ships — and the second sentence says so out loud.
    @ViewBuilder
    private var autoStartOnLaunchSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(isOn: $autoStartOnLaunch) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Auto-start model on launch")
                        .font(.callout.weight(.medium))
                    Text("On launch, Rapid-MLX loads your last-used model into memory so the chat is interactive immediately. Nothing loads while first-run setup is still open. Turn off if you sometimes open Rapid-MLX just to browse past conversations — you can still start a model manually by picking one in the message box and sending.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            .accessibilityLabel("Auto-start model on launch")
            .accessibilityHint("When off, opening Rapid-MLX will not load a model until you start one manually from the picker.")
            .accessibilityIdentifier("Settings.Models.AutoStartOnLaunchToggle")
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                .fill(RapidTheme.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                .stroke(RapidTheme.hairline, lineWidth: 1)
        )
    }

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

    /// True while a pull for this alias is actually running. Terminal
    /// jobs (completed / failed / cancelled) don't count — those rows
    /// keep their Dismiss affordance in the available section.
    private func isDownloading(_ alias: String) -> Bool {
        guard let job = downloads.jobs[alias] else { return false }
        return Self.isRunning(job)
    }

    private func bucket(_ entry: ModelEntry) -> Bucket {
        Self.bucket(
            entry: entry,
            servingAlias: server.servingAlias,
            isDownloading: isDownloading(entry.alias)
        )
    }

    @ViewBuilder
    private var inUseSection: some View {
        let active = catalog.first { entry in
            bucket(entry) == .inUse
        }
        if let active {
            cardSection(title: "In use") {
                modelRow(active, action: .inUseBadge)
            }
        }
    }

    @ViewBuilder
    private var downloadedSection: some View {
        let cached = catalog.filter { entry in
            bucket(entry) == .downloaded
        }
        let totalLine = totalCachedLine(cached: cached)
        cardSection(title: "Downloaded", subtitle: totalLine) {
            if cached.isEmpty {
                Text("No other models cached on disk.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 4)
            } else {
                ForEach(Array(cached.enumerated()), id: \.element.alias) { idx, entry in
                    modelRow(entry, action: .delete)
                    if idx < cached.count - 1 { Divider() }
                }
            }
        }
    }

    @ViewBuilder
    private var availableSection: some View {
        let available = catalog.filter { bucket($0) == .available }
        cardSection(
            title: "Available to download",
            subtitle: "\(available.count) model\(available.count == 1 ? "" : "s") · downloads run in the background"
        ) {
            if available.isEmpty {
                Text("Every model in the catalog is already downloaded.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 4)
            } else {
                ForEach(Array(available.enumerated()), id: \.element.alias) { idx, entry in
                    modelRow(entry, action: .download)
                    if idx < available.count - 1 { Divider() }
                }
            }
        }
    }

    // MARK: - Row

    private enum RowAction {
        case inUseBadge
        case delete
        case download
    }

    @ViewBuilder
    private func modelRow(_ entry: ModelEntry, action: RowAction) -> some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.alias)
                    .font(.body.weight(.medium))
                Text(sizeCaption(for: entry))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Spacer()
            actionCluster(for: entry, action: action)
        }
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private func actionCluster(for entry: ModelEntry, action: RowAction) -> some View {
        switch action {
        case .inUseBadge:
            Text("In use")
                .font(.caption.weight(.medium))
                .foregroundStyle(RapidTheme.green)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(
                    Capsule(style: .continuous)
                        .fill(RapidTheme.green.opacity(0.15))
                )
        case .delete:
            Button(role: .destructive) {
                pendingDeletion = entry
            } label: {
                Text("Delete")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        case .download:
            if let job = downloads.jobs[entry.alias] {
                inFlightDownloadCluster(entry: entry, job: job)
            } else {
                Button {
                    _ = downloads.startDownload(alias: entry.alias, hfPath: entry.hfRepo)
                } label: {
                    Text("Download")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }

    @ViewBuilder
    private func inFlightDownloadCluster(entry: ModelEntry, job: DownloadManager.Job) -> some View {
        HStack(spacing: 6) {
            Text(Self.captionForJob(job))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            if Self.isRunning(job) {
                Button {
                    downloads.cancelDownload(alias: entry.alias)
                } label: {
                    Text("Cancel")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Button {
                    downloads.dismissJob(alias: entry.alias)
                    Task { await refreshCatalog() }
                } label: {
                    Text("Dismiss")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }

    // MARK: - Card

    @ViewBuilder
    private func cardSection<Content: View>(
        title: String,
        subtitle: String? = nil,
        @ViewBuilder _ content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                if let subtitle {
                    Text("· \(subtitle)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            VStack(alignment: .leading, spacing: 0) {
                content()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
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

    // MARK: - Banners

    @ViewBuilder
    private func errorBanner(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.callout)
                .foregroundStyle(.red)
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

    // MARK: - Helpers

    private func sizeCaption(for entry: ModelEntry) -> String {
        // Mid-pull, `sizeOnDisk` is however many bytes have landed so
        // far — rendering it as "N on disk" reads as the finished size
        // and undercounts badly (1.6 GiB shown for a ~5.7 GB model).
        // Fall through to the estimate so the row advertises the TARGET
        // while the action cluster next to it shows live progress.
        if let size = entry.sizeOnDisk, !isDownloading(entry.alias) {
            return "\(size) on disk"
        }
        let fp = ModelSizing.estimate(alias: entry.alias)
        if fp.weightsGB > 0 {
            return String(format: "~%.1f GB to download", fp.weightsGB)
        }
        return "Size unknown"
    }

    private func deleteAlertMessage(for entry: ModelEntry) -> String {
        if let size = entry.sizeOnDisk {
            return "Removes \(entry.alias) from your Mac and frees \(size). You can download it again later from this tab."
        }
        return "Removes \(entry.alias) from your Mac. You can download it again later from this tab."
    }

    private func totalCachedLine(cached: [ModelEntry]) -> String? {
        let countLabel = "\(cached.count) model\(cached.count == 1 ? "" : "s")"
        return countLabel
    }

    /// Pure helper so the truth table is testable without a SwiftUI
    /// host. Tied to ``DownloadManager.Job.Status`` shape (running /
    /// completed / failed / cancelled) and the nested
    /// ``DownloadProgress.Phase`` for the in-flight percentage.
    static func captionForJob(_ job: DownloadManager.Job) -> String {
        switch job.status {
        case .running:
            return runningCaption(
                phase: job.progress.phase,
                bytesSubtitle: job.progress.hasDiskObservation
                    ? job.progress.progressSubtitle
                    : nil
            )
        case .completed:
            return "Downloaded"
        case .failed(let message):
            return "Failed: \(message)"
        case .cancelled:
            return "Cancelled"
        }
    }

    /// Split out so the running-phase branch can be exhaustively
    /// pinned by tests without us having to manufacture a fake
    /// ``DownloadProgress`` object first.
    ///
    /// ``bytesSubtitle`` carries the cache-dir monitor's byte-based
    /// subtitle (``"1.2 / 6.8 GB · 18%"``) when bytes have been
    /// observed on disk; the row prefers it over the file-count copy
    /// so an HF "Fetching N files: 0/9" stall doesn't read as "stuck."
    static func runningCaption(
        phase: DownloadProgress.Phase,
        bytesSubtitle: String? = nil
    ) -> String {
        if let bytesSubtitle, !bytesSubtitle.isEmpty {
            switch phase {
            case .idle, .preparing, .fetching:
                return bytesSubtitle
            case .downloading, .warmingUp:
                break  // tqdm phase has richer per-file info; fall through
            }
        }
        switch phase {
        case .idle:
            return "Starting…"
        case .preparing:
            return "Preparing…"
        case .fetching(let done, let total, _):
            return "Downloading \(done)/\(total) file\(total == 1 ? "" : "s")"
        case .downloading(_, _, _, let percent, _, _):
            return "\(percent)%"
        case .warmingUp:
            return "Finalising…"
        }
    }

    static func isRunning(_ job: DownloadManager.Job) -> Bool {
        if case .running = job.status { return true }
        return false
    }

    /// Which section a catalog entry belongs in.
    enum Bucket: Equatable {
        case inUse
        case downloaded
        case available
    }

    /// Classify one catalog entry. Pure so the rules can be pinned
    /// without standing up the panel.
    ///
    /// **An in-flight pull outranks ``ModelEntry.cached``.** `cached`
    /// comes from `rapid-mlx ls`, which lists whatever is present in the
    /// Hugging Face cache — including a directory that is still being
    /// written. So a model at 30 % reported as "Downloaded", with its
    /// current partial byte count rendered as if it were the finished
    /// size (dogfood: qwen3.5-4b-4bit listed as 1.6 GiB on disk while
    /// still pulling toward ~5.7 GB). Keeping it in `available` while a
    /// job runs is also what puts the progress caption and Cancel button
    /// on the row, which is the control the user actually wants there.
    static func bucket(
        entry: ModelEntry,
        servingAlias: String?,
        isDownloading: Bool
    ) -> Bucket {
        if isDownloading { return .available }
        guard entry.cached else { return .available }
        return servingAlias == entry.alias ? .inUse : .downloaded
    }

    // MARK: - Actions

    private func refreshCatalog() async {
        guard let binary = server.binaryPath else {
            catalog = []
            loading = false
            return
        }
        let generation = downloads.cacheGeneration
        // Cached snapshot → paint it and skip the spinner (see
        // ``ModelCatalogCache``); only a genuine miss shows a loading state.
        if let hit = await ModelCatalogCache.shared.cached(
            binary: binary, generation: generation
        ) {
            catalog = hit
            loading = false
            return
        }
        loading = true
        defer { loading = false }
        catalog = await ModelCatalogCache.shared.entries(
            binary: binary, generation: generation
        )
    }

    private func deleteAlias(_ entry: ModelEntry) async {
        lastError = nil
        lastFreed = nil
        let outcome = await ModelDeletion.deleteCachedModel(
            binaryPath: server.binaryPath,
            alias: entry.alias
        )
        switch outcome {
        case .freed(let bytes, _):
            if let bytes {
                lastFreed = "Freed \(Self.humanBytes(bytes)) — removed \(entry.alias)."
            } else {
                lastFreed = "Removed \(entry.alias)."
            }
            // Tell the rest of the app too — the picker dropdown and the
            // upgrade banner keep their own catalog snapshots and would
            // otherwise go on showing this model as downloaded.
            downloads.markCacheChanged()
            await refreshCatalog()
        case .failed(let message):
            lastError = message
        }
    }

    /// Pure helper — KiB-style formatter for the "Freed X GB" banner.
    /// Kept here so tests can pin the rounding without standing up an
    /// NSLocale-aware Formatter dance.
    static func humanBytes(_ bytes: Int64) -> String {
        let gib = Double(bytes) / (1024.0 * 1024.0 * 1024.0)
        if gib >= 1 {
            return String(format: "%.1f GB", gib)
        }
        let mib = Double(bytes) / (1024.0 * 1024.0)
        return String(format: "%.0f MB", mib)
    }
}

// ``ModelEntry`` already conforms to ``Identifiable`` in
// ``ModelCatalog.swift`` (id = alias) — used by the ``alert(item:)``
// modifier above to key the destructive sheet on the selected row.
