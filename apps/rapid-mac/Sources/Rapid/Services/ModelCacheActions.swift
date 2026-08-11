import Foundation

/// Shared deletion + filter + sort + aggregation primitives for
/// every surface that lets the user manage the on-disk model cache.
///
/// **Why this exists (issue #210).** The bottom-bar
/// ``ModelPickerBar`` already owned a context-menu "Delete from
/// disk" affordance, with its toast message + freed-bytes math
/// inlined in the view's ``runDeletion`` method. When the
/// dedicated ``SettingsModelManagementPanel`` landed (the
/// file-manager-style surface that is the canonical home for
/// download / delete / disk-usage state), the obvious shape was
/// to copy/adapt the toast formatting verbatim — but that would
/// give us TWO sites computing "Deleted X — freed Y" copy that
/// drift apart on the next polish pass.
///
/// ``ModelCacheActions`` lifts the pure helpers out:
///   * ``deletionConfirmation(for:)`` — the title + message a
///     confirmation dialog should render, key off the same
///     ``ModelEntry.sizeOnDisk`` field both surfaces have.
///   * ``successMessage(...)`` / ``failureMessage(...)`` — the
///     "Deleted X — freed Y" / "Couldn't delete X: …" copy the
///     picker shows as a toast and the management panel shows as
///     a banner.
///   * ``StatusBadge`` — the per-row state badge
///     (``.cached / .notCached / .downloading(percent:) /
///     .inUse / .failed``) derived from the catalog entry + the
///     ``DownloadManager`` snapshot.
///   * ``filter(_:by:query:)``, ``sorted(_:order:)``, and
///     ``aggregateOnDiskBytes(_:)`` — list-shaping primitives
///     used by the management panel's filter row and
///     disk-usage footer.
///
/// The actual rm-rf still lives in ``ModelDeletion`` (a static
/// enum with no state) because the path-traversal hardening is
/// expensive to set up and ``ModelCacheActions`` is meant to be
/// trivially testable without the FS. The picker + the panel
/// both go through ``runDeletion`` here, which is a thin
/// dispatcher that calls ``ModelDeletion.deleteCachedModel`` and
/// formats the outcome into the same human strings.
///
/// All methods are ``nonisolated`` / pure so unit tests pin the
/// truth tables without standing up a SwiftUI host or spawning a
/// rapid-mlx subprocess.
enum ModelCacheActions {

    // MARK: - Status badge

    /// Per-row badge that summarises an alias's current state on
    /// disk + in the download manager. Exposed as ``Equatable``
    /// so ``SettingsModelManagementPanel`` can drive a
    /// ``.changeMonitoring`` task on the badge if it ever needs
    /// to (today: not needed; the @Bindable on the manager is
    /// enough).
    enum StatusBadge: Equatable, Sendable {
        /// Alias is on disk, not currently being served.
        case cached
        /// Alias is on disk AND is the active serving alias.
        /// rapid-mlx has the weights mmap'd; deletion must stop the server
        /// before removing the cache directory.
        case inUse
        /// Alias is not on disk and no download is in flight.
        case notCached
        /// A download is currently running. ``percent`` is the
        /// best-guess byte percent ingested from tqdm; absent
        /// when the phase is still ``.idle / .preparing /
        /// .warmingUp``.
        case downloading(percent: Int?)
        /// Most recent download attempt finished with an error.
        /// The panel offers a Retry action that re-fires
        /// ``DownloadManager.startDownload``.
        case failed(message: String)
    }

    /// Resolve the badge for one row given the catalog entry, the
    /// download manager's job (if any), and the serving alias.
    ///
    /// ``@MainActor`` because ``DownloadManager.Job`` is itself
    /// main-actor isolated (the manager owns it and mutates ``status``
    /// from the main loop), so reaching into ``job.status`` /
    /// ``job.progress.phase`` is only safe on the main actor. The
    /// other helpers in this file stay nonisolated.
    @MainActor
    static func statusBadge(
        for entry: ModelEntry,
        downloadJob: DownloadManager.Job?,
        servingAlias: String?
    ) -> StatusBadge {
        // An in-flight download wins over every other branch — a
        // race where the alias also happens to be cached (e.g. the
        // user kicked a refresh) is benign because the panel
        // re-loads the catalog when the job completes.
        if let job = downloadJob {
            switch job.status {
            case .running:
                return .downloading(percent: runningPercent(progress: job.progress))
            case .failed(let message):
                return .failed(message: message)
            case .completed, .cancelled:
                // Fall through to the cached / not-cached branches —
                // the catalog refresh after a completed pull is what
                // flips the badge to ``.cached``.
                break
            }
        }
        if entry.cached {
            if let serving = servingAlias, serving == entry.alias {
                return .inUse
            }
            return .cached
        }
        return .notCached
    }

    /// Pull the byte-percent out of a running phase. Returns
    /// ``nil`` for non-byte phases (idle / preparing / warming up
    /// — the badge then renders ``"Downloading…"`` without a
    /// number, which is honest for the user.
    static func runningPercent(phase: DownloadProgress.Phase) -> Int? {
        switch phase {
        case .downloading(_, _, _, let percent, _, _):
            return percent
        case .fetching(_, _, let percent):
            // ``fetching`` is the outer "Fetching N files" tqdm — its
            // percent reflects file count, not bytes. Surface it so
            // the badge isn't blank while the first weight shard
            // initialises, but the management panel's progress bar
            // treats it as approximate.
            return percent
        case .idle, .preparing, .warmingUp:
            return nil
        }
    }

    /// Same shape as ``runningPercent(phase:)`` but consults the
    /// cache-dir byte monitor first — when bytes-on-disk are known
    /// the bytes percent supersedes the tqdm file-count percent. Used
    /// by callers that already hold a ``DownloadProgress`` instance
    /// (the management panel's row progress bar). Returns ``nil``
    /// when neither signal is available.
    @MainActor
    static func runningPercent(progress: DownloadProgress) -> Int? {
        if let fraction = progress.progressFraction {
            return Int((fraction * 100.0).rounded())
        }
        return runningPercent(phase: progress.phase)
    }

    // MARK: - Deletion confirmation copy

    /// Title + message pair for the destructive-action
    /// ``confirmationDialog``. Both surfaces use the same wording
    /// so a user who learns the picker's dialog isn't surprised
    /// by the management panel's. Mirrors the existing
    /// ``ModelPickerBar.deletionTitle`` shape — size front-loaded,
    /// alias in quotes — and extends it with a body line that
    /// states the consequence + the recovery path (re-download).
    struct DeletionConfirmation: Equatable {
        let title: String
        let message: String
    }

    static func deletionConfirmation(
        for entry: ModelEntry,
        isServing: Bool = false
    ) -> DeletionConfirmation {
        let title: String
        if let size = entry.sizeOnDisk {
            title = "Delete \"\(entry.alias)\"? This frees \(size)."
        } else {
            title = "Delete \"\(entry.alias)\"?"
        }
        let suffix = entry.sizeOnDisk.map { " Frees \($0)." } ?? ""
        let stopPrefix = isServing ? "Stops the currently serving model first. " : ""
        let message = "\(stopPrefix)Removes this model from your Mac. You can download it again later by selecting it.\(suffix)"
        return DeletionConfirmation(title: title, message: message)
    }

    // MARK: - Outcome → toast / banner copy

    /// Toast / banner copy after ``ModelDeletion`` returns. Pure
    /// so both the picker (toast at the top of the menu) and the
    /// management panel (inline green banner) format the same
    /// way. Mirrors the bytes-formatting the picker had inlined
    /// in its ``runDeletion`` body.
    static func successMessage(alias: String, freedBytes: Int64?, fallbackSize: String?) -> String {
        let freedLabel: String
        if let bytes = freedBytes, bytes > 0 {
            freedLabel = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
        } else if let fallback = fallbackSize {
            freedLabel = fallback
        } else {
            freedLabel = ""
        }
        if freedLabel.isEmpty {
            return "Deleted \(alias)."
        }
        return "Deleted \(alias) — freed \(freedLabel)."
    }

    static func failureMessage(alias: String, error: String) -> String {
        "Couldn't delete \(alias): \(error)"
    }

    // MARK: - Run delete

    /// Outcome shape both surfaces consume. Carries the formatted
    /// human string so view code doesn't need to know about
    /// ``ModelDeletion.Outcome`` internals.
    enum RunDeleteOutcome: Equatable {
        case success(message: String, freedBytes: Int64?)
        case failure(message: String)
    }

    /// Dispatch a cache delete and format the outcome into a
    /// ready-to-render string. The caller must have already
    /// presented a confirmation dialog (see
    /// ``deletionConfirmation(for:)``).
    ///
    /// ``binaryPath`` mirrors ``ServerManager.binaryPath`` — both
    /// surfaces read it from the same observable so a missing
    /// binary fails fast with a recognisable error message.
    static func runDeletion(
        for entry: ModelEntry,
        binaryPath: URL?,
        delete: (URL?, String, String?) async -> ModelDeletion.Outcome = {
            await ModelDeletion.deleteCachedModel(binaryPath: $0, alias: $1, knownRepo: $2)
        }
    ) async -> RunDeleteOutcome {
        // Defence in depth for #1718. The Settings panel already omits the
        // delete affordance for an external model, but that is a UI
        // decision one missed `if` away from a data-loss bug: deletion
        // rebuilds ``<hub-root>/models--<repo>``, so an external model's
        // delete would either miss or remove an unrelated hub entry that
        // happens to share the name. Refuse at the dispatcher, where every
        // present and future delete path has to pass.
        guard !entry.isExternal else {
            return .failure(
                message: "\(entry.alias) was downloaded by another app. "
                    + "Rapid can't remove it — delete it where it was installed."
            )
        }
        // Audio (and image) snapshots list as `(unmapped)`, so pass the
        // catalog's known repo for non-chat rows; `deleteCachedModel` reverse
        // maps chat aliases itself when the repo is nil.
        let knownRepo = entry.kind == .chat ? nil : entry.hfRepo
        let outcome = await delete(binaryPath, entry.alias, knownRepo)
        switch outcome {
        case .freed(let bytes, _):
            let msg = successMessage(
                alias: entry.alias,
                freedBytes: bytes,
                fallbackSize: entry.sizeOnDisk
            )
            return .success(message: msg, freedBytes: bytes)
        case .failed(let message):
            return .failure(message: failureMessage(alias: entry.alias, error: message))
        }
    }

    // MARK: - Filter / sort / aggregate

    /// Which subset of aliases the management panel surfaces.
    /// Mirrors the ``All / Cached / Not cached`` segmented
    /// picker at the top of the panel.
    enum FilterMode: String, CaseIterable, Identifiable, Sendable {
        case all
        case cached
        case notCached

        var id: String { rawValue }
        var displayLabel: String {
            switch self {
            case .all: return "All"
            case .cached: return "Cached"
            case .notCached: return "Not cached"
            }
        }
    }

    /// Filter + (optional) substring-search the catalog. The
    /// substring match is case-insensitive and applied to BOTH
    /// the alias and (if present) the HF repo — so the user can
    /// type "qwen3.6" to find every Qwen 3.6 alias regardless of
    /// the size suffix.
    static func filter(
        _ entries: [ModelEntry],
        by mode: FilterMode,
        query: String
    ) -> [ModelEntry] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return entries.filter { entry in
            switch mode {
            case .all: break
            case .cached: if !entry.cached { return false }
            case .notCached: if entry.cached { return false }
            }
            if trimmed.isEmpty { return true }
            if entry.alias.lowercased().contains(trimmed) { return true }
            if let repo = entry.hfRepo, repo.lowercased().contains(trimmed) { return true }
            return false
        }
    }

    /// The heading above the models table: what the table is showing,
    /// and how many rows that actually is.
    struct ListHeading: Equatable, Sendable {
        /// Names the subset on screen. Rendered uppercased.
        let title: String
        /// "175" when nothing is narrowing the list, "4 of 175" when
        /// something is.
        let countText: String

        /// One sentence for VoiceOver, since the rendered form is two
        /// fragments separated by a middle dot.
        var accessibilityLabel: String { "\(title), \(countText)" }
    }

    /// Derive the models-table heading.
    ///
    /// The panel used to render a fixed "All models" beside
    /// ``catalog.count`` — the size of the WHOLE catalog — no matter what
    /// the filter or the search box had done to the rows underneath. Type
    /// three characters and the table showed four rows under a heading
    /// that still said 175.
    ///
    /// So the count always describes what is on screen, and when
    /// something has narrowed it the total follows as context ("4 of
    /// 175") rather than the number silently changing meaning. The title
    /// names the segment, which is the other half of "what am I looking
    /// at". The search term is deliberately NOT echoed here: it is
    /// legible in the field a few points above, and repeating it would be
    /// the same duplication this panel is being cleaned of.
    static func listHeading(
        filter: FilterMode,
        query: String,
        visibleCount: Int,
        totalCount: Int
    ) -> ListHeading {
        let title: String = {
            switch filter {
            case .all: return "All models"
            case .cached: return "Cached"
            case .notCached: return "Not cached"
            }
        }()
        let searching = !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let narrowed = searching || filter != .all || visibleCount != totalCount
        return ListHeading(
            title: title,
            countText: narrowed ? "\(visibleCount) of \(totalCount)" : "\(totalCount)"
        )
    }

    /// How the panel orders the filtered list.
    enum SortOrder: String, CaseIterable, Identifiable, Sendable {
        /// Family A→Z, then ascending size within each family.
        /// This is the default — clusters every Qwen / Gemma /
        /// Llama row together so the panel reads like a file
        /// browser sorted by "kind".
        case familyThenSize
        /// Plain alphabetical order over the alias name —
        /// useful if the user knows the alias they want.
        case nameAscending
        /// Largest models first. Useful for the "what's eating
        /// my disk" use case.
        case sizeDescending

        var id: String { rawValue }
        var displayLabel: String {
            switch self {
            case .familyThenSize: return "Family · size"
            case .nameAscending: return "Name"
            case .sizeDescending: return "Size (largest first)"
            }
        }
    }

    /// Sort a (filtered) list per ``order``. Pure / stable so
    /// the UI render order is deterministic across re-renders
    /// — important when the user is mid-Delete and the panel
    /// re-resolves the row binding.
    static func sorted(_ entries: [ModelEntry], order: SortOrder) -> [ModelEntry] {
        switch order {
        case .familyThenSize:
            return entries.sorted { lhs, rhs in
                let lf = ModelInfoCatalog.familyAndContext(for: lhs.alias).family
                let rf = ModelInfoCatalog.familyAndContext(for: rhs.alias).family
                if lf != rf { return lf.localizedStandardCompare(rf) == .orderedAscending }
                let lp = ModelSizing.estimate(alias: lhs.alias).paramsBillions ?? .infinity
                let rp = ModelSizing.estimate(alias: rhs.alias).paramsBillions ?? .infinity
                if lp != rp { return lp < rp }
                return lhs.alias.localizedStandardCompare(rhs.alias) == .orderedAscending
            }
        case .nameAscending:
            return entries.sorted { $0.alias.localizedStandardCompare($1.alias) == .orderedAscending }
        case .sizeDescending:
            return entries.sorted { lhs, rhs in
                let lhsBytes = parseSizeBytes(lhs.sizeOnDisk) ?? -1
                let rhsBytes = parseSizeBytes(rhs.sizeOnDisk) ?? -1
                if lhsBytes != rhsBytes { return lhsBytes > rhsBytes }
                return lhs.alias.localizedStandardCompare(rhs.alias) == .orderedAscending
            }
        }
    }

    /// Disk-usage footer aggregation. Walks the cached entries,
    /// parses each ``sizeOnDisk`` string back into bytes, and
    /// returns a ``(count, bytes, missingSizeCount)`` triple.
    ///
    /// codex r2 P3 (#210): ``missingSizeCount`` is the number of
    /// cached entries whose ``sizeOnDisk`` was nil / unparseable.
    /// When that's > 0 the footer ADDS a ``(+N unmeasured)``
    /// suffix instead of silently pretending the byte total
    /// represents every cached model — a partial sum dressed as
    /// a complete one is a worse UX than the truth.
    struct DiskUsage: Equatable {
        let cachedCount: Int
        let totalBytes: Int64?
        let missingSizeCount: Int
    }

    /// Largest cache entry that this app can actually manage. External
    /// runtime entries are visible in the inventory but live outside Rapid's
    /// selected models folder, so including them would make the overview's
    /// total and cleanup signal contradict its own delete surface (#1818).
    static func largestManagedEntry(_ entries: [ModelEntry]) -> ModelEntry? {
        entries
            .filter { $0.cached && !$0.isExternal && parseSizeBytes($0.sizeOnDisk) != nil }
            .max {
                let lhs = parseSizeBytes($0.sizeOnDisk) ?? 0
                let rhs = parseSizeBytes($1.sizeOnDisk) ?? 0
                if lhs != rhs { return lhs < rhs }
                return $0.alias.localizedStandardCompare($1.alias) == .orderedDescending
            }
    }

    static func storageSummary(usage: DiskUsage, freeBytes: Int64?) -> String {
        let used = usage.totalBytes.map {
            ByteCountFormatter.string(fromByteCount: $0, countStyle: .file)
        } ?? "size unavailable"
        let models = "\(usage.cachedCount) model\(usage.cachedCount == 1 ? "" : "s")"
        guard let freeBytes else { return "\(used) · \(models)" }
        let free = ByteCountFormatter.string(fromByteCount: freeBytes, countStyle: .file)
        return "\(used) · \(models) · \(free) free"
    }

    /// Conservative keep-signals shown beside chat models. They make the two
    /// easy-to-regret deletions visible without claiming that everything else
    /// is automatically safe to remove (#1818).
    static func retentionBadges(
        alias: String,
        starterAlias: String,
        lastServedAlias: String?
    ) -> [String] {
        var badges: [String] = []
        if alias == starterAlias { badges.append("STARTER") }
        if alias == lastServedAlias { badges.append("LAST USED") }
        return badges
    }

    static func aggregateOnDiskBytes(_ entries: [ModelEntry]) -> DiskUsage {
        var total: Int64 = 0
        var anyParsed = false
        var cachedCount = 0
        var missing = 0
        for entry in entries where entry.cached {
            cachedCount += 1
            if let bytes = parseSizeBytes(entry.sizeOnDisk) {
                total += bytes
                anyParsed = true
            } else {
                missing += 1
            }
        }
        return DiskUsage(
            cachedCount: cachedCount,
            totalBytes: anyParsed ? total : nil,
            missingSizeCount: missing
        )
    }

    /// Format the disk-usage footer line. Returns ``nil`` when
    /// there is nothing cached — the panel hides the footer in
    /// that state rather than showing a "Total: 0 across 0
    /// models" line that reads as accidentally negative.
    ///
    /// When some cached entries have a parseable size and others
    /// don't, the footer reads ``Total: X across N models (+M
    /// unmeasured)`` so the user sees the sum applies to a
    /// proper subset, not the whole cache.
    static func diskUsageFooter(_ usage: DiskUsage) -> String? {
        guard usage.cachedCount > 0 else { return nil }
        let label = "\(usage.cachedCount) model\(usage.cachedCount == 1 ? "" : "s")"
        let unmeasuredSuffix: String = {
            guard usage.missingSizeCount > 0 else { return "" }
            return " (+\(usage.missingSizeCount) unmeasured)"
        }()
        if let bytes = usage.totalBytes {
            let size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            return "Total: \(size) across \(label)\(unmeasuredSuffix)"
        }
        return "Total: \(label)\(unmeasuredSuffix)"
    }

    // MARK: - Size parsing

    /// Best-effort parse of an ``rapid-mlx ls`` size column back
    /// into bytes — the column is human-formatted ("5.6 GB", "812
    /// MB", "1024 KiB"), so we accept the common units. Pulled out
    /// as a static so the sort + footer math can be pinned by a
    /// unit test, and so the surface tolerates a future ``ls``
    /// format drift gracefully (parse failure → row sorts last,
    /// footer falls back to count-only).
    static func parseSizeBytes(_ raw: String?) -> Int64? {
        guard let raw = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        // Strip the trailing " GB" / " GiB" / " MB" / etc. Accept
        // both decimal (GB) and binary (GiB) units — rapid-mlx
        // currently emits decimal but a future format change to
        // binary shouldn't silently zero the footer.
        let lower = raw.lowercased()
        let units: [(String, Double)] = [
            ("tib", 1024 * 1024 * 1024 * 1024),
            ("gib", 1024 * 1024 * 1024),
            ("mib", 1024 * 1024),
            ("kib", 1024),
            ("tb", 1_000_000_000_000),
            ("gb", 1_000_000_000),
            ("mb", 1_000_000),
            ("kb", 1_000),
            ("t", 1_000_000_000_000),
            ("g", 1_000_000_000),
            ("m", 1_000_000),
            ("k", 1_000),
            ("b", 1),
        ]
        for (suffix, multiplier) in units {
            if lower.hasSuffix(suffix) {
                let numPart = lower.dropLast(suffix.count).trimmingCharacters(in: .whitespaces)
                if let value = Double(numPart) {
                    return Int64(value * multiplier)
                }
                return nil
            }
        }
        // Bare number with no unit — treat as bytes.
        if let bytes = Int64(lower) { return bytes }
        return nil
    }

    // MARK: - Family / quant chip labels

    /// "Qwen 3.6 · 4-bit" — single string used as the per-row
    /// family/quant chip. Pulled to a helper because the picker
    /// and the panel may someday want different chip layouts but
    /// must agree on the displayed text.
    static func familyQuantChip(for alias: String) -> String {
        let family = ModelInfoCatalog.familyAndContext(for: alias).family
        let bits = ModelSizing.parseBitsPerWeight(alias)
        return "\(family) · \(bits)-bit"
    }
}
