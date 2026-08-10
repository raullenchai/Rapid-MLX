import Foundation
import Testing
@testable import Rapid

/// Pin every pure helper inside ``ModelCacheActions``. The picker
/// + the management panel both go through these for their
/// confirmation copy, toast text, status-badge derivation, and
/// filter/sort/aggregate primitives — so a silent drift here
/// would show up as a UI inconsistency between the two surfaces.
@MainActor
@Suite("ModelCacheActions — shared cache primitives (#210)")
struct ModelCacheActionsTests {

    // MARK: - Fixtures

    private func entry(_ alias: String, cached: Bool = false, size: String? = nil, repo: String? = nil) -> ModelEntry {
        ModelEntry(alias: alias, hfRepo: repo, sizeOnDisk: size, cached: cached)
    }

    // MARK: - Status badge

    @Test("status badge: inUse wins over cached when serving matches")
    func statusBadgeInUse() {
        let e = entry("qwen3.5-4b-4bit", cached: true, size: "2.3 GB")
        let badge = ModelCacheActions.statusBadge(
            for: e,
            downloadJob: nil,
            servingAlias: "qwen3.5-4b-4bit"
        )
        #expect(badge == .inUse)
    }

    @Test("status badge: cached when on disk and not serving")
    func statusBadgeCached() {
        let e = entry("qwen3.5-4b-4bit", cached: true)
        let badge = ModelCacheActions.statusBadge(
            for: e,
            downloadJob: nil,
            servingAlias: "qwen3.5-9b-4bit"
        )
        #expect(badge == .cached)
    }

    @Test("status badge: notCached when no disk, no job")
    func statusBadgeNotCached() {
        let e = entry("phi-4-4bit", cached: false)
        let badge = ModelCacheActions.statusBadge(
            for: e,
            downloadJob: nil,
            servingAlias: nil
        )
        #expect(badge == .notCached)
    }

    @Test("status badge: in-flight download overrides cached state")
    func statusBadgeDownloadingOverrides() {
        let e = entry("qwen3.5-4b-4bit", cached: false)
        let job = DownloadManager.Job(alias: "qwen3.5-4b-4bit")
        let badge = ModelCacheActions.statusBadge(
            for: e,
            downloadJob: job,
            servingAlias: nil
        )
        // A freshly-made job is .running. The badge should be
        // .downloading regardless of the catalog's cached flag —
        // a download lookup against an already-cached alias
        // (the user re-firing the pull manually) reads as
        // "downloading" to the user.
        if case .downloading = badge { return }
        Issue.record("Expected .downloading badge, got \(badge)")
    }

    // MARK: - Running percent

    @Test("runningPercent: surfaces percent for byte-level phases")
    func runningPercentDownloading() {
        let phase = DownloadProgress.Phase.downloading(
            file: "x.safetensors",
            done: "1G",
            total: "4G",
            percent: 42,
            speed: "30MB/s",
            eta: "01:00"
        )
        #expect(ModelCacheActions.runningPercent(phase: phase) == 42)
    }

    @Test("runningPercent: nil for idle/preparing/warmingUp")
    func runningPercentNilPhases() {
        #expect(ModelCacheActions.runningPercent(phase: .idle) == nil)
        #expect(ModelCacheActions.runningPercent(phase: .preparing) == nil)
        #expect(ModelCacheActions.runningPercent(phase: .warmingUp) == nil)
    }

    // MARK: - Deletion confirmation copy

    @Test("deletionConfirmation: sized entry names the model and surfaces freed size")
    func deletionConfirmationSized() {
        let e = entry("qwen3.5-4b-4bit", cached: true, size: "2.3 GB")
        let copy = ModelCacheActions.deletionConfirmation(for: e)
        #expect(copy.title == "Delete \"qwen3.5-4b-4bit\"? This frees 2.3 GB.")
        #expect(copy.message.contains("Frees 2.3 GB"))
    }

    @Test("deletionConfirmation: unsized entry still names the alias")
    func deletionConfirmationUnsized() {
        let e = entry("phi-4-4bit", cached: true, size: nil)
        let copy = ModelCacheActions.deletionConfirmation(for: e)
        #expect(copy.title == "Delete \"phi-4-4bit\"?")
        // Body must NOT say "Frees " with no number — we deliberately
        // skip the freed-line when we don't have one.
        #expect(!copy.message.contains("Frees "))
    }

    @Test("deletionConfirmation: serving model explains stop-before-delete")
    func deletionConfirmationServing() {
        let e = entry("qwen3-tts-4bit", cached: true, size: "2.2 GiB")
        let copy = ModelCacheActions.deletionConfirmation(for: e, isServing: true)
        #expect(copy.message.hasPrefix("Stops the currently serving model first."))
    }

    @Test("deletionConfirmation: matches ModelPickerBar.deletionTitle shape")
    func deletionConfirmationParityWithPicker() {
        // Issue #210 is explicit that the two surfaces present the
        // SAME confirmation dialog. Pin the title side-by-side so a
        // future refactor to one helper can't silently diverge.
        let cases: [(String, String?)] = [
            ("qwen3.5-4b-4bit", "2.3 GB"),
            ("phi-4-4bit", nil),
            ("gpt-oss-20b-mxfp4-q8", "12.4 GB"),
        ]
        for (alias, size) in cases {
            let e = entry(alias, cached: true, size: size)
            #expect(
                ModelCacheActions.deletionConfirmation(for: e).title
                    == ModelPickerBar.deletionTitle(for: e)
            )
        }
    }

    // MARK: - Outcome messages

    @Test("successMessage: prefers freedBytes when present")
    func successMessagePrefersBytes() {
        let msg = ModelCacheActions.successMessage(
            alias: "qwen3.5-4b-4bit",
            freedBytes: 2_400_000_000,
            fallbackSize: "2.3 GB"
        )
        // ByteCountFormatter for ~2.4 GB at .file produces "2.4 GB"
        // on macOS — we don't pin the exact formatted token (locale
        // could shift), only that the alias + a freed-X clause is
        // present.
        #expect(msg.hasPrefix("Deleted qwen3.5-4b-4bit — freed "))
    }

    @Test("successMessage: falls back to ls size when no bytes")
    func successMessageFallback() {
        let msg = ModelCacheActions.successMessage(
            alias: "phi-4-4bit",
            freedBytes: nil,
            fallbackSize: "5.1 GB"
        )
        #expect(msg == "Deleted phi-4-4bit — freed 5.1 GB.")
    }

    @Test("successMessage: bare deletion when neither bytes nor fallback")
    func successMessageBare() {
        let msg = ModelCacheActions.successMessage(
            alias: "phi-4-4bit",
            freedBytes: nil,
            fallbackSize: nil
        )
        #expect(msg == "Deleted phi-4-4bit.")
    }

    @Test("successMessage: zero bytes falls back to size string")
    func successMessageZeroBytes() {
        // Bytes of 0 shouldn't print "freed 0 B" — that's a worse UX
        // than the fallback size string from the catalog.
        let msg = ModelCacheActions.successMessage(
            alias: "phi-4-4bit",
            freedBytes: 0,
            fallbackSize: "5.1 GB"
        )
        #expect(msg == "Deleted phi-4-4bit — freed 5.1 GB.")
    }

    @Test("failureMessage: surfaces alias + error")
    func failureMessageShape() {
        #expect(
            ModelCacheActions.failureMessage(alias: "phi-4-4bit", error: "rapid-mlx not found.")
                == "Couldn't delete phi-4-4bit: rapid-mlx not found."
        )
    }

    // MARK: - Filter

    @Test("filter: All keeps every entry; cached/notCached split correctly")
    func filterByMode() {
        let entries = [
            entry("qwen3.5-4b-4bit", cached: true),
            entry("qwen3.5-9b-4bit", cached: false),
            entry("phi-4-4bit", cached: true),
        ]
        let all = ModelCacheActions.filter(entries, by: .all, query: "")
        #expect(all.count == 3)
        let cached = ModelCacheActions.filter(entries, by: .cached, query: "")
        #expect(cached.map { $0.alias } == ["qwen3.5-4b-4bit", "phi-4-4bit"])
        let notCached = ModelCacheActions.filter(entries, by: .notCached, query: "")
        #expect(notCached.map { $0.alias } == ["qwen3.5-9b-4bit"])
    }

    @Test("filter: substring search is case-insensitive and matches alias + repo")
    func filterByQuery() {
        let entries = [
            entry("qwen3.5-4b-4bit", repo: "mlx-community/Qwen3.5-4B-MLX-4bit"),
            entry("phi-4-4bit", repo: "mlx-community/Phi-4-mlx-4bit"),
            entry("gemma-3-12b-4bit", repo: nil),
        ]
        let qwen = ModelCacheActions.filter(entries, by: .all, query: "QWEN")
        #expect(qwen.map { $0.alias } == ["qwen3.5-4b-4bit"])
        // Repo-side match.
        let phi = ModelCacheActions.filter(entries, by: .all, query: "phi-4-mlx")
        #expect(phi.map { $0.alias } == ["phi-4-4bit"])
        // Blank query treated as no-op.
        let blank = ModelCacheActions.filter(entries, by: .all, query: "   ")
        #expect(blank.count == 3)
        // No match.
        let none = ModelCacheActions.filter(entries, by: .all, query: "nonexistent")
        #expect(none.isEmpty)
    }

    @Test("filter: filter mode AND query intersect (not OR)")
    func filterModeAndQueryIntersect() {
        let entries = [
            entry("qwen3.5-4b-4bit", cached: true),
            entry("qwen3.5-9b-4bit", cached: false),
            entry("phi-4-4bit", cached: true),
        ]
        let cachedQwen = ModelCacheActions.filter(entries, by: .cached, query: "qwen")
        #expect(cachedQwen.map { $0.alias } == ["qwen3.5-4b-4bit"])
    }

    // MARK: - Sort

    @Test("sorted familyThenSize: clusters families, ascending size within")
    func sortFamilyThenSize() {
        let entries = [
            entry("qwen3.5-9b-4bit"),
            entry("phi-4-4bit"),
            entry("qwen3.5-4b-4bit"),
            entry("gemma-3-12b-4bit"),
        ]
        let sorted = ModelCacheActions.sorted(entries, order: .familyThenSize)
        let aliases = sorted.map { $0.alias }
        // Each family is contiguous; within Qwen 3.5 the 4b comes
        // before the 9b. The exact inter-family order is whatever
        // localizedStandardCompare returns on the family names —
        // we pin contiguity, not the family ordering itself.
        let qwenIdx = aliases.firstIndex(where: { $0.contains("qwen3.5-4b") })!
        let qwen9Idx = aliases.firstIndex(where: { $0.contains("qwen3.5-9b") })!
        #expect(qwenIdx < qwen9Idx)
        // Both Qwen rows are adjacent.
        #expect(abs(qwenIdx - qwen9Idx) == 1)
    }

    @Test("sorted nameAscending: localizedStandardCompare order")
    func sortNameAscending() {
        let entries = [
            entry("qwen3.5-9b-4bit"),
            entry("phi-4-4bit"),
            entry("gemma-3-12b-4bit"),
        ]
        let sorted = ModelCacheActions.sorted(entries, order: .nameAscending)
        #expect(sorted.map { $0.alias } == ["gemma-3-12b-4bit", "phi-4-4bit", "qwen3.5-9b-4bit"])
    }

    @Test("sorted sizeDescending: largest size first; unparseable sorts last")
    func sortSizeDescending() {
        let entries = [
            entry("a", cached: true, size: "1.0 GB"),
            entry("b", cached: true, size: "10.0 GB"),
            entry("c", cached: true, size: nil),
            entry("d", cached: true, size: "5.0 GB"),
        ]
        let sorted = ModelCacheActions.sorted(entries, order: .sizeDescending)
        #expect(sorted.map { $0.alias } == ["b", "d", "a", "c"])
    }

    // MARK: - parseSizeBytes

    @Test("parseSizeBytes: decimal units")
    func parseSizeDecimal() {
        #expect(ModelCacheActions.parseSizeBytes("1.0 GB") == 1_000_000_000)
        #expect(ModelCacheActions.parseSizeBytes("5.6 GB") == 5_600_000_000)
        #expect(ModelCacheActions.parseSizeBytes("812 MB") == 812_000_000)
        #expect(ModelCacheActions.parseSizeBytes("4 KB") == 4_000)
    }

    @Test("parseSizeBytes: binary units")
    func parseSizeBinary() {
        #expect(ModelCacheActions.parseSizeBytes("1 GiB") == 1_073_741_824)
        #expect(ModelCacheActions.parseSizeBytes("512 MiB") == Int64(512) * 1024 * 1024)
    }

    @Test("parseSizeBytes: case-insensitive + trims whitespace")
    func parseSizeCase() {
        #expect(ModelCacheActions.parseSizeBytes("  2.5 gb ") == 2_500_000_000)
    }

    @Test("parseSizeBytes: rejects garbage")
    func parseSizeRejects() {
        #expect(ModelCacheActions.parseSizeBytes(nil) == nil)
        #expect(ModelCacheActions.parseSizeBytes("") == nil)
        #expect(ModelCacheActions.parseSizeBytes("abc") == nil)
        #expect(ModelCacheActions.parseSizeBytes("?? GB") == nil)
    }

    // MARK: - Aggregate

    @Test("aggregateOnDiskBytes: sums cached entries, ignores uncached")
    func aggregate() {
        let entries = [
            entry("a", cached: true, size: "1 GB"),
            entry("b", cached: true, size: "2 GB"),
            entry("c", cached: false, size: "5 GB"),
        ]
        let usage = ModelCacheActions.aggregateOnDiskBytes(entries)
        #expect(usage.cachedCount == 2)
        #expect(usage.totalBytes == 3_000_000_000)
        #expect(usage.missingSizeCount == 0)
    }

    @Test("aggregateOnDiskBytes: nil totalBytes when no entry has a parseable size")
    func aggregateUnparseable() {
        let entries = [
            entry("a", cached: true, size: nil),
            entry("b", cached: true, size: "?? GB"),
        ]
        let usage = ModelCacheActions.aggregateOnDiskBytes(entries)
        #expect(usage.cachedCount == 2)
        #expect(usage.totalBytes == nil)
        #expect(usage.missingSizeCount == 2)
    }

    @Test("aggregateOnDiskBytes: counts unmeasured entries when sizes are mixed (codex r2 P3)")
    func aggregateMixed() {
        // Half-and-half: parseable + nil → totalBytes is the partial
        // sum, missingSizeCount tells the footer to caveat it.
        let entries = [
            entry("a", cached: true, size: "1 GB"),
            entry("b", cached: true, size: nil),
            entry("c", cached: true, size: "2 GB"),
        ]
        let usage = ModelCacheActions.aggregateOnDiskBytes(entries)
        #expect(usage.cachedCount == 3)
        #expect(usage.totalBytes == 3_000_000_000)
        #expect(usage.missingSizeCount == 1)
    }

    @Test("diskUsageFooter: hidden when nothing cached")
    func diskUsageFooterHidden() {
        let usage = ModelCacheActions.DiskUsage(cachedCount: 0, totalBytes: nil, missingSizeCount: 0)
        #expect(ModelCacheActions.diskUsageFooter(usage) == nil)
    }

    @Test("diskUsageFooter: renders size + pluralised count")
    func diskUsageFooterRenders() {
        let one = ModelCacheActions.DiskUsage(cachedCount: 1, totalBytes: 5_000_000_000, missingSizeCount: 0)
        let three = ModelCacheActions.DiskUsage(cachedCount: 3, totalBytes: 15_000_000_000, missingSizeCount: 0)
        // ByteCountFormatter shape is locale-dependent (typically "5
        // GB" / "5.0 GB" — leaving the exact number unpinned), but
        // pluralisation + the "Total: " prefix are fixed.
        let oneLabel = ModelCacheActions.diskUsageFooter(one)!
        #expect(oneLabel.hasPrefix("Total: "))
        #expect(oneLabel.contains("1 model"))
        #expect(!oneLabel.contains("1 models"))
        let threeLabel = ModelCacheActions.diskUsageFooter(three)!
        #expect(threeLabel.contains("3 models"))
        // No "unmeasured" suffix when every cached entry parsed.
        #expect(!oneLabel.contains("unmeasured"))
        #expect(!threeLabel.contains("unmeasured"))
    }

    @Test("diskUsageFooter: count-only when bytes unparseable")
    func diskUsageFooterCountOnly() {
        let usage = ModelCacheActions.DiskUsage(cachedCount: 2, totalBytes: nil, missingSizeCount: 2)
        let label = ModelCacheActions.diskUsageFooter(usage)!
        // Count-only line should still flag every entry as
        // unmeasured so the user knows the byte total is missing
        // intentionally.
        #expect(label == "Total: 2 models (+2 unmeasured)")
    }

    @Test("diskUsageFooter: appends (+N unmeasured) when totals are partial (codex r2 P3)")
    func diskUsageFooterPartial() {
        let usage = ModelCacheActions.DiskUsage(cachedCount: 3, totalBytes: 3_000_000_000, missingSizeCount: 1)
        let label = ModelCacheActions.diskUsageFooter(usage)!
        // Pin the suffix shape; ByteCountFormatter token stays
        // locale-flexible.
        #expect(label.contains("(+1 unmeasured)"))
        #expect(label.contains("3 models"))
        #expect(label.hasPrefix("Total: "))
    }

    // MARK: - Family/quant chip

    @Test("familyQuantChip: known family + bit-width")
    func familyChipKnown() {
        #expect(ModelCacheActions.familyQuantChip(for: "qwen3.6-27b-4bit") == "Qwen 3.6 · 4-bit")
        #expect(ModelCacheActions.familyQuantChip(for: "qwen3.5-9b-8bit") == "Qwen 3.5 · 8-bit")
        #expect(ModelCacheActions.familyQuantChip(for: "phi-4-4bit") == "Phi 4 · 4-bit")
    }

    @Test("familyQuantChip: unknown family falls back to Unknown label")
    func familyChipUnknown() {
        #expect(ModelCacheActions.familyQuantChip(for: "made-up-3b-4bit") == "Unknown · 4-bit")
    }

    // MARK: - FilterMode / SortOrder labels

    @Test("FilterMode display labels match the segmented picker copy")
    func filterModeLabels() {
        #expect(ModelCacheActions.FilterMode.all.displayLabel == "All")
        #expect(ModelCacheActions.FilterMode.cached.displayLabel == "Cached")
        #expect(ModelCacheActions.FilterMode.notCached.displayLabel == "Not cached")
    }

    @Test("SortOrder display labels match the sort menu copy")
    func sortOrderLabels() {
        #expect(ModelCacheActions.SortOrder.familyThenSize.displayLabel == "Family · size")
        #expect(ModelCacheActions.SortOrder.nameAscending.displayLabel == "Name")
        #expect(ModelCacheActions.SortOrder.sizeDescending.displayLabel == "Size (largest first)")
    }

    // MARK: - List heading

    @Test("listHeading: unfiltered, the count is the whole catalog")
    func listHeadingUnfiltered() {
        let heading = ModelCacheActions.listHeading(
            filter: .all, query: "", visibleCount: 175, totalCount: 175
        )
        #expect(heading.title == "All models")
        #expect(heading.countText == "175")
    }

    /// The defect: the heading rendered ``catalog.count`` regardless of
    /// what the search box had done to the rows beneath it, so four
    /// visible models sat under a header claiming 175.
    @Test("listHeading: a search query makes the count describe the rows shown")
    func listHeadingWithQuery() {
        let heading = ModelCacheActions.listHeading(
            filter: .all, query: "qwen", visibleCount: 4, totalCount: 175
        )
        #expect(heading.countText == "4 of 175")
        #expect(heading.countText.contains("4"))
    }

    /// Same defect through the other control: the segmented filter.
    @Test("listHeading: the segment names itself and counts only its rows")
    func listHeadingWithSegment() {
        let cached = ModelCacheActions.listHeading(
            filter: .cached, query: "", visibleCount: 3, totalCount: 175
        )
        #expect(cached.title == "Cached")
        #expect(cached.countText == "3 of 175")

        let notCached = ModelCacheActions.listHeading(
            filter: .notCached, query: "", visibleCount: 172, totalCount: 175
        )
        #expect(notCached.title == "Not cached")
        #expect(notCached.countText == "172 of 175")
    }

    /// A filter that happens to match everything still says so, rather
    /// than collapsing to a bare total that would read as "unfiltered".
    @Test("listHeading: a filter matching everything still reads N of N")
    func listHeadingFilterMatchingEverything() {
        let heading = ModelCacheActions.listHeading(
            filter: .notCached, query: "", visibleCount: 175, totalCount: 175
        )
        #expect(heading.countText == "175 of 175")
    }

    /// Whitespace is not a search. The clear button leaves an empty
    /// string but a stray space must not flip the heading into its
    /// narrowed form while every row is still on screen.
    @Test("listHeading: whitespace-only query is not narrowing")
    func listHeadingWhitespaceQuery() {
        let heading = ModelCacheActions.listHeading(
            filter: .all, query: "   ", visibleCount: 175, totalCount: 175
        )
        #expect(heading.countText == "175")
    }

    /// Zero matches is the state most likely to be read as a bug, so it
    /// has to be stated plainly rather than showing the catalog size.
    @Test("listHeading: no matches counts zero, not the catalog")
    func listHeadingNoMatches() {
        let heading = ModelCacheActions.listHeading(
            filter: .all, query: "zzz", visibleCount: 0, totalCount: 175
        )
        #expect(heading.countText == "0 of 175")
        #expect(heading.accessibilityLabel == "All models, 0 of 175")
    }

    /// The count and the rows come from one filter pass, so whatever
    /// ``filter`` returns is what the heading must report.
    @Test("listHeading: the count agrees with filter() for the same inputs")
    func listHeadingAgreesWithFilter() {
        let entries = [
            entry("qwen3.6-27b-4bit", cached: true),
            entry("qwen3.5-9b-4bit", cached: false),
            entry("phi-4-4bit", cached: true),
        ]
        for mode in ModelCacheActions.FilterMode.allCases {
            for query in ["", "qwen", "phi", "nope"] {
                let visible = ModelCacheActions.filter(entries, by: mode, query: query)
                let heading = ModelCacheActions.listHeading(
                    filter: mode,
                    query: query,
                    visibleCount: visible.count,
                    totalCount: entries.count
                )
                #expect(heading.countText.hasPrefix("\(visible.count)"))
            }
        }
    }
}
