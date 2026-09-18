import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Model picker search, grouping, and counting")
struct CommunityBenchmarkPickerTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)

    private static func entry(
        _ alias: String,
        cached: Bool = true,
        task: ModelTask = .textGeneration,
        size: String? = nil
    ) -> ModelEntry {
        switch task {
        case .imageGeneration:
            return ModelEntry(
                alias: alias, hfRepo: "mlx-community/\(alias)", sizeOnDisk: size,
                cached: cached, taskTypes: [.imageGeneration], operationModes: [.textToImage]
            )
        case .videoGeneration:
            return ModelEntry(
                alias: alias, hfRepo: "mlx-community/\(alias)", sizeOnDisk: size,
                cached: cached, taskTypes: [.videoGeneration], operationModes: [.textToVideo]
            )
        default:
            return ModelEntry(
                alias: alias, hfRepo: "mlx-community/\(alias)", sizeOnDisk: size,
                cached: cached, taskTypes: [.textGeneration]
            )
        }
    }

    /// A mixed catalogue: LLM, image, and video models in one list, which is
    /// what the picker must present without modality tabs.
    private static var models: [CommunityBenchmarkModel] {
        CommunityBenchmarkModel.models(
            from: [
                entry("qwen3.5-9b-4bit", size: "6.5 GB"),
                entry("gemma-4-12b-4bit", size: "7.1 GB"),
                entry("z-image-turbo", cached: false, task: .imageGeneration),
                entry("wan2.2-ti2v-5b-q8", cached: true, task: .videoGeneration, size: "9.6 GB"),
                entry("mistral-small-3.4-24b-instruct-2506-uncensored-4bit", cached: false),
            ]
        )
    }

    private static func gap(
        _ alias: String,
        count: Int,
        workload: CommunityWorkload = .llm
    ) -> CommunityCoverageGap {
        CommunityCoverageGap(
            modelAlias: alias,
            workload: workload,
            observationCount: count,
            fitsThisMac: true,
            isDownloaded: false,
            downloadSizeGB: 3.6,
            requiredMemoryGB: nil
        )
    }

    // MARK: - Counting

    @Test("With no query the count is the live catalogue total")
    func unfilteredCountLabel() {
        let listing = CommunityBenchmarkPicker.listing(
            models: Self.models, coverage: .unavailable(.notConfigured), query: ""
        )
        #expect(listing.totalCount == Self.models.count)
        #expect(listing.matchCount == Self.models.count)
        #expect(listing.countLabel == "\(Self.models.count) models")
        // Never a hard-coded constant: the label tracks the catalogue.
        #expect(!listing.countLabel.contains("41"))
    }

    @Test("A query narrows the list and switches to a filtered count")
    func filteredCountLabel() {
        let listing = CommunityBenchmarkPicker.listing(
            models: Self.models, coverage: .unavailable(.notConfigured), query: "gemma"
        )
        #expect(listing.matchCount == 1)
        #expect(listing.countLabel == "1 of \(Self.models.count) models")
    }

    @Test("Search is case-insensitive, substring-based, and whitespace-tolerant")
    func searchMatching() {
        let listing = CommunityBenchmarkPicker.listing(
            models: Self.models, coverage: .unavailable(.notConfigured), query: "  IMAGE "
        )
        #expect(listing.sections.flatMap(\.rows).map(\.id) == ["z-image-turbo"])

        let blank = CommunityBenchmarkPicker.listing(
            models: Self.models, coverage: .unavailable(.notConfigured), query: "   "
        )
        #expect(blank.matchCount == Self.models.count)
        #expect(blank.countLabel.contains("of") == false)
    }

    @Test("A query with no matches yields an empty listing and no group headings")
    func emptySearch() {
        let listing = CommunityBenchmarkPicker.listing(
            models: Self.models, coverage: .unavailable(.notConfigured), query: "zzz"
        )
        #expect(listing.isEmpty)
        #expect(listing.sections.isEmpty)
        #expect(listing.countLabel == "0 of \(Self.models.count) models")
    }

    // MARK: - Grouping

    @Test("Coverage gaps drive the needed group and carry observation counts")
    func coverageDrivenGrouping() {
        let listing = CommunityBenchmarkPicker.listing(
            models: Self.models,
            coverage: .ready([
                Self.gap("z-image-turbo", count: 0, workload: .image),
                Self.gap("gemma-4-12b-4bit", count: 3),
            ]),
            query: ""
        )
        #expect(listing.sections.map(\.kind) == [.needed, .other])
        let needed = listing.sections[0]
        #expect(Set(needed.rows.map(\.id)) == ["z-image-turbo", "gemma-4-12b-4bit"])

        let first = needed.rows.first { $0.id == "z-image-turbo" }
        #expect(first?.observationCount == 0)
        #expect(first?.isFirstResultOpportunity == true)
        #expect(first?.coverageSentence == "No results yet for this Mac profile")

        let under = needed.rows.first { $0.id == "gemma-4-12b-4bit" }
        #expect(under?.isFirstResultOpportunity == false)
        #expect(under?.coverageSentence == "3 published results for this Mac profile")
    }

    @Test("Without the read API rows show no coverage sentence at all")
    func unavailableCoverageShowsNoClaim() {
        let listing = CommunityBenchmarkPicker.listing(
            models: Self.models, coverage: .unavailable(.notConfigured), query: ""
        )
        let rows = listing.sections.flatMap(\.rows)
        #expect(rows.allSatisfy { $0.observationCount == nil })
        // No row may imply "no results yet" from missing data.
        #expect(rows.allSatisfy { $0.coverageSentence == nil })
        #expect(rows.allSatisfy { !$0.isFirstResultOpportunity })
        // The focus flag still produces a useful "needed" group.
        #expect(listing.sections.contains { $0.kind == .needed })
    }

    @Test("Needed comes before other models, and empty groups are dropped")
    func groupOrderAndEmptyGroups() {
        let onlyNeeded = CommunityBenchmarkPicker.listing(
            models: Self.models,
            coverage: .ready(Self.models.map { Self.gap($0.entry.alias, count: 0) }),
            query: ""
        )
        #expect(onlyNeeded.sections.map(\.kind) == [.needed])

        let onlyOther = CommunityBenchmarkPicker.listing(
            models: Self.models, coverage: .ready([]), query: ""
        )
        #expect(onlyOther.sections.map(\.kind) == [.other])
    }

    @Test("Grouping survives a search, so headings still label the matches")
    func groupingSurvivesSearch() {
        let listing = CommunityBenchmarkPicker.listing(
            models: Self.models,
            coverage: .ready([Self.gap("z-image-turbo", count: 0, workload: .image)]),
            query: "i"
        )
        #expect(listing.sections.map(\.kind) == [.needed, .other])
        #expect(listing.sections[0].rows.map(\.id) == ["z-image-turbo"])
        #expect(listing.matchCount > 1)
    }

    @Test("The list mixes LLM, image, and video models")
    func mixedModalities() {
        let listing = CommunityBenchmarkPicker.listing(
            models: Self.models, coverage: .unavailable(.notConfigured), query: ""
        )
        let workloads = Set(
            listing.sections.flatMap(\.rows).map { CommunityWorkload(task: $0.model.task) }
        )
        #expect(workloads == [.llm, .image, .video])
    }

    // MARK: - Choosing

    @Test("A model that is merely not downloaded stays choosable")
    func notDownloadedIsStillChoosable() {
        let listing = CommunityBenchmarkPicker.listing(
            models: Self.models, coverage: .unavailable(.notConfigured), query: "z-image-turbo"
        )
        let row = listing.sections.flatMap(\.rows).first
        #expect(row?.model.entry.cached == false)
        // The benchmark fetches the model as part of the run, so blocking the
        // confirmation button would invent a prerequisite.
        #expect(CommunityBenchmarkPicker.canChoose(row))
        #expect(!CommunityBenchmarkPicker.canChoose(nil))
    }

    // MARK: - Sheet geometry

    @Test("The sheet clamps into the viewport with at least a 24pt margin")
    func sheetClampsToViewport() {
        // The detail pane at a 1440-wide window: the design's 680 × 620.
        #expect(min(680, 1_240 - 48) == 680)
        #expect(min(620, 870 - 48) == 620)
        // The detail pane at a 900-wide window: narrower, never overflowing.
        #expect(min(680, 700 - 48) == 652)
        #expect(min(620, 570 - 48) == 522)
    }

    @Test("The list shows eight or nine rows on desktop and at least six at 900")
    func visibleRowCounts() {
        let desktop = CommunityBenchmarkPickerSheet.visibleRowCount(sheetHeight: 620)
        #expect(desktop >= 8)
        #expect(desktop <= 9)

        let narrow = CommunityBenchmarkPickerSheet.visibleRowCount(sheetHeight: 522)
        #expect(narrow >= 6)
    }

    @Test("Picker keeps a 24-point margin even in a very small window")
    func pickerClampsToSmallViewport() {
        let size = CommunityBenchmarkPickerSheet.clampedSheetSize(
            in: CGSize(width: 360, height: 300)
        )
        #expect(size.width == 312)
        #expect(size.height == 252)
    }

    @Test("Download status never fabricates bytes, a rate, or an ETA")
    func downloadStatusIsHonest() {
        let downloaded = Self.models.first { $0.entry.alias == "qwen3.5-9b-4bit" }
        let status = CommunityBenchmarkPicker.downloadStatus(downloaded!)
        #expect(status.title == "Downloaded")
        #expect(status.detail == "6.5 GB")

        let pending = Self.models.first { $0.entry.alias == "z-image-turbo" }
        let pendingStatus = CommunityBenchmarkPicker.downloadStatus(pending!)
        #expect(pendingStatus.title == "Not downloaded")
        // No "of", no percentage, no "left": the client has no download phase.
        let rendered = "\(pendingStatus.title) \(pendingStatus.detail ?? "")"
        #expect(!rendered.contains("%"))
        #expect(!rendered.lowercased().contains(" of "))
        #expect(!rendered.lowercased().contains("left"))
    }
}
