import Foundation

/// Search, grouping, and counting for the model picker sheet.
///
/// Pure value logic, separate from the sheet view, because the behaviour worth
/// protecting is textual: which group a model lands in, what the count label
/// says once a query narrows the catalogue, and that an empty group never
/// renders a heading with nothing under it.
enum CommunityBenchmarkPicker {
    struct Row: Equatable, Identifiable, Sendable {
        let model: CommunityBenchmarkModel
        /// Published observations for this model on this Mac, when the
        /// community read API answered. Nil means "not known" — the row then
        /// shows no coverage sentence at all rather than implying zero.
        let observationCount: Int?
        /// Download size in GB, when the coverage feed knows it. Never the
        /// model's runtime memory requirement.
        var downloadSizeGB: Double?

        var id: String { model.entry.alias }

        /// The amber coverage line under the alias. Nil when coverage is
        /// unknown, so an unavailable API produces a quieter row, never a
        /// false "no results yet".
        var coverageSentence: String? {
            guard let observationCount else { return nil }
            if observationCount == 0 {
                return String(localized: "No results yet for this Mac profile")
            }
            return String(
                format: String(localized: "%1$d published %2$@ for this Mac profile"),
                observationCount,
                observationCount == 1
                    ? String(localized: "result")
                    : String(localized: "results")
            )
        }

        /// True when this row is a first-reference opportunity. Only ever true
        /// from a real zero, never from missing data.
        var isFirstResultOpportunity: Bool { observationCount == 0 }
    }

    struct Section: Equatable, Identifiable, Sendable {
        let kind: Kind
        let rows: [Row]

        enum Kind: String, Equatable, Sendable {
            case needed
            case other
        }

        var id: String { kind.rawValue }

        var title: String {
            switch kind {
            case .needed:
                return String(localized: "NEEDED BY THE COMMUNITY")
            case .other:
                return String(localized: "OTHER MODELS")
            }
        }
    }

    struct Listing: Equatable, Sendable {
        let sections: [Section]
        /// Every model matching the query, in display order.
        let matchCount: Int
        /// Every model in the catalogue, regardless of query.
        let totalCount: Int
        let query: String

        /// "41 models" with no query; "8 of 41 models" once one narrows the
        /// list. The total is always the live catalogue size, never a constant.
        var countLabel: String {
            guard !query.trimmingCharacters(in: .whitespaces).isEmpty else {
                return String(
                    format: String(localized: "%1$d models"),
                    totalCount
                )
            }
            return String(
                format: String(localized: "%1$d of %2$d models"),
                matchCount,
                totalCount
            )
        }

        var isEmpty: Bool { matchCount == 0 }
    }

    /// Builds the grouped, filtered listing.
    ///
    /// `coverage` supplies the "needed by the community" membership when the
    /// read API is available. When it is not, membership falls back to the
    /// benchmark catalogue's own focus flag — a static hint the client already
    /// ships — and rows carry no observation count, so the group still means
    /// "worth measuring" without asserting coverage the client cannot see.
    static func listing(
        models: [CommunityBenchmarkModel],
        coverage: CommunityDataState<[CommunityCoverageGap]>,
        query: String
    ) -> Listing {
        let needle = query
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let matches = needle.isEmpty
            ? models
            : models.filter { $0.entry.alias.lowercased().contains(needle) }

        let gapsByAlias: [String: CommunityCoverageGap]
        if let gaps = coverage.value {
            gapsByAlias = Dictionary(
                gaps.map { ($0.modelAlias, $0) },
                uniquingKeysWith: { first, _ in first }
            )
        } else {
            gapsByAlias = [:]
        }
        let coverageKnown = coverage.value != nil

        var needed: [Row] = []
        var other: [Row] = []
        for model in matches {
            let gap = gapsByAlias[model.entry.alias]
            let row = Row(
                model: model,
                observationCount: coverageKnown ? (gap?.observationCount ?? nil) : nil,
                downloadSizeGB: gap?.downloadSizeGB
            )
            let isNeeded = coverageKnown
                ? gap != nil
                : (model.isFocus && model.memoryFit != "does_not_fit")
            if isNeeded { needed.append(row) } else { other.append(row) }
        }

        let sections = [
            Section(kind: .needed, rows: needed),
            Section(kind: .other, rows: other),
        ].filter { !$0.rows.isEmpty }

        return Listing(
            sections: sections,
            matchCount: matches.count,
            totalCount: models.count,
            query: query
        )
    }

    /// Whether "Choose model" may be pressed for a row.
    ///
    /// A model that simply has not been downloaded stays choosable: the
    /// benchmark fetches it as part of the run, so blocking the button would
    /// invent a prerequisite the client does not have. Only a missing
    /// selection disables it.
    static func canChoose(_ row: Row?) -> Bool { row != nil }

    /// The trailing status column for a row: a title plus the size line under
    /// it. Never a byte count or percentage — the client has no download
    /// progress phase to read those from.
    ///
    /// `downloadSizeGB` is the only download figure this row may show.
    /// `estimatedMemoryGib` is the *runtime memory* the model needs while it
    /// executes; printing it under "Not downloaded" said "13 GB to fetch" for
    /// a model whose weights are a different size entirely. When the download
    /// size is unknown the detail line is omitted rather than substituted.
    static func downloadStatus(
        _ model: CommunityBenchmarkModel,
        downloadSizeGB: Double? = nil
    ) -> (title: String, detail: String?) {
        if model.entry.cached {
            return (String(localized: "Downloaded"), model.entry.sizeOnDisk)
        }
        let detail = downloadSizeGB.map { String(format: "%.1f GB", $0) }
        return (String(localized: "Not downloaded"), detail)
    }
}
