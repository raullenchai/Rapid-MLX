import Foundation

/// Reduces the server's `summary[]` cells to at most one row per model.
///
/// The feed groups by task, model, machine, protocol, **case** and
/// **execution**, so one model on one Mac routinely has several cells. Mapping
/// each cell to a row produced duplicate `CommunityObservationRow.id`s — which
/// SwiftUI's `ForEach` treats as undefined behaviour — and printed a
/// short-prompt median directly above a long-prompt one under a single
/// "Generation speed" column.
///
/// The projection therefore narrows rather than merges:
///
/// 1. Keep only cells whose metric is the one the column actually displays.
/// 2. Keep only the newest protocol version present for that model; results
///    measured under different protocol versions are not comparable.
/// 3. Keep only the canonical workload case for the column (the short prompt
///    for generation speed), so the number under the header is the number the
///    header names.
/// 4. If execution variants remain after that, **sum the counts and withhold
///    the median**. Each run is in exactly one cell, so the count is exact;
///    a median across bf16 and fp16 would describe no real population.
///
/// Nothing is ever picked arbitrarily: every step is a stated rule, and where
/// the rules cannot single out one population the statistic is dropped instead
/// of guessed.
enum CommunityTableProjection {
    /// One cell, reduced to just what the projections need.
    struct Cell: Equatable, Sendable {
        let modelAlias: String
        /// The full contract identity behind this cell. Two cells that differ
        /// by revision, subfolder, identity strength or quantization are
        /// different models and are never merged, even when they share an
        /// alias.
        let modelIdentity: CommunityModelIdentity
        let workload: CommunityWorkload
        let protocolID: String
        let protocolVersion: Int
        let caseID: String
        let metricName: String
        /// Distinguishes execution variants within an otherwise identical cell.
        let executionKey: String
        let samples: Int
        let median: Double?
        let unit: String?
        /// The pseudonyms whose runs are in this cell, as the server reported
        /// them. `summary[].contributors` is a real field — see
        /// `atomicBenchmarkSummary` in `landing/src/index.js` — so "does this
        /// aggregate already contain one of mine?" is answerable from a plain
        /// read, with no local publication state involved.
        let contributorSlugs: Set<String>

        init(
            modelAlias: String,
            modelIdentity: CommunityModelIdentity? = nil,
            workload: CommunityWorkload,
            protocolID: String,
            protocolVersion: Int,
            caseID: String,
            metricName: String,
            executionKey: String,
            samples: Int,
            median: Double?,
            unit: String?,
            contributorSlugs: Set<String> = []
        ) {
            self.modelAlias = modelAlias
            self.modelIdentity = modelIdentity ?? CommunityModelIdentity(repoID: modelAlias)
            self.workload = workload
            self.protocolID = protocolID
            self.protocolVersion = protocolVersion
            self.caseID = caseID
            self.metricName = metricName
            self.executionKey = executionKey
            self.samples = samples
            self.median = median
            self.unit = unit
            self.contributorSlugs = contributorSlugs
        }
    }

    /// The workload case the named metric is defined over.
    ///
    /// "Generation speed" means the short prompt: both LLM cases report
    /// `decode_tps`, and a 2 048-token prompt is materially slower, so a table
    /// that mixed them would rank models by which case happened to be listed
    /// first. Image and video declare a single case, so any is canonical.
    static func canonicalCaseID(for metric: CommunityMetric) -> String? {
        switch metric {
        case .generationSpeed, .timeToFirstToken:
            return CommunityBenchmarkMetrics.shortTextCaseID
        case .renderTime, .videoTime, .peakMemory:
            return nil
        }
    }

    /// Builds the displayed rows.
    ///
    /// `viewerSlug` is this installation's pseudonym when the service has
    /// issued one. A row is marked ``CommunityObservationSummary/includesYours``
    /// when that slug appears among the contributors of the cells the row is
    /// actually counting — so the badge survives a restart, a reinstall, and a
    /// failed local receipt write, none of which the server knows about.
    static func rows(
        from cells: [Cell],
        workload: CommunityWorkload,
        metric: CommunityMetric,
        viewerSlug: String? = nil
    ) -> [CommunityObservationRow] {
        let metricName = metric.serverMetricName
        let canonicalCase = canonicalCaseID(for: metric)

        var byModel: [String: [Cell]] = [:]
        for cell in cells where cell.workload == workload && cell.metricName == metricName {
            byModel[cell.modelAlias, default: []].append(cell)
        }

        return byModel.compactMap { alias, modelCells -> CommunityObservationRow? in
            // Newest protocol version only.
            guard let newest = modelCells.map(\.protocolVersion).max() else { return nil }
            var candidates = modelCells.filter { $0.protocolVersion == newest }

            // Canonical case, when the metric defines one.
            if let canonicalCase {
                let matching = candidates.filter { $0.caseID == canonicalCase }
                // A model that simply has not been measured on the canonical
                // case has nothing truthful to show under this column.
                guard !matching.isEmpty else { return nil }
                candidates = matching
            } else if Set(candidates.map(\.caseID)).count > 1 {
                // No canonical case and several present: the column cannot
                // name which one it is showing.
                let total = candidates.reduce(0) { $0 + max(0, $1.samples) }
                return row(
                    alias: alias, workload: workload, samples: total, median: nil,
                    unit: candidates.first?.unit,
                    includesYours: contains(viewerSlug, in: candidates)
                )
            }

            let total = candidates.reduce(0) { $0 + max(0, $1.samples) }
            // A median may be printed only when the surviving cells describe
            // ONE population. Two things can break that, and both are counted
            // exactly and reported without a median rather than averaged:
            //
            //  - several execution configurations (bf16 beside fp16), and
            //  - several model identities sharing an alias — a `4bit/`
            //    subfolder, a different snapshot revision, a differently
            //    quantized artifact. Those are different models; a median
            //    across them describes none of them.
            let singleVariant = Set(candidates.map(\.executionKey)).count == 1
            let singleIdentity = Set(candidates.map(\.modelIdentity.canonicalKey)).count == 1
            return row(
                alias: alias,
                workload: workload,
                samples: total,
                median: singleVariant && singleIdentity ? candidates.first?.median : nil,
                unit: candidates.first?.unit,
                includesYours: contains(viewerSlug, in: candidates)
            )
        }
        .sorted { $0.modelAlias.localizedStandardCompare($1.modelAlias) == .orderedAscending }
    }

    /// Whether this installation's pseudonym is among the contributors of the
    /// cells being counted. Compares canonical server-issued slugs only — the
    /// app must not invent a second spelling of an identity the website owns.
    static func contains(_ viewerSlug: String?, in cells: [Cell]) -> Bool {
        guard let viewerSlug, !viewerSlug.isEmpty else { return false }
        return cells.contains { $0.contributorSlugs.contains(viewerSlug) }
    }

    private static func row(
        alias: String,
        workload: CommunityWorkload,
        samples: Int,
        median: Double?,
        unit: String?,
        includesYours: Bool
    ) -> CommunityObservationRow {
        CommunityObservationRow(
            modelAlias: alias,
            workload: workload,
            summary: CommunityObservationSummary(
                observationCount: samples,
                median: median,
                observedMinimum: nil,
                observedMaximum: nil,
                unit: unit,
                includesYours: includesYours,
                isBounded: true
            )
        )
    }
}

extension CommunityMetric {
    /// The server's metric name for this column.
    var serverMetricName: String {
        switch self {
        case .generationSpeed: return "decode_tps"
        case .timeToFirstToken: return "ttft_ms"
        case .renderTime, .videoTime: return "total_seconds"
        case .peakMemory: return "peak_memory_mib"
        }
    }
}
