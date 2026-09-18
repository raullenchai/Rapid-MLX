import Foundation

/// The public atomic benchmark feed on rapidmlx.com, as a
/// ``CommunityBenchmarkDirectory``.
///
/// Three routes exist and this adapter uses all three:
///
/// | Route | Shape | Completeness |
/// | --- | --- | --- |
/// | `/api/benchmarks/atomic/public` | `{summary[], runs[]}` | **Bounded**: the newest 50 runs, and `summary` is aggregated over exactly those |
/// | `/api/benchmarks/atomic/contributions` | `{runs[], cursor, complete}` | Cursor-paginated over everything |
/// | `/api/benchmarks/atomic/contributors/<slug>` | same, pre-filtered | Cursor-paginated over one pseudonym |
///
/// ## What this adapter will and will not claim
///
/// The bounded feed can prove that results **exist**; it can never prove that
/// none do. A scope missing from `summary` might have fifty newer runs in front
/// of it, so its absence is `unavailable`, never zero — which is what keeps the
/// first-reference branch from firing on a quiet week. Counts that do come back
/// are marked ``CommunityObservationSummary/isBounded`` so the UI says
/// "at least N" instead of asserting an exact total.
///
/// Per-contributor totals are different: `/contributors/<slug>` paginates to
/// `complete == true`, so this adapter follows the cursor and reports an exact
/// number for the signed-in installation's own contribution count.
struct CommunityBenchmarkAPIDirectory: CommunityBenchmarkDirectory {
    /// Site root. Injectable so tests can point at a local stub.
    var baseURL: URL = URL(string: "https://rapidmlx.com")!
    /// Performs one request. Injected so tests exercise this adapter's parsing
    /// and pagination without a network.
    var transport: @Sendable (URLRequest) async throws -> (Data, URLResponse) = {
        try await URLSession.shared.data(for: $0)
    }
    /// Maps a Hugging Face repo id back to the product alias the rest of the
    /// module speaks. The feed identifies models by `repo_id`; Desktop's
    /// catalogue owns the alias.
    var aliasForRepoID: @Sendable (String) -> String = { $0 }
    /// Hard ceiling on pages followed when counting one contributor's history,
    /// so a pathological cursor loop cannot spin forever.
    var maximumContributionPages = 40

    // MARK: - Directory

    func observations(
        for scope: CommunityBenchmarkScope,
        viewerSlug: String?
    ) async -> CommunityDataState<CommunityObservationSummary> {
        do {
            let feed = try await publicFeed()
            // Every cell for this model + workload + protocol + machine,
            // across execution configurations and workload cases.
            let candidates = feed.summary.filter { $0.matchesCoverage(scope, aliasForRepoID) }
            guard !candidates.isEmpty else {
                // Absence from a bounded feed is not evidence of absence from
                // the corpus. Reporting zero here is precisely the bug that
                // would congratulate someone for being "first" when they are
                // not.
                return .unavailable(.boundedFeed)
            }

            guard let comparison = scope.comparison else {
                // Coverage question. One run can contribute a summary cell
                // for every workload case, so summing cell samples can count
                // the same submission more than once. Count the bounded
                // feed's distinct matching runs instead.
                //
                // The median is deliberately dropped: averaging medians drawn
                // from different execution configurations and prompt cases
                // would produce a number that describes no real population.
                let matchingRunIDs = Set(
                    feed.runs
                        .filter { $0.matchesCoverage(scope, aliasForRepoID) }
                        .map(\.submissionID)
                )
                guard !matchingRunIDs.isEmpty else {
                    return .unavailable(.boundedFeed)
                }
                return .ready(
                    CommunityObservationSummary(
                        observationCount: matchingRunIDs.count,
                        median: nil,
                        unit: candidates.first?.metric.unit,
                        includesYours: SummaryCell.contains(viewerSlug, in: candidates),
                        isBounded: true
                    )
                )
            }

            // Comparison question: only cells produced exactly the way this run
            // was. Picking `candidates.first` here would compare a short prompt
            // against a long one, or bf16 against fp16.
            let matching = candidates.filter { $0.matches(comparison) }
            guard !matching.isEmpty else { return .unavailable(.boundedFeed) }

            // A comparison is a claim that two numbers describe the same
            // artifact. When this run pins a revision or a quantization that a
            // candidate row does not state, that claim is unproven — the row
            // may be a different build of the same repo. One under-specified
            // row is not better evidence than several; it is the same missing
            // fact with a smaller sample. So the comparison is withheld
            // entirely rather than downgraded to a count.
            if let wanted = scope.modelIdentity {
                let undetermined = matching.contains { cell in
                    guard let identity = cell.identity else { return true }
                    return !wanted.facetsAreFullyDetermined(against: identity)
                }
                if undetermined { return .unavailable(.ambiguousIdentity) }
            }

            guard matching.count == 1 else {
                // Several fully-determined populations still match. That means
                // the comparison identity does not separate them, and no one
                // of them is the population this run belongs to.
                return .unavailable(.ambiguousIdentity)
            }
            let cell = matching[0]
            return .ready(
                cell.observationSummary(
                    includesYours: SummaryCell.contains(viewerSlug, in: [cell])
                )
            )
        } catch {
            return .unavailable(.failed(error.localizedDescription))
        }
    }

    func table(
        macProfile: CommunityMacProfile,
        workload: CommunityWorkload,
        metric: CommunityMetric,
        viewerSlug: String?
    ) async -> CommunityDataState<[CommunityObservationRow]> {
        do {
            let feed = try await publicFeed()
            // One row per model, narrowed by stated rules rather than by
            // picking whichever incompatible cell came first. See
            // `CommunityTableProjection`.
            let rows = CommunityTableProjection.rows(
                from: feed.summary
                    .filter { $0.machine.matches(macProfile) }
                    .compactMap { $0.projectionCell(aliasForRepoID) },
                workload: workload,
                metric: metric,
                viewerSlug: viewerSlug
            )
            // An empty filtered result is NOT "nobody has published anything
            // for this Mac and workload" — the feed only carries the newest
            // runs, so the rows may simply be further back. Returning
            // `.ready([])` let the table render "Yours would be the first",
            // which is the same false claim the branch rule exists to stop.
            guard !rows.isEmpty else { return .unavailable(.boundedFeed) }
            return .ready(rows)
        } catch {
            return .unavailable(.failed(error.localizedDescription))
        }
    }

    /// Coverage gaps need the set of models with **no** results on this Mac
    /// profile, which the bounded feed cannot establish — and a wrong answer
    /// here would put a "FIRST RESULT NEEDED" banner on a model that already
    /// has results.
    ///
    /// What it can honestly report is the opposite: models that DO appear, whose
    /// **total** across every execution and case cell is still small. Those are
    /// real under-represented pairings, and none of them is ever flagged as a
    /// first-result opportunity.
    ///
    /// The aggregation happens before the threshold, never after — see
    /// ``CommunityCoverageProjection`` for why the other order produced a count
    /// that belonged to no population.
    func coverageGaps(
        macProfile: CommunityMacProfile
    ) async -> CommunityDataState<[CommunityCoverageGap]> {
        do {
            let feed = try await publicFeed()
            let gaps = CommunityCoverageProjection.gaps(
                from: feed.summary
                    .filter { $0.machine.matches(macProfile) }
                    .compactMap { $0.projectionCell(aliasForRepoID) },
                below: Self.underRepresentedBelow
            )
            // An empty list means "nothing in the newest runs looks thin", not
            // "every catalogue model is covered" — the caller must say so.
            return .ready(gaps)
        } catch {
            return .unavailable(.failed(error.localizedDescription))
        }
    }

    /// A pairing with fewer than this many samples in the recent feed is worth
    /// another measurement.
    static let underRepresentedBelow = 5

    func pulse() async -> CommunityDataState<CommunityPulse> {
        do {
            let feed = try await publicFeed()
            var identities: [String: CommunityBenchmarkContributor] = [:]
            var models = Set<String>()
            var latest: Date?
            for run in feed.runs {
                if let contributor = run.contributor {
                    identities[contributor.slug] = contributor
                }
                models.insert(run.modelKey)
                if let accepted = Self.parseTimestamp(run.acceptedAt) {
                    latest = max(latest ?? accepted, accepted)
                }
            }
            return .ready(
                CommunityPulse(
                    contributors: identities.values.sorted { $0.slug < $1.slug },
                    contributorCount: identities.count,
                    publishedRunCount: feed.runs.count,
                    modelCount: models.count,
                    lastContributionAt: latest,
                    // The feed is the newest 50 runs, so every total here is a
                    // floor, not a census. The pulse band renders "at least".
                    isBounded: true
                )
            )
        } catch {
            return .unavailable(.failed(error.localizedDescription))
        }
    }

    // MARK: - Exact per-contributor totals

    /// The exact number of published runs and distinct models for one
    /// pseudonym, by following `cursor` until `complete == true`.
    ///
    /// This is the one total the public API can answer precisely, which is why
    /// the My Results identity band uses it instead of counting local receipts
    /// — a Mac that published before a reinstall, or whose receipt write
    /// failed, would otherwise undercount its own public contributions.
    func contributions(
        forSlug slug: String
    ) async -> CommunityDataState<CommunityContributorTotals> {
        var cursor: String?
        var runs: [AtomicRun] = []
        var pages = 0
        do {
            repeat {
                let endpoint = baseURL
                    .appendingPathComponent("api")
                    .appendingPathComponent("benchmarks")
                    .appendingPathComponent("atomic")
                    .appendingPathComponent("contributors")
                var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)
                var segmentCharacters = CharacterSet.alphanumerics
                segmentCharacters.formUnion(CharacterSet(charactersIn: "-._~"))
                guard let encodedSlug = slug.addingPercentEncoding(
                    withAllowedCharacters: segmentCharacters
                ) else { return .unavailable(.failed("bad contributor slug")) }
                components?.percentEncodedPath += "/" + encodedSlug
                var query = [URLQueryItem(name: "limit", value: "50")]
                if let cursor { query.append(URLQueryItem(name: "cursor", value: cursor)) }
                components?.queryItems = query
                guard let url = components?.url else { return .unavailable(.failed("bad url")) }
                let page: ContributionsPage = try await get(url)
                runs.append(contentsOf: page.runs)
                pages += 1
                if page.complete {
                    cursor = nil
                } else if let next = page.cursor, !next.isEmpty {
                    cursor = next
                } else {
                    return .unavailable(.incompleteAggregate)
                }
                if pages >= maximumContributionPages, !page.complete {
                    // Stopped early: the number would be a floor, and this
                    // screen promises an exact one.
                    return .unavailable(.incompleteAggregate)
                }
            } while cursor != nil

            return .ready(
                CommunityContributorTotals(
                    slug: slug,
                    publishedRunCount: runs.count,
                    modelCount: Set(runs.map(\.modelKey)).count,
                    lastContributionAt: runs
                        .compactMap { Self.parseTimestamp($0.acceptedAt) }
                        .max()
                )
            )
        } catch {
            return .unavailable(.failed(error.localizedDescription))
        }
    }

    // MARK: - Transport

    private func publicFeed() async throws -> PublicFeed {
        try await get(baseURL.appendingPathComponent("/api/benchmarks/atomic/public"))
    }

    private func get<Value: Decodable>(_ url: URL) async throws -> Value {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 15
        let (data, response) = try await transport(request)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw CommunityBenchmarkCommand.Failure(
                message: String(
                    format: String(localized: "The community service returned %1$d."),
                    http.statusCode
                )
            )
        }
        return try JSONDecoder().decode(Value.self, from: data)
    }

    static func parseTimestamp(_ raw: String) -> Date? {
        CommunityBenchmarkResult.parseTimestamp(raw)
    }
}

/// Exact totals for one pseudonym, read from the paginated contributions
/// endpoint.
struct CommunityContributorTotals: Hashable, Sendable {
    let slug: String
    let publishedRunCount: Int
    let modelCount: Int
    let lastContributionAt: Date?
}

// MARK: - Wire types

/// `/api/benchmarks/atomic/public`.
///
/// Only the fields Desktop reads are decoded; the projection carries more
/// (execution knobs, identity strength, per-case token counts) and adding a
/// field to the worker must not break the client.
private struct PublicFeed: Decodable {
    let summary: [SummaryCell]
    let runs: [AtomicRun]
}

private struct ContributionsPage: Decodable {
    let runs: [AtomicRun]
    let cursor: String?
    let complete: Bool
}

private struct AtomicRun: Decodable {
    let submissionID: String
    let acceptedAt: String
    let contributor: CommunityBenchmarkContributor?
    let taskType: String
    let model: CommunityModelIdentity.Wire
    let machine: AtomicMachine?
    let `protocol`: AtomicProtocol?

    enum CodingKeys: String, CodingKey {
        case submissionID = "submission_id"
        case acceptedAt = "accepted_at"
        case contributor
        case taskType = "task_type"
        case model
        case machine
        case `protocol`
    }

    /// Distinct-model counting keys on the full identity, so two variants of
    /// one repo are two models, as they are everywhere else.
    var modelKey: String {
        model.identity?.canonicalKey ?? model.repoID ?? "unknown"
    }

    func matchesCoverage(
        _ scope: CommunityBenchmarkScope,
        _ aliasForRepoID: (String) -> String
    ) -> Bool {
        guard let identity = model.identity,
              CommunityWorkload(taskType: taskType) == scope.workload,
              let machine, machine.matches(scope.macProfile),
              let `protocol`,
              `protocol`.id == scope.protocolID,
              `protocol`.version == scope.protocolVersion
        else { return false }
        if let wanted = scope.modelIdentity {
            return wanted.isCompatible(withPublished: identity)
        }
        return aliasForRepoID(identity.repoID) == scope.modelAlias
    }
}

private struct AtomicMachine: Decodable {
    let chip: String
    let memoryGiB: Int

    enum CodingKeys: String, CodingKey {
        case chip
        case memoryGiB = "memory_gib"
    }

    func matches(_ profile: CommunityMacProfile) -> Bool {
        chip == profile.chip && memoryGiB == profile.memoryGiB
    }
}

/// One `summary[]` cell: a median over the runs sharing a task, model, machine,
/// protocol, case and execution configuration.
private struct SummaryCell: Decodable {
    let taskType: String
    let model: CommunityModelIdentity.Wire
    let machine: AtomicMachine
    let metric: SummaryMetric
    let samples: Int
    let caseID: String
    let `protocol`: AtomicProtocol
    let execution: AtomicExecution
    /// The distinct pseudonyms behind this cell's runs. The worker builds this
    /// from `run.contributor` (`atomicBenchmarkSummary`), so it is the same
    /// identity the leaderboard shows — which is what makes "includes yours"
    /// answerable from a plain read after a restart or a reinstall.
    ///
    /// Optional on the wire: a cell whose runs were all submitted anonymously
    /// carries an empty list, and an older worker may omit the field entirely.
    let contributors: [CommunityBenchmarkContributor]?

    enum CodingKeys: String, CodingKey {
        case taskType = "task_type"
        case model, machine, metric, samples, execution, contributors
        case caseID = "case_id"
        case `protocol`
    }

    /// Canonical slugs only. `CommunityBenchmarkContributor.slug` is the API's
    /// slug when it sent one and exactly `name + "-" + tag` otherwise, which is
    /// the string `normalize()` composes on the website — so both sides key off
    /// one spelling of an identity.
    var contributorSlugs: Set<String> {
        Set((contributors ?? []).map(\.slug))
    }

    /// Whether this installation's pseudonym appears in any of these cells.
    static func contains(_ viewerSlug: String?, in cells: [SummaryCell]) -> Bool {
        guard let viewerSlug, !viewerSlug.isEmpty else { return false }
        return cells.contains { $0.contributorSlugs.contains(viewerSlug) }
    }

    var workload: CommunityWorkload? { CommunityWorkload(taskType: taskType) }

    /// The published identity of the model this cell aggregates.
    var identity: CommunityModelIdentity? { model.identity }

    /// Model + workload + protocol + machine. Everything that must agree
    /// before two results are even about the same thing.
    func matches(
        _ scope: CommunityBenchmarkScope,
        _ aliasForRepoID: (String) -> String
    ) -> Bool {
        matchesCoverage(scope, aliasForRepoID)
    }

    func matchesCoverage(
        _ scope: CommunityBenchmarkScope,
        _ aliasForRepoID: (String) -> String
    ) -> Bool {
        guard let identity else { return false }
        guard workload == scope.workload,
              machine.matches(scope.macProfile),
              `protocol`.id == scope.protocolID,
              `protocol`.version == scope.protocolVersion
        else { return false }
        // A scope derived from a real record knows exactly which artifact it
        // measured, so it is matched on the full contract identity: a cell
        // that states a different revision, subfolder or quantization is a
        // different model, not the same one seen twice.
        if let wanted = scope.modelIdentity {
            return wanted.isCompatible(withPublished: identity)
        }
        // A catalogue-entry scope has no resolved identity, so the repo id is
        // all there is to match on, via the alias the catalogue owns.
        return aliasForRepoID(identity.repoID) == scope.modelAlias
    }

    /// The remaining identity a *comparison* needs: same workload case, same
    /// measured quantity, same execution configuration.
    func matches(_ comparison: CommunityComparisonIdentity) -> Bool {
        caseID == comparison.caseID
            && metric.name == comparison.metricName
            && execution.identity == comparison.execution
    }

    /// The feed publishes a median and a best, but no observed minimum or
    /// maximum, so the range stays absent rather than being approximated from
    /// `best` (which is one end only).
    /// Flattened for ``CommunityTableProjection``. `executionKey` only has to
    /// be stable and distinct per configuration, so the decoded identity's
    /// description is enough.
    func projectionCell(
        _ aliasForRepoID: (String) -> String
    ) -> CommunityTableProjection.Cell? {
        guard let identity, let workload else { return nil }
        return CommunityTableProjection.Cell(
            modelAlias: aliasForRepoID(identity.repoID),
            modelIdentity: identity,
            workload: workload,
            protocolID: `protocol`.id,
            protocolVersion: `protocol`.version,
            caseID: caseID,
            metricName: metric.name,
            executionKey: String(describing: execution.identity),
            samples: samples,
            median: metric.median,
            unit: metric.unit,
            contributorSlugs: contributorSlugs
        )
    }

    func observationSummary(includesYours: Bool = false) -> CommunityObservationSummary {
        CommunityObservationSummary(
            observationCount: samples,
            median: metric.median,
            observedMinimum: nil,
            observedMaximum: nil,
            unit: metric.unit,
            includesYours: includesYours,
            isBounded: true
        )
    }
}

private struct AtomicProtocol: Decodable {
    let id: String
    let version: Int
}

/// The allowlisted execution projection the worker publishes and groups by.
private struct AtomicExecution: Decodable {
    let rapidMLX: String
    let computeDType: String?
    let speculativeDecoding: SpeculativeDecoding?
    let kvCache: KVCache?
    let prefillBackend: String?

    struct SpeculativeDecoding: Decodable { let method: String? }
    struct KVCache: Decodable {
        let mode: String?
        let dtype: String?
    }

    enum CodingKeys: String, CodingKey {
        case rapidMLX = "rapid_mlx"
        case computeDType = "compute_dtype"
        case speculativeDecoding = "speculative_decoding"
        case kvCache = "kv_cache"
        case prefillBackend = "prefill_backend"
    }

    var identity: CommunityExecutionIdentity {
        CommunityExecutionIdentity(
            rapidMLX: rapidMLX,
            computeDType: computeDType ?? "unknown",
            speculativeDecodingMethod: speculativeDecoding?.method,
            kvCacheMode: kvCache?.mode,
            kvCacheDType: kvCache?.dtype,
            prefillBackend: prefillBackend
        )
    }
}

private struct SummaryMetric: Decodable {
    let name: String
    let median: Double?
    let better: String?

    /// `decode_tps` → tok/s, `total_seconds` → s. Unknown metric names carry no
    /// unit rather than a guessed one.
    var unit: String? {
        switch name {
        case "decode_tps": return "tok/s"
        case "total_seconds": return "s"
        default: return nil
        }
    }
}
