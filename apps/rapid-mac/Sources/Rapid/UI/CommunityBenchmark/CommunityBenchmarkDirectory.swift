import Foundation

/// The read side of Community Benchmark: aggregate observations the Desktop
/// client can only *read*, never compute.
///
/// ## Why this seam exists
///
/// `rapid-mlx benchmark …` can run a measurement, list local history, build a
/// publish payload, and upload it (`POST /api/benchmarks`), but it has no read
/// subcommand — the client cannot ask it "how many public results exist for
/// this model, on this workload, on a Mac like mine?".
///
/// rapidmlx.com *does* answer that, over three public routes, and
/// ``CommunityBenchmarkAPIDirectory`` binds this protocol to them. The seam
/// exists because the routes answer some questions exactly and others only as a
/// floor: the shape of the answer, not its availability, is what the UI has to
/// reason about. ``UnavailableCommunityBenchmarkDirectory`` remains the
/// offline / not-configured fallback, where run and publish keep working and
/// community claims are withheld.
///
/// Every query is scoped by ``CommunityBenchmarkScope`` because a comparison —
/// and especially a "first result" claim — is only meaningful for one model on
/// one benchmark protocol on one Mac profile. Model alias alone is not enough.
protocol CommunityBenchmarkDirectory: Sendable {
    /// Aggregate for exactly one model + workload + Mac profile.
    ///
    /// `viewerSlug` is this installation's server-issued pseudonym, when it has
    /// one. It is used solely to answer "is one of my published runs already in
    /// this aggregate?" from the server's own `contributors` list, so the badge
    /// is a read of public data rather than a memory of this session.
    func observations(
        for scope: CommunityBenchmarkScope,
        viewerSlug: String?
    ) async -> CommunityDataState<CommunityObservationSummary>

    /// Every model with published observations on this Mac profile, for one
    /// workload and metric — the "Performance on Macs like yours" table.
    func table(
        macProfile: CommunityMacProfile,
        workload: CommunityWorkload,
        metric: CommunityMetric,
        viewerSlug: String?
    ) async -> CommunityDataState<[CommunityObservationRow]>

    /// Models on this Mac profile that have no observations, or few enough
    /// that another sample materially improves the comparison.
    func coverageGaps(
        macProfile: CommunityMacProfile
    ) async -> CommunityDataState<[CommunityCoverageGap]>

    /// Contributor / run / model totals plus recent activity for the pulse
    /// band. Server-provided aggregates only: never inferred from local
    /// receipts, which would make one Mac look like a community.
    func pulse() async -> CommunityDataState<CommunityPulse>
}

extension CommunityBenchmarkDirectory {
    /// The same queries asked by a caller with no identity yet — a fresh
    /// install, or any context where "is this mine?" simply does not apply.
    func observations(
        for scope: CommunityBenchmarkScope
    ) async -> CommunityDataState<CommunityObservationSummary> {
        await observations(for: scope, viewerSlug: nil)
    }

    func table(
        macProfile: CommunityMacProfile,
        workload: CommunityWorkload,
        metric: CommunityMetric
    ) async -> CommunityDataState<[CommunityObservationRow]> {
        await table(
            macProfile: macProfile, workload: workload, metric: metric, viewerSlug: nil
        )
    }
}

// MARK: - Query scope

/// Mac hardware identity a comparison is scoped to. Deliberately coarse —
/// chip family plus unified memory — so it groups comparable machines without
/// becoming a fingerprint.
struct CommunityMacProfile: Hashable, Sendable {
    let chip: String
    let memoryGiB: Int

    /// "Apple M3 Pro · 18 GB" — the label shown wherever the scope must be
    /// visible to the user (which is everywhere a "first" claim is made).
    var displayName: String { "\(chip) · \(memoryGiB) GB" }

    /// The profile of the Mac the app is running on.
    static func current(_ hardware: MacHardware = .detect()) -> Self {
        Self(
            chip: hardware.brandString,
            memoryGiB: max(1, Int(hardware.physicalRAMGB.rounded()))
        )
    }
}

/// The benchmark workload family. Mirrors the three registered protocols;
/// results from different families are never comparable.
enum CommunityWorkload: String, Hashable, Sendable, CaseIterable {
    case llm
    case image
    case video

    init(task: ModelTask) {
        switch task {
        case .imageGeneration: self = .image
        case .videoGeneration: self = .video
        default: self = .llm
        }
    }

    /// The contract's `workload.task_type`, as both local records and
    /// `summary[]` spell it.
    init?(taskType: String) {
        switch taskType {
        case "text_generation": self = .llm
        case "image_generation": self = .image
        case "video_generation": self = .video
        default: return nil
        }
    }

    var displayName: String {
        switch self {
        case .llm: return String(localized: "LLM")
        case .image: return String(localized: "Image")
        case .video: return String(localized: "Video")
        }
    }
}

/// Which measured quantity a table column ranks by. The unit travels with the
/// value from the server so the client never guesses a unit for a metric it
/// does not understand.
enum CommunityMetric: String, Hashable, Sendable {
    case generationSpeed
    case timeToFirstToken
    case renderTime
    case videoTime
    case peakMemory

    /// The metric a workload leads with, per the benchmark specification.
    static func primary(for workload: CommunityWorkload) -> Self {
        switch workload {
        case .llm: return .generationSpeed
        case .image: return .renderTime
        case .video: return .videoTime
        }
    }

    var displayName: String {
        switch self {
        case .generationSpeed: return String(localized: "Generation speed")
        case .timeToFirstToken: return String(localized: "Time to first token")
        case .renderTime: return String(localized: "Render time")
        case .videoTime: return String(localized: "Seconds per video")
        case .peakMemory: return String(localized: "Peak memory")
        }
    }
}

/// model + workload + protocol + Mac profile, and — when a specific run is
/// being compared — the exact case, execution configuration and metric behind
/// that run.
///
/// The two-level shape matters. A *coverage* question ("has anyone measured
/// this model on a Mac like mine?") is answered across every execution
/// configuration. A *comparison* ("is my number typical?") is only meaningful
/// against results produced the same way: a different protocol version,
/// prompt case, dtype, KV-cache mode or speculative-decoding setting can move
/// the number materially, so matching on model + workload + Mac alone would
/// silently compare a run against an unrelated population.
struct CommunityBenchmarkScope: Hashable, Sendable {
    /// The product alias, for display. Two variants of one repo can share an
    /// alias, so this is **not** the identity — ``modelIdentity`` is.
    let modelAlias: String
    let workload: CommunityWorkload
    /// Registered protocol identifier, e.g. `rapid-community-speed`. This is
    /// the server's `protocol.id`, not a display string.
    let protocolID: String
    /// Registered protocol version. Two results measured under different
    /// versions are not comparable.
    let protocolVersion: Int
    let macProfile: CommunityMacProfile
    /// The full contract identity of the model, when the scope was derived
    /// from a record that carries one. Nil on the Ready screen, where the
    /// question is about a catalogue entry the user has not measured yet and
    /// no revision or quantization has been resolved.
    ///
    /// Part of the scope's identity, so a `4bit/` subfolder and a repo-root
    /// build never share a count, a median, or a publication floor.
    var modelIdentity: CommunityModelIdentity?
    /// Present only when comparing one completed run. Absent on Ready, where
    /// no run exists yet and the question is coverage, not comparison.
    var comparison: CommunityComparisonIdentity?

    init(
        modelAlias: String,
        workload: CommunityWorkload,
        protocolID: String,
        protocolVersion: Int,
        macProfile: CommunityMacProfile,
        modelIdentity: CommunityModelIdentity? = nil,
        comparison: CommunityComparisonIdentity? = nil
    ) {
        self.modelAlias = modelAlias
        self.workload = workload
        self.protocolID = protocolID
        self.protocolVersion = protocolVersion
        self.macProfile = macProfile
        self.modelIdentity = modelIdentity
        self.comparison = comparison
    }

    /// Human-readable protocol label for technical details.
    var protocolName: String { "\(protocolID) v\(protocolVersion)" }

    /// A user-facing description of the scope, used verbatim in first-result
    /// copy so the claim is never broader than the data behind it.
    ///
    /// Names the variant when there is one to name: claiming "no one has
    /// published qwen3.5-9b-4bit" while holding a result for a specific
    /// subfolder or snapshot would be a broader claim than the evidence.
    var scopeDescription: String {
        let facets = modelIdentity?.distinguishingFacets ?? []
        let model = facets.isEmpty
            ? modelAlias
            : "\(modelAlias) (\(facets.joined(separator: ", ")))"
        return "\(model) · \(workload.displayName) · \(macProfile.displayName)"
    }
}

/// Everything beyond model/workload/protocol/machine that has to agree before
/// two numbers may be compared.
struct CommunityComparisonIdentity: Hashable, Sendable {
    /// The workload case the headline metric came from, e.g. `pp512-tg128`.
    /// The short and long prompt cases produce different tok/s.
    let caseID: String
    /// The server's metric name: `decode_tps` or `total_seconds`.
    let metricName: String
    let execution: CommunityExecutionIdentity
}

/// The execution knobs the service groups by.
///
/// Mirrors the allowlisted projection in `atomicPublicProjection`: runtime
/// version, compute dtype, and — for language tasks — speculative decoding,
/// KV-cache policy and prefill backend. These are exactly the fields the
/// worker folds into its summary group key, so matching on the same set is
/// what makes a client-side cell lookup agree with the server's grouping.
struct CommunityExecutionIdentity: Hashable, Sendable {
    let rapidMLX: String
    let computeDType: String
    let speculativeDecodingMethod: String?
    let kvCacheMode: String?
    let kvCacheDType: String?
    let prefillBackend: String?

    init(
        rapidMLX: String,
        computeDType: String,
        speculativeDecodingMethod: String? = nil,
        kvCacheMode: String? = nil,
        kvCacheDType: String? = nil,
        prefillBackend: String? = nil
    ) {
        self.rapidMLX = rapidMLX
        self.computeDType = computeDType
        self.speculativeDecodingMethod = speculativeDecodingMethod
        self.kvCacheMode = kvCacheMode
        self.kvCacheDType = kvCacheDType
        self.prefillBackend = prefillBackend
    }
}

// MARK: - Returned values

/// Aggregate for one scope. `observationCount == 0` is a real, *known* answer
/// and is the only thing that may drive first-reference language; not knowing
/// is represented by ``CommunityDataState`` instead.
struct CommunityObservationSummary: Hashable, Sendable {
    let observationCount: Int
    /// Median of the primary metric. Absent when the server has a count but
    /// not enough completed samples to publish a median.
    let median: Double?
    let observedMinimum: Double?
    let observedMaximum: Double?
    /// Unit string supplied by the server (e.g. "tok/s", "s"). Never invented.
    let unit: String?
    /// True when one of this installation's published results is in the
    /// aggregate. Comes from the server, matched by receipt.
    let includesYours: Bool
    /// True when the count is a **floor** rather than a census — it was read
    /// from the bounded public feed, which carries only the newest runs.
    ///
    /// The distinction changes what the UI is allowed to say: an exact count
    /// reads "7 published results exist", a bounded one reads "at least 7".
    /// It never changes the branch, because a bounded count is still positive
    /// evidence that results exist.
    let isBounded: Bool

    init(
        observationCount: Int,
        median: Double? = nil,
        observedMinimum: Double? = nil,
        observedMaximum: Double? = nil,
        unit: String? = nil,
        includesYours: Bool = false,
        isBounded: Bool = false
    ) {
        self.observationCount = observationCount
        self.median = median
        self.observedMinimum = observedMinimum
        self.observedMaximum = observedMaximum
        self.unit = unit
        self.includesYours = includesYours
        self.isBounded = isBounded
    }

    /// The aggregate after this installation publishes one more result into
    /// the same scope. Used only after a receipt confirms the upload, so the
    /// displayed count moves for a real publication and nothing else.
    ///
    /// The median and range are *dropped* rather than recomputed: the client
    /// holds no sample population, so any locally derived median would be a
    /// fabrication. The next read refreshes them.
    func incrementedAfterPublishing() -> Self {
        Self(
            observationCount: observationCount + 1,
            median: nil,
            observedMinimum: nil,
            observedMaximum: nil,
            unit: unit,
            includesYours: true,
            isBounded: isBounded
        )
    }
}

/// One row of "Performance on Macs like yours".
struct CommunityObservationRow: Hashable, Sendable, Identifiable {
    let modelAlias: String
    let workload: CommunityWorkload
    let summary: CommunityObservationSummary

    var id: String { "\(modelAlias)#\(workload.rawValue)" }
}

/// A model this Mac profile could usefully measure. `observationCount == 0`
/// selects the first-reference mission; anything larger is an
/// under-represented model.
struct CommunityCoverageGap: Hashable, Sendable, Identifiable {
    let modelAlias: String
    let workload: CommunityWorkload
    let observationCount: Int
    /// True when the count came from a newest-N public feed and is therefore
    /// a lower bound rather than an exact all-time total.
    let isBounded: Bool
    /// Whether the model fits this Mac, as reported by the benchmark catalog.
    let fitsThisMac: Bool
    let isDownloaded: Bool
    /// Approximate download size in GB when the model is not cached.
    let downloadSizeGB: Double?
    /// Memory the model needs, when the catalog knows it.
    let requiredMemoryGB: Int?

    init(
        modelAlias: String,
        workload: CommunityWorkload,
        observationCount: Int,
        isBounded: Bool = true,
        fitsThisMac: Bool,
        isDownloaded: Bool,
        downloadSizeGB: Double?,
        requiredMemoryGB: Int?
    ) {
        self.modelAlias = modelAlias
        self.workload = workload
        self.observationCount = observationCount
        self.isBounded = isBounded
        self.fitsThisMac = fitsThisMac
        self.isDownloaded = isDownloaded
        self.downloadSizeGB = downloadSizeGB
        self.requiredMemoryGB = requiredMemoryGB
    }

    var id: String { "\(modelAlias)#\(workload.rawValue)" }

    var isFirstResultOpportunity: Bool { observationCount == 0 }
}

/// Community-wide activity for the pulse band.
struct CommunityPulse: Hashable, Sendable {
    /// Real pseudonymous identities, in slug order. These are what the
    /// portraits are derived from — the same slug the website hashes — so one
    /// installation shows one face in the app and on the leaderboard.
    ///
    /// Not opaque "seeds": an arbitrary seed would render a cheetah plate that
    /// belongs to nobody, and the same contributor would look different here
    /// than on their own profile page.
    let contributors: [CommunityBenchmarkContributor]
    let contributorCount: Int
    let publishedRunCount: Int
    let modelCount: Int
    let lastContributionAt: Date?
    /// True when the totals were read from the bounded public feed, so they
    /// are floors. The band then renders "at least N".
    let isBounded: Bool

    init(
        contributors: [CommunityBenchmarkContributor] = [],
        contributorCount: Int,
        publishedRunCount: Int,
        modelCount: Int,
        lastContributionAt: Date?,
        isBounded: Bool = false
    ) {
        self.contributors = contributors
        self.contributorCount = contributorCount
        self.publishedRunCount = publishedRunCount
        self.modelCount = modelCount
        self.lastContributionAt = lastContributionAt
        self.isBounded = isBounded
    }

    /// Community portraits drawn before the overflow tile.
    static let renderedAvatarLimit = 5

    var renderedContributors: [CommunityBenchmarkContributor] {
        Array(contributors.prefix(Self.renderedAvatarLimit))
    }

    /// Contributors not represented by a rendered tile. Counted from the
    /// authoritative contributor total, not from the seed list, so a service
    /// that sends five seeds and a total of 24,000 still overflows correctly.
    var overflowCount: Int {
        max(0, contributorCount - renderedContributors.count)
    }

    /// "+24.0k" — abbreviated past a thousand so the fixed-width overflow
    /// tile cannot be outgrown by a healthy community.
    var overflowLabel: String? {
        let remaining = overflowCount
        guard remaining > 0 else { return nil }
        if remaining < 1_000 { return "+\(remaining)" }
        let thousands = Double(remaining) / 1_000
        if thousands < 10 { return String(format: "+%.1fk", thousands) }
        if thousands < 1_000 { return String(format: "+%.0fk", thousands) }
        return String(format: "+%.1fM", thousands / 1_000)
    }
}

// MARK: - Availability

/// Three-way result for every community read.
///
/// The distinction that matters: ``unavailable`` and ``loading`` are NOT an
/// observation count of zero. Treating them as zero is what would produce a
/// false "you would be the first" claim, so the type makes that conflation
/// impossible to express.
enum CommunityDataState<Value: Sendable>: Sendable {
    case loading
    case unavailable(CommunityUnavailableReason)
    case ready(Value)

    var value: Value? {
        if case let .ready(value) = self { return value }
        return nil
    }

    var isLoading: Bool {
        if case .loading = self { return true }
        return false
    }

    var unavailableReason: CommunityUnavailableReason? {
        if case let .unavailable(reason) = self { return reason }
        return nil
    }

    func map<Other: Sendable>(
        _ transform: (Value) -> Other
    ) -> CommunityDataState<Other> {
        switch self {
        case .loading: return .loading
        case let .unavailable(reason): return .unavailable(reason)
        case let .ready(value): return .ready(transform(value))
        }
    }
}

extension CommunityDataState: Equatable where Value: Equatable {}

enum CommunityUnavailableReason: Hashable, Sendable {
    /// No community read API is configured in this build. The shipping
    /// default until the endpoint exists.
    case notConfigured
    /// The endpoint exists but could not be reached.
    case offline
    /// The endpoint answered with an error or an unreadable body.
    case failed(String)
    /// The public feed is bounded to its newest runs, so it can prove that
    /// results exist but never that none do. Absence from it means "unknown",
    /// which is what stops a quiet week from reading as "you would be first".
    case boundedFeed
    /// Pagination did not reach `complete`, so an exact total is not available.
    case incompleteAggregate
    /// The server returned a count that cannot be interpreted — negative, or
    /// otherwise malformed. Never silently treated as zero.
    case invalidCount
    /// Published rows exist for this repo, but none of them can be shown to be
    /// the same artifact as this run: they state fewer identity facets than it
    /// pins (no revision, no quantization), or several fully-determined
    /// variants match at once. A comparison would be a claim about provenance
    /// the data does not support, so the statistic is withheld.
    case ambiguousIdentity

    var message: String {
        switch self {
        case .notConfigured:
            return String(localized: "Community comparisons are not available in this build.")
        case .offline:
            return String(localized: "Community data unavailable. Running and publishing still work.")
        case let .failed(detail):
            return detail
        case .boundedFeed:
            return String(
                localized: "The public feed only covers the most recent runs, so the total for this model and Mac is not known yet."
            )
        case .incompleteAggregate:
            return String(localized: "The published history was too long to total exactly.")
        case .invalidCount:
            return String(localized: "The community service returned a count that could not be read.")
        case .ambiguousIdentity:
            return String(
                localized: "Published results exist for this model, but not enough detail to confirm they used the same build as your run."
            )
        }
    }
}

// MARK: - Shipping default

/// The directory the app ships with today: every query is unavailable because
/// no community read API exists in this repository.
///
/// This is deliberately not a stub returning zeros. Zero would be a factual
/// claim ("nobody has published this") that the client cannot substantiate,
/// and it would light up first-reference language everywhere.
struct UnavailableCommunityBenchmarkDirectory: CommunityBenchmarkDirectory {
    let reason: CommunityUnavailableReason

    init(reason: CommunityUnavailableReason = .notConfigured) {
        self.reason = reason
    }

    func observations(
        for scope: CommunityBenchmarkScope,
        viewerSlug: String?
    ) async -> CommunityDataState<CommunityObservationSummary> {
        .unavailable(reason)
    }

    func table(
        macProfile: CommunityMacProfile,
        workload: CommunityWorkload,
        metric: CommunityMetric,
        viewerSlug: String?
    ) async -> CommunityDataState<[CommunityObservationRow]> {
        .unavailable(reason)
    }

    func coverageGaps(
        macProfile: CommunityMacProfile
    ) async -> CommunityDataState<[CommunityCoverageGap]> {
        .unavailable(reason)
    }

    func pulse() async -> CommunityDataState<CommunityPulse> {
        .unavailable(reason)
    }
}

/// A directory backed by explicit in-memory fixtures. Used by SwiftUI previews
/// and tests to exercise both contribution branches without inventing a wire
/// format; never wired into a shipping build.
struct StaticCommunityBenchmarkDirectory: CommunityBenchmarkDirectory {
    var summaries: [CommunityBenchmarkScope: CommunityObservationSummary] = [:]
    var rows: [CommunityObservationRow] = []
    var gaps: [CommunityCoverageGap] = []
    var pulseValue: CommunityPulse?
    /// Applied to every query, so a test can exercise the loading and offline
    /// paths without a live endpoint.
    var forcedState: CommunityUnavailableReason?
    var isLoadingForever = false

    private func state<Value: Sendable>(_ value: Value) -> CommunityDataState<Value> {
        if isLoadingForever { return .loading }
        if let forcedState { return .unavailable(forcedState) }
        return .ready(value)
    }

    func observations(
        for scope: CommunityBenchmarkScope,
        viewerSlug: String?
    ) async -> CommunityDataState<CommunityObservationSummary> {
        guard let summary = summaries[scope] else {
            if isLoadingForever { return .loading }
            if let forcedState { return .unavailable(forcedState) }
            return .unavailable(.notConfigured)
        }
        return state(summary)
    }

    func table(
        macProfile: CommunityMacProfile,
        workload: CommunityWorkload,
        metric: CommunityMetric,
        viewerSlug: String?
    ) async -> CommunityDataState<[CommunityObservationRow]> {
        state(rows.filter { $0.workload == workload })
    }

    func coverageGaps(
        macProfile: CommunityMacProfile
    ) async -> CommunityDataState<[CommunityCoverageGap]> {
        state(gaps)
    }

    func pulse() async -> CommunityDataState<CommunityPulse> {
        guard let pulseValue else {
            if isLoadingForever { return .loading }
            return .unavailable(forcedState ?? .notConfigured)
        }
        return state(pulseValue)
    }
}
