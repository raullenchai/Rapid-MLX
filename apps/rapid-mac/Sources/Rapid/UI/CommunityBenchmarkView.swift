import Darwin
import Foundation
import SwiftUI

private let communityBenchmarkLeaderboardURL = URL(
    string: "https://rapidmlx.com/leaderboard"
)!

struct CommunityBenchmarkModel: Identifiable, Hashable {
    let entry: ModelEntry
    let task: ModelTask
    let protocolName: String
    /// The registered protocol identifier the service uses, e.g.
    /// `rapid-community-speed`. Supplied by `benchmark catalog --json`; the
    /// fallback mirrors `_TASK_PROTOCOL` in `community_bench/workspace.py`.
    var protocolID: String = "rapid-community-speed"
    var protocolVersion: Int = 2
    let isFocus: Bool
    let estimatedMemoryGib: Int?
    let memoryFit: String
    var runtimeStatus: String? = nil
    var runtimeMessage: String? = nil

    var id: String { entry.alias }
    var runtimeCanRun: Bool { runtimeStatus != "unavailable" }

    static let focusAliases: Set<String> = [
        "qwen3.8-27b-4bit", "qwen3.5-9b-4bit", "gemma-4-e4b-4bit",
        "flux2-klein-4b", "z-image-turbo", "qwen-image",
        "wan2.2-ti2v-5b-q8"
    ]
    /// Mirrors `_TASK_PROTOCOL` in `rapid_mlx/community_bench/workspace.py`.
    static func defaultProtocolID(for task: ModelTask) -> String {
        switch task {
        case .imageGeneration: return "rapid-image-speed"
        case .videoGeneration: return "rapid-video-speed"
        default: return "rapid-community-speed"
        }
    }

    static let registeredWanAliases: Set<String> = [
        "wan2.2-t2v-a14b-bf16", "wan2.2-ti2v-5b-bf16", "wan2.2-ti2v-5b-q8"
    ]

    static func models(
        from catalog: [ModelEntry],
        metadata: [String: CommunityBenchmarkCatalogModel] = [:]
    ) -> [Self] {
        catalog.compactMap { entry in
            let catalogModel = metadata[entry.alias]
            if !metadata.isEmpty, catalogModel == nil { return nil }
            let task: ModelTask?
            if entry.taskTypes.contains(.imageGeneration),
               entry.operationModes.contains(.textToImage) {
                task = .imageGeneration
            } else if entry.taskTypes.contains(.videoGeneration),
                      entry.operationModes.contains(.textToVideo),
                      registeredWanAliases.contains(entry.alias) {
                task = .videoGeneration
            } else if entry.taskTypes.contains(.textGeneration) {
                task = .textGeneration
            } else if entry.taskTypes.isEmpty {
                // A pre-atomic Desktop row has no capability evidence. Text
                // keeps its historical fallback, but diffusion/video rows
                // must not advertise a registered protocol based on `kind`
                // alone: the shared CLI may reject their operation/family.
                switch entry.kind {
                case .image, .video: task = nil
                case .chat: task = .textGeneration
                case .audio: task = nil
                }
            } else {
                task = nil
            }
            guard let task else { return nil }
            let protocolVersion = catalogModel?.protocolVersion ?? {
                switch task {
                case .textGeneration: 2
                case .imageGeneration, .videoGeneration: 1
                default: 1
                }
            }()
            let protocolName: String
            switch task {
            case .imageGeneration: protocolName = "Rapid Image Speed v\(protocolVersion)"
            case .videoGeneration: protocolName = "Rapid Video Speed v\(protocolVersion)"
            case .textGeneration: protocolName = "Rapid Community Speed v\(protocolVersion)"
            default: return nil
            }
            return Self(
                entry: entry,
                task: task,
                protocolName: protocolName,
                protocolID: catalogModel?.protocolID ?? Self.defaultProtocolID(for: task),
                protocolVersion: protocolVersion,
                isFocus: catalogModel?.focus ?? focusAliases.contains(entry.alias),
                estimatedMemoryGib: catalogModel?.estimatedMemoryGib,
                memoryFit: catalogModel?.memoryFit ?? "unknown",
                runtimeStatus: catalogModel?.runtime?.status,
                runtimeMessage: catalogModel?.runtime?.message
            )
        }
        .sorted {
            if $0.isFocus != $1.isFocus { return $0.isFocus }
            if $0.entry.cached != $1.entry.cached { return $0.entry.cached }
            return $0.entry.alias.localizedStandardCompare($1.entry.alias) == .orderedAscending
        }
    }

    /// One labelled group of the model picker menu.
    struct PickerSection: Equatable {
        let title: String
        let models: [CommunityBenchmarkModel]
    }

    static let recommendedSectionTitle = String(
        localized: String.LocalizationValue("Recommended for this Mac")
    )
    static let downloadedSectionTitle = String(
        localized: String.LocalizationValue("Downloaded")
    )
    static let allModelsSectionTitle = String(
        localized: String.LocalizationValue("All models")
    )

    /// Splits the flat (already sorted) model list into the three menu
    /// groups: focus models that fit this Mac first, then anything else that
    /// is already on disk, then the long tail. Empty groups are dropped so the
    /// menu never shows a header with nothing under it. Every alias appears in
    /// exactly one section, so `Picker` tags stay unique.
    static func pickerSections(_ models: [Self]) -> [PickerSection] {
        var recommended: [Self] = []
        var downloaded: [Self] = []
        var rest: [Self] = []
        for model in models {
            if model.isFocus, model.memoryFit != "does_not_fit" {
                recommended.append(model)
            } else if model.entry.cached {
                downloaded.append(model)
            } else {
                rest.append(model)
            }
        }
        return [
            PickerSection(title: recommendedSectionTitle, models: recommended),
            PickerSection(title: downloadedSectionTitle, models: downloaded),
            PickerSection(title: allModelsSectionTitle, models: rest),
        ].filter { !$0.models.isEmpty }
    }

    static func reconciledSelection(current: String, models: [Self]) -> String {
        if models.contains(where: { $0.entry.alias == current }) { return current }
        return models.first?.entry.alias ?? ""
    }

    static func resolvedCatalog(
        product: [ModelEntry]?,
        fallback: [ModelEntry]
    ) -> [ModelEntry] {
        product ?? fallback
    }
}

struct CommunityBenchmarkCatalogModel: Decodable, Sendable {
    struct RuntimeReadiness: Decodable, Sendable {
        let status: String
        let message: String?
    }

    let alias: String
    let focus: Bool
    let estimatedMemoryGib: Int?
    let memoryFit: String
    let protocolVersion: Int?
    /// `protocol_id` from `benchmark catalog --json`, e.g.
    /// `rapid-community-speed`. Optional so an older runtime still parses.
    let protocolID: String?
    let runtime: RuntimeReadiness?

    enum CodingKeys: String, CodingKey {
        case alias, focus, runtime
        case estimatedMemoryGib = "estimated_memory_gib"
        case memoryFit = "memory_fit"
        case protocolVersion = "protocol_version"
        case protocolID = "protocol_id"
    }

    /// Explicit memberwise init so `protocolID` can default to nil: it was
    /// added after the existing construction sites, and an older runtime's
    /// catalog simply does not carry it.
    init(
        alias: String,
        focus: Bool,
        estimatedMemoryGib: Int?,
        memoryFit: String,
        protocolVersion: Int?,
        protocolID: String? = nil,
        runtime: RuntimeReadiness? = nil
    ) {
        self.alias = alias
        self.focus = focus
        self.estimatedMemoryGib = estimatedMemoryGib
        self.memoryFit = memoryFit
        self.protocolVersion = protocolVersion
        self.protocolID = protocolID
        self.runtime = runtime
    }
}

private struct CommunityBenchmarkCatalogEnvelope: Decodable {
    let models: [CommunityBenchmarkCatalogModel]
}

struct CommunityBenchmarkResults: Decodable {
    let runs: [CommunityBenchmarkResult]
    let receipts: [String: CommunityBenchmarkReceipt]?
}

struct CommunityBenchmarkContributor: Decodable, Hashable, Sendable {
    let name: String
    let tag: String
    /// The canonical slug the API issues. Optional because older receipts and
    /// the CLI's own payload predate it.
    private let rawSlug: String?
    /// The profile route the API supplied, when it supplied one.
    private let rawURL: String?

    enum CodingKeys: String, CodingKey {
        case name, tag
        case rawSlug = "slug"
        case rawURL = "url"
    }

    init(name: String, tag: String, slug: String? = nil, url: String? = nil) {
        self.name = name
        self.tag = tag
        rawSlug = slug
        rawURL = url
    }

    var displayName: String { "\(name) ·\(tag)" }

    /// The identity key everything else keys off — the avatar plate, the
    /// profile route, and the contributions query.
    ///
    /// Mirrors `normalize()` in `landing/public/community-identity.js`
    /// exactly: the API's slug when present, otherwise `name + "-" + tag`,
    /// which is the same string the API composes. Desktop must not invent a
    /// third spelling, or the app and the website would disagree about who
    /// this installation is.
    var slug: String {
        if let rawSlug, !rawSlug.isEmpty { return rawSlug }
        return "\(name)-\(tag)"
    }

    /// `/leaderboard/contributors/<slug>` unless the API sent its own route.
    var profileURL: URL? {
        if let rawURL, !rawURL.isEmpty {
            if let absolute = URL(string: rawURL), absolute.scheme != nil { return absolute }
            return URL(string: "https://rapidmlx.com\(rawURL.hasPrefix("/") ? "" : "/")\(rawURL)")
        }
        return legacyProfileURL
    }

    private var legacyProfileURL: URL? {
        // Percent-encode the identifier so an embedded "/" in a server-assigned
        // name/tag cannot become a path separator — mirrors the CLI client's
        // urllib `quote(f"{name}-{tag}", safe="-")`. (Nothing but ASCII
        // alphanumerics, "_", ".", "-", "~" stays literal; everything else is
        // percent-encoded, so the joined slug is safe to drop into a URL path.)
        var allowed = CharacterSet.alphanumerics
        allowed.formUnion(CharacterSet(charactersIn: "_.-~"))
        let encoded = slug.addingPercentEncoding(
            withAllowedCharacters: allowed
        ) ?? ""
        return URL(string: "https://rapidmlx.com/leaderboard/contributors/\(encoded)")
    }
}

struct CommunityBenchmarkReceipt: Decodable, Identifiable, Equatable, Sendable {
    let submissionID: String
    let alreadyExists: Bool
    let acceptedAt: String
    let contributor: CommunityBenchmarkContributor?

    var id: String { submissionID }
    var contributionLinkTitle: String {
        contributor?.displayName ?? "View Community Benchmark"
    }

    var contributionURL: URL {
        contributor?.profileURL ?? communityBenchmarkLeaderboardURL
    }

    var contributionAccessibilityLabel: String {
        contributor.map { "View contributions by \($0.displayName)" }
            ?? "View Community Benchmark"
    }

    enum CodingKeys: String, CodingKey {
        case contributor
        case submissionID = "submission_id"
        case alreadyExists = "already_exists"
        case acceptedAt = "accepted_at"
    }
}

private struct CommunityBenchmarkShareResponse: Decodable {
    let uploaded: Bool
    let receiptSaved: Bool
    let receipt: CommunityBenchmarkReceipt

    enum CodingKeys: String, CodingKey {
        case uploaded, receipt
        case receiptSaved = "receipt_saved"
    }
}

struct CommunityBenchmarkUploadPreview: Identifiable {
    /// One fact the local archive keeps that the submission does not carry.
    ///
    /// The CLI projects the archived record before sending it, because the
    /// ingestion validator allowlists a narrower model identity than a warm
    /// cache produces (see `rapid_mlx/community_bench/publication.py`). The
    /// projection is only honest if the user can see it, so the exact facts —
    /// path, value and the reason each was held back — travel with the preview
    /// and are shown before Publish.
    struct WithheldFact: Equatable, Sendable, Identifiable {
        let path: String
        /// Rendered from whatever JSON the CLI sent: a string, a number, or a
        /// nested object such as a quantization block.
        let value: String
        let reason: String

        var id: String { path }

        /// The leaf the path names, for a compact label.
        var fieldName: String {
            path.split(separator: ".").last.map(String.init) ?? path
        }
    }

    let runID: String
    let target: String
    let installID: String
    let payloadDigest: String
    let bodyDigest: String
    let payloadJSON: String
    /// Everything the projection removed. Empty when the record already
    /// satisfied the allowlist — a cold-cache run, typically.
    var withheld: [WithheldFact] = []
    /// The model identity actually on the wire, decoded from `payloadJSON`, so
    /// the disclosure describes the submission rather than the archive.
    var publishedIdentity: CommunityModelIdentity?

    var id: String { runID }
}

struct CommunityBenchmarkResult: Decodable, Identifiable {
    struct Workload: Decodable {
        struct Case: Decodable {
            let caseID: String
            let targetPromptTokens: Int?
            let targetOutputTokens: Int?
            let warmupRounds: Int?
            let measuredRounds: Int?
            enum CodingKeys: String, CodingKey {
                case caseID = "case_id"
                case targetPromptTokens = "target_prompt_tokens"
                case targetOutputTokens = "target_output_tokens"
                case warmupRounds = "warmup_rounds"
                case measuredRounds = "measured_rounds"
            }
        }
        let taskType: String
        let cases: [Case]?
        /// `registered_workload` writes both on every record, and the worker
        /// requires them. Optional so a record written before they existed
        /// still decodes — it simply cannot be scoped for comparison.
        let protocolID: String?
        let protocolVersion: Int?
        enum CodingKeys: String, CodingKey {
            case taskType = "task_type"
            case cases
            case protocolID = "protocol_id"
            case protocolVersion = "protocol_version"
        }
    }
    struct Outcome: Decodable { let status: String }
    struct Measurement: Decodable {
        let caseID: String
        let completed: Bool?
        let outputTokens: Int?
        let ttftMS: Double?
        let decodeDurationMS: Double?
        let totalDurationMS: Double?
        /// `measurementBase.peak_active_memory_mib` — the high-water unified
        /// memory for this round. Optional because a runtime that cannot
        /// sample it omits the field rather than reporting zero.
        let peakActiveMemoryMiB: Double?
        enum CodingKeys: String, CodingKey {
            case caseID = "case_id"
            case completed
            case outputTokens = "output_tokens"
            case ttftMS = "ttft_ms"
            case decodeDurationMS = "decode_duration_ms"
            case totalDurationMS = "total_duration_ms"
            case peakActiveMemoryMiB = "peak_active_memory_mib"
        }

        /// Explicit memberwise init so `peakActiveMemoryMiB` can default to
        /// nil: it was added after the existing construction sites, and a
        /// round that predates the field is "unknown", not "zero bytes".
        init(
            caseID: String,
            completed: Bool?,
            outputTokens: Int?,
            ttftMS: Double?,
            decodeDurationMS: Double?,
            totalDurationMS: Double?,
            peakActiveMemoryMiB: Double? = nil
        ) {
            self.caseID = caseID
            self.completed = completed
            self.outputTokens = outputTokens
            self.ttftMS = ttftMS
            self.decodeDurationMS = decodeDurationMS
            self.totalDurationMS = totalDurationMS
            self.peakActiveMemoryMiB = peakActiveMemoryMiB
        }
    }
    /// The full contract identity, not just a repo id: a `4bit/` subfolder,
    /// the resolved snapshot revision and the artifact's quantization all
    /// distinguish one measured artifact from another.
    typealias Model = CommunityModelIdentity.Wire
    struct Machine: Decodable {
        struct Profile: Decodable {
            let chip: String
            let memoryGib: Int
            let cpuCores: Int
            let gpuCores: Int?

            enum CodingKeys: String, CodingKey {
                case chip
                case memoryGib = "memory_gib"
                case cpuCores = "cpu_cores"
                case gpuCores = "gpu_cores"
            }
        }
        struct OS: Decodable { let version: String }
        let profile: Profile
        let os: OS
    }
    struct Execution: Decodable {
        struct Runtime: Decodable {
            let rapidMLX: String
            let mlx: String
            let python: String

            enum CodingKeys: String, CodingKey {
                case rapidMLX = "rapid_mlx"
                case mlx, python
            }
        }
        /// `execution.resources` — only `compute_dtype` takes part in the
        /// server's summary grouping.
        struct Resources: Decodable {
            let computeDType: String?
            enum CodingKeys: String, CodingKey { case computeDType = "compute_dtype" }
        }
        /// `execution.task` — language knobs are what the worker projects for
        /// text runs. Image/video tasks carry no `language` block.
        struct Task: Decodable {
            struct Language: Decodable {
                struct SpeculativeDecoding: Decodable { let method: String? }
                struct KVCache: Decodable {
                    let mode: String?
                    let dtype: String?
                }
                let speculativeDecoding: SpeculativeDecoding?
                let kvCache: KVCache?
                let prefillBackend: String?

                enum CodingKeys: String, CodingKey {
                    case speculativeDecoding = "speculative_decoding"
                    case kvCache = "kv_cache"
                    case prefillBackend = "prefill_backend"
                }
            }
            let language: Language?
        }
        let runtime: Runtime
        let configDigest: String
        let resources: Resources?
        let task: Task?

        enum CodingKeys: String, CodingKey {
            case runtime, resources, task
            case configDigest = "config_digest"
        }
    }
    let id: String
    let completedAt: String
    let workload: Workload
    let outcome: Outcome
    let measurements: [Measurement]?
    let model: Model
    let machine: Machine?
    let execution: Execution

    enum CodingKeys: String, CodingKey {
        case id = "run_id"
        case completedAt = "completed_at"
        case workload, outcome, measurements, model, machine, execution
    }

    /// Per-case medians over completed rounds — the same shape the CLI prints
    /// after `benchmark run` (`summarize_measurements`): decode tok/s + TTFT
    /// for text cases, wall seconds for image/video.
    struct CaseSummary: Equatable {
        let caseID: String
        let rounds: Int
        /// Median decode throughput using the leaderboard's formula
        /// `(output_tokens - 1) / decode_duration` (the first decode token is
        /// counted in TTFT), so the number matches what the public board will
        /// show for a shared run. The CLI's text summary divides by
        /// `output_tokens`, so it reads ~1/128 higher on the short case.
        let decodeTokensPerSecond: Double?
        let ttftMS: Double?
        /// Median wall time per round; the headline for image/video cases.
        let wallSeconds: Double?

        var headline: String {
            if let decodeTokensPerSecond {
                var text = String(format: "%.1f tok/s", decodeTokensPerSecond)
                if let ttftMS {
                    text += " · TTFT " + Self.formatMilliseconds(ttftMS)
                }
                return text
            }
            if let wallSeconds {
                return String(format: "%.1f s per run", wallSeconds)
            }
            // Unreachable for summaries produced by `summarize`, which drops
            // cases without a task-appropriate metric.
            return "\(rounds) rounds"
        }

        static func formatMilliseconds(_ value: Double) -> String {
            value >= 10_000
                ? String(format: "%.1f s", value / 1_000)
                : String(format: "%.0f ms", value)
        }
    }

    /// Case order follows the workload declaration (short case first), then
    /// first appearance for measurements the workload did not declare.
    var caseSummaries: [CaseSummary] {
        Self.summarize(
            measurements: measurements ?? [],
            declaredOrder: workload.cases?.map(\.caseID) ?? [],
            taskType: workload.taskType
        )
    }

    /// `taskType` decides the headline family: text workloads report decode
    /// tok/s (+ TTFT) and never fall back to wall time, so a text record
    /// missing decode fields shows its status instead of an image-style
    /// "s per run"; image/video workloads report median wall seconds.
    static func summarize(
        measurements: [Measurement],
        declaredOrder: [String],
        taskType: String
    ) -> [CaseSummary] {
        let isText = taskType == "text_generation"
        var order = declaredOrder
        var byCase: [String: [Measurement]] = [:]
        // Records written before the `completed` flag existed carry only
        // finished rounds, so a missing flag means completed.
        for sample in measurements where sample.completed ?? true {
            if byCase[sample.caseID] == nil, !order.contains(sample.caseID) {
                order.append(sample.caseID)
            }
            byCase[sample.caseID, default: []].append(sample)
        }
        return order.compactMap { caseID in
            guard let samples = byCase[caseID], !samples.isEmpty else { return nil }
            // Decode and TTFT medians come from the same text rounds so the
            // headline never pairs numbers from different populations: TTFT
            // is reported only when every decode round carries it.
            let textRounds = isText ? samples.filter { sample in
                (sample.outputTokens ?? 0) > 1 && (sample.decodeDurationMS ?? 0) > 0
            } : []
            let decode = textRounds.map { sample in
                Double(sample.outputTokens! - 1) / sample.decodeDurationMS! * 1_000
            }
            let ttft = textRounds.compactMap(\.ttftMS)
            let decodeMedian = median(decode)
            let wall = isText ? [] : samples.compactMap(\.totalDurationMS)
            let wallMedian = median(wall).map { $0 / 1_000 }
            // Like the CLI, a case with neither metric contributes nothing,
            // so a run made only of such cases falls back to its status.
            guard decodeMedian != nil || wallMedian != nil else { return nil }
            return CaseSummary(
                caseID: caseID,
                rounds: samples.count,
                decodeTokensPerSecond: decodeMedian,
                ttftMS: ttft.count == textRounds.count ? median(ttft) : nil,
                wallSeconds: wallMedian
            )
        }
    }

    static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid]
    }

    /// Only a run the CLI marked completed gets numbers; a run that failed
    /// after some rounds keeps showing its outcome so partial medians are
    /// never mistaken for a finished benchmark.
    var isCompleted: Bool { outcome.status == "completed" }

    /// The number shown on the result row: the first declared (short) case.
    /// If that case produced no usable metric and another case is promoted,
    /// its ID is kept in the headline so it is never mistaken for the short
    /// case.
    var headline: String? {
        guard isCompleted, let first = caseSummaries.first else { return nil }
        let declaredFirst = workload.cases?.first?.caseID
        return declaredFirst == nil || declaredFirst == first.caseID
            ? first.headline
            : "\(first.caseID): \(first.headline)"
    }

    /// Remaining cases, one per line, for the secondary line / tooltip.
    var secondaryLines: [String] {
        guard isCompleted else { return [] }
        return caseSummaries.dropFirst().map { "\($0.caseID): \($0.headline)" }
    }

    /// The full identity of the artifact this run measured. Nil when the
    /// record names no model at all.
    var modelIdentity: CommunityModelIdentity? { model.identity }

    var repoID: String { modelIdentity?.repoID ?? "Local model" }

    /// The Mac this run was measured on — the record's own machine, never the
    /// one the app happens to be running on now.
    var macProfile: CommunityMacProfile? {
        machine.map {
            CommunityMacProfile(chip: $0.profile.chip, memoryGiB: $0.profile.memoryGib)
        }
    }

    /// The complete community scope this run belongs to, derived **only** from
    /// the record.
    ///
    /// Publishing an older run from My Results used to build its scope from the
    /// *currently selected* model, so a receipt for model X was attributed to
    /// whatever Y the picker was on: Y's alias in the celebration, Y's protocol
    /// version in the query, Y's count incremented, Y's floor confirmed. Every
    /// field here now comes from the run itself.
    ///
    /// `alias` maps the repo id to the product alias for display; it is the
    /// only thing the caller supplies, because the catalogue owns that mapping
    /// and the record does not carry it.
    ///
    /// Nil when the record cannot be scoped truthfully — no model, no machine,
    /// no registered protocol, or no primary case. A run that cannot say what
    /// it is about must not be compared with anything.
    func communityScope(alias: (String) -> String) -> CommunityBenchmarkScope? {
        guard let modelIdentity,
              let macProfile,
              let communityWorkload = CommunityWorkload(taskType: workload.taskType),
              let protocolID = workload.protocolID,
              let protocolVersion = workload.protocolVersion,
              let comparisonIdentity
        else { return nil }
        return CommunityBenchmarkScope(
            modelAlias: alias(modelIdentity.repoID),
            workload: communityWorkload,
            protocolID: protocolID,
            protocolVersion: protocolVersion,
            macProfile: macProfile,
            modelIdentity: modelIdentity,
            comparison: comparisonIdentity
        )
    }

    /// The execution configuration this run was measured under, in exactly the
    /// fields the service groups by. Used to pick the ONE summary cell this
    /// result may honestly be compared against.
    var executionIdentity: CommunityExecutionIdentity {
        let language = execution.task?.language
        return CommunityExecutionIdentity(
            rapidMLX: execution.runtime.rapidMLX,
            computeDType: execution.resources?.computeDType ?? "unknown",
            speculativeDecodingMethod: language?.speculativeDecoding?.method,
            kvCacheMode: language?.kvCache?.mode,
            kvCacheDType: language?.kvCache?.dtype,
            prefillBackend: language?.prefillBackend
        )
    }

    /// The declared primary case — the one the headline metric came from, and
    /// the one the service keys its summary group on (`run.cases[0]`).
    var primaryCaseID: String? {
        workload.cases?.first?.caseID ?? caseSummaries.first?.caseID
    }

    /// The server's metric name for this workload.
    var primaryMetricName: String {
        workload.taskType == "text_generation" ? "decode_tps" : "total_seconds"
    }

    /// Everything beyond model/workload/protocol/machine that must agree
    /// before this run may be compared with a published aggregate. Nil when
    /// the record does not name its primary case.
    var comparisonIdentity: CommunityComparisonIdentity? {
        guard let primaryCaseID else { return nil }
        return CommunityComparisonIdentity(
            caseID: primaryCaseID,
            metricName: primaryMetricName,
            execution: executionIdentity
        )
    }

    /// `completed_at` is a UTC ISO-8601 stamp with or without fractional
    /// seconds, depending on the CLI version that wrote the record.
    static func parseTimestamp(_ raw: String) -> Date? {
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: raw) { return date }
        return ISO8601DateFormatter().date(from: raw)
    }

    /// "Today 21:33", "Yesterday 09:10", "Sep 5, 21:33" or "Sep 5, 2025,
    /// 21:33" in the user's locale and time zone; falls back to the raw
    /// stamp when it cannot be parsed so a malformed record still renders.
    static func formatCompletedAt(
        _ raw: String,
        now: Date = Date(),
        calendar: Calendar = .current,
        locale: Locale = .current,
        timeZone: TimeZone = .current
    ) -> String {
        guard let date = parseTimestamp(raw) else { return raw }
        var calendar = calendar
        calendar.timeZone = timeZone
        calendar.locale = locale
        let time = DateFormatter()
        time.locale = locale
        time.timeZone = timeZone
        time.setLocalizedDateFormatFromTemplate("jm")
        // "Today"/"Yesterday" already have catalog entries (zh-Hans), so the
        // relative day follows the same locale as the rest of the row.
        if calendar.isDate(date, inSameDayAs: now) {
            let today = String(
                localized: String.LocalizationValue("Today"), locale: locale
            )
            return "\(today) \(time.string(from: date))"
        }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: now),
           calendar.isDate(date, inSameDayAs: yesterday) {
            let label = String(
                localized: String.LocalizationValue("Yesterday"), locale: locale
            )
            return "\(label) \(time.string(from: date))"
        }
        let day = DateFormatter()
        day.locale = locale
        day.timeZone = timeZone
        let sameYear = calendar.component(.year, from: date)
            == calendar.component(.year, from: now)
        day.setLocalizedDateFormatFromTemplate(sameYear ? "MMMd" : "yMMMd")
        return "\(day.string(from: date)), \(time.string(from: date))"
    }
}

/// Monotonic stamp for stderr progress lines, taken on the reader thread in
/// arrival order so the main actor can discard out-of-order deliveries.
final class ProgressSequencer: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    func next() -> Int {
        lock.lock()
        defer { lock.unlock() }
        value += 1
        return value
    }
}

/// Copy shown next to the spinner while `benchmark run` is measuring, so the
/// user knows what is being measured, how much work that is, and roughly how
/// long to expect before the result row appears.
enum CommunityBenchmarkRunStatus {
    /// `Measuring qwen3.5-9b-4bit · 2 cases × (1 warmup + 5 rounds) · usually 2–5 minutes`
    static func description(for model: CommunityBenchmarkModel) -> String {
        var parts = ["Measuring \(model.entry.alias)", scope(for: model.task)]
        parts.append(expectedDuration(for: model.task))
        if !model.entry.cached { parts.append("plus the download") }
        return parts.joined(separator: " · ")
    }

    static func scope(for task: ModelTask) -> String {
        switch task {
        case .imageGeneration: return "1 warmup + 1 measured render"
        case .videoGeneration: return "1 measured render"
        default: return "2 cases × (1 warmup + 5 rounds)"
        }
    }

    static func expectedDuration(for task: ModelTask) -> String {
        switch task {
        // Image time is dominated by the model: a small SD-class model lands
        // in a couple of minutes, a flux-class one can take ten. Keep the
        // up-front hint wide and honest; the live ETA below carries accuracy.
        case .imageGeneration: return "usually 2–10 minutes"
        case .videoGeneration: return "usually 5–15 minutes"
        default: return "usually 2–5 minutes"
        }
    }

    /// Record-separator prefix the CLI puts on machine-readable progress
    /// lines under `--json --progress`. Mirrors `PROGRESS_TAG` in
    /// `rapid_mlx/community_bench/cli.py`.
    static let progressTag = "\u{1e}"

    /// A tagged progress line with its marker removed and whitespace
    /// collapsed, or nil for any other stderr (the untagged failure
    /// document, warnings, tracebacks) so the view never mirrors it.
    ///
    /// The 200-character cap is a *display* bound: this string is rendered
    /// verbatim in the status row. Structured events are not displayed and are
    /// routinely longer than a sentence, so they go through
    /// ``strippedEvent(from:)`` instead — a plan event for a three-case
    /// protocol is well past 200 bytes, and silently dropping it is how the
    /// denominator stayed hardcoded.
    static func strippedProgress(from line: String) -> String? {
        guard let body = taggedBody(of: line) else { return nil }
        guard !body.hasPrefix("{"), body.count <= 200 else { return nil }
        return body.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// A tagged **structured** event body — compact JSON — or nil.
    ///
    /// Prose never begins with `{`, so the two streams are separated on the
    /// first character. `LineSplitter` already discards anything over 4 KB, so
    /// this bound only guards against a pathological single event.
    static func strippedEvent(from line: String) -> String? {
        guard let body = taggedBody(of: line) else { return nil }
        guard body.hasPrefix("{"), body.utf8.count <= 4 * 1_024 else { return nil }
        return body
    }

    private static func taggedBody(of line: String) -> String? {
        guard line.hasPrefix(progressTag) else { return nil }
        let body = line.dropFirst(progressTag.count)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return body.isEmpty ? nil : body
    }

    /// True when a (stripped) progress line marks one COMPLETED unit of work.
    /// Completion lines are always `<case-id> warmup …` or
    /// `<case-id> round N/M …`, i.e. the phase is the SECOND token. Matching
    /// the token position (not a bare "warmup"/"round" substring) is what
    /// keeps plan/status lines — "Benchmarking … 1 warmup + 5 measured
    /// rounds", "Estimated time remaining … from the warmup rate" — from
    /// wrongly advancing the bar.
    static func isStepLine(_ stripped: String) -> Bool {
        let tokens = stripped.split(separator: " ")
        guard tokens.count >= 2 else { return false }
        if tokens[1] == "warmup" { return true }
        return tokens[1] == "round" && tokens.count >= 3
            && tokens[2].range(of: #"^\d+/\d+$"#, options: .regularExpression) != nil
    }

    /// Total warmup + measured passes for a task, i.e. how many step lines to
    /// expect. Returns nil for shapes too coarse for a determinate bar (a
    /// single measured pass), where the spinner + elapsed clock read better.
    static func totalSteps(for task: ModelTask) -> Int? {
        switch task {
        case .imageGeneration: return 2   // 1 warmup + 1 measured
        case .videoGeneration: return nil // 1 measured render — no useful bar
        default: return 12                // 2 buckets × (1 warmup + 5 measured)
        }
    }

    /// A `~m:ss left` estimate. The average step time comes only from
    /// COMPLETION timestamps — the span `lastStepAt - firstStepAt` over
    /// `stepsDone - 1` intervals — so it is stable between steps (dividing by
    /// intervals, not steps, excludes the one-off model-load + first-step
    /// time). `now` is used only to count the projection DOWN as time passes,
    /// never to inflate the per-step average. Text estimates wait for two
    /// completions. A two-step image run uses its warmup duration from run
    /// start; otherwise its ETA would first become available at completion.
    static func eta(
        stepsDone: Int,
        totalSteps: Int,
        runStartedAt: Date,
        firstStepAt: Date,
        lastStepAt: Date,
        now: Date
    ) -> String? {
        guard stepsDone >= 1, stepsDone < totalSteps else { return nil }
        let perStep: TimeInterval
        if stepsDone == 1, totalSteps == 2 {
            perStep = max(0, lastStepAt.timeIntervalSince(runStartedAt))
        } else {
            guard stepsDone >= 2 else { return nil }
            perStep = max(0, lastStepAt.timeIntervalSince(firstStepAt))
                / Double(stepsDone - 1)
        }
        let projected = perStep * Double(totalSteps - stepsDone)
        let remaining = projected - max(0, now.timeIntervalSince(lastStepAt))
        // Past the projection with no new step: don't sit on a stale
        // "~0:00 left" — say we're finishing the last pass(es).
        guard remaining > 0 else { return "wrapping up…" }
        let secs = Int(remaining.rounded())
        return String(format: "~%d:%02d left", secs / 60, secs % 60)
    }

    /// `m:ss` elapsed clock, clamped at zero so a clock adjustment mid-run
    /// cannot render a negative time.
    static func elapsed(from start: Date, to now: Date) -> String {
        let seconds = Int(max(0, now.timeIntervalSince(start)))
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }

    /// Picks per-round progress lines of the form
    /// `pp512-tg128  round 3/5  46.1 tok/s` out of everything else on the
    /// CLI's stderr (warnings, tracebacks, download logs). The shipped CLI
    /// does not emit them yet — the runner is silent until the final JSON —
    /// so today the row simply stays hidden and the status line + elapsed
    /// clock carry the feedback; a CLI that starts emitting them lights the
    /// row up without a Desktop change. Returns nil for anything that is not
    /// a progress line so the view never mirrors arbitrary stderr, and
    /// collapses whitespace so the row renders on one line.
    static func progressLine(from line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= 200,
              trimmed.range(of: #"\bround \d+/\d+\b"#, options: .regularExpression) != nil
        else { return nil }
        return trimmed.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}

final class BenchmarkProcessBox: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var child: ProcessGroupChild?

    func start(
        binary: URL,
        arguments: [String],
        standardOutput: Pipe,
        standardError: Pipe
    ) throws -> ProcessGroupChild {
        lock.lock()
        defer { lock.unlock() }
        if cancelled { throw CancellationError() }
        let spawned = try ProcessGroupChild.spawn(
            executableURL: binary,
            arguments: arguments,
            standardInput: .nullDevice,
            standardOutput: standardOutput,
            standardError: standardError
        )
        child = spawned
        return spawned
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let runningChild = child
        lock.unlock()
        // Wake the detached waiter immediately. It owns TERM/KILL escalation
        // and final liveness confirmation, so this callback never blocks
        // AppKit while the server reservation remains held by `startRun`.
        runningChild?.signalProcessGroup(SIGTERM)
    }

    func waitForCompletion(_ child: ProcessGroupChild) -> pid_t? {
        defer { clearTrackedChild(child) }
        while child.isRunning {
            lock.lock()
            let shouldCancel = cancelled
            lock.unlock()
            if shouldCancel { return Self.terminateAndReap(child) }
            // Also drives ProcessGroupChild's non-blocking waitpid fallback
            // when its dispatch exit source is delayed on a saturated host.
            _ = child.isProcessGroupAlive
            Thread.sleep(forTimeInterval: 0.01)
        }
        // A crashed/cancelled CLI can exit before one of its serve descendants.
        // Never return control (and release the memory reservation) with any
        // member of the benchmark process group still alive.
        if child.isProcessGroupAlive {
            return Self.terminateAndReap(child)
        }
        return nil
    }

    private func clearTrackedChild(_ completedChild: ProcessGroupChild) {
        lock.lock()
        if child === completedChild { child = nil }
        lock.unlock()
    }

    internal var _testHasTrackedChild: Bool {
        lock.lock()
        defer { lock.unlock() }
        return child != nil
    }

    private static func terminateAndReap(_ child: ProcessGroupChild) -> pid_t? {
        let exited = boundedTermination(
            isAlive: { child.isProcessGroupAlive },
            signal: { child.signalProcessGroup($0) },
            termGrace: 2,
            killGrace: 1
        )
        return exited ? nil : child.processGroupID
    }

    static func boundedTermination(
        isAlive: () -> Bool,
        signal: (Int32) -> Void,
        termGrace: TimeInterval,
        killGrace: TimeInterval,
        now: () -> Date = Date.init,
        sleep: (TimeInterval) -> Void = Thread.sleep(forTimeInterval:)
    ) -> Bool {
        signal(SIGTERM)
        let termDeadline = now().addingTimeInterval(termGrace)
        while isAlive(), now() < termDeadline { sleep(0.01) }
        if isAlive() {
            signal(SIGKILL)
            let killDeadline = now().addingTimeInterval(killGrace)
            while isAlive(), now() < killDeadline { sleep(0.01) }
        }
        return !isAlive()
    }
}

enum CommunityBenchmarkCommand {
    struct Failure: LocalizedError {
        let message: String
        /// True when the CLI declined on purpose rather than failing.
        ///
        /// Carried on the error itself because the message the user sees is
        /// the extracted `error` sentence — by the time a caller has that
        /// string the surrounding document, and its `refused` flag, are gone.
        var isRefusal: Bool = false
        var errorDescription: String? { message }
    }

    private enum RunOutcome {
        case output(Data)
        case deferredReap(pid_t)
    }

    private struct PipeCapture: Sendable {
        let data: Data
        let truncated: Bool
    }

    private static let maxStdoutBytes = 8 * 1_024 * 1_024
    private static let maxStderrBytes = 256 * 1_024
    private static let pipeChunkBytes = 64 * 1_024

    static func benchmarkRunArguments(alias: String) -> [String] {
        [
            "benchmark", "run", alias, "--json",
            // Stream RS-tagged live progress so the run shows a determinate
            // bar + ETA; the untagged failure document stays clean.
            "--progress",
            "--inherit-process-group",
        ]
    }

    static func benchmarkResultsArguments(limit: Int = 8) -> [String] {
        ["benchmark", "results", "--limit", String(limit), "--json"]
    }

    /// The `run_id` from a `benchmark run --json` payload, so the caller can
    /// act on the exact run that finished rather than inferring it from list
    /// order. nil for any payload without one (e.g. a deferred-reap result).
    static func runID(from data: Data) -> String? {
        struct RunID: Decodable {
            let runID: String
            enum CodingKeys: String, CodingKey { case runID = "run_id" }
        }
        return try? JSONDecoder().decode(RunID.self, from: data).runID
    }

    /// True when the CLI refused to publish rather than failing to.
    ///
    /// Read from the raw failure document's `refused` flag, not from the
    /// message text: the wording is user-facing and will change, the flag is
    /// a contract.
    static func isRefusal(_ detail: String) -> Bool {
        struct Doc: Decodable { let refused: Bool? }
        let candidates = [detail]
            + detail.split(separator: "\n").reversed().map(String.init)
        for candidate in candidates {
            if let data = candidate.data(using: .utf8),
               let doc = try? JSONDecoder().decode(Doc.self, from: data) {
                return doc.refused == true
            }
        }
        return false
    }

    /// A human sentence for a failed run. Under `--json` the CLI prints a
    /// `{"error": …, "run": {…}}` failure document; surface just its `error`
    /// so the user sees one clear line instead of a wall of raw JSON. Falls
    /// back to the raw text for any non-JSON failure (a crash, a traceback).
    static func failureSummary(from detail: String) -> String {
        struct Doc: Decodable { let error: String }
        // The document is one JSON line, but tracebacks/warnings may precede
        // it — try the whole string, then each line newest-first.
        let candidates = [detail]
            + detail.split(separator: "\n").reversed().map(String.init)
        for candidate in candidates {
            if let data = candidate.data(using: .utf8),
               let doc = try? JSONDecoder().decode(Doc.self, from: data),
               !doc.error.isEmpty {
                return doc.error
            }
        }
        return detail
    }

    static func benchmarkSharePreviewArguments(runID: String) -> [String] {
        ["benchmark", "share", runID, "--preview", "--json"]
    }

    static func benchmarkShareArguments(
        runID: String, installID: String, payloadDigest: String,
        bodyDigest: String, target: String
    ) -> [String] {
        [
            "benchmark", "share", runID, "--yes", "--install-id", installID,
            "--payload-digest", payloadDigest, "--body-digest", bodyDigest,
            "--target", target, "--json",
        ]
    }

    static func decodeSharePreview(_ data: Data, runID: String) throws
        -> CommunityBenchmarkUploadPreview
    {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let target = root["target"] as? String,
              let installID = root["install_id"] as? String,
              let payloadDigest = root["payload_digest"] as? String,
              let bodyDigest = root["body_digest"] as? String,
              let payloadJSON = root["payload_json"] as? String
        else {
            throw Failure(message: "The benchmark preview was incomplete.")
        }
        return CommunityBenchmarkUploadPreview(
            runID: runID,
            target: target,
            installID: installID,
            payloadDigest: payloadDigest,
            bodyDigest: bodyDigest,
            payloadJSON: payloadJSON,
            withheld: decodeWithheld(root["withheld"]),
            publishedIdentity: decodePublishedIdentity(from: payloadJSON)
        )
    }

    /// `withheld` from `benchmark share --preview --json`.
    ///
    /// A preview from an older CLI has no such key, which is not an error —
    /// that CLI does not project, so nothing was withheld.
    static func decodeWithheld(
        _ raw: Any?
    ) -> [CommunityBenchmarkUploadPreview.WithheldFact] {
        // Element-wise, not `as? [[String: Any]]`: one malformed entry must
        // not discard the disclosure for all the others.
        guard let items = raw as? [Any] else { return [] }
        return items.compactMap { element in
            guard let item = element as? [String: Any],
                  let path = item["path"] as? String,
                  let reason = item["reason"] as? String
            else { return nil }
            return CommunityBenchmarkUploadPreview.WithheldFact(
                path: path,
                value: describeWithheldValue(item["value"]),
                reason: reason
            )
        }
    }

    /// Renders a withheld value for display.
    ///
    /// The values are heterogeneous — a revision string, a whole quantization
    /// object — and the point is for the user to recognise what is being held
    /// back, so an object is shown as its `key: value` pairs rather than as
    /// raw JSON braces.
    static func describeWithheldValue(_ raw: Any?) -> String {
        switch raw {
        case let text as String: return text
        case let number as NSNumber: return number.stringValue
        case let object as [String: Any]:
            return object.keys.sorted()
                .map { "\($0): \(describeWithheldValue(object[$0]))" }
                .joined(separator: ", ")
        case let list as [Any]:
            return list.map(describeWithheldValue).joined(separator: ", ")
        case is NSNull, nil: return "—"
        default: return String(describing: raw ?? "")
        }
    }

    /// The model identity on the wire, so the disclosure can describe what is
    /// actually published instead of what the archive happens to hold.
    static func decodePublishedIdentity(from payloadJSON: String) -> CommunityModelIdentity? {
        guard let data = payloadJSON.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"],
              let modelData = try? JSONSerialization.data(withJSONObject: model),
              let wire = try? JSONDecoder().decode(
                  CommunityModelIdentity.Wire.self, from: modelData
              )
        else { return nil }
        return wire.identity
    }

    @MainActor
    static func run(
        binary: URL,
        arguments: [String],
        onDeferredReap: ((pid_t) -> Void)? = nil,
        onStandardErrorLine: (@Sendable (String) -> Void)? = nil
    ) async throws -> Data {
        let box = BenchmarkProcessBox()
        return try await withTaskCancellationHandler {
            let outcome: RunOutcome
            do {
                outcome = try await Task.detached(priority: .userInitiated) {
                    let stdout = Pipe()
                    let stderr = Pipe()
                    let child = try box.start(
                        binary: binary,
                        arguments: arguments,
                        standardOutput: stdout,
                        standardError: stderr
                    )
                    // On dedicated threads, NOT the Swift cooperative pool.
                    // These block in `read(upToCount:)` until the child writes,
                    // and `waitForCompletion` below blocks another thread for
                    // the whole run in a 10 ms sleep loop. Hosting all three on
                    // the cooperative pool starved the readers: every progress
                    // line for a 34-second benchmark arrived in one burst when
                    // the waiter finally returned, so the Running screen sat on
                    // "Getting ready" and then jumped straight to Result.
                    let outputTask = Task {
                        await runBlocking {
                            readBoundedPipe(
                                stdout.fileHandleForReading,
                                maxBytes: maxStdoutBytes,
                                retainTail: false
                            )
                        }
                    }
                    let errorTask = Task {
                        await runBlocking {
                            let lines = LineSplitter(onLine: onStandardErrorLine)
                            let capture = readBoundedPipe(
                                stderr.fileHandleForReading,
                                maxBytes: maxStderrBytes,
                                retainTail: true,
                                onChunk: lines.consume
                            )
                            lines.finish()
                            return capture
                        }
                    }
                    // The child owns duplicated write descriptors after spawn.
                    // Drop the parent's copies so both readers observe EOF when
                    // the process group exits.
                    try? stdout.fileHandleForWriting.close()
                    try? stderr.fileHandleForWriting.close()
                    defer {
                        // A read error on either stream must not strand the
                        // sibling detached reader or its descriptor. Closing
                        // first wakes a blocking read; cancellation then
                        // prevents any remaining detached work from escaping
                        // this command invocation.
                        try? stdout.fileHandleForReading.close()
                        try? stderr.fileHandleForReading.close()
                        outputTask.cancel()
                        errorTask.cancel()
                    }
                    // Also off the cooperative pool: this spins on
                    // `Thread.sleep` until the child exits.
                    let waited = await runBlocking { box.waitForCompletion(child) }
                    if let processGroupID = waited {
                        return RunOutcome.deferredReap(processGroupID)
                    }
                    let output = await outputTask.value
                    let errorCapture = await errorTask.value
                    guard child.terminationStatus == 0 else {
                        // Drop RS-tagged live-progress lines so the failure
                        // document the user sees stays clean.
                        let detail = String(data: errorCapture.data, encoding: .utf8)?
                            .split(separator: "\n", omittingEmptySubsequences: false)
                            .filter {
                                !$0.hasPrefix(CommunityBenchmarkRunStatus.progressTag)
                            }
                            .joined(separator: "\n")
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        let message = detail
                            .flatMap { $0.isEmpty ? nil : Self.failureSummary(from: $0) }
                            ?? "Benchmark exited with code \(child.terminationStatus)."
                        throw Failure(
                            message: message,
                            isRefusal: detail.map(Self.isRefusal) ?? false
                        )
                    }
                    guard !output.truncated else {
                        throw Failure(
                            message: "Benchmark output exceeded the 8 MiB safety limit."
                        )
                    }
                    return RunOutcome.output(output.data)
                }.value
            } catch {
                try Task.checkCancellation()
                throw error
            }
            guard case let .output(data) = outcome else {
                if case let .deferredReap(processGroupID) = outcome {
                    if let onDeferredReap {
                        onDeferredReap(processGroupID)
                    } else {
                        ProcessGroupChild.reapProcessGroupInBackground(
                            processGroupID: processGroupID
                        )
                    }
                }
                throw CancellationError()
            }
            try Task.checkCancellation()
            return data
        } onCancel: {
            box.cancel()
        }
    }

    /// Re-assembles newline-delimited text out of arbitrary pipe chunks and
    /// hands each complete line to the observer. Purely advisory: the bounded
    /// capture above is still what error messages are built from, and a
    /// partial line longer than `maxLineBytes` is dropped rather than buffered
    /// without limit, so a chatty or malformed stream cannot grow memory.
    final class LineSplitter: @unchecked Sendable {
        private let onLine: (@Sendable (String) -> Void)?
        private var pending = Data()
        private let maxLineBytes = 4 * 1_024

        init(onLine: (@Sendable (String) -> Void)?) {
            self.onLine = onLine
        }

        func consume(_ chunk: Data) {
            guard let onLine else { return }
            var rest = chunk[...]
            while let newline = rest.firstIndex(of: UInt8(ascii: "\n")) {
                append(rest[rest.startIndex..<newline])
                if pending.count <= maxLineBytes,
                   let line = String(data: pending, encoding: .utf8) {
                    onLine(line)
                }
                pending.removeAll(keepingCapacity: true)
                rest = rest[rest.index(after: newline)...]
            }
            append(rest)
        }

        /// Never buffers more than `maxLineBytes`: an oversized line is
        /// poisoned (one byte past the cap) without copying its payload, so
        /// its eventual tail is dropped too instead of being reported as a
        /// fresh, truncated line.
        private func append(_ slice: Data.SubSequence) {
            if pending.count + slice.count <= maxLineBytes {
                pending.append(slice)
            } else if pending.count <= maxLineBytes {
                pending = Data(count: maxLineBytes + 1)
            }
        }

        internal var _testPendingBytes: Int { pending.count }

        /// EOF: a final line without a trailing newline is still a line.
        func finish() {
            guard let onLine, !pending.isEmpty else { return }
            if pending.count <= maxLineBytes,
               let line = String(data: pending, encoding: .utf8) {
                onLine(line)
            }
            pending.removeAll(keepingCapacity: false)
        }
    }

    /// Runs blocking work on a dedicated thread, off the Swift cooperative
    /// pool.
    ///
    /// The pool is sized to the core count and is not allowed to block: a task
    /// that sits in `read()` or `Thread.sleep` holds a thread that other tasks
    /// need to make progress on. Pipe draining and process waiting are both
    /// unavoidably blocking, so they get real threads and hand their result
    /// back through a continuation.
    private static func runBlocking<Value: Sendable>(
        _ work: @escaping @Sendable () -> Value
    ) async -> Value {
        await withCheckedContinuation { continuation in
            let thread = Thread { continuation.resume(returning: work()) }
            thread.name = "rapid.benchmark.blocking"
            thread.stackSize = 512 * 1024
            thread.start()
        }
    }

    private static func readBoundedPipe(
        _ handle: FileHandle,
        maxBytes: Int,
        retainTail: Bool,
        onChunk: ((Data) -> Void)? = nil
    ) -> PipeCapture {
        var data = Data()
        var truncated = false
        var buffer = [UInt8](repeating: 0, count: pipeChunkBytes)
        let descriptor = handle.fileDescriptor
        while true {
            // `FileHandle.read(upToCount:)` is NOT a streaming read: it loops
            // internally until it has the full count or hits EOF, so asking for
            // a 64 KB chunk of a slow trickle returns nothing until the child
            // exits. That is what froze the Running screen on "Getting ready"
            // for an entire 34-second benchmark and then delivered every
            // progress line at once, too late to render.
            //
            // `read(2)` returns as soon as any bytes are available, which is
            // what a live progress stream needs.
            let count = buffer.withUnsafeMutableBytes { raw in
                read(descriptor, raw.baseAddress, raw.count)
            }
            if count < 0 {
                // Retry a signal-interrupted read; anything else ends the stream.
                if errno == EINTR { continue }
                break
            }
            guard count > 0 else { break }
            let chunk = Data(buffer[0..<count])
            onChunk?(chunk)
            if chunk.count >= maxBytes {
                truncated = truncated || !data.isEmpty || chunk.count > maxBytes
                data = retainTail ? Data(chunk.suffix(maxBytes)) : Data(chunk.prefix(maxBytes))
                continue
            }
            let overflow = data.count + chunk.count - maxBytes
            if overflow > 0 {
                truncated = true
                if retainTail {
                    data.removeFirst(overflow)
                    data.append(chunk)
                } else {
                    data.append(chunk.prefix(maxBytes - data.count))
                }
            } else if retainTail || data.count < maxBytes {
                data.append(chunk)
            }
        }
        return PipeCapture(data: data, truncated: truncated)
    }

    internal static func _testReadBoundedPipe(
        _ handle: FileHandle,
        maxBytes: Int,
        retainTail: Bool
    ) -> (data: Data, truncated: Bool) {
        let capture = readBoundedPipe(
            handle, maxBytes: maxBytes, retainTail: retainTail
        )
        return (capture.data, capture.truncated)
    }
}

/// The Community Benchmark workspace: Run, My Results, and Community.
///
/// Owns the module's state and wires the three tabs, the Ready → Running →
/// Result phases, and the four sheets (model picker, test method, publish
/// confirmation, published). The measurement pipeline underneath — process
/// spawn, progress parsing, result decoding, publish payload — is unchanged.
struct CommunityBenchmarkView: View {
    let catalog: [ModelEntry]
    let binary: URL?
    let prepareServer: () async throws -> UUID
    let releaseServer: (UUID) -> Void
    let retainServerDuringDeferredReap: (pid_t) -> Void
    /// The community read API. Defaults to the unavailable directory because
    /// this repository contains no read endpoint; tests and previews inject a
    /// static one. See ``CommunityBenchmarkDirectory``.
    var directory: any CommunityBenchmarkDirectory = UnavailableCommunityBenchmarkDirectory()
    var macProfile: CommunityMacProfile = .current()

    // MARK: Run pipeline state (unchanged behaviour)

    @State private var selectedAlias = ""
    @State private var results: [CommunityBenchmarkResult] = []
    @State private var isRunning = false
    @State private var runStartedAt: Date?
    @State private var runningModel: CommunityBenchmarkModel?
    @State private var currentRunID: UUID?
    @State private var appliedProgressSequence = 0
    /// Live state reduced from the CLI's RS-tagged progress stream: stage,
    /// completed passes, newest measurement, time left. Replaces the two bare
    /// scalars the screen used to derive everything from.
    @State private var runProgress = CommunityRunProgress()
    /// The protocol shape the stepper renders. The assumed one until the run
    /// declares its own in its first `plan` event.
    @State private var runProgressPlan = CommunityRunPlan.assumed(for: .textGeneration)
    @State private var errorMessage: String?
    @State private var errorTone: InlineNotice.Tone = .error
    @State private var latestResultID: String?
    @State private var runTask: Task<Void, Never>?
    @State private var shareTask: Task<Void, Never>?
    @State private var shareCandidate: CommunityBenchmarkUploadPreview?
    @State private var sharingRunID: String?
    @State private var shareSuccess: CommunityBenchmarkReceipt?
    @State private var receiptNotSavedWarning: String?
    /// Identity + confirmed-count bookkeeping for this session's uploads.
    /// Survives a failed local receipt write and defends the confirmed count
    /// against the public feed's 30-second edge cache.
    @State private var publication = CommunityPublicationState()
    /// Scheduled re-read once the edge cache can have expired.
    @State private var staleFeedRetryTask: Task<Void, Never>?
    @State private var receipts: [String: CommunityBenchmarkReceipt] = [:]
    @State private var benchmarkMetadata: [String: CommunityBenchmarkCatalogModel] = [:]
    @State private var benchmarkCLIAvailable = false
    @State private var productCatalog: [ModelEntry]?

    // MARK: Redesign state

    @State private var tab: CommunityBenchmarkTab = .run
    @State private var showsPicker = false
    @State private var showsTestMethod = false
    @State private var pickerQuery = ""
    @State private var pickerSelection = ""
    @State private var containerSize: CGSize = .init(width: 1_240, height: 820)
    @State private var communityWorkload: CommunityWorkload = .llm

    /// Observation aggregate for the currently selected model's scope.
    @State private var observations: CommunityDataState<CommunityObservationSummary> = .loading
    @State private var communityTable: CommunityDataState<[CommunityObservationRow]> = .loading
    @State private var coverage: CommunityDataState<[CommunityCoverageGap]> = .loading
    @State private var pulse: CommunityDataState<CommunityPulse> = .loading
    /// Everything about the run being published, frozen when Publish was
    /// pressed: its scope, the count that scope had, and the branch the user
    /// saw. Nothing here is re-read after the upload starts, so a Run again
    /// mid-upload cannot redirect the receipt onto another run's scope.
    @State private var publishContext: CommunityPublicationContext?
    /// The post-publish count for the *published* scope, used by the
    /// celebration sheet. Held separately from `observations`, which tracks
    /// whatever is on screen now.
    @State private var publishedObservationCount: Int?
    /// Stamps every community read so a slow response for a model or workload
    /// the user has since changed cannot overwrite the current one. Per-query,
    /// so changing the model never orphans the pulse or coverage requests.
    @State private var readGenerations = CommunityRequestGenerations()
    /// Exact published totals for this installation's own pseudonym, read from
    /// the paginated contributions endpoint.
    @State private var contributorTotals: CommunityDataState<CommunityContributorTotals> = .loading

    private var resolvedCatalog: [ModelEntry] {
        CommunityBenchmarkModel.resolvedCatalog(product: productCatalog, fallback: catalog)
    }

    private var models: [CommunityBenchmarkModel] {
        CommunityBenchmarkModel.models(from: resolvedCatalog, metadata: benchmarkMetadata)
    }

    private var selected: CommunityBenchmarkModel? {
        models.first { $0.entry.alias == selectedAlias }
    }

    /// Coverage scope for the selected model: no comparison identity, because
    /// no run exists yet.
    private var scope: CommunityBenchmarkScope? {
        CommunityBenchmarkView.observationScope(
            selected: selected, macProfile: macProfile, latestResult: nil
        )
    }

    /// Comparison scope for a finished run, built **entirely from that run**.
    ///
    /// Nothing here reads `selectedAlias`, `selected`, or this Mac's current
    /// hardware: publishing a stored result from My Results is a legitimate
    /// action on a run whose model, protocol version and Mac may all differ
    /// from whatever the Run tab is showing.
    private func comparisonScope(for result: CommunityBenchmarkResult) -> CommunityBenchmarkScope? {
        result.communityScope(alias: alias(for:))
    }

    /// The scope the observations query must actually use.
    ///
    /// When a completed run is on screen the question is "how does THIS run
    /// compare?", which is only answerable against the cell produced with the
    /// same case, metric and execution configuration. Querying the generic
    /// coverage scope there returned a count across every execution variant
    /// and no median, so the Result screen could never show a comparison.
    private var activeObservationScope: CommunityBenchmarkScope? {
        CommunityBenchmarkView.observationScope(
            selected: selected,
            macProfile: macProfile,
            latestResult: latestResult,
            alias: alias(for:)
        )
    }

    /// The scope-selection rule, as a free function so the refresh path's
    /// behaviour can be asserted directly instead of being re-implemented in a
    /// test double.
    ///
    /// Ready (no completed run) asks a *coverage* question and gets the generic
    /// scope. As soon as a completed run is on screen the question becomes a
    /// *comparison* and the scope must carry that run's exact case, metric and
    /// execution configuration.
    static func observationScope(
        selected: CommunityBenchmarkModel?,
        macProfile: CommunityMacProfile,
        latestResult: CommunityBenchmarkResult?,
        alias: (String) -> String = { $0 }
    ) -> CommunityBenchmarkScope? {
        // A displayed run answers for itself — model identity, protocol,
        // machine, case, metric and execution all come from the record. The
        // selected model is irrelevant to a result that already exists.
        if let latestResult, let scope = latestResult.communityScope(alias: alias) {
            return scope
        }
        guard let selected else { return nil }
        // Coverage question about a catalogue entry: no run has been measured,
        // so no revision or quantization has been resolved and there is no
        // identity beyond the repo to claim.
        return CommunityBenchmarkScope(
            modelAlias: selected.entry.alias,
            workload: CommunityWorkload(task: selected.task),
            protocolID: selected.protocolID,
            protocolVersion: selected.protocolVersion,
            macProfile: macProfile
        )
    }

    private var branch: CommunityContributionBranch {
        CommunityContributionBranch.select(from: observations)
    }

    /// The freshest completed run for the selected model, which is what the
    /// Run tab shows after a measurement finishes.
    private var latestResult: CommunityBenchmarkResult? {
        guard let latestResultID else { return nil }
        return results.first { $0.id == latestResultID }
    }

    private var isNarrow: Bool { containerSize.width < 880 }

    var body: some View {
        GeometryReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                    header
                    tabBar
                    if let errorMessage {
                        InlineNotice(
                            message: errorMessage,
                            tone: errorTone,
                            actionTitle: String(localized: "Dismiss"),
                            action: { self.errorMessage = nil }
                        )
                    }
                    content
                }
                .padding(isNarrow ? RapidTheme.Space.xl : 40)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(RapidTheme.surfaceCanvas)
            .onAppear { containerSize = proxy.size }
            .onChange(of: proxy.size) { _, newValue in containerSize = newValue }
        }
        .task {
            await refreshProductCatalog()
            if selectedAlias.isEmpty { selectedAlias = models.first?.entry.alias ?? "" }
            await refreshBenchmarkCatalog()
            await refreshResults()
            await refreshCommunity()
        }
        .onChange(of: activeObservationScope) { _, _ in
            // Bump synchronously whenever the actual query scope changes —
            // including Result → Ready transitions for the same model. That
            // makes the previous scope's in-flight answer stale before the
            // replacement starts, without invalidating unrelated reads.
            let token = readGenerations.begin(.observations)
            staleFeedRetryTask?.cancel()
            staleFeedRetryTask = nil
            // The floors stay. A confirmed publication into A's scope is still
            // true while B is on screen, and coming back to A with the feed
            // still cached must not walk A's number back down.
            Task { await refreshObservations(token: token) }
        }
        .onChange(of: communityWorkload) { _, _ in
            let token = readGenerations.begin(.table)
            Task { await refreshCommunityTable(token: token) }
        }
        .onDisappear {
            // `runTask` is intentionally unstructured so the button owns it;
            // navigation must explicitly cancel it before the only Stop
            // control disappears. ServerManager keeps the lease until the
            // subprocess tree has actually been reaped.
            runTask?.cancel()
            shareTask?.cancel()
        }
        .sheet(isPresented: $showsPicker) { pickerSheet }
        .sheet(isPresented: $showsTestMethod) { testMethodSheet }
        .sheet(item: $shareCandidate) { preview in
            CommunityBenchmarkShareConfirmationSheet(
                preview: preview,
                knownContributor: knownContributor,
                isPublishing: sharingRunID == preview.runID,
                onCancel: { shareCandidate = nil },
                onPublish: { share(preview) }
            )
        }
        .sheet(item: $shareSuccess) { receipt in
            // Every number and name here comes from the captured context, not
            // from current page state: the celebration describes the run that
            // was published, even if the user has since started another.
            CommunityBenchmarkPublishedSheet(
                celebration: CommunityBenchmarkCopy.publishedCelebration(
                    branchBeforePublishing: publishContext?.branch ?? branch,
                    scope: publishContext?.scope ?? scope ?? fallbackScope,
                    observationCountAfterPublishing: publishedObservationCount,
                    alreadyPublished: receipt.alreadyExists
                ),
                receipt: receipt,
                receiptNotSavedWarning: receiptNotSavedWarning,
                onDone: {
                    shareSuccess = nil
                    receiptNotSavedWarning = nil
                    publishContext = nil
                    publishedObservationCount = nil
                }
            )
        }
    }

    /// Used only when no model is selected (an empty catalogue), so the
    /// celebration sheet always has a scope to describe.
    private var fallbackScope: CommunityBenchmarkScope {
        CommunityBenchmarkScope(
            modelAlias: selectedAlias,
            workload: .llm,
            protocolID: "rapid-community-speed",
            protocolVersion: 2,
            macProfile: macProfile
        )
    }

    // MARK: - Chrome

    private var header: some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.lg) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                Text("Community Benchmark")
                    .font(RapidFont.pageTitle)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("Measure a model on this Mac. Keep the result private, or publish it to the Community Benchmark on rapidmlx.com.")
                    .font(RapidFont.body)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            // Quieted on the Community tab, where the page already ends with a
            // prominent leaderboard link; two competing external destinations
            // on one screen is what made the earlier layout ambiguous.
            Link(destination: communityBenchmarkLeaderboardURL) {
                HStack(spacing: 6) {
                    Text("Open rapidmlx.com")
                    Image(systemName: "arrow.up.right.square").font(.system(size: 11))
                }
                .font(RapidFont.body)
            }
            .buttonStyle(tab == .community ? AnyButtonStyle(.rapidLink) : AnyButtonStyle(.rapidSecondaryCompact))
            .accessibilityIdentifier("CommunityBenchmark.OpenWebsite")
        }
    }

    private var tabBar: some View {
        Picker(String(localized: "Section"), selection: $tab) {
            ForEach(CommunityBenchmarkTab.allCases) { candidate in
                Text(candidate.title).tag(candidate)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(width: 280)
        .accessibilityIdentifier("CommunityBenchmark.Tabs")
    }

    @ViewBuilder
    private var content: some View {
        switch tab {
        case .run: runTab
        case .myResults: myResultsTab
        case .community: communityTab
        }
    }

    // MARK: - Run

    @ViewBuilder
    private var runTab: some View {
        if isRunning, let runningModel, let runStartedAt {
            CommunityBenchmarkRunningView(
                model: runningModel,
                scope: CommunityBenchmarkScope(
                    modelAlias: runningModel.entry.alias,
                    workload: CommunityWorkload(task: runningModel.task),
                    protocolID: runningModel.protocolID,
                    protocolVersion: runningModel.protocolVersion,
                    macProfile: macProfile
                ),
                runStartedAt: runStartedAt,
                progress: runProgress,
                plan: runProgressPlan,
                isNarrow: isNarrow,
                onStop: stopRun
            )
        } else if let result = latestResult, let scope = comparisonScope(for: result) {
            CommunityBenchmarkResultView(
                result: result,
                modelAlias: alias(for: result.repoID),
                scope: scope,
                branch: branch,
                observations: observations,
                receipt: effectiveReceipts[result.id],
                isPublishing: sharingRunID == result.id,
                isNarrow: isNarrow,
                onPublish: { prepareShare(result) },
                // Repeats THIS run, not whatever the picker is on. The result
                // on screen may be a stored one for another model, and "Run
                // again" has to mean what it says.
                onRunAgain: { runAgain(result) },
                onBenchmarkAnother: {
                    latestResultID = nil
                    pickerSelection = selectedAlias
                    pickerQuery = ""
                    showsPicker = true
                }
            )
        } else if let selected, let scope {
            CommunityBenchmarkReadyView(
                model: selected,
                scope: scope,
                branch: branch,
                isRunEnabled: binary != nil && benchmarkCLIAvailable && selected.runtimeCanRun,
                serverImpactNote: serverImpactNote,
                isNarrow: isNarrow,
                onRun: startRun,
                onChangeModel: {
                    pickerSelection = selectedAlias
                    pickerQuery = ""
                    showsPicker = true
                },
                onShowTestMethod: { showsTestMethod = true }
            )
        } else {
            CommunityUnavailableBand(
                title: String(localized: "No benchmark models available"),
                message: String(localized: "Community Benchmark needs a current rapid-mlx runtime. Update or restart Rapid, then try again.")
            )
        }
    }

    private var serverImpactNote: String? {
        guard let selected else { return nil }
        return String(
            format: String(
                localized: "Chat and Images pause while %1$@ is measured, then your model reloads automatically."
            ),
            selected.entry.alias
        )
    }

    // MARK: - My Results

    private var myResultsTab: some View {
        CommunityBenchmarkMyResultsView(
            results: results,
            receipts: effectiveReceipts,
            aliasForRepo: alias(for:),
            workloadForResult: { result in
                CommunityWorkload(
                    task: ModelTask(rawValue: result.workload.taskType) ?? .textGeneration
                )
            },
            contributor: knownContributor,
            publishedTotals: contributorTotals,
            localOnlyCount: results.filter {
                effectiveReceipts[$0.id] == nil && $0.isCompleted
            }.count,
            sharingRunID: sharingRunID,
            onPublish: prepareShare,
            onRunFirstBenchmark: { tab = .run }
        )
    }

    private var effectiveReceipts: [String: CommunityBenchmarkReceipt] {
        receipts.merging(publication.sessionReceipts) { persisted, _ in persisted }
    }

    // MARK: - Community

    private var communityTab: some View {
        CommunityBenchmarkCommunityView(
            macProfile: macProfile,
            pulse: pulse,
            table: communityTable,
            coverage: coverage,
            workload: $communityWorkload,
            metric: .primary(for: communityWorkload),
            isNarrow: isNarrow,
            youContributor: knownContributor,
            leaderboardURL: communityBenchmarkLeaderboardURL,
            onRunModel: { alias in
                selectedAlias = alias
                latestResultID = nil
                tab = .run
            }
        )
    }

    // MARK: - Sheets

    private var pickerSheet: some View {
        CommunityBenchmarkPickerSheet(
            listing: CommunityBenchmarkPicker.listing(
                models: models,
                coverage: coverage,
                query: pickerQuery
            ),
            containerSize: containerSize,
            query: $pickerQuery,
            selectedAlias: $pickerSelection,
            onCancel: { showsPicker = false },
            onChoose: { alias in
                selectedAlias = alias
                latestResultID = nil
                showsPicker = false
            }
        )
    }

    @ViewBuilder
    private var testMethodSheet: some View {
        if let scope {
            CommunityBenchmarkTestMethodSheet(
                modelAlias: scope.modelAlias,
                workload: scope.workload,
                onDone: { showsTestMethod = false }
            )
        }
    }

    // MARK: - Community reads

    private func refreshCommunity() async {
        async let observationsTask: Void = refreshObservations()
        async let tableTask: Void = refreshCommunityTable()
        async let coverageTask: Void = refreshCoverage()
        async let pulseTask: Void = refreshPulse()
        async let totalsTask: Void = refreshContributorTotals()
        _ = await (observationsTask, tableTask, coverageTask, pulseTask, totalsTask)
    }

    /// Reads the aggregate for the selected model. `token` is the generation
    /// at request time; if the selection moved while the request was in
    /// flight, the answer describes a model that is no longer on screen and is
    /// dropped rather than rendered beside the new one.
    private func refreshObservations(token: Int? = nil) async {
        let token = token ?? readGenerations.begin(.observations)
        // The scope is captured with the token, so a late answer is matched
        // against the model that asked for it, not whatever is selected now.
        guard let requested = activeObservationScope else {
            if readGenerations.isCurrent(.observations, token) {
                observations = .unavailable(.notConfigured)
            }
            return
        }
        if readGenerations.isCurrent(.observations, token) { observations = .loading }
        // The pseudonym goes with the query so "includes yours" is answered
        // from the server's own contributor list. It then survives a restart,
        // a reinstall, and a failed local receipt write — none of which the
        // optimistic publication state outlives.
        let answer = await directory.observations(
            for: requested, viewerSlug: knownContributor?.slug
        )
        guard readGenerations.isCurrent(.observations, token),
              activeObservationScope == requested else { return }
        // Fold through the publication state so a stale cached feed cannot
        // walk the confirmed post-publish count backwards.
        observations = publication.merge(answer, scope: requested)
        scheduleStaleFeedRetryIfNeeded(scope: requested)
    }

    /// Re-reads once the public feed's edge cache can have expired, so the
    /// optimistic count is replaced by a real server number rather than
    /// persisting for the rest of the session.
    ///
    /// Every projection backed by `/atomic/public` is re-read, not just the
    /// count. One cached body feeds the observation aggregate, the Community
    /// table, the coverage list and the pulse band, so refreshing the count
    /// alone left the table row that should have gained "INCLUDES YOURS"
    /// showing pre-publish data until the user toggled a workload or relaunched.
    private func scheduleStaleFeedRetryIfNeeded(scope: CommunityBenchmarkScope) {
        guard publication.confirmedFloor(for: scope) != nil else {
            staleFeedRetryTask?.cancel()
            staleFeedRetryTask = nil
            return
        }
        guard staleFeedRetryTask == nil else { return }
        staleFeedRetryTask = Task {
            try? await Task.sleep(
                nanoseconds: UInt64(
                    CommunityPublicationState.publicFeedCacheSeconds * 1_000_000_000
                )
            )
            guard !Task.isCancelled else { return }
            staleFeedRetryTask = nil
            guard publication.mayRetryAnything(at: Date()) else { return }
            await refreshPublicFeedBackedReads()
        }
    }

    /// Re-reads exactly the queries `/api/benchmarks/atomic/public` serves.
    ///
    /// Driven by `CommunityPublicationState.publicFeedBackedReads` so the set
    /// is stated once, next to the cache duration it exists because of.
    private func refreshPublicFeedBackedReads() async {
        let kinds = CommunityPublicationState.publicFeedBackedReads
        async let observationsTask: Void = kinds.contains(.observations)
            ? refreshObservations() : ()
        async let tableTask: Void = kinds.contains(.table) ? refreshCommunityTable() : ()
        async let coverageTask: Void = kinds.contains(.coverage) ? refreshCoverage() : ()
        async let pulseTask: Void = kinds.contains(.pulse) ? refreshPulse() : ()
        _ = await (observationsTask, tableTask, coverageTask, pulseTask)
    }

    private func refreshCommunityTable(token: Int? = nil) async {
        let token = token ?? readGenerations.begin(.table)
        let workload = communityWorkload
        if readGenerations.isCurrent(.table, token) { communityTable = .loading }
        let result = await directory.table(
            macProfile: macProfile,
            workload: workload,
            metric: .primary(for: workload),
            viewerSlug: knownContributor?.slug
        )
        guard readGenerations.isCurrent(.table, token) else { return }
        communityTable = result
    }

    private func refreshCoverage() async {
        let token = readGenerations.begin(.coverage)
        coverage = .loading
        let result = await directory.coverageGaps(macProfile: macProfile)
        guard readGenerations.isCurrent(.coverage, token) else { return }
        coverage = result
    }

    private func refreshPulse() async {
        let token = readGenerations.begin(.pulse)
        let result = await directory.pulse()
        guard readGenerations.isCurrent(.pulse, token) else { return }
        pulse = result
    }

    /// The exact public contribution total for this installation's pseudonym.
    ///
    /// Read from the server rather than counted from `receipts`: a local
    /// receipt can be missing (the upload succeeded but the write failed) or
    /// lost (reinstall), and My Results must not under-report what is publicly
    /// attributed to this contributor.
    private func refreshContributorTotals() async {
        let token = readGenerations.begin(.contributorTotals)
        guard let slug = knownContributor?.slug,
              let api = directory as? CommunityBenchmarkAPIDirectory
        else {
            if readGenerations.isCurrent(.contributorTotals, token) {
                contributorTotals = .unavailable(.notConfigured)
            }
            return
        }
        contributorTotals = .loading
        let result = await api.contributions(forSlug: slug)
        guard readGenerations.isCurrent(.contributorTotals, token) else { return }
        contributorTotals = result
    }

    /// The pseudonym this installation publishes under, once the service has
    /// issued one. Never invented locally.
    private var knownContributor: CommunityBenchmarkContributor? {
        // The session identity first: an upload whose local receipt could not
        // be written still returned a server-issued pseudonym, and dropping it
        // would leave the session with no portrait, no profile link, and no
        // slug to ask for contributor totals with.
        publication.sessionContributor ?? receipts.values.compactMap(\.contributor).first
    }

    // MARK: - Run pipeline

    private func stopRun() {
        // Invalidate the run token first so stderr that arrives during
        // teardown cannot update the row.
        currentRunID = nil
        runTask?.cancel()
    }

    /// Repeats the model a displayed result was measured on.
    ///
    /// The result may be a stored one for a model the picker is not on, so the
    /// selection is moved to it first. `startRun` reads `selected`, and
    /// `selectedAlias` is `@AppStorage`-backed, so the alias is set and the
    /// run is started in the same actor step only when the model resolves —
    /// starting a run for the wrong model would be worse than not starting one.
    private func runAgain(_ result: CommunityBenchmarkResult) {
        latestResultID = nil
        let alias = alias(for: result.repoID)
        guard models.contains(where: { $0.entry.alias == alias }) else {
            // The catalogue no longer offers this model (removed, or renamed).
            // Say so rather than silently benchmarking something else.
            errorTone = .info
            errorMessage = String(
                format: String(localized: "%1$@ is no longer in the benchmark catalogue, so it can’t be run again. Choose another model."),
                alias
            )
            pickerSelection = selectedAlias
            pickerQuery = ""
            showsPicker = true
            return
        }
        selectedAlias = alias
        startRun(alias: alias)
    }

    private func startRun() { startRun(alias: nil) }

    /// `alias` pins the model when the caller already knows it, because
    /// `selectedAlias` was just written and `selected` is derived from it.
    private func startRun(alias pinned: String?) {
        let target = pinned.flatMap { alias in
            models.first { $0.entry.alias == alias }
        } ?? selected
        guard benchmarkCLIAvailable, let selected = target, let binary else { return }
        errorMessage = nil
        isRunning = true
        runningModel = selected
        runStartedAt = Date()
        let assumedPlan = CommunityRunPlan.assumed(for: selected.task)
        runProgressPlan = assumedPlan
        runProgress = CommunityRunProgress(totalPasses: assumedPlan.totalPasses)
        appliedProgressSequence = 0
        latestResultID = nil
        let activeRunID = UUID()
        currentRunID = activeRunID
        // Reduces the stderr stream off the main actor, in arrival order
        // (LineSplitter delivers lines sequentially), and hands the view a
        // value snapshot.
        let reducer = CommunityRunProgressBox(plan: assumedPlan)
        // One ordered channel instead of a Task per line. `Task { @MainActor }`
        // per line is unordered and fire-and-forget: a hop could be applied out
        // of order, or still be queued when the run task set `isRunning = false`
        // and the Result screen replaced Running — in which case the update was
        // simply lost. A stream is delivered in yield order by construction, and
        // the run task awaits its drain before transitioning.
        let (progressStream, progressFeed) = AsyncStream<CommunityRunProgress>
            .makeStream(bufferingPolicy: .unbounded)
        runTask = Task {
            var acquiredReservation = false
            // Applies every state in yield order, on the main actor, for as
            // long as the stream is open. A child task rather than a detached
            // one so cancellation propagates with the run.
            let delivery = Task { @MainActor in
                for await state in progressStream {
                    guard currentRunID == activeRunID else { continue }
                    runProgress = state
                    runProgressPlan = reducer.currentPlan
                }
            }
            do {
                let reservation = try await prepareServer()
                acquiredReservation = true
                defer { releaseServer(reservation) }
                try Task.checkCancellation()
                let runOutput = try await CommunityBenchmarkCommand.run(
                    binary: binary,
                    arguments: CommunityBenchmarkCommand.benchmarkRunArguments(
                        alias: selected.entry.alias
                    ),
                    onDeferredReap: retainServerDuringDeferredReap,
                    onStandardErrorLine: { line in
                        guard let state = reducer.apply(line: line, at: Date()) else {
                            return
                        }
                        progressFeed.yield(state)
                    }
                )
                // Everything the reader handed over is applied BEFORE the
                // Result screen replaces Running. `run` only returns once both
                // pipes have hit EOF, so no further lines can arrive; closing
                // the stream lets the consumer finish its backlog and exit.
                progressFeed.finish()
                await delivery.value
                await refreshProductCatalog()
                await refreshResults()
                // Show the run that just finished — only when the CLI payload
                // names it. A payload without a run_id (e.g. deferred reap)
                // must not surface an unrelated historical run as "your
                // result".
                latestResultID = CommunityBenchmarkCommand.runID(from: runOutput)
                // Only now does a comparison identity exist, so this re-query
                // is the exact one — same case, metric and execution as the
                // run just finished.
                await refreshObservations()
            } catch is CancellationError {
                errorTone = .info
                errorMessage = acquiredReservation
                    ? String(localized: "Benchmark stopped. Nothing was saved or published.")
                    : String(localized: "Benchmark request stopped before it started.")
            } catch {
                errorTone = .error
                errorMessage = error.localizedDescription
            }
            // Idempotent: already finished on the success path, and the only
            // thing that ends the consumer on a throw or a cancellation.
            progressFeed.finish()
            await delivery.value
            isRunning = false
            runningModel = nil
            runStartedAt = nil
            runProgress = CommunityRunProgress()
            runTask = nil
        }
    }

    private func prepareShare(_ result: CommunityBenchmarkResult) {
        guard let binary else { return }
        sharingRunID = result.id
        errorMessage = nil
        shareTask = Task {
            do {
                let data = try await CommunityBenchmarkCommand.run(
                    binary: binary,
                    arguments: CommunityBenchmarkCommand.benchmarkSharePreviewArguments(
                        runID: result.id
                    )
                )
                shareCandidate = try CommunityBenchmarkCommand.decodeSharePreview(
                    data, runID: result.id
                )
            } catch is CancellationError {
                // Navigation cancelled the preview command.
            } catch {
                // A refusal is a decision the CLI made on purpose — a result
                // measured by a modified build cannot be attributed to any
                // commit — so it reads as information, not as a crash, and
                // its own sentence already explains what to do.
                let detail = error.localizedDescription
                let refused = (error as? CommunityBenchmarkCommand.Failure)?
                    .isRefusal ?? false
                errorTone = refused ? .info : .error
                errorMessage = refused
                    ? detail
                    : String(
                        format: String(localized: "Couldn’t prepare the publication: %1$@"),
                        detail
                    )
            }
            sharingRunID = nil
            shareTask = nil
        }
    }

    private func share(_ preview: CommunityBenchmarkUploadPreview) {
        guard let binary else { return }
        // Freeze the publish context BEFORE the first await. `share` awaits a
        // CLI subprocess for seconds, and Run again / Benchmark another / the
        // model picker can all change which result is on screen during it.
        // Reading the scope afterwards applied this receipt to whatever run
        // happened to be visible when it landed.
        let context = CommunityPublicationContext.capture(
            runID: preview.runID,
            resultScope: results.first { $0.id == preview.runID }.flatMap(comparisonScope(for:)),
            visibleScope: activeObservationScope,
            observations: observations,
            branch: branch
        )
        publishContext = context
        publishedObservationCount = nil
        shareCandidate = nil
        sharingRunID = preview.runID
        errorMessage = nil
        receiptNotSavedWarning = nil
        shareTask = Task {
            do {
                let data = try await CommunityBenchmarkCommand.run(
                    binary: binary,
                    arguments: CommunityBenchmarkCommand.benchmarkShareArguments(
                        runID: preview.runID,
                        installID: preview.installID,
                        payloadDigest: preview.payloadDigest,
                        bodyDigest: preview.bodyDigest,
                        target: preview.target
                    )
                )
                let response = try JSONDecoder().decode(
                    CommunityBenchmarkShareResponse.self, from: data
                )
                guard response.uploaded else {
                    throw CommunityBenchmarkCommand.Failure(
                        message: String(localized: "The benchmark was not uploaded.")
                    )
                }
                if response.receiptSaved {
                    receipts[preview.runID] = response.receipt
                }
                // One place records the identity, the increment, the
                // duplicate rule and the stale-feed floor — all against the
                // captured scope, never against whatever is on screen now.
                let outcome = publication.recordPublication(
                    receipt: response.receipt,
                    receiptSaved: response.receiptSaved,
                    context: context,
                    visibleScope: activeObservationScope
                )
                publishedObservationCount = outcome.observations.value?.observationCount
                // Only paint the count when the screen is still showing the
                // run it belongs to. The floor is recorded either way, so
                // navigating back to that run still shows the confirmed number.
                if outcome.appliesToVisibleScope {
                    observations = outcome.observations
                }
                receiptNotSavedWarning = outcome.receiptNotSavedWarning
                shareSuccess = response.receipt
                // The publication changed server state: this installation may
                // have just been issued its first pseudonym, its public total
                // moved, and the aggregates now include this run. Re-read them
                // rather than leaving the screen on pre-publish numbers.
                //
                // Runs even for a duplicate: the receipt may be the first one
                // this install has seen (so the identity is new to us) even
                // though the run itself was already public.
                await refreshAfterPublishing()
            } catch is CancellationError {
                // Navigation cancelled the upload command and its subprocess.
                publishContext = nil
            } catch {
                publishContext = nil
                errorTone = .error
                errorMessage = String(
                    format: String(localized: "Couldn’t publish: %1$@. Your local result is unchanged."),
                    error.localizedDescription
                )
            }
            sharingRunID = nil
            shareTask = nil
        }
    }

    /// Re-reads everything a successful publication can change.
    ///
    /// Deliberately not fire-and-forget: the Published sheet is on screen and
    /// reads `observations` for its celebration, so the refresh has to be part
    /// of the same task that set the receipt.
    private func refreshAfterPublishing() async {
        await refreshResults()
        async let totals: Void = refreshContributorTotals()
        async let observationsTask: Void = refreshObservations()
        async let table: Void = refreshCommunityTable()
        async let coverageTask: Void = refreshCoverage()
        async let pulseTask: Void = refreshPulse()
        _ = await (totals, observationsTask, table, coverageTask, pulseTask)
    }

    // MARK: - Local data

    private func alias(for repoID: String) -> String {
        resolvedCatalog.first { $0.hfRepo == repoID }?.alias ?? repoID
    }

    private func refreshResults() async {
        guard benchmarkCLIAvailable, let binary else { return }
        do {
            let data = try await CommunityBenchmarkCommand.run(
                binary: binary,
                arguments: CommunityBenchmarkCommand.benchmarkResultsArguments()
            )
            let envelope = try JSONDecoder().decode(CommunityBenchmarkResults.self, from: data)
            results = envelope.runs
            receipts = envelope.receipts ?? [:]
        } catch {
            if results.isEmpty {
                errorTone = .error
                errorMessage = String(
                    format: String(localized: "Couldn’t read local results: %1$@"),
                    error.localizedDescription
                )
            }
        }
    }

    private func refreshProductCatalog() async {
        guard let binary,
              let entries = await ModelCatalog.productEntries(binary: binary),
              !Task.isCancelled
        else { return }
        productCatalog = entries
        selectedAlias = CommunityBenchmarkModel.reconciledSelection(
            current: selectedAlias,
            models: models
        )
    }

    private func refreshBenchmarkCatalog() async {
        guard let binary else {
            benchmarkCLIAvailable = false
            errorTone = .error
            errorMessage = String(localized: "Community Benchmark needs the bundled rapid-mlx runtime. Restart Rapid, then try again.")
            return
        }
        let memory = max(1, Int(MacHardware.detect().physicalRAMGB.rounded()))
        do {
            let data = try await CommunityBenchmarkCommand.run(
                binary: binary,
                arguments: ["benchmark", "catalog", "--memory-gib", String(memory), "--json"]
            )
            let envelope = try JSONDecoder().decode(
                CommunityBenchmarkCatalogEnvelope.self, from: data
            )
            var metadata: [String: CommunityBenchmarkCatalogModel] = [:]
            for model in envelope.models {
                guard metadata.updateValue(model, forKey: model.alias) == nil else {
                    throw CommunityBenchmarkCommand.Failure(
                        message: "Benchmark catalog contains duplicate alias \(model.alias)."
                    )
                }
            }
            benchmarkCLIAvailable = true
            benchmarkMetadata = metadata
            selectedAlias = CommunityBenchmarkModel.reconciledSelection(
                current: selectedAlias,
                models: models
            )
        } catch {
            benchmarkCLIAvailable = false
            benchmarkMetadata = [:]
            errorTone = .error
            errorMessage = String(localized: "Community Benchmark needs a current rapid-mlx runtime. Update or restart Rapid, then try again.")
        }
    }
}
