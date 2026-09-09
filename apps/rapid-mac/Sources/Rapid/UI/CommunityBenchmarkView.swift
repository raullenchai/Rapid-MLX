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
    let isFocus: Bool
    let estimatedMemoryGib: Int?
    let memoryFit: String

    var id: String { entry.alias }

    static let focusAliases: Set<String> = [
        "qwen3.8-27b-4bit", "qwen3.5-9b-4bit", "gemma-4-e4b-4bit",
        "flux2-klein-4b", "z-image-turbo", "qwen-image",
        "wan2.2-ti2v-5b-q8"
    ]
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
                isFocus: catalogModel?.focus ?? focusAliases.contains(entry.alias),
                estimatedMemoryGib: catalogModel?.estimatedMemoryGib,
                memoryFit: catalogModel?.memoryFit ?? "unknown"
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
    let alias: String
    let focus: Bool
    let estimatedMemoryGib: Int?
    let memoryFit: String
    let protocolVersion: Int?

    enum CodingKeys: String, CodingKey {
        case alias, focus
        case estimatedMemoryGib = "estimated_memory_gib"
        case memoryFit = "memory_fit"
        case protocolVersion = "protocol_version"
    }
}

private struct CommunityBenchmarkCatalogEnvelope: Decodable {
    let models: [CommunityBenchmarkCatalogModel]
}

struct CommunityBenchmarkResults: Decodable {
    let runs: [CommunityBenchmarkResult]
    let receipts: [String: CommunityBenchmarkReceipt]?
}

struct CommunityBenchmarkContributor: Decodable, Equatable {
    let name: String
    let tag: String

    var displayName: String { "\(name) ·\(tag)" }

    var profileURL: URL? {
        // Percent-encode the identifier so an embedded "/" in a server-assigned
        // name/tag cannot become a path separator — mirrors the CLI client's
        // urllib `quote(f"{name}-{tag}", safe="-")`. (Nothing but ASCII
        // alphanumerics, "_", ".", "-", "~" stays literal; everything else is
        // percent-encoded, so the joined slug is safe to drop into a URL path.)
        var allowed = CharacterSet.alphanumerics
        allowed.formUnion(CharacterSet(charactersIn: "_.-~"))
        let encoded = ("\(name)-\(tag)").addingPercentEncoding(
            withAllowedCharacters: allowed
        ) ?? ""
        return URL(string: "https://rapidmlx.com/leaderboard/contributors/\(encoded)")
    }
}

struct CommunityBenchmarkReceipt: Decodable, Identifiable {
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
    let runID: String
    let target: String
    let installID: String
    let payloadDigest: String
    let bodyDigest: String
    let payloadJSON: String

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
        enum CodingKeys: String, CodingKey {
            case taskType = "task_type"
            case cases
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
        enum CodingKeys: String, CodingKey {
            case caseID = "case_id"
            case completed
            case outputTokens = "output_tokens"
            case ttftMS = "ttft_ms"
            case decodeDurationMS = "decode_duration_ms"
            case totalDurationMS = "total_duration_ms"
        }
    }
    struct Model: Decodable {
        struct Component: Decodable {
            struct Source: Decodable { let repoID: String?
                enum CodingKeys: String, CodingKey { case repoID = "repo_id" }
            }
            let source: Source
        }
        let components: [Component]
    }
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
        let runtime: Runtime
        let configDigest: String

        enum CodingKeys: String, CodingKey {
            case runtime
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

    var repoID: String { model.components.first?.source.repoID ?? "Local model" }

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
    /// `vllm_mlx/community_bench/cli.py`.
    static let progressTag = "\u{1e}"

    /// A tagged progress line with its marker removed and whitespace
    /// collapsed, or nil for any other stderr (the untagged failure
    /// document, warnings, tracebacks) so the view never mirrors it.
    static func strippedProgress(from line: String) -> String? {
        guard line.hasPrefix(progressTag) else { return nil }
        let body = line.dropFirst(progressTag.count)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty, body.count <= 200 else { return nil }
        return body.split(whereSeparator: \.isWhitespace).joined(separator: " ")
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
            payloadJSON: payloadJSON
        )
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
                    let outputTask = Task.detached {
                        readBoundedPipe(
                            stdout.fileHandleForReading,
                            maxBytes: maxStdoutBytes,
                            retainTail: false
                        )
                    }
                    let errorTask = Task.detached {
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
                    if let processGroupID = box.waitForCompletion(child) {
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
                        throw Failure(message: message)
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

    private static func readBoundedPipe(
        _ handle: FileHandle,
        maxBytes: Int,
        retainTail: Bool,
        onChunk: ((Data) -> Void)? = nil
    ) -> PipeCapture {
        var data = Data()
        var truncated = false
        while true {
            let chunk: Data?
            do {
                chunk = try handle.read(upToCount: pipeChunkBytes)
            } catch {
                break
            }
            guard let chunk, !chunk.isEmpty else { break }
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

struct CommunityBenchmarkView: View {
    let catalog: [ModelEntry]
    let binary: URL?
    let prepareServer: () async throws -> UUID
    let releaseServer: (UUID) -> Void
    let retainServerDuringDeferredReap: (pid_t) -> Void

    @State private var selectedAlias = ""
    @State private var results: [CommunityBenchmarkResult] = []
    @State private var isRunning = false
    @State private var runStartedAt: Date?
    @State private var runningModel: CommunityBenchmarkModel?
    @State private var runProgressLine: String?
    @State private var currentRunID: UUID?
    /// Highest progress sequence applied so far; lines are stamped in
    /// arrival order off the main actor and applied only if newer, so the
    /// unordered main-actor hops can never show an older round.
    @State private var appliedProgressSequence = 0
    /// Completed warmup + measured passes, and when the first one landed, for
    /// the determinate progress bar and the live ETA.
    @State private var stepsDone = 0
    @State private var firstStepAt: Date?
    /// The most recent step completion, so the ETA divides by real
    /// inter-step time and stays stable between steps.
    @State private var lastStepAt: Date?
    @State private var errorMessage: String?
    /// The result id of the run that just finished, so a prominent CTA can
    /// invite the user to share it (instead of relying on the small per-row
    /// link). Cleared when the row is shared, dismissed, or a new run starts.
    @State private var pendingShareResultID: String?
    @State private var runTask: Task<Void, Never>?
    @State private var shareTask: Task<Void, Never>?
    @State private var shareCandidate: CommunityBenchmarkUploadPreview?
    @State private var sharingRunID: String?
    @State private var shareSuccess: CommunityBenchmarkReceipt?
    @State private var receipts: [String: CommunityBenchmarkReceipt] = [:]
    @State private var benchmarkMetadata: [String: CommunityBenchmarkCatalogModel] = [:]
    @State private var benchmarkCLIAvailable = false
    @State private var productCatalog: [ModelEntry]?

    private var resolvedCatalog: [ModelEntry] {
        CommunityBenchmarkModel.resolvedCatalog(
            product: productCatalog,
            fallback: catalog
        )
    }

    private var models: [CommunityBenchmarkModel] {
        CommunityBenchmarkModel.models(
            from: resolvedCatalog,
            metadata: benchmarkMetadata
        )
    }

    private var selected: CommunityBenchmarkModel? {
        models.first { $0.entry.alias == selectedAlias }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header
                setupCard
                postRunShareCTA
                recentResults
            }
            .frame(maxWidth: 760, alignment: .leading)
            .padding(32)
            .frame(maxWidth: .infinity, alignment: .top)
        }
        .background(RapidTheme.surfaceCanvas)
        .task {
            await refreshProductCatalog()
            if selectedAlias.isEmpty { selectedAlias = models.first?.entry.alias ?? "" }
            await refreshBenchmarkCatalog()
            await refreshResults()
        }
        .onDisappear {
            // `runTask` is intentionally unstructured so the button owns it;
            // navigation must explicitly cancel it before the only Stop
            // control disappears. ServerManager keeps the lease until the
            // subprocess tree has actually been reaped.
            runTask?.cancel()
            shareTask?.cancel()
        }
        .sheet(item: $shareCandidate) { preview in
            VStack(alignment: .leading, spacing: 16) {
                Text("Share benchmark result?")
                    .font(.title2.weight(.semibold))
                Label(
                    "Every shared result makes the community leaderboard more "
                        + "complete — helping everyone compare models across real "
                        + "Macs and find faster local AI for their machine.",
                    systemImage: "mappin.and.ellipse"
                )
                .font(.callout)
                .foregroundStyle(RapidTheme.textSecondary)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RapidTheme.brandPrimaryDeep.opacity(0.08),
                    in: RoundedRectangle(cornerRadius: 10)
                )
                Text("Everything in the JSON below will be sent to \(preview.target).")
                    .foregroundStyle(.secondary)
                ScrollView {
                    Text(preview.payloadJSON)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                }
                .background(RapidTheme.surfaceCanvas)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                Text(
                    "No name, hostname, serial number, hardware UUID, prompts, "
                        + "outputs, file paths, or IP-address field are included in the JSON. "
                        + "The service observes the source IP for short-lived rate limiting "
                        + "but does not put it in the benchmark record."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                HStack {
                    Spacer()
                    Button("Cancel", role: .cancel) { shareCandidate = nil }
                        .accessibilityIdentifier("CommunityBenchmark.Share.Cancel")
                    Button("Share") { share(preview) }
                        .buttonStyle(.borderedProminent)
                        .accessibilityIdentifier("CommunityBenchmark.Share.Confirm")
                }
            }
            .padding(24)
            .frame(minWidth: 680, minHeight: 600)
        }
        .sheet(item: $shareSuccess, content: shareSuccessSheet)
    }

    private func shareSuccessSheet(_ receipt: CommunityBenchmarkReceipt) -> some View {
        VStack(spacing: 0) {
            VStack(spacing: 24) {
                Image(systemName: "mappin.and.ellipse")
                    .font(.system(size: 32, weight: .medium))
                    .foregroundStyle(RapidTheme.brandPrimaryDeep)
                    .frame(width: 72, height: 72)
                    .background(RapidTheme.brandPrimaryDeep.opacity(0.10), in: Circle())
                    .overlay(alignment: .bottomTrailing) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 22, weight: .semibold))
                            .foregroundStyle(RapidTheme.brandPrimaryDeep)
                            .padding(3)
                            .background(RapidTheme.surfaceRaised, in: Circle())
                    }
                    .accessibilityHidden(true)

                VStack(spacing: 10) {
                    Text(receipt.alreadyExists ? "Already on the map" : "You added a point to the map")
                        .font(.title2.weight(.semibold))
                        .foregroundStyle(RapidTheme.textPrimary)
                    Text(receipt.alreadyExists
                         ? "This result is already part of the community leaderboard. Thanks for contributing."
                         : "Your model’s performance on this Mac now has a place on the community leaderboard.")
                        .font(.body)
                        .foregroundStyle(RapidTheme.textSecondary)
                }

                if let contributor = receipt.contributor {
                    VStack(spacing: 10) {
                        Text("Your community identity")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(RapidTheme.textSecondary)
                        Text(contributor.displayName)
                            .font(.system(.callout, design: .monospaced).weight(.medium))
                            .foregroundStyle(RapidTheme.textPrimary)
                            .textSelection(.enabled)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 10)
                            .background(RapidTheme.surfaceCanvas, in: Capsule())
                            .overlay {
                                Capsule().strokeBorder(RapidTheme.hairline, lineWidth: 1)
                            }
                            .accessibilityIdentifier("CommunityBenchmark.Share.Identity")
                    }
                }

                Text("Together, we’re building a clearer picture of local AI performance—so everyone can find faster models for their Mac.")
                    .font(.callout)
                    .foregroundStyle(RapidTheme.textSecondary)
            }
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
            .padding(32)

            Rectangle()
                .fill(RapidTheme.hairline)
                .frame(height: 1)

            HStack(spacing: 12) {
                if let contributor = receipt.contributor {
                    if let url = contributor.profileURL {
                        Link("View my contributions", destination: url)
                            .buttonStyle(.bordered)
                            .accessibilityIdentifier("CommunityBenchmark.Share.Profile")
                    }
                } else {
                    Link(
                        "View Community Benchmark",
                        destination: communityBenchmarkLeaderboardURL
                    )
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("CommunityBenchmark.Share.Leaderboard")
                }
                Spacer(minLength: 0)
                Button("Done") { shareSuccess = nil }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .accessibilityIdentifier("CommunityBenchmark.Share.Done")
            }
            .controlSize(.large)
            .padding(.horizontal, 24)
            .padding(.vertical, 18)
            .background(RapidTheme.surfaceCanvas)
        }
        .frame(width: 480)
        .background(RapidTheme.surfaceRaised)
        .accessibilityIdentifier("CommunityBenchmark.Share.Success")
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Benchmark")
                .font(.system(size: 28, weight: .semibold))
            Text("How fast is local AI on this Mac? Find out in minutes — then share your result to help build an open leaderboard of real models on real Macs, so everyone can pick the fastest local AI for their machine.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var setupCard: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Run a benchmark").font(.headline)
            Picker("Model", selection: $selectedAlias) {
                ForEach(
                    CommunityBenchmarkModel.pickerSections(models), id: \.title
                ) { section in
                    Section(section.title) {
                        ForEach(section.models) { model in
                            Text("\(model.isFocus ? "★ " : "")\(model.entry.alias)")
                                .tag(model.entry.alias)
                        }
                    }
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityIdentifier("CommunityBenchmark.ModelPicker")

            if let selected {
                HStack(spacing: 10) {
                    Label(selected.protocolName, systemImage: "gauge.with.dots.needle.50percent")
                    if selected.entry.cached {
                        Text("Downloaded").foregroundStyle(.green)
                    } else {
                        Text("Download required").foregroundStyle(.secondary)
                    }
                    if let memory = selected.estimatedMemoryGib {
                        Text(memoryCopy(memory, fit: selected.memoryFit))
                            .foregroundStyle(selected.memoryFit == "does_not_fit" ? .orange : .secondary)
                    }
                }
                .font(.callout)
                Text(protocolDescription(selected.task))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let errorMessage {
                Text(errorMessage).font(.callout).foregroundStyle(.red)
            }

            HStack(alignment: .top) {
                Button(isRunning ? "Stop" : "Run locally") {
                    if isRunning {
                        // Invalidate the run token first so stderr that
                        // arrives during teardown cannot update the row.
                        currentRunID = nil
                        runTask?.cancel()
                    } else {
                        startRun()
                    }
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier("CommunityBenchmark.RunOrStop")
                .disabled(
                    !isRunning
                        && (selected == nil || binary == nil || !benchmarkCLIAvailable)
                )
                if isRunning {
                    ProgressView().controlSize(.small)
                    runningStatus
                }
            }
        }
        .padding(20)
        .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 14))
    }

    /// What is being measured, its scope, the expected wall time, and a
    /// live elapsed clock — the only feedback the user gets for several
    /// minutes while the CLI owns the machine.
    private var runningStatus: some View {
        // A determinate bar is only honest when we know how many passes to
        // expect (text, image); other shapes keep the spinner + clock.
        let totalSteps = runningModel.flatMap {
            CommunityBenchmarkRunStatus.totalSteps(for: $0.task)
        }
        return VStack(alignment: .leading, spacing: 6) {
            if let runningModel {
                Text(CommunityBenchmarkRunStatus.description(for: runningModel))
                    .font(.callout)
                    .accessibilityIdentifier("CommunityBenchmark.RunStatus")
            }
            if let totalSteps {
                ProgressView(
                    value: Double(min(stepsDone, totalSteps)),
                    total: Double(totalSteps)
                )
                .progressViewStyle(.linear)
                .frame(maxWidth: 360)
                .accessibilityIdentifier("CommunityBenchmark.RunProgressBar")
            }
            HStack(spacing: 8) {
                if let runStartedAt {
                    TimelineView(.periodic(from: runStartedAt, by: 1)) { context in
                        HStack(spacing: 8) {
                            Text("Elapsed \(CommunityBenchmarkRunStatus.elapsed(from: runStartedAt, to: context.date))")
                                .monospacedDigit()
                            if let totalSteps, let firstStepAt, let lastStepAt,
                               let eta = CommunityBenchmarkRunStatus.eta(
                                   stepsDone: stepsDone,
                                   totalSteps: totalSteps,
                                   runStartedAt: runStartedAt,
                                   firstStepAt: firstStepAt,
                                   lastStepAt: lastStepAt,
                                   now: context.date
                               ) {
                                Text(eta)
                                    .monospacedDigit()
                                    .accessibilityIdentifier("CommunityBenchmark.RunETA")
                            }
                        }
                    }
                    .accessibilityIdentifier("CommunityBenchmark.RunElapsed")
                }
                if let runProgressLine {
                    Text(runProgressLine)
                        .font(.caption.monospaced())
                        .lineLimit(1)
                        .accessibilityIdentifier("CommunityBenchmark.RunProgress")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            Text("The active server will stop while this model is measured.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    /// A prominent invitation to contribute the run that just finished — the
    /// per-row "Share" link is easy to miss, and a result is only useful to
    /// the community once it is shared. Hidden once the run is shared (a
    /// receipt appears), dismissed, or superseded by a new run.
    @ViewBuilder
    private var postRunShareCTA: some View {
        if let id = pendingShareResultID,
           receipts[id] == nil,
           !isRunning,
           let result = results.first(where: { $0.id == id }) {
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: "mappin.and.ellipse")
                    .font(.title2)
                    .foregroundStyle(RapidTheme.brandPrimaryDeep)
                VStack(alignment: .leading, spacing: 6) {
                    Text("Put your Mac on the map").font(.headline)
                    Text(
                        "Share this \(alias(for: result.repoID)) result to help the "
                            + "community compare models and find faster local AI."
                    )
                    .font(.callout)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 12) {
                        Button(sharingRunID == result.id ? "Sharing…" : "Share benchmark result") {
                            prepareShare(result)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(sharingRunID != nil || binary == nil)
                        .accessibilityIdentifier("CommunityBenchmark.ShareCTA.Confirm")
                        Button("Not now") { pendingShareResultID = nil }
                            .buttonStyle(.link)
                            .accessibilityIdentifier("CommunityBenchmark.ShareCTA.Dismiss")
                    }
                    .padding(.top, 2)
                }
                Spacer(minLength: 0)
            }
            .padding(16)
            .background(
                RapidTheme.brandPrimaryDeep.opacity(0.06),
                in: RoundedRectangle(cornerRadius: 14)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(RapidTheme.brandPrimaryDeep.opacity(0.25))
            )
            .accessibilityIdentifier("CommunityBenchmark.ShareCTA")
        }
    }

    private var recentResults: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent local results").font(.headline)
            if results.isEmpty {
                Text("No benchmarks yet. Your first result will appear here.")
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 12)
            } else {
                ForEach(results.prefix(8)) { result in
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(alias(for: result.repoID)).fontWeight(.medium)
                            Text(
                                "\(result.workload.taskType.replacingOccurrences(of: "_", with: " ")) · "
                                    + CommunityBenchmarkResult.formatCompletedAt(result.completedAt)
                            )
                            .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        VStack(alignment: .trailing, spacing: 5) {
                            resultHeadline(result)
                            if let receipt = receipts[result.id] {
                                Label("Shared", systemImage: "checkmark.circle.fill")
                                    .font(.caption)
                                    .foregroundStyle(.green)
                                Link(
                                    receipt.contributionLinkTitle,
                                    destination: receipt.contributionURL
                                )
                                .font(.caption.monospaced())
                                .accessibilityLabel(receipt.contributionAccessibilityLabel)
                                .accessibilityIdentifier(
                                    "CommunityBenchmark.Contributor.\(result.id)"
                                )
                            } else {
                                Button(sharingRunID == result.id ? "Sharing…" : "Share") {
                                    prepareShare(result)
                                }
                                .buttonStyle(.link)
                                .font(.caption)
                                .disabled(sharingRunID != nil || binary == nil)
                                .accessibilityIdentifier("CommunityBenchmark.Share.\(result.id)")
                            }
                        }
                    }
                    .padding(.vertical, 8)
                    Divider()
                }
            }
        }
    }

    /// Median decode tok/s + TTFT for the short case (or wall seconds for
    /// image/video), with the remaining cases underneath and in the tooltip.
    /// Failed or incomplete runs keep showing their outcome status instead.
    @ViewBuilder
    private func resultHeadline(_ result: CommunityBenchmarkResult) -> some View {
        if let headline = result.headline {
            let secondary = result.secondaryLines
            VStack(alignment: .trailing, spacing: 2) {
                Text(headline)
                    .monospacedDigit()
                    .accessibilityIdentifier("CommunityBenchmark.Result.\(result.id)")
                ForEach(secondary, id: \.self) { line in
                    Text(line)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }
            .help(
                result.caseSummaries
                    .map { "\($0.caseID): \($0.headline) (\($0.rounds) rounds)" }
                    .joined(separator: "\n")
            )
        } else {
            Text(result.outcome.status.capitalized)
                .accessibilityIdentifier("CommunityBenchmark.Result.\(result.id)")
        }
    }

    private func protocolDescription(_ task: ModelTask) -> String {
        switch task {
        case .imageGeneration: return "1 warmup + 1 measured 1024×1024 render · fixed prompt, seed and 20 steps"
        case .videoGeneration: return "1 measured 832×480, 81-frame render · fixed prompt and seed"
        default: return "Two fixed token workloads · 1 warmup + 5 measured rounds each · concurrency 1"
        }
    }

    private func alias(for repoID: String) -> String {
        resolvedCatalog.first { $0.hfRepo == repoID }?.alias ?? repoID
    }

    private func memoryCopy(_ memory: Int, fit: String) -> String {
        fit == "does_not_fit" ? "Needs about \(memory) GB" : "About \(memory) GB"
    }

    private func startRun() {
        guard benchmarkCLIAvailable, let selected, let binary else { return }
        errorMessage = nil
        isRunning = true
        runningModel = selected
        runStartedAt = Date()
        runProgressLine = nil
        appliedProgressSequence = 0
        stepsDone = 0
        firstStepAt = nil
        lastStepAt = nil
        pendingShareResultID = nil
        let activeRunID = UUID()
        currentRunID = activeRunID
        let sequencer = ProgressSequencer()
        // Cumulative step index, stamped off the main actor in arrival order
        // (LineSplitter delivers lines sequentially). The view applies it as
        // a monotonic max, so a late/out-of-order hop can never drop a step.
        let stepCounter = ProgressSequencer()
        runTask = Task {
            var acquiredReservation = false
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
                        guard let progress = CommunityBenchmarkRunStatus.strippedProgress(
                            from: line
                        ) else { return }
                        let sequence = sequencer.next()
                        // Cumulative step index (0 for non-step lines), assigned
                        // in arrival order so the count survives unordered hops.
                        let isStep = CommunityBenchmarkRunStatus.isStepLine(progress)
                        let stepIndex = isStep ? stepCounter.next() : 0
                        // Timestamp the step at emission (arrival order), not
                        // when its main-actor hop lands, so the ETA baseline is
                        // the first step's real completion time.
                        let stepAt = isStep ? Date() : nil
                        Task { @MainActor in
                            guard isRunning, runStartedAt != nil,
                                  currentRunID == activeRunID
                            else { return }
                            // Display: show only the newest line — a hop that
                            // lands late must not overwrite a newer status.
                            if sequence > appliedProgressSequence {
                                appliedProgressSequence = sequence
                                runProgressLine = progress
                            }
                            // Count: monotonic max, so an out-of-order hop can
                            // never discard a completed step (undercounting the
                            // bar/ETA). The baseline keeps the earliest step
                            // timestamp (step 1) regardless of hop order.
                            if stepIndex > 0, let stepAt {
                                firstStepAt = min(firstStepAt ?? stepAt, stepAt)
                                lastStepAt = max(lastStepAt ?? stepAt, stepAt)
                                stepsDone = max(stepsDone, stepIndex)
                            }
                        }
                    }
                )
                await refreshProductCatalog()
                await refreshResults()
                // Invite the user to contribute the run that just finished —
                // only when the CLI payload names it. No fallback to "whatever
                // sorts first": a payload without a run_id (e.g. deferred reap)
                // must not surface the CTA for an unrelated historical run.
                pendingShareResultID = CommunityBenchmarkCommand.runID(from: runOutput)
            } catch is CancellationError {
                errorMessage = acquiredReservation
                    ? "Benchmark stopped. No incomplete result was shared."
                    : "Benchmark request stopped before it started."
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
            runningModel = nil
            runStartedAt = nil
            runProgressLine = nil
            stepsDone = 0
            firstStepAt = nil
            lastStepAt = nil
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
                errorMessage = "Couldn’t prepare benchmark upload: \(error.localizedDescription)"
            }
            sharingRunID = nil
            shareTask = nil
        }
    }

    private func share(_ preview: CommunityBenchmarkUploadPreview) {
        guard let binary else { return }
        shareCandidate = nil
        sharingRunID = preview.runID
        errorMessage = nil
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
                        message: "The benchmark was not uploaded."
                    )
                }
                if response.receiptSaved {
                    receipts[preview.runID] = response.receipt
                } else {
                    errorMessage = "Uploaded, but Rapid couldn’t save the local receipt."
                }
                shareSuccess = response.receipt
            } catch is CancellationError {
                // Navigation cancelled the upload command and its subprocess.
            } catch {
                errorMessage = "Couldn’t share benchmark: \(error.localizedDescription)"
            }
            sharingRunID = nil
            shareTask = nil
        }
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
            if results.isEmpty { errorMessage = "Couldn’t read local results: \(error.localizedDescription)" }
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
            errorMessage = "Community Benchmark needs the bundled rapid-mlx runtime. Restart Rapid, then try again."
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
            errorMessage = "Community Benchmark needs a current rapid-mlx runtime. Update or restart Rapid, then try again."
        }
    }
}
