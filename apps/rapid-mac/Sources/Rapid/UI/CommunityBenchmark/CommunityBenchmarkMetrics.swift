import Foundation

/// Projects one saved benchmark record into the metrics its workload is
/// allowed to display.
///
/// The rule this type exists to enforce: a workload shows only the quantities
/// it actually measures. An image run has no tokens per second and no time to
/// first token; a video run measures a single render, so it has no per-round
/// spread either. Earlier surfaces leaked the text-shaped metric set onto
/// every task, which produced blank or nonsensical cells for image and video.
enum CommunityBenchmarkMetrics {
    /// One displayed quantity: a formatted value, its unit, and the label
    /// underneath.
    struct Metric: Equatable, Identifiable, Sendable {
        let key: String
        /// Formatted magnitude, e.g. "25.8". Kept separate from `unit` so the
        /// view can typeset the unit smaller without re-parsing a string.
        let value: String
        let unit: String?
        let label: String

        var id: String { key }

        var combined: String {
            guard let unit else { return value }
            return "\(value) \(unit)"
        }
    }

    /// The headline metric plus everything shown under "Also measured".
    struct MetricSet: Equatable, Sendable {
        let workload: CommunityWorkload
        let headline: Metric?
        /// Sentence under the headline naming the workload that produced it.
        let headlineCaption: String
        let supporting: [Metric]
        /// Present when the run did not complete; the view shows this instead
        /// of any number so a partial run is never read as a measurement.
        let incompleteStatus: String?
    }

    /// Case IDs of the two declared LLM cases. The short case is the headline;
    /// the long case becomes the "on long prompts" supporting metric.
    static let shortTextCaseID = "pp512-tg128"
    static let longTextCaseID = "pp2048-tg512"

    static func metricSet(
        for result: CommunityBenchmarkResult,
        workload: CommunityWorkload
    ) -> MetricSet {
        guard result.isCompleted else {
            return MetricSet(
                workload: workload,
                headline: nil,
                headlineCaption: "",
                supporting: [],
                incompleteStatus: result.outcome.status.capitalized
            )
        }
        switch workload {
        case .llm: return textMetrics(result)
        case .image: return imageMetrics(result)
        case .video: return videoMetrics(result)
        }
    }

    // MARK: - Per workload

    private static func textMetrics(_ result: CommunityBenchmarkResult) -> MetricSet {
        let summaries = result.caseSummaries
        guard let short = summaries.first(where: { $0.caseID == shortTextCaseID }),
              let long = summaries.first(where: { $0.caseID == longTextCaseID }) else {
            return MetricSet(
                workload: .llm,
                headline: nil,
                headlineCaption: "",
                supporting: [],
                incompleteStatus: String(localized: "Unsupported benchmark protocol")
            )
        }

        var supporting: [Metric] = []
        if let ttft = short.ttftMS {
            supporting.append(
                Metric(
                    key: "ttft",
                    value: formatMilliseconds(ttft),
                    unit: nil,
                    label: String(localized: "Time to first token")
                )
            )
        }
        if let memory = peakMemoryGB(result) {
            supporting.append(
                Metric(
                    key: "memory",
                    value: String(format: "%.1f", memory),
                    unit: "GB",
                    label: String(localized: "Peak memory")
                )
            )
        }
        if let longSpeed = long.decodeTokensPerSecond {
            supporting.append(
                Metric(
                    key: "long",
                    value: String(format: "%.1f", longSpeed),
                    unit: "tok/s",
                    label: String(localized: "On long prompts")
                )
            )
        }
        if let duration = totalDuration(result) {
            supporting.append(
                Metric(
                    key: "duration",
                    value: duration,
                    unit: nil,
                    label: String(localized: "Total duration")
                )
            )
        }

        let headline = short.decodeTokensPerSecond.map { speed in
            Metric(
                key: "speed",
                value: String(format: "%.1f", speed),
                unit: "tok/s",
                label: String(localized: "Generation speed")
            )
        }
        let rounds = short.rounds
        return MetricSet(
            workload: .llm,
            headline: headline,
            headlineCaption: rounds > 0
                ? String(
                    format: String(localized: "Median of %1$d measured %2$@ on short prompts."),
                    rounds,
                    rounds == 1
                        ? String(localized: "pass")
                        : String(localized: "passes")
                )
                : String(localized: "Median of the measured passes on short prompts."),
            supporting: supporting,
            incompleteStatus: headline == nil ? result.outcome.status.capitalized : nil
        )
    }

    private static func imageMetrics(_ result: CommunityBenchmarkResult) -> MetricSet {
        let summary = result.caseSummaries.first
        var supporting: [Metric] = []
        if let memory = peakMemoryGB(result) {
            supporting.append(
                Metric(
                    key: "memory",
                    value: String(format: "%.1f", memory),
                    unit: "GB",
                    label: String(localized: "Peak memory")
                )
            )
        }
        if let duration = totalDuration(result) {
            supporting.append(
                Metric(
                    key: "duration",
                    value: duration,
                    unit: nil,
                    label: String(localized: "Total benchmark duration")
                )
            )
        }
        let headline = summary?.wallSeconds.map { seconds in
            Metric(
                key: "render",
                value: String(format: "%.1f", seconds),
                unit: String(localized: "s / image"),
                label: String(localized: "Render time")
            )
        }
        return MetricSet(
            workload: .image,
            headline: headline,
            headlineCaption: String(
                localized: "One measured 1024 × 1024 render at 20 steps, after a warm-up render."
            ),
            supporting: supporting,
            incompleteStatus: headline == nil ? result.outcome.status.capitalized : nil
        )
    }

    private static func videoMetrics(_ result: CommunityBenchmarkResult) -> MetricSet {
        let summary = result.caseSummaries.first
        var supporting: [Metric] = []
        if let memory = peakMemoryGB(result) {
            supporting.append(
                Metric(
                    key: "memory",
                    value: String(format: "%.1f", memory),
                    unit: "GB",
                    label: String(localized: "Peak memory")
                )
            )
        }
        if let duration = totalDuration(result) {
            supporting.append(
                Metric(
                    key: "duration",
                    value: duration,
                    unit: nil,
                    label: String(localized: "Total benchmark duration")
                )
            )
        }
        let headline = summary?.wallSeconds.map { seconds in
            Metric(
                key: "video",
                value: String(format: "%.1f", seconds),
                unit: String(localized: "s / video"),
                label: String(localized: "Seconds per video")
            )
        }
        return MetricSet(
            workload: .video,
            headline: headline,
            headlineCaption: String(
                localized: "One measured 832 × 480, 81-frame render with a fixed prompt and seed."
            ),
            supporting: supporting,
            incompleteStatus: headline == nil ? result.outcome.status.capitalized : nil
        )
    }

    // MARK: - Shared derivations

    /// High-water unified memory across every completed round, in GB. Nil when
    /// no round reported the field — the runtime could not sample it, and a
    /// zero would read as "used no memory".
    static func peakMemoryGB(_ result: CommunityBenchmarkResult) -> Double? {
        let peaks = (result.measurements ?? [])
            .filter { $0.completed ?? true }
            .compactMap(\.peakActiveMemoryMiB)
            .filter { $0 > 0 }
        guard let maximum = peaks.max() else { return nil }
        return maximum / 1_024
    }

    /// Wall time for the whole run as `m:ss`, summed over completed rounds.
    /// Warm-up rounds are included because the user waited for them.
    static func totalDuration(_ result: CommunityBenchmarkResult) -> String? {
        let total = (result.measurements ?? [])
            .filter { $0.completed ?? true }
            .compactMap(\.totalDurationMS)
            .reduce(0, +)
        guard total > 0 else { return nil }
        let seconds = Int((total / 1_000).rounded())
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }

    static func formatMilliseconds(_ value: Double) -> String {
        value >= 10_000
            ? String(format: "%.1f s", value / 1_000)
            : String(format: "%.2f s", value / 1_000)
    }

    /// The fixed workload sentence shown on Ready and in Test Method. These
    /// are the parameters the shipped CLI actually runs; they are not
    /// configurable, which is what makes results comparable.
    static func protocolDescription(for workload: CommunityWorkload) -> String {
        switch workload {
        case .image:
            return String(
                localized: "1 warm-up + 1 measured 1024×1024 render · fixed prompt, seed and 20 steps"
            )
        case .video:
            return String(
                localized: "1 measured 832×480, 81-frame render · fixed prompt and seed"
            )
        case .llm:
            return String(
                localized: "Two fixed token workloads · 1 warm-up + 5 measured rounds each · concurrency 1"
            )
        }
    }

    /// What "What this measures" lists, per workload. Image and video never
    /// list generation speed or time to first token.
    static func measuredQuantities(
        for workload: CommunityWorkload
    ) -> [(title: String, detail: String)] {
        switch workload {
        case .llm:
            return [
                (
                    String(localized: "Generation speed"),
                    String(localized: "How many tokens per second the model writes once it starts.")
                ),
                (
                    String(localized: "Time to first token"),
                    String(localized: "How long you wait before the first word appears.")
                ),
                (
                    String(localized: "Peak memory"),
                    String(localized: "The most unified memory the model held at once.")
                ),
            ]
        case .image:
            return [
                (
                    String(localized: "Render time"),
                    String(localized: "How long one 1024 × 1024 image takes to generate.")
                ),
                (
                    String(localized: "Peak memory"),
                    String(localized: "The most unified memory the model held at once.")
                ),
                (
                    String(localized: "Total benchmark duration"),
                    String(localized: "How long the whole run takes, warm-up render included.")
                ),
            ]
        case .video:
            return [
                (
                    String(localized: "Seconds per video"),
                    String(localized: "How long one 832 × 480, 81-frame render takes.")
                ),
                (
                    String(localized: "Peak memory"),
                    String(localized: "The most unified memory the model held at once.")
                ),
                (
                    String(localized: "Total benchmark duration"),
                    String(localized: "How long the whole run takes.")
                ),
            ]
        }
    }
}
