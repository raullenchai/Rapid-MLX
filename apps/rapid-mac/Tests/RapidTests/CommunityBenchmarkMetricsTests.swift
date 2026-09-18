import Foundation
import Testing
@testable import Rapid

@Suite("Workload-specific result metrics")
struct CommunityBenchmarkMetricsTests {
    /// Two text cases × five completed rounds, with `peak_active_memory_mib`
    /// on every round — the shape `benchmark results --json` writes today.
    private static let textRun = #"""
    {"completed_at":"2026-09-06T04:37:42.144907Z","execution":{"config_digest":"sha256:abc","runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"}},"machine":{"os":{"version":"15.6.1"},"profile":{"chip":"Apple M3 Pro","cpu_cores":12,"gpu_cores":18,"memory_gib":18}},"measurements":[
    {"case_id":"pp512-tg128","completed":true,"decode_duration_ms":5000,"output_tokens":129,"peak_active_memory_mib":6875,"round_index":1,"total_duration_ms":6500,"ttft_ms":1480},
    {"case_id":"pp512-tg128","completed":true,"decode_duration_ms":5000,"output_tokens":129,"peak_active_memory_mib":6875,"round_index":2,"total_duration_ms":6500,"ttft_ms":1480},
    {"case_id":"pp2048-tg512","completed":true,"decode_duration_ms":20000,"output_tokens":513,"peak_active_memory_mib":7168,"round_index":1,"total_duration_ms":26000,"ttft_ms":5800}
    ],"model":{"components":[{"source":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"}}]},"outcome":{"status":"completed"},"run_id":"run-text","workload":{"cases":[{"case_id":"pp512-tg128","measured_rounds":5,"target_output_tokens":128,"target_prompt_tokens":512,"warmup_rounds":1},{"case_id":"pp2048-tg512","measured_rounds":5,"target_output_tokens":512,"target_prompt_tokens":2048,"warmup_rounds":1}],"task_type":"text_generation"}}
    """#

    /// One measured render. No token fields at all, which is exactly why an
    /// image run must not display generation speed or time to first token.
    private static let imageRun = #"""
    {"completed_at":"2026-09-06T04:37:42Z","execution":{"config_digest":"sha256:def","runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"}},"machine":{"os":{"version":"15.6.1"},"profile":{"chip":"Apple M3 Pro","cpu_cores":12,"gpu_cores":18,"memory_gib":18}},"measurements":[
    {"case_id":"render-1024","completed":true,"peak_active_memory_mib":6246,"round_index":1,"total_duration_ms":4600}
    ],"model":{"components":[{"source":{"repo_id":"mlx-community/z-image-turbo"}}]},"outcome":{"status":"completed"},"run_id":"run-image","workload":{"cases":[{"case_id":"render-1024","measured_rounds":1,"warmup_rounds":1}],"task_type":"image_generation"}}
    """#

    private static let failedRun = #"""
    {"completed_at":"2026-09-06T04:37:42Z","execution":{"config_digest":"sha256:ghi","runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"}},"machine":{"os":{"version":"15.6.1"},"profile":{"chip":"Apple M3 Pro","cpu_cores":12,"gpu_cores":18,"memory_gib":18}},"measurements":[
    {"case_id":"render-832","completed":false,"round_index":1,"total_duration_ms":9000}
    ],"model":{"components":[{"source":{"repo_id":"mlx-community/wan"}}]},"outcome":{"status":"failed"},"run_id":"run-failed","workload":{"cases":[{"case_id":"render-832","measured_rounds":1,"warmup_rounds":0}],"task_type":"video_generation"}}
    """#

    private static func decode(_ json: String) throws -> CommunityBenchmarkResult {
        try JSONDecoder().decode(CommunityBenchmarkResult.self, from: Data(json.utf8))
    }

    @Test("An LLM run leads with generation speed and lists the text-only metrics")
    func textMetrics() throws {
        let set = CommunityBenchmarkMetrics.metricSet(
            for: try Self.decode(Self.textRun), workload: .llm
        )
        #expect(set.workload == .llm)
        #expect(set.headline?.label == "Generation speed")
        #expect(set.headline?.unit == "tok/s")
        // (129 - 1) / 5.0 s = 25.6 tok/s
        #expect(set.headline?.value == "25.6")
        #expect(set.headlineCaption.contains("short prompts"))

        let labels = set.supporting.map(\.label)
        #expect(labels.contains("Time to first token"))
        #expect(labels.contains("Peak memory"))
        #expect(labels.contains("On long prompts"))
        #expect(labels.contains("Total duration"))
        #expect(set.incompleteStatus == nil)
    }

    @Test("Unknown text cases are not mislabeled as short and long prompts")
    func unknownTextCasesAreUnsupported() throws {
        let unknown = Self.textRun
            .replacingOccurrences(of: "pp512-tg128", with: "vendor-short")
            .replacingOccurrences(of: "pp2048-tg512", with: "vendor-long")
        let set = CommunityBenchmarkMetrics.metricSet(
            for: try Self.decode(unknown), workload: .llm
        )

        #expect(set.headline == nil)
        #expect(set.supporting.isEmpty)
        #expect(set.headlineCaption.isEmpty)
        #expect(set.incompleteStatus == "Unsupported benchmark protocol")
    }

    @Test("An image run reports render time and never tokens per second")
    func imageMetrics() throws {
        let set = CommunityBenchmarkMetrics.metricSet(
            for: try Self.decode(Self.imageRun), workload: .image
        )
        #expect(set.headline?.label == "Render time")
        #expect(set.headline?.value == "4.6")
        #expect(set.headline?.unit == "s / image")

        let labels = set.supporting.map(\.label)
        #expect(labels == ["Peak memory", "Total benchmark duration"])
        // The two text-only metrics must be entirely absent, not blank cells.
        #expect(!labels.contains("Time to first token"))
        #expect(!labels.contains("On long prompts"))
        #expect(set.supporting.allSatisfy { $0.unit != "tok/s" })
    }

    @Test("A video run reports seconds per video with no per-round spread")
    func videoMetricsListedForWorkload() {
        let quantities = CommunityBenchmarkMetrics.measuredQuantities(for: .video)
        #expect(quantities.map(\.title) == [
            "Seconds per video", "Peak memory", "Total benchmark duration",
        ])
        let imageQuantities = CommunityBenchmarkMetrics.measuredQuantities(for: .image)
        #expect(imageQuantities.map(\.title).contains("Render time"))
        #expect(!imageQuantities.map(\.title).contains("Generation speed"))
        let textQuantities = CommunityBenchmarkMetrics.measuredQuantities(for: .llm)
        #expect(textQuantities.map(\.title).contains("Time to first token"))
    }

    @Test("A failed run shows its status instead of any number")
    func failedRunShowsStatus() throws {
        let set = CommunityBenchmarkMetrics.metricSet(
            for: try Self.decode(Self.failedRun), workload: .video
        )
        #expect(set.headline == nil)
        #expect(set.supporting.isEmpty)
        #expect(set.incompleteStatus == "Failed")
    }

    @Test("Peak memory takes the high-water round and converts MiB to GB")
    func peakMemoryDerivation() throws {
        let run = try Self.decode(Self.textRun)
        let peak = CommunityBenchmarkMetrics.peakMemoryGB(run)
        // 7168 MiB is the maximum across rounds → 7.0 GB.
        #expect(peak != nil)
        #expect(abs((peak ?? 0) - 7.0) < 0.001)
    }

    @Test("A run with no memory samples reports no peak rather than zero")
    func missingPeakMemoryIsAbsent() throws {
        let json = #"""
        {"completed_at":"2026-09-06T04:37:42Z","execution":{"config_digest":"sha256:x","runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"}},"measurements":[
        {"case_id":"render-1024","completed":true,"round_index":1,"total_duration_ms":4600}
        ],"model":{"components":[{"source":{"repo_id":"r"}}]},"outcome":{"status":"completed"},"run_id":"run-nomem","workload":{"cases":[{"case_id":"render-1024","measured_rounds":1,"warmup_rounds":1}],"task_type":"image_generation"}}
        """#
        let set = CommunityBenchmarkMetrics.metricSet(
            for: try Self.decode(json), workload: .image
        )
        #expect(!set.supporting.map(\.label).contains("Peak memory"))
        #expect(set.headline?.value == "4.6")
    }

    @Test("Total duration sums completed rounds as m:ss")
    func totalDuration() throws {
        // 6500 + 6500 + 26000 ms = 39 s
        #expect(CommunityBenchmarkMetrics.totalDuration(try Self.decode(Self.textRun)) == "0:39")
    }

    @Test("Protocol descriptions match the shipped fixed workloads")
    func protocolDescriptions() {
        #expect(
            CommunityBenchmarkMetrics.protocolDescription(for: .llm)
                .contains("1 warm-up + 5 measured rounds each")
        )
        #expect(
            CommunityBenchmarkMetrics.protocolDescription(for: .image)
                .contains("1024×1024")
        )
        #expect(
            CommunityBenchmarkMetrics.protocolDescription(for: .video)
                .contains("832×480, 81-frame")
        )
    }
}
