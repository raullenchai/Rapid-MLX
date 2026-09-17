import AppKit
import Foundation
import SwiftUI
import Testing
@testable import Rapid

/// Layout regressions for the redesigned Community Benchmark surfaces.
///
/// Rendered at the two widths the design was reviewed at — 1440 × 900 and the
/// 900 × 600 narrow window — because the failures this module kept shipping
/// were layout failures rather than logic ones: a facts row running past its
/// card, a picker growing off screen, a side panel pushed off the trailing
/// edge at a narrow width.
///
/// Deliberately a focused set rather than one baseline per screen. Each PNG is
/// a full-page @2x bitmap, so the directory grows fast; these seven cover the
/// distinct layout risks — narrow stacking, sheet clamping, the row budget, a
/// `Divider` that used to inflate its row, the two-column information
/// architecture, the shipping unavailable state, and dark mode. Copy and
/// branch behaviour are covered far more cheaply by the value tests.
///
/// The detail pane is what these views occupy, so the sizes below subtract the
/// 200pt sidebar from the window width.
///
/// Gated behind ``.uiSnapshot`` like the other pixel suites: these bitmaps are
/// host- and OS-specific, so an ordinary `swift test` must not fail on a
/// machine whose font rendering differs. Run with
/// `RAPID_UI_SNAPSHOT_TESTS=1`.
@MainActor
@Suite("Community Benchmark layout", .serialized, .uiSnapshot)
struct CommunityBenchmarkLayoutSnapshotTests {
    private static let desktop = CGSize(width: 1_240, height: 870)
    private static let narrow = CGSize(width: 700, height: 570)

    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)

    private static func model(
        alias: String,
        cached: Bool,
        task: ModelTask,
        memory: Int?,
        fit: String = "fits"
    ) -> CommunityBenchmarkModel {
        CommunityBenchmarkModel(
            entry: ModelEntry(
                alias: alias,
                hfRepo: "mlx-community/\(alias)",
                sizeOnDisk: cached ? "6.5 GB" : nil,
                cached: cached,
                taskTypes: task == .imageGeneration ? [.imageGeneration] : [.textGeneration],
                operationModes: task == .imageGeneration ? [.textToImage] : [.chat]
            ),
            task: task,
            protocolName: task == .imageGeneration
                ? "Rapid Image Speed v1"
                : "Rapid Community Speed v2",
            isFocus: true,
            estimatedMemoryGib: memory,
            memoryFit: fit
        )
    }

    private static func scope(
        _ alias: String,
        _ workload: CommunityWorkload
    ) -> CommunityBenchmarkScope {
        CommunityBenchmarkScope(
            modelAlias: alias,
            workload: workload,
            protocolID: workload == .image
                ? "rapid-image-speed"
                : "rapid-community-speed",
            protocolVersion: workload == .image ? 1 : 2,
            macProfile: profile
        )
    }

    private static func page<Content: View>(
        _ size: CGSize,
        @ViewBuilder content: () -> Content
    ) -> some View {
        content()
            .padding(size.width < 880 ? 24 : 40)
            .frame(width: size.width, height: size.height, alignment: .topLeading)
            .background(RapidTheme.surfaceCanvas)
    }

    // MARK: - Ready

    private static func readyView(
        branch: CommunityContributionBranch,
        alias: String = "qwen3.5-9b-4bit",
        workload: CommunityWorkload = .llm,
        cached: Bool = true,
        isNarrow: Bool = false
    ) -> some View {
        CommunityBenchmarkReadyView(
            model: model(
                alias: alias,
                cached: cached,
                task: workload == .image ? .imageGeneration : .textGeneration,
                memory: 6
            ),
            scope: scope(alias, workload),
            branch: branch,
            isRunEnabled: true,
            serverImpactNote:
                "Chat and Images pause while \(alias) is measured, then your model reloads automatically.",
            isNarrow: isNarrow,
            onRun: {},
            onChangeModel: {},
            onShowTestMethod: {}
        )
    }

    /// The regression this pins: at a 900pt window the fixed-width "What this
    /// measures" panel used to be pushed off the trailing edge and clipped.
    @Test("Ready — a narrow window stacks instead of clipping the side panel")
    func readyNarrow() {
        assertSnapshot(
            of: Self.page(Self.narrow) {
                Self.readyView(branch: .strengthen(observationCount: 7), isNarrow: true)
            },
            size: Self.narrow,
            name: "community-benchmark-ready-narrow"
        )
    }

    @Test("Ready — dark appearance")
    func readyDark() {
        assertSnapshot(
            of: Self.page(Self.desktop) {
                Self.readyView(branch: .strengthen(observationCount: 7))
            },
            size: Self.desktop,
            name: "community-benchmark-ready-strengthen-dark",
            appearance: .darkAqua
        )
    }

    // MARK: - Model picker

    private static var pickerModels: [CommunityBenchmarkModel] {
        [
            model(alias: "z-image-turbo", cached: false, task: .imageGeneration, memory: 4),
            model(alias: "qwen3.5-9b-4bit", cached: true, task: .textGeneration, memory: 6),
            model(alias: "gemma-4-12b-4bit", cached: true, task: .textGeneration, memory: 7),
            model(alias: "qwen3.5-4b-4bit", cached: true, task: .textGeneration, memory: 5),
            model(
                alias: "mistral-small-3.4-24b-instruct-2506-uncensored-4bit",
                cached: false, task: .textGeneration, memory: 13
            ),
            model(alias: "phi-5-mini-4bit", cached: true, task: .textGeneration, memory: 2),
            model(alias: "gemma-4-2b-4bit", cached: true, task: .textGeneration, memory: 2),
            model(
                alias: "llama-4.2-3b-instruct-4bit", cached: false,
                task: .textGeneration, memory: 2
            ),
            model(
                alias: "qwen3.8-27b-4bit", cached: false, task: .textGeneration,
                memory: 20, fit: "does_not_fit"
            ),
            model(
                alias: "gemma-4-27b-4bit", cached: false, task: .textGeneration,
                memory: 21, fit: "does_not_fit"
            ),
        ]
    }

    private static var pickerCoverage: CommunityDataState<[CommunityCoverageGap]> {
        .ready([
            CommunityCoverageGap(
                modelAlias: "z-image-turbo", workload: .image, observationCount: 0,
                fitsThisMac: true, isDownloaded: false, downloadSizeGB: 3.6,
                requiredMemoryGB: nil
            ),
            CommunityCoverageGap(
                modelAlias: "qwen3.5-9b-4bit", workload: .llm, observationCount: 7,
                fitsThisMac: true, isDownloaded: true, downloadSizeGB: nil,
                requiredMemoryGB: nil
            ),
            CommunityCoverageGap(
                modelAlias: "gemma-4-12b-4bit", workload: .llm, observationCount: 3,
                fitsThisMac: true, isDownloaded: true, downloadSizeGB: nil,
                requiredMemoryGB: nil
            ),
        ])
    }

    private static func pickerSheet(size: CGSize, query: String) -> some View {
        StatefulPickerHarness(
            models: pickerModels,
            coverage: pickerCoverage,
            initialQuery: query,
            containerSize: size
        )
        .frame(width: size.width, height: size.height)
        .background(RapidTheme.surfaceCanvas)
    }

    /// Pins the row budget, the sticky group headings, long-alias truncation,
    /// and the fixed header and footer.
    @Test("Model picker — unfiltered, desktop")
    func pickerDesktop() {
        assertSnapshot(
            of: Self.pickerSheet(size: Self.desktop, query: ""),
            size: Self.desktop,
            name: "community-benchmark-picker-desktop"
        )
    }

    /// Pins the viewport clamp: the sheet shrinks into a short window with a
    /// margin instead of overflowing it.
    @Test("Model picker — clamped into a 900 window, still scrolling")
    func pickerNarrow() {
        assertSnapshot(
            of: Self.pickerSheet(size: Self.narrow, query: ""),
            size: Self.narrow,
            name: "community-benchmark-picker-narrow"
        )
    }

    // MARK: - Result

    private static let textRunJSON = #"""
    {"completed_at":"2026-09-06T04:37:42Z","execution":{"config_digest":"sha256:069529b6e4cc5059","runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"}},"machine":{"os":{"version":"15.6.1"},"profile":{"chip":"Apple M3 Pro","cpu_cores":12,"gpu_cores":18,"memory_gib":18}},"measurements":[
    {"case_id":"pp512-tg128","completed":true,"decode_duration_ms":5000,"output_tokens":129,"peak_active_memory_mib":6875,"round_index":1,"total_duration_ms":6500,"ttft_ms":1490},
    {"case_id":"pp2048-tg512","completed":true,"decode_duration_ms":20000,"output_tokens":513,"peak_active_memory_mib":6875,"round_index":1,"total_duration_ms":26000,"ttft_ms":5810}
    ],"model":{"components":[{"source":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"}}]},"outcome":{"status":"completed"},"run_id":"run-text","workload":{"cases":[{"case_id":"pp512-tg128","measured_rounds":5,"target_output_tokens":128,"target_prompt_tokens":512,"warmup_rounds":1},{"case_id":"pp2048-tg512","measured_rounds":5,"target_output_tokens":512,"target_prompt_tokens":2048,"warmup_rounds":1}],"task_type":"text_generation"}}
    """#

    private static func decode(_ json: String) -> CommunityBenchmarkResult {
        // Force-decoded: a literal fixture in the test file, so a failure here
        // is a broken test rather than a runtime path.
        try! JSONDecoder().decode(CommunityBenchmarkResult.self, from: Data(json.utf8))
    }

    /// Pins the metrics row height. The column separator is a `Divider`, which
    /// expands to whatever height the parent offers; without `fixedSize` the
    /// row inflated to fill the window and left a dead band above the
    /// comparison area.
    @Test("Result — metrics, comparison and publish invitation stack tightly")
    func resultWithComparison() {
        assertSnapshot(
            of: Self.page(Self.desktop) {
                CommunityBenchmarkResultView(
                    result: Self.decode(Self.textRunJSON),
                    modelAlias: "qwen3.5-9b-4bit",
                    scope: Self.scope("qwen3.5-9b-4bit", .llm),
                    branch: .strengthen(observationCount: 7),
                    observations: .ready(
                        CommunityObservationSummary(
                            observationCount: 7,
                            median: 25.9,
                            observedMinimum: 24.6,
                            observedMaximum: 26.9,
                            unit: "tok/s"
                        )
                    ),
                    receipt: nil,
                    isPublishing: false,
                    onPublish: {}, onRunAgain: {}, onBenchmarkAnother: {}
                )
            },
            size: Self.desktop,
            name: "community-benchmark-result-strengthen"
        )
    }

    /// The screen a user lands on after finishing a run they have not
    /// published. It must offer a way off itself; the previous build rendered
    /// `Benchmark another model` only for a published result.
    @Test("Result — unpublished, with both exits and publish")
    func resultUnpublishedExits() {
        assertSnapshot(
            of: Self.page(Self.desktop) {
                CommunityBenchmarkResultView(
                    result: Self.decode(Self.textRunJSON),
                    modelAlias: "qwen3.5-9b-4bit",
                    scope: Self.scope("qwen3.5-9b-4bit", .llm),
                    branch: .firstReference,
                    observations: .ready(CommunityObservationSummary(observationCount: 0)),
                    receipt: nil,
                    isPublishing: false,
                    onPublish: {}, onRunAgain: {}, onBenchmarkAnother: {}
                )
            },
            size: Self.desktop,
            name: "community-benchmark-result-unpublished-exits"
        )
    }

    /// Taller than the other narrow baselines on purpose: the card is what is
    /// under test, and clipping it at the window height would hide the action
    /// row this test exists to check.
    private static let narrowTall = CGSize(width: 700, height: 900)

    @Test("Result — narrow window keeps every action reachable")
    func resultNarrowExits() {
        assertSnapshot(
            of: Self.page(Self.narrowTall) {
                CommunityBenchmarkResultView(
                    result: Self.decode(Self.textRunJSON),
                    modelAlias: "qwen3.5-9b-4bit",
                    scope: Self.scope("qwen3.5-9b-4bit", .llm),
                    branch: .strengthen(observationCount: 7, isAtLeast: true),
                    observations: .ready(
                        CommunityObservationSummary(
                            observationCount: 7, median: 25.9, observedMinimum: 24.6,
                            observedMaximum: 26.9, unit: "tok/s", isBounded: true
                        )
                    ),
                    receipt: nil,
                    isPublishing: false,
                    isNarrow: true,
                    onPublish: {}, onRunAgain: {}, onBenchmarkAnother: {}
                )
            },
            size: Self.narrowTall,
            name: "community-benchmark-result-narrow-exits"
        )
    }

    // MARK: - Running

    /// Mid-run, eight passes in: the stage stepper, the pass count, the ETA
    /// and the newest measured rate are all live state, not a clock.
    @Test("Running — stage stepper, pass count and latest measurement")
    func runningMidRun() {
        assertSnapshot(
            of: Self.page(Self.desktop) {
                CommunityBenchmarkRunningView(
                    model: Self.model(alias: "qwen3.5-9b-4bit", cached: true, task: .textGeneration, memory: 6),
                    scope: Self.scope("qwen3.5-9b-4bit", .llm),
                    // `Date()`, not a fixed epoch: the elapsed clock is a real
                    // `TimelineView` reading wall time, so a pinned start date
                    // would bake an ever-growing number into the baseline.
                    runStartedAt: Date(),
                    progress: CommunityRunProgress(
                        stage: .longReplies,
                        passesComplete: 8,
                        totalPasses: 12,
                        statusLine: "pp2048-tg512 round 2/5 18.9 tok/s",
                        latestMeasurement: .init(value: "18.9 tok/s", passNumber: 8),
                        timeLeft: "~2:40 left"
                    ),
                    plan: CommunityRunPlan.assumed(for: .textGeneration),
                    onStop: {}
                )
            },
            size: Self.desktop,
            name: "community-benchmark-running-mid"
        )
    }

    /// Before the first completed pass there is no determinate progress to
    /// show and no ETA to quote.
    @Test("Running — getting ready makes no claim it cannot support")
    func runningGettingReady() {
        assertSnapshot(
            of: Self.page(Self.desktop) {
                CommunityBenchmarkRunningView(
                    model: Self.model(alias: "qwen3.5-9b-4bit", cached: true, task: .textGeneration, memory: 6),
                    scope: Self.scope("qwen3.5-9b-4bit", .llm),
                    // `Date()`, not a fixed epoch: the elapsed clock is a real
                    // `TimelineView` reading wall time, so a pinned start date
                    // would bake an ever-growing number into the baseline.
                    runStartedAt: Date(),
                    progress: CommunityRunProgress(
                        stage: .gettingReady,
                        passesComplete: 0,
                        totalPasses: 12,
                        statusLine: "Loading mlx-community/Qwen3.5-9B-4bit…"
                    ),
                    plan: CommunityRunPlan.assumed(for: .textGeneration),
                    onStop: {}
                )
            },
            size: Self.desktop,
            name: "community-benchmark-running-getting-ready"
        )
    }

    @Test("Running — a narrow window keeps the stepper legible")
    func runningNarrow() {
        assertSnapshot(
            of: Self.page(Self.narrow) {
                CommunityBenchmarkRunningView(
                    model: Self.model(alias: "qwen3.5-9b-4bit", cached: true, task: .textGeneration, memory: 6),
                    scope: Self.scope("qwen3.5-9b-4bit", .llm),
                    // `Date()`, not a fixed epoch: the elapsed clock is a real
                    // `TimelineView` reading wall time, so a pinned start date
                    // would bake an ever-growing number into the baseline.
                    runStartedAt: Date(),
                    progress: CommunityRunProgress(
                        stage: .shortReplies,
                        passesComplete: 3,
                        totalPasses: 12,
                        statusLine: "pp512-tg128 round 2/5 46.2 tok/s",
                        latestMeasurement: .init(value: "46.2 tok/s", passNumber: 3),
                        timeLeft: "~3:10 left"
                    ),
                    plan: CommunityRunPlan.assumed(for: .textGeneration),
                    isNarrow: true,
                    onStop: {}
                )
            },
            size: Self.narrow,
            name: "community-benchmark-running-narrow"
        )
    }

    // MARK: - Share confirmation

    /// The consent dialog for a warm-cache run, whose submission is narrowed
    /// before it is sent. The SHARED column names only what the payload really
    /// carries, and the withheld facts are shown rather than buried.
    @Test("Share confirmation — the disclosure matches the payload")
    func shareConfirmationDisclosure() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/community-share-preview.json")
        let preview = try CommunityBenchmarkCommand.decodeSharePreview(
            try Data(contentsOf: url), runID: "e1390322-5f48-41cf-bd37-b24391953baf"
        )
        assertSnapshot(
            of: CommunityBenchmarkShareConfirmationSheet(
                preview: preview,
                knownContributor: CommunityBenchmarkContributor(
                    name: "swift-otter", tag: "4417", slug: "swift-otter-4417"
                ),
                isPublishing: false,
                onCancel: {}, onPublish: {}
            )
            .frame(width: 620)
            .background(RapidTheme.surfaceCanvas),
            size: CGSize(width: 620, height: 900),
            name: "community-benchmark-share-confirmation"
        )
    }

    // MARK: - Community

    private static var communityRows: [CommunityObservationRow] {
        [
            ("gemma-4-12b-4bit", 3, 19.4, 18.9, 20.1, false),
            ("gemma-4-e4b-4bit", 6, 31.2, 29.8, 32.6, false),
            ("qwen3.8-27b-4bit", 1, nil, 9.8, 9.8, false),
            ("qwen3.5-4b-4bit", 9, 37.6, 35.1, 39.8, true),
            ("qwen3.5-9b-4bit", 8, 25.9, 24.6, 26.9, true),
        ].map { alias, count, median, low, high, yours in
            CommunityObservationRow(
                modelAlias: alias,
                workload: .llm,
                summary: CommunityObservationSummary(
                    observationCount: count,
                    median: median,
                    observedMinimum: low,
                    observedMaximum: high,
                    unit: "tok/s",
                    includesYours: yours
                )
            )
        }
    }

    private static var communityGaps: [CommunityCoverageGap] {
        [
            CommunityCoverageGap(
                modelAlias: "z-image-turbo", workload: .image, observationCount: 0,
                fitsThisMac: true, isDownloaded: false, downloadSizeGB: 3.6,
                requiredMemoryGB: nil
            ),
            CommunityCoverageGap(
                modelAlias: "gemma-4-12b-4bit", workload: .llm, observationCount: 3,
                fitsThisMac: true, isDownloaded: true, downloadSizeGB: nil,
                requiredMemoryGB: nil
            ),
            CommunityCoverageGap(
                modelAlias: "qwen3.8-27b-4bit", workload: .llm, observationCount: 1,
                fitsThisMac: false, isDownloaded: false, downloadSizeGB: nil,
                requiredMemoryGB: 20
            ),
        ]
    }

    private static var readyPulse: CommunityDataState<CommunityPulse> {
        .ready(
            CommunityPulse(
                contributors: [
                    ("swift-otter", "4417"), ("modest-slate-wombat", "545"),
                    ("sleepy-alpine-okapi", "e22"), ("brisk-amber-lynx", "1a4"),
                    ("quiet-harbor-ibex", "9c0"), ("lucid-marble-tapir", "77b"),
                ].map { CommunityBenchmarkContributor(name: $0.0, tag: $0.1) },
                contributorCount: 24,
                publishedRunCount: 63,
                modelCount: 17,
                lastContributionAt: Date(timeIntervalSince1970: 1_780_000_000)
            )
        )
    }

    private static func communityView(
        size: CGSize,
        pulse: CommunityDataState<CommunityPulse>,
        table: CommunityDataState<[CommunityObservationRow]>,
        coverage: CommunityDataState<[CommunityCoverageGap]>
    ) -> some View {
        StatefulCommunityHarness(
            profile: profile,
            pulse: pulse,
            table: table,
            coverage: coverage,
            isNarrow: size.width < 880
        )
    }

    /// Pins the information architecture: a compact left-aligned pulse band,
    /// observations as the primary left area, contribution on the right, and
    /// the leaderboard link last.
    @Test("Community — desktop, observations primary and leaderboard last")
    func communityDesktop() {
        assertSnapshot(
            of: Self.page(Self.desktop) {
                Self.communityView(
                    size: Self.desktop,
                    pulse: Self.readyPulse,
                    table: .ready(Self.communityRows),
                    coverage: .ready(Self.communityGaps)
                )
            },
            size: Self.desktop,
            name: "community-benchmark-community-desktop"
        )
    }

    /// Offline, or the community service not configured. Nothing on this
    /// screen may imply an observation count — a missing answer is not zero.
    @Test("Community — an unavailable service degrades without making any claim")
    func communityUnavailable() {
        assertSnapshot(
            of: Self.page(Self.desktop) {
                Self.communityView(
                    size: Self.desktop,
                    pulse: .unavailable(.notConfigured),
                    table: .unavailable(.notConfigured),
                    coverage: .unavailable(.notConfigured)
                )
            },
            size: Self.desktop,
            name: "community-benchmark-community-unavailable"
        )
    }
}

// MARK: - Harnesses

/// Wraps the picker so its `@Binding`s have real storage in a snapshot.
private struct StatefulPickerHarness: View {
    let models: [CommunityBenchmarkModel]
    let coverage: CommunityDataState<[CommunityCoverageGap]>
    let initialQuery: String
    let containerSize: CGSize

    @State private var query: String = ""
    @State private var selection = "z-image-turbo"

    var body: some View {
        CommunityBenchmarkPickerSheet(
            listing: CommunityBenchmarkPicker.listing(
                models: models, coverage: coverage, query: query
            ),
            containerSize: containerSize,
            query: $query,
            selectedAlias: $selection,
            onCancel: {},
            onChoose: { _ in }
        )
        .onAppear { query = initialQuery }
    }
}

/// Wraps the Community view so its workload `@Binding` has storage.
private struct StatefulCommunityHarness: View {
    let profile: CommunityMacProfile
    let pulse: CommunityDataState<CommunityPulse>
    let table: CommunityDataState<[CommunityObservationRow]>
    let coverage: CommunityDataState<[CommunityCoverageGap]>
    let isNarrow: Bool

    @State private var workload: CommunityWorkload = .llm

    var body: some View {
        CommunityBenchmarkCommunityView(
            macProfile: profile,
            pulse: pulse,
            table: table,
            coverage: coverage,
            workload: $workload,
            metric: .primary(for: workload),
            isNarrow: isNarrow,
            leaderboardURL: URL(string: "https://rapidmlx.com/leaderboard")!,
            onRunModel: { _ in }
        )
    }
}
