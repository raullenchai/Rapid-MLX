import Foundation
import Testing
@testable import Rapid

@Suite("Community read API seam and pulse")
struct CommunityBenchmarkDirectoryTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)
    private static let scope = CommunityBenchmarkScope(
        modelAlias: "qwen3.5-9b-4bit",
        workload: .llm,
        protocolID: "rapid-community-speed",
        protocolVersion: 2,
        macProfile: profile
    )

    @Test("The shipping directory reports every query as unavailable, never as zero")
    func unavailableDirectoryNeverReportsZero() async {
        let directory = UnavailableCommunityBenchmarkDirectory()

        let observations = await directory.observations(for: Self.scope)
        #expect(observations.value == nil)
        #expect(observations.unavailableReason == .notConfigured)
        // The critical consequence: no first-result language anywhere.
        #expect(
            !CommunityContributionBranch.select(from: observations)
                .allowsFirstReferenceLanguage
        )

        let table = await directory.table(
            macProfile: Self.profile, workload: .llm, metric: .generationSpeed
        )
        #expect(table.value == nil)
        let gaps = await directory.coverageGaps(macProfile: Self.profile)
        #expect(gaps.value == nil)
        let pulse = await directory.pulse()
        #expect(pulse.value == nil)
    }

    @Test("A static directory answers the scope it was given and nothing else")
    func staticDirectoryIsScoped() async {
        let other = CommunityBenchmarkScope(
            modelAlias: "gemma-4-12b-4bit",
            workload: .llm,
            protocolID: "rapid-community-speed",
        protocolVersion: 2,
            macProfile: Self.profile
        )
        let directory = StaticCommunityBenchmarkDirectory(
            summaries: [Self.scope: CommunityObservationSummary(observationCount: 7, median: 25.9)]
        )
        #expect(await directory.observations(for: Self.scope).value?.observationCount == 7)
        // A different model in the same profile is a different scope, and an
        // unanswered scope is unavailable — not zero.
        let miss = await directory.observations(for: other)
        #expect(miss.value == nil)
        #expect(miss.unavailableReason != nil)
    }

    @Test("The same model on a different Mac profile is a different scope")
    func macProfileIsPartOfTheScope() async {
        let bigger = CommunityBenchmarkScope(
            modelAlias: Self.scope.modelAlias,
            workload: Self.scope.workload,
            protocolID: Self.scope.protocolID,
            protocolVersion: Self.scope.protocolVersion,
            macProfile: CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 36)
        )
        let directory = StaticCommunityBenchmarkDirectory(
            summaries: [Self.scope: CommunityObservationSummary(observationCount: 7)]
        )
        #expect(await directory.observations(for: bigger).value == nil)
        #expect(Self.scope != bigger)
    }

    @Test("Scope descriptions name model, workload, and Mac profile")
    func scopeDescriptionIsComplete() {
        let description = Self.scope.scopeDescription
        #expect(description.contains("qwen3.5-9b-4bit"))
        #expect(description.contains("LLM"))
        #expect(description.contains("Apple M3 Pro · 18 GB"))
    }

    @Test("Loading and offline are distinguishable from each other and from data")
    func dataStateDistinctions() async {
        var directory = StaticCommunityBenchmarkDirectory(
            summaries: [Self.scope: CommunityObservationSummary(observationCount: 4)]
        )
        directory.isLoadingForever = true
        #expect(await directory.observations(for: Self.scope).isLoading)

        directory.isLoadingForever = false
        directory.forcedState = .offline
        #expect(await directory.observations(for: Self.scope).unavailableReason == .offline)
    }

    // MARK: - Publication accounting

    @Test("Publishing increments the count and drops statistics the client cannot recompute")
    func incrementAfterPublishing() {
        let before = CommunityObservationSummary(
            observationCount: 7,
            median: 25.9,
            observedMinimum: 24.6,
            observedMaximum: 26.9,
            unit: "tok/s",
            includesYours: false
        )
        let after = before.incrementedAfterPublishing()
        #expect(after.observationCount == 8)
        #expect(after.includesYours)
        #expect(after.unit == "tok/s")
        // A median over a population the client does not hold would be invented.
        #expect(after.median == nil)
        #expect(after.observedMinimum == nil)
        #expect(after.observedMaximum == nil)
    }

    @Test("A first publication moves the count from zero to one")
    func firstPublicationCount() {
        let after = CommunityObservationSummary(observationCount: 0).incrementedAfterPublishing()
        #expect(after.observationCount == 1)
        #expect(after.includesYours)
        #expect(
            CommunityContributionBranch.select(from: .ready(after))
                == .strengthen(observationCount: 1)
        )
    }

    // MARK: - Pulse cap

    /// Real identities, so the portraits are the ones the website shows.
    private static func identities(_ count: Int) -> [CommunityBenchmarkContributor] {
        (0..<count).map {
            CommunityBenchmarkContributor(
                name: "swift-quiet-noun\($0)", tag: String(format: "%03x", $0)
            )
        }
    }

    @Test("The avatar cluster renders a fixed number of portraits at any community size")
    func avatarClusterIsCapped() {
        let people = Self.identities(40)
        for contributors in [24, 240, 24_000] {
            let pulse = CommunityPulse(
                contributors: people,
                contributorCount: contributors,
                publishedRunCount: contributors * 3,
                modelCount: 17,
                lastContributionAt: nil
            )
            // Always five portraits, whatever the service sends or the
            // community grows to, so the band's width never changes.
            #expect(pulse.renderedContributors.count == 5)
            #expect(pulse.overflowCount == contributors - 5)
        }
    }

    @Test("Rendered portraits come from real slugs, so they match the website")
    func portraitsUseRealSlugs() {
        let pulse = CommunityPulse(
            contributors: [
                CommunityBenchmarkContributor(name: "swift-otter", tag: "4417"),
                CommunityBenchmarkContributor(name: "modest-slate-wombat", tag: "545"),
            ],
            contributorCount: 2,
            publishedRunCount: 6,
            modelCount: 2,
            lastContributionAt: nil
        )
        #expect(pulse.renderedContributors.map(\.slug) == [
            "swift-otter-4417", "modest-slate-wombat-545",
        ])
        #expect(
            pulse.renderedContributors.map { CommunityContributorAvatar.plate(for: $0) }
                == [12, 2]
        )
    }

    @Test("Overflow labels abbreviate so the fixed-width tile cannot be outgrown")
    func overflowLabelAbbreviates() {
        func label(_ contributors: Int) -> String? {
            CommunityPulse(
                contributors: Self.identities(5),
                contributorCount: contributors,
                publishedRunCount: 0,
                modelCount: 0,
                lastContributionAt: nil
            ).overflowLabel
        }
        #expect(label(24) == "+19")
        #expect(label(240) == "+235")
        // One decimal below ten thousand, none above, so the label stays
        // short: 23,995 remaining reads "+24k", not "+23995".
        #expect(label(5_500) == "+5.5k")
        #expect(label(24_000) == "+24k")
        #expect(label(2_400_000) == "+2.4M")
        // Whatever the size, the label fits the fixed-width overflow tile.
        for contributors in [24, 240, 5_500, 24_000, 2_400_000, 900_000_000] {
            #expect((label(contributors) ?? "").count <= 7)
        }
        // Five contributors exactly fill the rendered tiles: no overflow chip.
        #expect(label(5) == nil)
        #expect(label(3) == nil)
    }

    @Test("A service that sends fewer seeds than contributors still overflows correctly")
    func overflowCountsFromContributorTotal() {
        let pulse = CommunityPulse(
            contributors: Self.identities(2),
            contributorCount: 24_000,
            publishedRunCount: 90_000,
            modelCount: 210,
            lastContributionAt: nil
        )
        #expect(pulse.renderedContributors.count == 2)
        #expect(pulse.overflowCount == 23_998)
    }

    // MARK: - Workload mapping

    @Test("Model tasks map onto the three registered benchmark workloads")
    func workloadMapping() {
        #expect(CommunityWorkload(task: .imageGeneration) == .image)
        #expect(CommunityWorkload(task: .videoGeneration) == .video)
        #expect(CommunityWorkload(task: .textGeneration) == .llm)
        #expect(CommunityMetric.primary(for: .llm) == .generationSpeed)
        #expect(CommunityMetric.primary(for: .image) == .renderTime)
        #expect(CommunityMetric.primary(for: .video) == .videoTime)
    }
}
