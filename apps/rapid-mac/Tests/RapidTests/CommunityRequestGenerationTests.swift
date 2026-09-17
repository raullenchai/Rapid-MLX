import Foundation
import Testing
@testable import Rapid

/// Stale community reads must never overwrite fresher ones — and invalidating
/// one read must never strand another.
///
/// Two failures are covered here. The first is the obvious race: the user
/// selects model A, A's request is slow, they switch to B, B's answer renders,
/// and then A's late answer lands and paints A's count beside B's name. If A
/// had no observations that is a first-reference banner for an unselected
/// model.
///
/// The second is what a *single shared* token caused: bumping it because the
/// model changed also invalidated the in-flight coverage, pulse and
/// contributor-totals requests, none of which depend on the model — and
/// nothing restarted them, so those panels sat on `.loading` forever.
@Suite("Out-of-order community responses")
struct CommunityRequestGenerationTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)

    private static func scope(_ alias: String) -> CommunityBenchmarkScope {
        CommunityBenchmarkScope(
            modelAlias: alias, workload: .llm,
            protocolID: "rapid-community-speed", protocolVersion: 2, macProfile: profile
        )
    }

    // MARK: - Independence

    @Test("Beginning one read never invalidates a different one")
    func kindsAreIndependent() {
        var generations = CommunityRequestGenerations()
        let pulse = generations.begin(.pulse)
        let coverage = generations.begin(.coverage)
        let totals = generations.begin(.contributorTotals)

        // The user changes model: only observations may be invalidated.
        generations.begin(.observations)

        #expect(generations.isCurrent(.pulse, pulse))
        #expect(generations.isCurrent(.coverage, coverage))
        #expect(generations.isCurrent(.contributorTotals, totals))
    }

    @Test("A workload change leaves the model-scoped read alone")
    func workloadChangeDoesNotTouchObservations() {
        var generations = CommunityRequestGenerations()
        let observations = generations.begin(.observations)
        generations.begin(.table)
        #expect(generations.isCurrent(.observations, observations))
    }

    @Test("Beginning a read invalidates only the previous request of that kind")
    func sameKindIsInvalidated() {
        var generations = CommunityRequestGenerations()
        let first = generations.begin(.observations)
        let second = generations.begin(.observations)
        #expect(!generations.isCurrent(.observations, first))
        #expect(generations.isCurrent(.observations, second))
    }

    @Test("Tokens are monotonic per kind, so one is never accidentally revived")
    func tokensAreMonotonic() {
        var generations = CommunityRequestGenerations()
        for kind in CommunityReadKind.allCases {
            var seen: Set<Int> = [generations.current(kind)]
            for _ in 0..<50 {
                let token = generations.begin(kind)
                #expect(!seen.contains(token))
                seen.insert(token)
            }
        }
    }

    /// Every invalidation has a restart attached by construction: the only way
    /// to invalidate is `begin`, which returns the token the replacement
    /// carries. There is no API that invalidates without producing a successor.
    @Test("Invalidation always yields the token of a restarted request")
    func invalidationAlwaysProducesASuccessor() {
        var generations = CommunityRequestGenerations()
        for kind in CommunityReadKind.allCases {
            let old = generations.current(kind)
            let new = generations.begin(kind)
            #expect(!generations.isCurrent(kind, old))
            #expect(generations.isCurrent(kind, new))
        }
    }

    // MARK: - Out-of-order application

    /// A tiny model of the view's apply rule, so the race can be replayed
    /// deterministically instead of depending on scheduler luck.
    private struct Screen {
        var generations = CommunityRequestGenerations()
        var observations: CommunityDataState<CommunityObservationSummary> = .loading
        var table: CommunityDataState<[CommunityObservationRow]> = .loading
        var coverage: CommunityDataState<[CommunityCoverageGap]> = .loading
        var pulse: CommunityDataState<CommunityPulse> = .loading

        mutating func applyObservations(
            _ value: CommunityDataState<CommunityObservationSummary>, token: Int
        ) {
            guard generations.isCurrent(.observations, token) else { return }
            observations = value
        }

        mutating func applyTable(
            _ value: CommunityDataState<[CommunityObservationRow]>, token: Int
        ) {
            guard generations.isCurrent(.table, token) else { return }
            table = value
        }

        mutating func applyCoverage(
            _ value: CommunityDataState<[CommunityCoverageGap]>, token: Int
        ) {
            guard generations.isCurrent(.coverage, token) else { return }
            coverage = value
        }

        mutating func applyPulse(_ value: CommunityDataState<CommunityPulse>, token: Int) {
            guard generations.isCurrent(.pulse, token) else { return }
            pulse = value
        }
    }

    @Test("A slow first answer cannot overwrite a fast second one")
    func slowFirstAnswerDoesNotOverwrite() async {
        let directory = StaticCommunityBenchmarkDirectory(
            summaries: [
                // The model the user navigated away from has no observations —
                // the value that would light up a first-reference banner.
                Self.scope("model-a"): CommunityObservationSummary(observationCount: 0),
                Self.scope("model-b"): CommunityObservationSummary(observationCount: 9),
            ]
        )
        var screen = Screen()

        let tokenA = screen.generations.begin(.observations)   // model A requested
        let tokenB = screen.generations.begin(.observations)   // user switches to B

        screen.applyObservations(
            await directory.observations(for: Self.scope("model-b")), token: tokenB
        )
        #expect(screen.observations.value?.observationCount == 9)

        // A's answer lands late. It is stale and must be dropped.
        screen.applyObservations(
            await directory.observations(for: Self.scope("model-a")), token: tokenA
        )
        #expect(
            screen.observations.value?.observationCount == 9,
            "a stale answer overwrote the current one"
        )
        #expect(
            !CommunityContributionBranch.select(from: screen.observations)
                .allowsFirstReferenceLanguage,
            "a stale zero produced a first-reference claim for an unselected model"
        )
    }

    /// Rapid model AND workload changes while the initial reads are still in
    /// flight — the real sequence a user produces by clicking through the
    /// picker and the workload tabs quickly.
    @Test("Rapid model and workload changes leave every panel on fresh data")
    func rapidChangesWhileInitialReadsInFlight() async {
        let directory = StaticCommunityBenchmarkDirectory(
            summaries: [
                Self.scope("model-a"): CommunityObservationSummary(observationCount: 0),
                Self.scope("model-b"): CommunityObservationSummary(observationCount: 4),
                Self.scope("model-c"): CommunityObservationSummary(observationCount: 11),
            ],
            rows: [
                CommunityObservationRow(
                    modelAlias: "qwen3.5-9b-4bit", workload: .llm,
                    summary: CommunityObservationSummary(observationCount: 8)
                ),
                CommunityObservationRow(
                    modelAlias: "z-image-turbo", workload: .image,
                    summary: CommunityObservationSummary(observationCount: 2)
                ),
            ],
            gaps: [
                CommunityCoverageGap(
                    modelAlias: "z-image-turbo", workload: .image, observationCount: 1,
                    fitsThisMac: true, isDownloaded: false, downloadSizeGB: 3.6,
                    requiredMemoryGB: nil
                )
            ],
            pulseValue: CommunityPulse(
                contributors: [], contributorCount: 3, publishedRunCount: 9,
                modelCount: 2, lastContributionAt: nil
            )
        )

        var screen = Screen()

        // Initial page load: all four reads start.
        let obsA = screen.generations.begin(.observations)
        let tableLLM = screen.generations.begin(.table)
        let coverageToken = screen.generations.begin(.coverage)
        let pulseToken = screen.generations.begin(.pulse)

        // …and before any of them return, the user changes the model twice
        // and the workload once.
        _ = screen.generations.begin(.observations)            // → model B
        let obsC = screen.generations.begin(.observations)     // → model C
        let tableImage = screen.generations.begin(.table)      // → Image tab

        // Now everything lands, in the worst possible order.
        screen.applyObservations(
            await directory.observations(for: Self.scope("model-a")), token: obsA
        )
        screen.applyTable(
            await directory.table(macProfile: Self.profile, workload: .llm, metric: .generationSpeed),
            token: tableLLM
        )
        screen.applyCoverage(
            await directory.coverageGaps(macProfile: Self.profile), token: coverageToken
        )
        screen.applyPulse(await directory.pulse(), token: pulseToken)
        screen.applyObservations(
            await directory.observations(for: Self.scope("model-c")), token: obsC
        )
        screen.applyTable(
            await directory.table(macProfile: Self.profile, workload: .image, metric: .renderTime),
            token: tableImage
        )

        // The selected model's answer won, not the two it superseded.
        #expect(screen.observations.value?.observationCount == 11)
        #expect(
            !CommunityContributionBranch.select(from: screen.observations)
                .allowsFirstReferenceLanguage
        )
        // The selected workload's table won.
        #expect(screen.table.value?.map(\.modelAlias) == ["z-image-turbo"])
        // And the reads that never depended on either input still landed —
        // this is what a single shared token broke.
        #expect(screen.coverage.value?.count == 1)
        #expect(screen.pulse.value?.contributorCount == 3)
        #expect(!screen.coverage.isLoading)
        #expect(!screen.pulse.isLoading)
    }

    @Test("A workload change discards the previous workload's table")
    func workloadChangeDiscardsTable() async {
        let directory = StaticCommunityBenchmarkDirectory(
            rows: [
                CommunityObservationRow(
                    modelAlias: "qwen3.5-9b-4bit", workload: .llm,
                    summary: CommunityObservationSummary(observationCount: 8)
                ),
                CommunityObservationRow(
                    modelAlias: "z-image-turbo", workload: .image,
                    summary: CommunityObservationSummary(observationCount: 2)
                ),
            ]
        )
        var screen = Screen()
        let llmToken = screen.generations.begin(.table)
        let imageToken = screen.generations.begin(.table)

        screen.applyTable(
            await directory.table(macProfile: Self.profile, workload: .image, metric: .renderTime),
            token: imageToken
        )
        #expect(screen.table.value?.map(\.modelAlias) == ["z-image-turbo"])

        screen.applyTable(
            await directory.table(macProfile: Self.profile, workload: .llm, metric: .generationSpeed),
            token: llmToken
        )
        #expect(
            screen.table.value?.map(\.modelAlias) == ["z-image-turbo"],
            "the previous workload's table overwrote the selected one"
        )
    }
}
