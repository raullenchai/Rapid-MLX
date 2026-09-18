import Foundation
import Testing
@testable import Rapid

@Suite("Community Benchmark contribution branches")
struct CommunityBenchmarkBranchTests {
    private static let scope = CommunityBenchmarkScope(
        modelAlias: "z-image-turbo",
        workload: .image,
        protocolID: "rapid-image-speed",
        protocolVersion: 1,
        macProfile: CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)
    )

    // MARK: - Selection

    @Test("A known zero is the only state that selects the first-reference branch")
    func knownZeroSelectsFirstReference() {
        let branch = CommunityContributionBranch.select(
            from: .ready(CommunityObservationSummary(observationCount: 0))
        )
        #expect(branch == .firstReference)
        #expect(branch.allowsFirstReferenceLanguage)
        // Nothing to compare against, so no median/range may render.
        #expect(!branch.allowsComparisonStatistics)
    }

    @Test("A positive count selects the strengthen branch and permits comparisons")
    func positiveCountSelectsStrengthen() {
        let branch = CommunityContributionBranch.select(
            from: .ready(CommunityObservationSummary(observationCount: 7, median: 25.9))
        )
        #expect(branch == .strengthen(observationCount: 7))
        #expect(!branch.allowsFirstReferenceLanguage)
        #expect(branch.allowsComparisonStatistics)
        #expect(branch.observationCount == 7)
    }

    @Test("Loading, offline, failed, and unconfigured never infer a first result")
    func unknownStatesNeverClaimFirst() {
        let unknownStates: [CommunityDataState<CommunityObservationSummary>] = [
            .loading,
            .unavailable(.notConfigured),
            .unavailable(.offline),
            .unavailable(.failed("500")),
        ]
        for state in unknownStates {
            let branch = CommunityContributionBranch.select(from: state)
            #expect(!branch.allowsFirstReferenceLanguage, "\(state) claimed a first result")
            #expect(!branch.allowsComparisonStatistics, "\(state) rendered comparisons")
            #expect(branch.observationCount == nil)
        }
        #expect(
            CommunityContributionBranch.select(from: .loading)
                == .unknown(.loading)
        )
        #expect(
            CommunityContributionBranch.select(from: .unavailable(.offline))
                == .unknown(.unavailable(.offline))
        )
    }

    @Test("A negative count is invalid data, never \"you are first\"")
    func negativeCountsAreInvalidNotZero() {
        // Clamping a malformed count to zero would turn a service bug into a
        // confident "nobody has published this" — the one claim that is
        // hardest to take back once the user has acted on it.
        let branch = CommunityContributionBranch.select(
            from: .ready(CommunityObservationSummary(observationCount: -3))
        )
        #expect(branch == .unknown(.unavailable(.invalidCount)))
        #expect(!branch.allowsFirstReferenceLanguage)
        #expect(!branch.allowsComparisonStatistics)
        #expect(branch.observationCount == nil)
    }

    @Test("Only an exact, unbounded zero selects the first-reference branch")
    func onlyExactZeroIsFirstReference() {
        #expect(
            CommunityContributionBranch.select(
                from: .ready(CommunityObservationSummary(observationCount: 0))
            ) == .firstReference
        )
        // A bounded feed can prove results exist but never that none do, so a
        // zero arriving from one is incoherent and is not trusted.
        let bounded = CommunityContributionBranch.select(
            from: .ready(CommunityObservationSummary(observationCount: 0, isBounded: true))
        )
        #expect(bounded == .unknown(.unavailable(.boundedFeed)))
        #expect(!bounded.allowsFirstReferenceLanguage)
    }

    @Test("A bounded count reads as a floor, never as an exact total")
    func boundedCountsSayAtLeast() {
        let branch = CommunityContributionBranch.select(
            from: .ready(CommunityObservationSummary(observationCount: 7, isBounded: true))
        )
        #expect(branch == .strengthen(observationCount: 7, isAtLeast: true))
        #expect(branch.observationCountIsApproximate)
        #expect(branch.allowsComparisonStatistics)

        let ready = CommunityBenchmarkCopy.readyInvitation(branch: branch, scope: Self.scope)
        #expect(ready.body.contains("at least 7 published results exist"))

        // An exact count must NOT be hedged.
        let exact = CommunityContributionBranch.select(
            from: .ready(CommunityObservationSummary(observationCount: 7))
        )
        #expect(!exact.observationCountIsApproximate)
        #expect(
            !CommunityBenchmarkCopy.readyInvitation(branch: exact, scope: Self.scope)
                .body.contains("at least")
        )
    }

    @Test("A bounded count never numbers the next observation")
    func boundedPublishInvitationAvoidsOrdinals() {
        // "add observation 8" asserts that exactly 7 exist. From a bounded
        // feed the true total may be larger, so the ordinal is dropped.
        let bounded = CommunityBenchmarkCopy.publishInvitation(
            branch: .strengthen(observationCount: 7, isAtLeast: true), scope: Self.scope
        )
        #expect(bounded.headline.contains("another observation"))
        #expect(!bounded.headline.contains("observation 8"))

        let exact = CommunityBenchmarkCopy.publishInvitation(
            branch: .strengthen(observationCount: 7), scope: Self.scope
        )
        #expect(exact.headline.contains("observation 8"))
    }

    @Test("A bounded publication celebrates without quoting a new total")
    func boundedPublishedCelebration() {
        let celebration = CommunityBenchmarkCopy.publishedCelebration(
            branchBeforePublishing: .strengthen(observationCount: 7, isAtLeast: true),
            scope: Self.scope,
            observationCountAfterPublishing: 8,
            alreadyPublished: false
        )
        #expect(celebration.headline == "You made this comparison stronger")
        // The count was never exact, so "there are now 8" would be invented.
        #expect(!celebration.body.contains("8 published observations"))
    }

    // MARK: - Copy scoping

    @Test("Every first-reference sentence names the model and the Mac profile")
    func firstReferenceCopyIsScoped() {
        let ready = CommunityBenchmarkCopy.readyInvitation(
            branch: .firstReference, scope: Self.scope
        )
        #expect(ready.eyebrow == "FIRST RESULT NEEDED")
        #expect(ready.body.contains("z-image-turbo"))
        #expect(ready.body.contains("Image"))
        #expect(ready.body.contains("Apple M3 Pro · 18 GB"))

        let publish = CommunityBenchmarkCopy.publishInvitation(
            branch: .firstReference, scope: Self.scope
        )
        #expect(publish.headline == "Create the first public reference")
        #expect(publish.body.contains("z-image-turbo"))
        #expect(publish.body.contains("Image"))
        #expect(publish.body.contains("Apple M3 Pro · 18 GB"))

        let published = CommunityBenchmarkCopy.publishedCelebration(
            branchBeforePublishing: .firstReference,
            scope: Self.scope,
            observationCountAfterPublishing: 1,
            alreadyPublished: false
        )
        #expect(published.headline == "Published to Community Benchmark")
        #expect(!published.headline.localizedCaseInsensitiveContains("first"))
        #expect(!published.body.localizedCaseInsensitiveContains("first"))
        #expect(published.body.contains("z-image-turbo"))
        #expect(published.body.contains("Image"))
        #expect(published.body.contains("Apple M3 Pro · 18 GB"))
    }

    @Test("The strengthen branch states the existing count and the next observation")
    func strengthenCopyStatesCounts() {
        let ready = CommunityBenchmarkCopy.readyInvitation(
            branch: .strengthen(observationCount: 7), scope: Self.scope
        )
        #expect(ready.eyebrow == "COMMUNITY NEEDS ANOTHER RESULT")
        #expect(ready.headline == "Make this comparison stronger")
        #expect(ready.body.contains("7 published results exist"))

        let publish = CommunityBenchmarkCopy.publishInvitation(
            branch: .strengthen(observationCount: 7), scope: Self.scope
        )
        // Publishing the 8th observation when 7 already exist.
        #expect(publish.headline.contains("observation 8"))
    }

    @Test("A singular count reads as one result, not \"1 results\"")
    func singularCopy() {
        let ready = CommunityBenchmarkCopy.readyInvitation(
            branch: .strengthen(observationCount: 1), scope: Self.scope
        )
        #expect(ready.body.contains("1 published result exists"))
        let publish = CommunityBenchmarkCopy.publishInvitation(
            branch: .strengthen(observationCount: 1), scope: Self.scope
        )
        #expect(publish.body.contains("1 published result exists"))
    }

    @Test("An unknown branch never says first, and never promises a comparison")
    func unknownCopyMakesNoClaim() {
        for cause in [
            CommunityBranchUnknownCause.loading,
            .unavailable(.notConfigured),
            .unavailable(.offline),
        ] {
            let branch = CommunityContributionBranch.unknown(cause)
            let ready = CommunityBenchmarkCopy.readyInvitation(branch: branch, scope: Self.scope)
            #expect(!ready.eyebrow.lowercased().contains("first"))
            #expect(!ready.headline.lowercased().contains("first"))
            #expect(!ready.body.lowercased().contains("first"))
            #expect(!ready.body.lowercased().contains("no one has published"))

            let publish = CommunityBenchmarkCopy.publishInvitation(
                branch: branch, scope: Self.scope
            )
            #expect(!publish.headline.lowercased().contains("first"))
        }
    }

    @Test("A publication under an unknown branch confirms only that it published")
    func unknownPublishedCelebrationMakesNoClaim() {
        let celebration = CommunityBenchmarkCopy.publishedCelebration(
            branchBeforePublishing: .unknown(.unavailable(.offline)),
            scope: Self.scope,
            observationCountAfterPublishing: nil,
            alreadyPublished: false
        )
        #expect(celebration.headline == "Published to Community Benchmark")
        #expect(!celebration.headline.lowercased().contains("first"))
        #expect(!celebration.body.lowercased().contains("stronger"))
    }

    @Test("A duplicate publication never claims a first reference or a new count")
    func duplicatePublishCelebration() {
        let celebration = CommunityBenchmarkCopy.publishedCelebration(
            branchBeforePublishing: .firstReference,
            scope: Self.scope,
            observationCountAfterPublishing: 1,
            alreadyPublished: true
        )
        #expect(celebration.headline == "This result is already published")
        #expect(!celebration.headline.lowercased().contains("first reference"))
        #expect(celebration.body.contains("Nothing was added a second time"))
    }

    @Test("The celebration follows the branch shown before publishing, not after")
    func celebrationUsesPrePublishBranch() {
        // The Result screen knew 7 observations existed; after the receipt the
        // client's own count is 8. The headline must still be "stronger", and
        // must never become a first-reference claim.
        let celebration = CommunityBenchmarkCopy.publishedCelebration(
            branchBeforePublishing: .strengthen(observationCount: 7),
            scope: Self.scope,
            observationCountAfterPublishing: 8,
            alreadyPublished: false
        )
        #expect(celebration.headline == "You made this comparison stronger")
        #expect(celebration.body.contains("8 published observations"))
    }

    // MARK: - Comparison placeholder

    @Test("Comparison statistics are replaced by a reason on every non-strengthen branch")
    func comparisonPlaceholderPresence() {
        #expect(
            CommunityBenchmarkCopy.comparisonPlaceholder(
                branch: .strengthen(observationCount: 4), scope: Self.scope
            ) == nil
        )
        let first = CommunityBenchmarkCopy.comparisonPlaceholder(
            branch: .firstReference, scope: Self.scope
        )
        #expect(first?.title == "Comparison statistics are not available yet")

        let loading = CommunityBenchmarkCopy.comparisonPlaceholder(
            branch: .unknown(.loading), scope: Self.scope
        )
        #expect(loading?.title == "Loading community observations")

        let offline = CommunityBenchmarkCopy.comparisonPlaceholder(
            branch: .unknown(.unavailable(.offline)), scope: Self.scope
        )
        #expect(offline?.title == "Community data unavailable")
        #expect(offline?.body.contains("Running and publishing still work") == true)
    }
}
