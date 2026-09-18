import Foundation

/// Which contribution story the module tells for one scope.
///
/// Exactly three outcomes, and the third one is load-bearing: when the
/// community read is loading, offline, failed, or absent, the module must
/// select NEITHER branch. A missing answer is not "nobody has published this",
/// and inferring the first-reference branch from missing data is the single
/// worst failure this surface can have — it tells the user something flattering
/// and false, and it does so exactly when the service is least able to correct
/// it.
enum CommunityContributionBranch: Equatable, Sendable {
    /// Known: the scope has no published observations. The only state allowed
    /// to use first-reference language.
    case firstReference
    /// Known: at least `observationCount` published observations exist.
    /// Invite another sample that strengthens the comparison.
    ///
    /// `isAtLeast` records that the number came from the bounded public feed,
    /// so the copy says "at least 7" rather than asserting an exact total.
    case strengthen(observationCount: Int, isAtLeast: Bool = false)
    /// Not known. Run and publish stay available; every community claim,
    /// comparison statistic, and "first" phrase is withheld.
    case unknown(CommunityBranchUnknownCause)

    /// The one place the branch is derived. Views must not re-derive it from a
    /// count, because an `Int?` cannot express "loading" and a caller that
    /// defaults nil to zero reintroduces the false-first bug.
    static func select(
        from state: CommunityDataState<CommunityObservationSummary>
    ) -> Self {
        switch state {
        case .loading:
            return .unknown(.loading)
        case let .unavailable(reason):
            return .unknown(.unavailable(reason))
        case let .ready(summary):
            // Exactly zero — and nothing else — selects the first-reference
            // branch. A negative or otherwise malformed count is a broken
            // answer, not an empty one; clamping it to zero would turn a
            // service bug into a confident "nobody has published this" on the
            // one screen where that claim is hardest to take back.
            if summary.observationCount < 0 {
                return .unknown(.unavailable(.invalidCount))
            }
            if summary.observationCount == 0 {
                // A bounded feed cannot observe zero — it can only fail to
                // find a scope, which the adapter already reports as
                // unavailable. A zero that arrives flagged bounded is
                // therefore incoherent and is not trusted.
                return summary.isBounded
                    ? .unknown(.unavailable(.boundedFeed))
                    : .firstReference
            }
            return .strengthen(
                observationCount: summary.observationCount,
                isAtLeast: summary.isBounded
            )
        }
    }

    /// True only for the branch that may claim a first public reference.
    var allowsFirstReferenceLanguage: Bool { self == .firstReference }

    /// True only when a median / observed range / difference-from-median may
    /// be rendered. Both `.firstReference` (nothing to compare against) and
    /// `.unknown` (nothing trustworthy to show) suppress comparison stats.
    var allowsComparisonStatistics: Bool {
        if case .strengthen = self { return true }
        return false
    }

    var observationCount: Int? {
        switch self {
        case .firstReference: return 0
        case let .strengthen(count, _): return count
        case .unknown: return nil
        }
    }

    /// True when ``observationCount`` is a floor rather than an exact total.
    var observationCountIsApproximate: Bool {
        if case let .strengthen(_, isAtLeast) = self { return isAtLeast }
        return false
    }
}

enum CommunityBranchUnknownCause: Equatable, Sendable {
    case loading
    case unavailable(CommunityUnavailableReason)
}

// MARK: - Copy

/// The user-facing sentences for each branch and stage.
///
/// Centralised for two reasons. First, every first-reference sentence must
/// name its scope (model, workload, Mac profile) — a claim scoped only by
/// model name is wrong, and reviewing that is far easier when the sentences
/// sit together. Second, the three stages have to agree: a Ready screen that
/// promises a first reference and a Published screen that congratulates a
/// stronger comparison is the cross-screen inconsistency this module kept
/// regressing into.
enum CommunityBenchmarkCopy {
    /// "7" for an exact count, "at least 7" for one read from the bounded feed.
    static func countPhrase(_ count: Int, isAtLeast: Bool) -> String {
        isAtLeast
            ? String(format: String(localized: "at least %1$d"), count)
            : "\(count)"
    }

    // MARK: Coverage

    struct Band: Equatable {
        let title: String
        let message: String
    }

    /// What "Where your Mac can help" says when the feed reports no thin
    /// pairings.
    ///
    /// Deliberately not "every model the catalogue offers already has results
    /// on this Mac profile". The feed carries only the newest runs, and a model
    /// with no results at all is precisely the model that never appears in it,
    /// so an empty gap list is the one piece of evidence that cannot support a
    /// full-coverage claim.
    static let coverageAllCovered = Band(
        title: String(localized: "Nothing looks thin right now"),
        message: String(localized: "None of the models in the recent published results needs more samples on this Mac profile. That only covers what has been published lately — it isn’t a statement about every model in the catalogue.")
    )

    // MARK: Ready

    struct ReadyInvitation: Equatable {
        let eyebrow: String
        let headline: String
        let body: String
    }

    static func readyInvitation(
        branch: CommunityContributionBranch,
        scope: CommunityBenchmarkScope
    ) -> ReadyInvitation {
        switch branch {
        case .firstReference:
            return ReadyInvitation(
                eyebrow: String(localized: "FIRST RESULT NEEDED"),
                headline: String(localized: "Your Mac can create the first reference"),
                body: String(
                    format: String(
                        localized:
                            "No one has published a benchmark for %1$@ yet. Run it once to give the community its first point of comparison."
                    ),
                    scope.scopeDescription
                )
            )
        case let .strengthen(count, isAtLeast):
            return ReadyInvitation(
                eyebrow: String(localized: "COMMUNITY NEEDS ANOTHER RESULT"),
                headline: String(localized: "Make this comparison stronger"),
                body: String(
                    format: String(
                        localized:
                            "%1$@ published %2$@ for %3$@ on an %4$@. Run it once to add another reference and tighten the range."
                    ),
                    Self.countPhrase(count, isAtLeast: isAtLeast),
                    count == 1 && !isAtLeast
                        ? String(localized: "result exists")
                        : String(localized: "results exist"),
                    scope.modelAlias,
                    scope.macProfile.displayName
                )
            )
        case .unknown:
            // No coverage claim of any kind. The invitation falls back to what
            // the client can always substantiate on its own: this Mac, this
            // model, a fixed workload, and a private result.
            return ReadyInvitation(
                eyebrow: String(localized: "MEASURE THIS MODEL"),
                headline: String(localized: "Measure this model on your Mac"),
                body: String(
                    format: String(
                        localized:
                            "Run a fixed %1$@ workload for %2$@ and keep the result on this Mac. You can publish it to the Community Benchmark afterwards."
                    ),
                    scope.workload.displayName,
                    scope.modelAlias
                )
            )
        }
    }

    // MARK: Result — the publish invitation band

    struct PublishInvitation: Equatable {
        let headline: String
        let body: String
    }

    static func publishInvitation(
        branch: CommunityContributionBranch,
        scope: CommunityBenchmarkScope
    ) -> PublishInvitation {
        switch branch {
        case .firstReference:
            return PublishInvitation(
                headline: String(localized: "Create the first public reference"),
                body: String(
                    format: String(
                        localized:
                            "No published result exists yet for %1$@. Publishing yours gives other Mac users a real baseline."
                    ),
                    scope.scopeDescription
                )
            )
        case let .strengthen(count, isAtLeast):
            return PublishInvitation(
                headline: isAtLeast
                    ? String(
                        format: String(
                            localized: "Publish this result to add another observation for %1$@ on an %2$@"
                        ),
                        scope.modelAlias,
                        scope.macProfile.displayName
                    )
                    : String(
                        format: String(
                            localized: "Publish this result to add observation %1$d for %2$@ on an %3$@"
                        ),
                        count + 1,
                        scope.modelAlias,
                        scope.macProfile.displayName
                    ),
                body: String(
                    format: String(
                        localized:
                            "%1$@ %2$@ for this pairing. Yours makes the comparison harder to argue with."
                    ),
                    Self.countPhrase(count, isAtLeast: isAtLeast),
                    count == 1 && !isAtLeast
                        ? String(localized: "published result exists")
                        : String(localized: "published results exist")
                )
            )
        case .unknown:
            return PublishInvitation(
                headline: String(localized: "Publish this result to the Community Benchmark"),
                body: String(
                    localized:
                        "Your result stays on this Mac until you publish it. Publishing adds it to the public observations for this model and Mac profile."
                )
            )
        }
    }

    // MARK: Published

    struct PublishedCelebration: Equatable {
        let headline: String
        let body: String
    }

    /// `branchBeforePublishing` is the branch the Result screen showed, and
    /// `summaryAfterPublishing` is derived from the receipt. A celebration is
    /// never allowed to claim an ordinal based only on the pre-publish branch:
    /// another contributor can publish before this request commits.
    static func publishedCelebration(
        branchBeforePublishing: CommunityContributionBranch,
        scope: CommunityBenchmarkScope,
        observationCountAfterPublishing: Int?,
        alreadyPublished: Bool
    ) -> PublishedCelebration {
        if alreadyPublished {
            return PublishedCelebration(
                headline: String(localized: "This result is already published"),
                body: String(
                    localized:
                        "It is already on your contributor page. Nothing was added a second time."
                )
            )
        }
        switch branchBeforePublishing {
        case .firstReference:
            return PublishedCelebration(
                headline: String(localized: "Published to Community Benchmark"),
                body: String(
                    format: String(
                        localized:
                            "Your result is now part of the public observations for %1$@."
                    ),
                    scope.scopeDescription
                )
            )
        case let .strengthen(_, isAtLeast):
            guard !isAtLeast, let count = observationCountAfterPublishing else {
                return PublishedCelebration(
                    headline: String(localized: "You made this comparison stronger"),
                    body: String(
                        format: String(
                            localized: "Your %1$@ result is now part of the public observations for an %2$@."
                        ),
                        scope.modelAlias,
                        scope.macProfile.displayName
                    )
                )
            }
            return PublishedCelebration(
                headline: String(localized: "You made this comparison stronger"),
                body: String(
                    format: String(
                        localized:
                            "There are now %1$d published observations for %2$@ on an %3$@, including yours."
                    ),
                    count,
                    scope.modelAlias,
                    scope.macProfile.displayName
                )
            )
        case .unknown:
            // The count was never known, so neither "first" nor "stronger" can
            // be claimed. Confirm the fact that is certain: it is published.
            return PublishedCelebration(
                headline: String(localized: "Published to Community Benchmark"),
                body: String(
                    format: String(
                        localized: "Your %1$@ result is live on rapidmlx.com."
                    ),
                    scope.modelAlias
                )
            )
        }
    }

    // MARK: Comparison area

    /// The replacement shown where median / range / difference would go when
    /// the branch does not permit comparison statistics.
    static func comparisonPlaceholder(
        branch: CommunityContributionBranch,
        scope: CommunityBenchmarkScope
    ) -> (title: String, body: String)? {
        switch branch {
        case .strengthen:
            return nil
        case .firstReference:
            return (
                String(localized: "Comparison statistics are not available yet"),
                String(
                    format: String(
                        localized:
                            "Medians and observed ranges appear here once more people publish %1$@ results from an %2$@."
                    ),
                    scope.modelAlias,
                    scope.macProfile.displayName
                )
            )
        case let .unknown(cause):
            switch cause {
            case .loading:
                return (
                    String(localized: "Loading community observations"),
                    String(localized: "Comparison statistics appear when the community data loads.")
                )
            case let .unavailable(reason):
                return (
                    String(localized: "Community data unavailable"),
                    reason.message
                )
            }
        }
    }
}
