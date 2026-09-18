import Foundation

/// The community reads that can be in flight at once.
///
/// Each is guarded independently. A single shared token looked equivalent and
/// was not: bumping it because the *model* changed also invalidated the
/// in-flight coverage, pulse and contributor-totals requests — none of which
/// depend on the model — and nothing restarted them, so those panels sat on
/// `.loading` forever. The generation must only invalidate the queries whose
/// inputs actually moved.
enum CommunityReadKind: CaseIterable, Sendable {
    /// Aggregate for the selected model. Depends on the model.
    case observations
    /// "Performance on Macs like yours". Depends on the selected workload.
    case table
    /// "Where your Mac can help". Depends on the Mac profile only.
    case coverage
    /// Community pulse. Depends on nothing the user can change here.
    case pulse
    /// Exact published totals for this installation's pseudonym.
    case contributorTotals
}

/// Per-query request generations.
///
/// Guards against a slow read landing after the thing it described has
/// changed: the user selects model A, A's request is slow, they switch to B,
/// B's (fast) answer renders, then A's answer arrives and overwrites it. The
/// screen would then show B's name beside A's observation count — and if A had
/// no observations, a first-reference banner for a model that is not selected.
///
/// Two rules make this safe:
///
/// 1. **Independence.** Bumping `observations` leaves `pulse` alone, so an
///    unrelated request in flight is never orphaned.
/// 2. **Bump-then-start.** Callers take a token from ``begin(_:)``, which
///    invalidates any earlier request for that kind and returns the token the
///    new one carries. Every invalidation therefore has a restart attached to
///    it by construction.
struct CommunityRequestGenerations: Sendable {
    private var values: [CommunityReadKind: Int] = [:]

    init() {
        for kind in CommunityReadKind.allCases { values[kind] = 0 }
    }

    /// Invalidates any in-flight request of this kind and returns the token
    /// the replacement must carry.
    @discardableResult
    mutating func begin(_ kind: CommunityReadKind) -> Int {
        let next = (values[kind] ?? 0) + 1
        values[kind] = next
        return next
    }

    /// Whether a response stamped `token` may still be applied.
    func isCurrent(_ kind: CommunityReadKind, _ token: Int) -> Bool {
        values[kind] == token
    }

    /// The token a request of this kind would currently carry. Test-facing.
    func current(_ kind: CommunityReadKind) -> Int { values[kind] ?? 0 }
}
