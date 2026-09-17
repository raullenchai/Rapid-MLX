import Foundation

/// Which model/workload pairings on this Mac profile would genuinely benefit
/// from another measurement.
///
/// The previous implementation asked the threshold question of *individual*
/// summary cells and then kept whichever cell of the surviving set came first:
///
/// ```swift
/// feed.summary.filter { $0.machine.matches(profile) && $0.samples < 5 }
///     .reduce(into: []) { unique, cell in /* first cell per model wins */ }
/// ```
///
/// A model with a 4-sample fp16 cell and a 5-sample bf16 cell has **nine**
/// published results on that Mac. The filter dropped the 5-sample cell for
/// failing the threshold, kept the 4-sample one, and the screen read "Only 4
/// published results on an Apple M3 Pro so far" — a number that is not the
/// model's coverage, is not any cell's coverage as displayed, and depends on
/// the order the worker happened to emit its groups in.
///
/// So the order is inverted: **aggregate first, threshold second.** Each run
/// contributes to exactly one summary group (the worker keys on
/// `run.cases[0]`), so summing `samples` across a model's cells counts every
/// run once and the total is a defensible bounded count.
enum CommunityCoverageProjection {
    /// Under-represented pairings, fewest observations first.
    ///
    /// Only the newest protocol version present for a pairing is counted:
    /// results measured under an older protocol are not evidence that the
    /// current one is well covered.
    ///
    /// A pairing that reaches `threshold` is simply absent from the result. It
    /// is never reported with a partial count, because a partial count reads as
    /// a claim about the model's whole coverage.
    static func gaps(
        from cells: [CommunityTableProjection.Cell],
        below threshold: Int
    ) -> [CommunityCoverageGap] {
        // Keyed on the full model identity, not the alias: a `4bit/` subfolder
        // and a repo-root build are different artifacts, and folding their
        // samples together would report one as well covered because the other
        // is. The alias travels alongside for display only.
        struct Key: Hashable {
            let alias: String
            let identity: CommunityModelIdentity
            let workload: CommunityWorkload
            let protocolID: String
        }

        var byPairing: [Key: [CommunityTableProjection.Cell]] = [:]
        for cell in cells {
            let key = Key(
                alias: cell.modelAlias,
                identity: cell.modelIdentity,
                workload: cell.workload,
                protocolID: cell.protocolID
            )
            byPairing[key, default: []].append(cell)
        }

        return byPairing.compactMap { key, pairingCells -> CommunityCoverageGap? in
            guard let newest = pairingCells.map(\.protocolVersion).max() else { return nil }
            let counted = pairingCells.filter { $0.protocolVersion == newest }
            let total = counted.reduce(0) { $0 + max(0, $1.samples) }

            // A pairing present in the feed has at least one run behind it. A
            // zero here would be malformed data, and zero is the one value
            // that turns on "FIRST RESULT NEEDED" — a claim a bounded feed can
            // never support. Drop it rather than make it.
            guard total > 0 else { return nil }
            guard total < threshold else { return nil }

            return CommunityCoverageGap(
                modelAlias: key.alias,
                workload: key.workload,
                observationCount: total,
                fitsThisMac: true,
                isDownloaded: false,
                downloadSizeGB: nil,
                requiredMemoryGB: nil
            )
        }
        .sorted {
            $0.observationCount == $1.observationCount
                ? $0.modelAlias.localizedStandardCompare($1.modelAlias) == .orderedAscending
                : $0.observationCount < $1.observationCount
        }
    }
}
