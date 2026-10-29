import Foundation

// MARK: - Local availability

/// A pool model joined to what this Mac actually has on disk.
///
/// ``ShareComputeModel/supported`` is the catalog QuickSilver will accept; the
/// ``ModelEntry`` list is what `rapid-mlx ls` found locally. Share may only
/// offer the intersection, so this type exists to make that join explicit and
/// testable rather than re-deriving `catalog.first { $0.alias == … }` in four
/// views.
struct ShareComputeLocalModel: Identifiable, Hashable, Sendable {
    let model: ShareComputeModel
    /// The matching catalog row, when the alias is known to the engine at all.
    let entry: ModelEntry?

    var id: String { model.catalogID }
    var title: String { model.title }
    var detail: String { model.detail }

    /// Downloaded and ready to serve right now.
    var isReady: Bool { entry?.cached == true }

    /// MEASURED on-disk size, straight from `rapid-mlx ls`. `nil` rather than
    /// an estimate — a ready model shows what it really occupies, matching
    /// Settings → Models so the two surfaces can never print different numbers
    /// for the same model.
    var onDiskSize: String? {
        guard let raw = entry?.sizeOnDisk?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else { return nil }
        return raw
    }

    /// Joins the pool catalog to the local catalog.
    ///
    /// Alias comparison is case-insensitive because `rapid-mlx ls` and the
    /// pool catalog have historically disagreed on casing for the same model.
    static func make(
        models: [ShareComputeModel] = ShareComputeModel.supported,
        catalog: [ModelEntry]
    ) -> [ShareComputeLocalModel] {
        models.map { model in
            ShareComputeLocalModel(
                model: model,
                entry: catalog.first {
                    $0.alias.caseInsensitiveCompare(model.alias) == .orderedSame
                }
            )
        }
    }

    /// The models Share is allowed to list: downloaded and locally ready.
    ///
    /// This is the single filter enforcing that, so a future picker cannot
    /// quietly widen the list.
    static func readyForSharing(_ all: [ShareComputeLocalModel]) -> [ShareComputeLocalModel] {
        all.filter(\.isReady)
    }
}

// MARK: - Provider destinations

/// Every QuickSilver link the module uses, in one place so "where does this
/// send the user?" has a single answer and no view builds a URL inline.
enum ShareComputeDestination {
    static let provider = URL(string: "https://quicksilverpro.io/")!

    /// Where a provider key comes from.
    ///
    /// The Dashboard now has a first-class **Share Compute** tab, and a signed-in
    /// user creates their own `qsppk-` key there — so the old "keys can only be
    /// minted by hand against the API" copy is gone.
    /// `#compute` is QuickSilver's published stable tab deep link. Keep the
    /// canonical trailing slash before the fragment: the dashboard redirects
    /// `/dashboard` to `/dashboard/`, and auth redirects have historically made
    /// fragment preservation harder to reason about from the non-canonical path.
    ///
    /// Rapid never collects QuickSilver credentials: the user signs in to
    /// QuickSilver in their own browser, and the key comes back to Rapid only
    /// by paste, only into the connection sheet, and only onward via the share
    /// subprocess's stdin.
    static let dashboard = URL(string: "https://quicksilverpro.io/dashboard/#compute")!

    /// QuickSilver account BALANCE and recharge.
    ///
    /// Not a contributor-ledger page — it shows what the account can spend,
    /// not what it earned. Rapid renders the ledger itself, so this link is
    /// labelled as balance and must never be presented as "see your ledger
    /// details", which is what the pre-ledger build implied.
    static let credits = URL(string: "https://quicksilverpro.io/dashboard#credits")!

    /// Reward activity and payout account live on the pay host, which is the
    /// same origin the provider client is configured against.
    static let rewards = URL(string: "https://pay.quicksilverpro.io/")!
}
