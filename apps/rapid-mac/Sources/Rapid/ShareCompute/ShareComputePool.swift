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

    /// Catalog download size, falling back to the measured or legacy size.
    var onDiskSize: String? {
        entry?.displayedModelSize?.displayText
    }

    var estimatedDownloadBytes: Int64? {
        guard !isReady else { return nil }
        if let bytes = entry?.displayedModelSize?.byteCount, bytes > 0 { return bytes }
        let footprint = ModelSizing.estimate(alias: model.alias)
        guard footprint.weightsGB > 0 else { return nil }
        return DownloadProgress.bytes(fromDouble: footprint.weightsGB * Double(1 << 30))
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

// MARK: - Storage eligibility

/// Whether this Mac can land a model download.
///
/// Wraps ``DiskSpaceProbe`` rather than reimplementing it so Share Compute and
/// Quickstart apply the same threshold, and so the "probe failed" case keeps
/// the app's established fail-open behaviour: an unreadable volume reports
/// ``unknown``, never a false "not enough space".
enum ShareComputeStorageEligibility: Equatable, Sendable {
    case enoughSpace(freeBytes: Int64)
    case notEnoughSpace(freeBytes: Int64, requiredBytes: Int64)
    /// Free space could not be read, so no claim is made either way.
    case unknown

    static func evaluate(downloadBytes: Int64?, freeBytes: Int64?) -> Self {
        guard let downloadBytes, downloadBytes > 0 else { return .unknown }
        guard let freeBytes else { return .unknown }
        let required = DiskSpaceProbe.requiredBytes(downloadBytes: downloadBytes)
        switch DiskSpaceProbe.decide(freeBytes: freeBytes, requiredBytes: required) {
        case .ok:
            return .enoughSpace(freeBytes: freeBytes)
        case .warn(let free, let requiredBytes):
            return .notEnoughSpace(freeBytes: free, requiredBytes: requiredBytes)
        }
    }

    /// `84 GB free · Enough space` — the exact shape Paper uses, built from
    /// measured bytes.
    var label: String? {
        let formatter: (Int64) -> String = {
            DownloadProgress.formatBytes($0)
        }
        switch self {
        case .enoughSpace(let free):
            return String(
                format: String(localized: "%1$@ free · Enough space"),
                formatter(free)
            )
        case .notEnoughSpace(let free, let required):
            return String(
                format: String(localized: "%1$@ free · Needs %2$@"),
                formatter(free),
                formatter(required)
            )
        case .unknown:
            return nil
        }
    }

    var isSufficient: Bool {
        if case .enoughSpace = self { return true }
        return false
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
