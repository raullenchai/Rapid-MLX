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

    /// ESTIMATED download size for a model that is not on disk, derived from
    /// the alias by ``ModelSizing``. Callers render it with a leading `~` —
    /// the repo-wide convention for a figure that is a guess, not a
    /// measurement.
    var estimatedDownloadBytes: Int64? {
        guard !isReady else { return nil }
        let footprint = ModelSizing.estimate(alias: model.alias)
        guard footprint.weightsGB > 0 else { return nil }
        return Int64(footprint.weightsGB * Double(1 << 30))
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
    /// Share never shows a Download Required state — downloads belong in Pool.
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
            ByteCountFormatter.string(fromByteCount: $0, countStyle: .file)
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

// MARK: - Live Pool rows

/// What the pool summary says about one model this Mac could serve.
///
/// Three outcomes, and they are not interchangeable:
///
/// * ``live`` — the pool published this model and has it switched on. Its
///   counters are real, including when every one of them is zero.
/// * ``disabled`` — published with `enabled: false`. The pool is not routing
///   to it, so this Mac must not connect for it.
/// * ``unreported`` — the summary said nothing about this model at all. The
///   counters are UNKNOWN, and the row renders `—`. Substituting zero here
///   would state, falsely, that nobody is serving a model the pool never
///   mentioned.
enum ShareComputePoolAvailability: Equatable, Sendable {
    case live(ShareComputePoolModelStats)
    case disabled(ShareComputePoolModelStats)
    case unreported

    /// The counters to render, or `nil` when there is nothing truthful to show.
    var stats: ShareComputePoolModelStats? {
        switch self {
        case .live(let stats), .disabled(let stats): return stats
        case .unreported: return nil
        }
    }

    /// Whether this Mac may join the pool for this model right now.
    var acceptsConnections: Bool {
        if case .live = self { return true }
        return false
    }

    /// The placeholder for a counter with no published value. Never `0`.
    static let unknownValue = "—"
}

/// One row of the Live Pool model list.
///
/// A join between three independent facts: the pool catalog Rapid supports,
/// what this Mac has on disk, and what the summary endpoint published. Every
/// one of the three can be absent, and the row keeps them distinguishable
/// instead of flattening them into a number.
struct ShareComputePoolRow: Identifiable, Equatable, Sendable {
    let local: ShareComputeLocalModel
    let availability: ShareComputePoolAvailability
    let storage: ShareComputeStorageEligibility

    var id: String { local.id }

    var isDisabledUpstream: Bool {
        if case .disabled = availability { return true }
        return false
    }

    /// Selectable when the pool is routing to it. A disabled model can still be
    /// inspected — it just cannot be connected for.
    var canConnect: Bool { availability.acceptsConnections && local.isReady }

    /// Builds the list from Rapid's supported catalog joined to the summary.
    ///
    /// The catalog is the SPINE, not the summary: a model Rapid supports stays
    /// on screen (as ``unreported``) even if the summary omits it, and a
    /// `model_id` Rapid has never heard of is ignored rather than rendered as a
    /// row nobody can act on. Neither side may shorten the list to whatever the
    /// other happens to contain — that assumption is exactly what made adding a
    /// fourth pool model a breaking change.
    ///
    /// Order is the catalog's own and is deliberately stable across refreshes:
    /// the summary's `models` array order is not a ranking, and reordering rows
    /// every 30 seconds would make the list jump under the pointer.
    static func make(
        locals: [ShareComputeLocalModel],
        summary: ShareComputePoolSummary?,
        freeBytes: Int64?
    ) -> [ShareComputePoolRow] {
        locals.map { local in
            let availability: ShareComputePoolAvailability
            switch summary?.stats(for: local.model.catalogID) {
            case .some(let stats) where stats.isEnabled:
                availability = .live(stats)
            case .some(let stats):
                availability = .disabled(stats)
            case .none:
                availability = .unreported
            }
            return ShareComputePoolRow(
                local: local,
                availability: availability,
                storage: .evaluate(
                    downloadBytes: local.estimatedDownloadBytes,
                    freeBytes: freeBytes
                )
            )
        }
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

    /// Read-key management — where a `qsprk-` ledger key is created and
    /// revoked. The Share Compute tab of the dashboard.
    ///
    /// QuickSilver currently exposes only panel-level hashes. The read-key
    /// card has no stable section id, so `#compute` is the most precise link
    /// Rapid can use until QuickSilver publishes a nested deep-link contract.
    static let readKeyManagement = URL(string: "https://quicksilverpro.io/dashboard/#compute")!

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

// MARK: - Live Pool action

/// What the Live Pool CTA says and does for the selected row.
///
/// Extracted from the view so the "selecting a model updates the detail and the
/// action" rule is a testable function rather than a claim about a SwiftUI body
/// nobody can assert on. The three cases are mutually exclusive and cover every
/// row the picker can produce.
enum ShareComputePoolAction: Equatable, Sendable {
    /// Ready locally and the pool is routing to it.
    case connect(modelTitle: String)
    /// Supported and enabled upstream, but not on this Mac yet.
    case download(modelTitle: String)
    /// The model is supported, but the cache volume cannot safely land it.
    case insufficientStorage
    /// The pool has it switched off, so neither connecting nor downloading
    /// achieves anything.
    case unavailable
    /// Nothing selected.
    case none

    static func make(for row: ShareComputePoolRow?) -> ShareComputePoolAction {
        guard let row else { return .none }
        // Order matters: a disabled model can be downloaded and still be
        // useless, so the upstream switch is checked FIRST.
        if !row.availability.acceptsConnections { return .unavailable }
        if !row.local.isReady,
           case .notEnoughSpace = row.storage {
            return .insufficientStorage
        }
        if !row.local.isReady { return .download(modelTitle: row.local.title) }
        return .connect(modelTitle: row.local.model.shortTitle)
    }

    var title: String {
        switch self {
        case .connect(let title):
            return String(format: String(localized: "Continue with %@"), title)
        case .download:
            return String(localized: "Download to serve")
        case .insufficientStorage:
            return String(localized: "Not enough storage")
        case .unavailable:
            return String(localized: "Unavailable right now")
        case .none:
            return String(localized: "Select a model")
        }
    }

    var isEnabled: Bool {
        switch self {
        case .connect, .download: return true
        case .insufficientStorage, .unavailable, .none: return false
        }
    }

    /// True when pressing the button starts a download rather than opening the
    /// connection review.
    var startsDownload: Bool {
        if case .download = self { return true }
        return false
    }
}

// MARK: - Aggregate pool wording

/// User-facing names for POOL-WIDE figures.
///
/// These describe `totals`, which counts every node on the pool. The QuickSilver
/// protocol makes no hardware assumption — the public reference client is a
/// generic OpenAI-compatible adapter, and a Linux box serving through vLLM
/// registers exactly the way this Mac does — so an aggregate label may not say
/// "Macs". Local wording is the opposite case and still says "this Mac",
/// because the machine Rapid is running on genuinely is one.
///
/// Centralised here so the distinction is testable rather than a convention
/// each view has to remember.
enum ShareComputePoolLabels {
    /// Column label under the pool-wide connected count.
    static var connectedTotal: String {
        String(localized: "Connected Machines")
    }

    /// Headline over the amber panel: `34 Machines online`.
    static func machinesOnline(_ count: Int) -> String {
        count == 1
            ? String(localized: "1 Machine online")
            : String(format: String(localized: "%d Machines online"), count)
    }

    /// The calm all-zero state. A real zero is a normal reading, not an error.
    static var noneConnected: String {
        String(localized: "No machines are connected right now.")
    }

    /// Every aggregate string this type can produce, for the test that pins
    /// their neutrality.
    static var allAggregateStrings: [String] {
        [connectedTotal, machinesOnline(0), machinesOnline(1), machinesOnline(34), noneConnected]
    }
}
