import Foundation

// MARK: - Reward record status

/// What Rapid can honestly say about the reward for one finished session.
///
/// Rapid never stores, derives, or displays a reward AMOUNT — QuickSilver owns
/// that, and there is no local API that returns it (see
/// ``ShareComputePoolDemand`` for the same boundary on demand data). What Rapid
/// *does* know is how far the session got before it ended, and that is the only
/// input to this value:
///
///   * the node reached `online`, so QuickSilver accepted it and a provider
///     record exists to go and look at → ``available``
///   * the node got as far as warm-up / connecting but never went online, so
///     whether the provider accepted it is genuinely unresolved here →
///     ``processing``
///   * the node never reached the pool at all, so there is no accepted work →
///     ``notEligible``
///
/// Deliberately NOT a mirror of a QuickSilver enum. If the provider ever
/// exposes a reward endpoint, that becomes a second, separate field and this
/// one keeps describing the local session.
enum ShareComputeRewardStatus: String, Codable, Sendable, CaseIterable {
    case available
    case processing
    case notEligible

    /// The compact tag shown in the history table. Uppercased at the call site
    /// so the stored value stays readable in the receipts file.
    ///
    /// The wording describes what THIS MAC observed, and nothing more. The old
    /// labels — "Available", "No accepted work" — asserted a settlement state
    /// Rapid cannot see: there is no contributor-earnings API, so "available"
    /// was a guess about QuickSilver's ledger wearing the clothes of a fact.
    /// The raw values are unchanged so receipts written by older builds still
    /// decode; only the rendering is honest now.
    var tagTitle: String {
        switch self {
        case .available: return String(localized: "Ended")
        case .processing: return String(localized: "Ended")
        case .notEligible: return String(localized: "Incomplete")
        }
    }

    /// The sentence shown in the selected-session panel. Same rule: local
    /// observation only.
    var detailTitle: String {
        switch self {
        case .available: return String(localized: "Served the pool · ended cleanly")
        case .processing: return String(localized: "Ended before serving began")
        case .notEligible: return String(localized: "Never reached the pool")
        }
    }

    /// True when linking the user to QuickSilver can actually show them
    /// something about this session.
    var hasProviderRecord: Bool { self != .notEligible }
}

/// Whether the model Rapid paused for the session came back afterwards.
///
/// Paper labels this slot `MODEL RESTORE`, not `Previous model` — the user is
/// being told the outcome of an action Rapid took on their behalf, so the label
/// has to name the action and carry a status.
enum ShareComputeRestoreStatus: String, Codable, Sendable {
    /// There was no model serving when sharing started, so nothing to restore.
    case notNeeded
    /// The previous alias was asked to start again.
    case complete
    /// Sharing ended as part of app shutdown; the restore never ran.
    case skipped

    var title: String {
        switch self {
        case .notNeeded: return String(localized: "Not needed")
        case .complete: return String(localized: "Complete")
        case .skipped: return String(localized: "Skipped")
        }
    }
}

// MARK: - Receipt

/// One completed local sharing session.
///
/// This is a RAPID-owned record. Everything in it was observed on this Mac:
/// the model that was shared, when it started and stopped, the worker name the
/// user chose, and the node id the provider published into the desktop status
/// file. Nothing here is fetched from QuickSilver, and nothing here is a
/// promise about payment.
struct ShareComputeReceipt: Codable, Identifiable, Hashable, Sendable {
    /// Stable local id. Rendered to the user as `Receipt <id>`.
    let id: String
    /// Pool catalog id, so "Share this model again" can reselect it even if the
    /// display title changes.
    let catalogID: String
    /// Display title captured at the time of the session.
    let modelTitle: String
    /// The worker label this Mac registered under.
    let worker: String
    /// The provider's node id, when the session got far enough to have one.
    let nodeID: String?
    let startedAt: Date
    let endedAt: Date
    let rewardStatus: ShareComputeRewardStatus
    let restoreStatus: ShareComputeRestoreStatus

    var duration: TimeInterval { max(0, endedAt.timeIntervalSince(startedAt)) }

    /// Receipt ids follow the provider's own short-hex shape so a user reading
    /// `QS-8A31` in Rapid and in QuickSilver is looking at the same string
    /// family. The value is generated locally; it is an identifier, not a
    /// claim that QuickSilver knows it.
    static func makeID(uuid: UUID = UUID()) -> String {
        let hex = uuid.uuidString.replacingOccurrences(of: "-", with: "")
        return "QS-" + hex.prefix(4).uppercased()
    }
}

// MARK: - Summary

/// The three recognition metrics above the history table.
struct ShareComputeContributionSummary: Equatable, Sendable {
    let sessionCount: Int
    let totalShared: TimeInterval
    let modelCount: Int

    static let empty = ShareComputeContributionSummary(
        sessionCount: 0,
        totalShared: 0,
        modelCount: 0
    )

    /// Distinct models are counted by catalog id, not display title: a model
    /// that was renamed between two sessions is still one model contributed.
    static func make(from receipts: [ShareComputeReceipt]) -> Self {
        ShareComputeContributionSummary(
            sessionCount: receipts.count,
            totalShared: receipts.reduce(0) { $0 + $1.duration },
            modelCount: Set(receipts.map(\.catalogID)).count
        )
    }
}

// MARK: - Pagination

/// Fixed-size paging over the history table.
///
/// Paper pages at five rows and the implementation handoff repeats it, so the
/// size is a constant here rather than a parameter every call site could get
/// wrong.
enum ShareComputeHistoryPage {
    static let size = 5

    static func pageCount(total: Int) -> Int {
        max(1, Int(ceil(Double(total) / Double(size))))
    }

    /// Clamps `page` into range before slicing, so a receipt list that shrinks
    /// (or a stale page index after a delete) can never trap the view on an
    /// empty page.
    static func slice(
        _ receipts: [ShareComputeReceipt],
        page: Int
    ) -> [ShareComputeReceipt] {
        guard !receipts.isEmpty else { return [] }
        let clamped = min(max(0, page), pageCount(total: receipts.count) - 1)
        let start = clamped * size
        let end = min(start + size, receipts.count)
        guard start < end else { return [] }
        return Array(receipts[start..<end])
    }

    /// "Showing 1–5 of 12" — 1-based and inclusive, matching Paper.
    static func rangeLabel(page: Int, total: Int) -> String {
        guard total > 0 else { return String(localized: "No local receipts yet") }
        let clamped = min(max(0, page), pageCount(total: total) - 1)
        let first = clamped * size + 1
        let last = min(first + size - 1, total)
        return String(
            format: String(localized: "Local receipts · Showing %1$d–%2$d of %3$d"),
            first,
            last,
            total
        )
    }
}

// MARK: - Store

/// Durable local contribution history.
///
/// Receipts live beside the app's other user data in
/// ``~/Library/Application Support/Rapid/share-compute-receipts.json`` rather
/// than in `~/.rapid-mlx/quicksilver/`, which is the PROVIDER's directory — the
/// registration marker and desktop status file there are written and deleted by
/// the provider process, and a Rapid-owned history that outlives any single
/// session must not share that lifecycle.
///
/// Reads are bounded and every failure degrades to "no history" rather than
/// throwing: a corrupt receipts file must not make the tab unopenable.
struct ShareComputeReceiptStore: Sendable {
    /// Newest receipts are the ones a user looks at, so a bounded file keeps
    /// the oldest from growing without limit. 500 sessions is far beyond any
    /// realistic history and still reads in one gulp.
    static let maximumReceipts = 500
    /// Matches the bound above with generous headroom per record.
    static let maximumFileBytes = 1 << 20

    let fileURL: URL

    init(fileURL: URL) {
        self.fileURL = fileURL
    }

    init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        self.fileURL = ApplicationSupportLocator
            .applicationSupportRoot(environment: environment)
            .appendingPathComponent("share-compute-receipts.json")
    }

    /// Everything on disk, newest first. Never throws.
    func load() -> [ShareComputeReceipt] {
        guard let handle = try? FileHandle(forReadingFrom: fileURL) else { return [] }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: Self.maximumFileBytes + 1),
              data.count <= Self.maximumFileBytes else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let envelope = try? decoder.decode(Envelope.self, from: data),
              envelope.schemaVersion == Envelope.currentSchemaVersion else { return [] }
        return Self.sorted(envelope.receipts)
    }

    /// Prepends `receipt` and rewrites the file. Returns the new history so the
    /// caller does not have to re-read what it just wrote.
    @discardableResult
    func append(_ receipt: ShareComputeReceipt) -> [ShareComputeReceipt] {
        var receipts = load()
        receipts.removeAll { $0.id == receipt.id }
        receipts.append(receipt)
        let trimmed = Array(Self.sorted(receipts).prefix(Self.maximumReceipts))
        save(trimmed)
        return trimmed
    }

    func save(_ receipts: [ShareComputeReceipt]) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(
            Envelope(schemaVersion: Envelope.currentSchemaVersion, receipts: receipts)
        ) else { return }
        try? FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        // Atomic so a crash mid-write leaves the previous history intact rather
        // than a truncated file that `load` would discard entirely.
        try? data.write(to: fileURL, options: .atomic)
    }

    /// Newest first, with the id as a tiebreaker so equal timestamps still
    /// produce a stable order across launches.
    static func sorted(_ receipts: [ShareComputeReceipt]) -> [ShareComputeReceipt] {
        receipts.sorted {
            $0.startedAt == $1.startedAt ? $0.id > $1.id : $0.startedAt > $1.startedAt
        }
    }

    private struct Envelope: Codable {
        static let currentSchemaVersion = 1
        let schemaVersion: Int
        let receipts: [ShareComputeReceipt]
    }
}
