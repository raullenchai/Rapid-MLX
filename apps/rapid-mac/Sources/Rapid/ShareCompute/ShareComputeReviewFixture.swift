import Foundation

// MARK: - Stages

/// A Share Compute lifecycle surface a visual-review harness asked to see.
///
/// Preparing, Online, Session Complete and Connection Review are reachable in
/// the shipping app only by actually joining the pool, which needs a
/// QuickSilver provider key and a live pool to join. Neither is available to a
/// screenshot run, so those four surfaces shipped unreviewed — the thing this
/// type exists to fix.
///
/// ## What it is not
///
/// It is not a mock provider. Nothing here answers an API call, populates
/// ``ShareComputeManager``, writes a receipt to disk, or changes what any
/// production code path believes about this Mac. It supplies VIEW INPUTS —
/// the same `let` parameters ``ShareComputeView`` would otherwise compute —
/// to the same views, so what gets reviewed is the shipping rendering of a
/// state, not a second implementation of it.
///
/// ## Why it cannot leak
///
/// Two environment keys, both required, and the first
/// (``RAPID_GUI_GOLDEN_MODE``) is the marker the app already uses to gate its
/// update, dictation, and initial-section fixtures. A normal launch never
/// reads ``RAPID_GUI_SHARE_COMPUTE_STAGE`` at all, so the tab behaves exactly
/// as it ships.
enum ShareComputeReviewStage: String, CaseIterable, Sendable {
    /// The five-step preparation rail, mid-sequence.
    case preparing
    /// A live session: elapsed clock, requests in flight, node id.
    case online
    /// The receipt surface shown after a session ends.
    case sessionComplete
    /// The pre-flight sheet, with the provider-key field.
    case connectionReview
    /// The ready workbench with the model picker open.
    case modelPicker

    static func requested(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ShareComputeReviewStage? {
        guard environment["RAPID_GUI_GOLDEN_MODE"] == "1",
              let name = environment["RAPID_GUI_SHARE_COMPUTE_STAGE"] else { return nil }
        return ShareComputeReviewStage(rawValue: name)
    }

    /// Whether this stage replaces the Share tab's own body.
    ///
    /// ``connectionReview`` and ``modelPicker`` do not: they are the READY
    /// workbench with something presented over it, so the tab renders
    /// normally and the stage only opens the sheet or the popover.
    var replacesShareTab: Bool {
        switch self {
        case .preparing, .online, .sessionComplete: return true
        case .connectionReview, .modelPicker: return false
        }
    }
}

// MARK: - Inputs

/// The view inputs each reviewable stage needs.
///
/// Every value is built from a real domain type — ``ShareComputeReceipt``,
/// ``ShareComputePreparationPlan``, ``ShareComputeRewardTracking`` — and the
/// step rail is PROJECTED from a real ``ShareComputeManager/State`` rather
/// than hand-assembled, so a change to the projection rules shows up in the
/// review captures instead of being papered over by a hard-coded row list.
enum ShareComputeReviewFixture {
    /// Matches the model the screenshot script's fake catalog marks as
    /// cached, so the reviewable stages and the ready workbench name the same
    /// model.
    static let modelTitle = "Qwen3.8 27B · 4-bit"
    static let catalogID = "qwen3.8-27b"
    static let worker = "Lori-Mac"
    static let nodeID = "qs-node-a8c1"

    /// Mid-sequence: registration already on file from a prior session, the
    /// shared model starting. Chosen because it is the only configuration
    /// that shows all four step statuses at once — complete, already
    /// registered, in progress, and waiting.
    static var preparationRows: [ShareComputePreparationRow] {
        ShareComputePreparationPlan.rows(
            state: .starting,
            isAlreadyRegistered: true
        )
    }

    static var preparingTracking: ShareComputeRewardTracking {
        .make(screen: .preparing)
    }

    /// An hour and a bit into the session, so the elapsed clock renders at
    /// its full `1h 42m 18s` width rather than as a two-digit stub.
    static func poolJoinedAt(now: Date = Date()) -> Date {
        now.addingTimeInterval(-6_138)
    }

    /// The finished-session receipt. A real ``ShareComputeReceipt``, built in
    /// memory and handed straight to the view — it is never appended to the
    /// store, so a review run leaves no trace in anyone's history.
    static func receipt(now: Date = Date()) -> ShareComputeReceipt {
        let startedAt = now.addingTimeInterval(-6_138)
        return ShareComputeReceipt(
            id: "QS-8A31",
            catalogID: catalogID,
            modelTitle: modelTitle,
            worker: worker,
            nodeID: nodeID,
            startedAt: startedAt,
            endedAt: now,
            rewardStatus: .available,
            restoreStatus: .complete
        )
    }
}

// MARK: - Pool summary fixtures

/// A fixed `GET /v1/pool/summary` reading for a visual-review capture.
///
/// A screenshot run must NEVER read the production pool. Two reasons, and both
/// have bitten:
///
/// 1. The live pool is currently all zeros, so every capture would show an
///    empty pool and the populated layout — long model names beside four
///    metric columns, the case most likely to overlap — would go unreviewed.
/// 2. Numbers that change between two runs make two screenshots of the same
///    code look like a regression.
///
/// Gated exactly like ``ShareComputeReviewStage``: both
/// ``RAPID_GUI_GOLDEN_MODE`` and ``RAPID_GUI_SHARE_COMPUTE_POOL_SUMMARY`` must
/// be set, so a normal launch always goes to the real client.
enum ShareComputePoolSummaryFixture: String, CaseIterable, Sendable {
    /// A busy pool with all four catalog models, GLM included. The GLM row is
    /// `enabled: false`, so the disabled state is on screen in every capture.
    case populated
    /// A real, healthy all-zero pool — the calm empty state, not an error.
    case empty
    /// The endpoint is down and there is no previous reading.
    case unavailable

    static func requested(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ShareComputePoolSummaryFixture? {
        guard environment["RAPID_GUI_GOLDEN_MODE"] == "1",
              let name = environment["RAPID_GUI_SHARE_COMPUTE_POOL_SUMMARY"] else {
            return nil
        }
        return ShareComputePoolSummaryFixture(rawValue: name)
    }

    /// Two minutes before the capture.
    ///
    /// RELATIVE, not a fixed epoch: a pinned instant drifts further from "now"
    /// every day, and a pool whose freshness badge reads "Updated 2d ago" is
    /// not reviewing the live surface. The rendered STRING is still stable
    /// across runs because ``ShareComputePoolClock`` buckets by whole minutes.
    static var updatedAt: Date { Date().addingTimeInterval(-120) }

    func load() throws -> ShareComputePoolSummary {
        switch self {
        case .populated:
            return ShareComputePoolSummary(
                updatedAt: Self.updatedAt,
                // Deliberately NOT the sum of the rows below (which is 32): the
                // totals are the server's own, and a capture whose numbers
                // happened to add up would hide a UI that secretly re-derived
                // them. The extra two nodes stand for the real case —
                // machines (of any hardware) serving a catalog model this build
                // does not support yet.
                totals: .init(connectedNodes: 34, readyNodes: 25, availableSlots: 15),
                models: [
                    .init(modelID: "qwen3.8-27b", isEnabled: true, connectedNodes: 18, readyNodes: 14, busyNodes: 4, availableSlots: 9),
                    .init(modelID: "qwen3.6-35b", isEnabled: true, connectedNodes: 9, readyNodes: 7, busyNodes: 2, availableSlots: 4),
                    .init(modelID: "nemotron-3.5-lightning", isEnabled: true, connectedNodes: 5, readyNodes: 4, busyNodes: 1, availableSlots: 2),
                    .init(modelID: "glm-5.3-flash", isEnabled: false, connectedNodes: 0, readyNodes: 0, busyNodes: 0, availableSlots: 0),
                ]
            )
        case .empty:
            return ShareComputePoolSummary(
                updatedAt: Self.updatedAt,
                totals: .init(connectedNodes: 0, readyNodes: 0, availableSlots: 0),
                models: ShareComputeModel.supported.map {
                    .init(
                        modelID: $0.catalogID,
                        isEnabled: true,
                        connectedNodes: 0,
                        readyNodes: 0,
                        busyNodes: 0,
                        availableSlots: 0
                    )
                }
            )
        case .unavailable:
            throw ShareComputePoolSummaryError.httpStatus(503)
        }
    }
}

// MARK: - Ledger fixtures

/// A fixed contributor-ledger state for a visual-review capture.
///
/// A screenshot run must never hold a real `qsprk-` key or call the live
/// ledger. Three reasons, all of which have teeth:
///
/// 1. The endpoint is authenticated and account-scoped — a capture would need
///    somebody's real credential, and the resulting PNG would show their real
///    earnings.
/// 2. The rate limit is 120 requests/hour/IP; a screenshot matrix that walked
///    the ledger for every variant would exhaust it.
/// 3. Amounts and timestamps change, so two captures of identical code would
///    differ.
///
/// Gated exactly like ``ShareComputeReviewStage``: both
/// ``RAPID_GUI_GOLDEN_MODE`` and ``RAPID_GUI_SHARE_COMPUTE_LEDGER`` must be
/// set, so a normal launch always goes to the Keychain and the real client.
enum ShareComputeLedgerFixture: String, CaseIterable, Sendable {
    /// No key saved — the onboarding surface.
    case noReadKey
    /// A populated ledger. Deliberately mixes a credited row, a pending row,
    /// a row with an unknown accrued amount, a deleted-model row, and two
    /// nodes, so one capture reviews every row variant at once.
    case loaded
    /// Authenticated, but the account has no windows yet. A successful 200.
    case empty
    /// The saved key was rejected.
    case revoked
    /// A refresh failed over good data — values and their real timestamps stay.
    case stale

    static func requested(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ShareComputeLedgerFixture? {
        guard environment["RAPID_GUI_GOLDEN_MODE"] == "1",
              let name = environment["RAPID_GUI_SHARE_COMPUTE_LEDGER"] else {
            return nil
        }
        return ShareComputeLedgerFixture(rawValue: name)
    }

    /// A non-secret placeholder label. NOT a key — there is no fixture key
    /// anywhere, because a fixture that carried a `qsprk-` shaped string is one
    /// careless copy away from looking like a real credential in a repo.
    static let keyLabel = "qsprk-…7f24"

    private static func window(
        cursor: Int,
        node: String,
        model: String?,
        minutesAgo: Int,
        requests: Int,
        input: Int,
        output: Int,
        accrued: Decimal?,
        final: Decimal,
        status: ShareComputeLedgerStatus
    ) -> ShareComputeLedgerWindow {
        let start = Date().addingTimeInterval(TimeInterval(-minutesAgo * 60))
        return ShareComputeLedgerWindow(
            nodeID: node,
            modelID: model,
            periodStart: start,
            periodEnd: start.addingTimeInterval(1_800),
            requestCount: requests,
            inputTokens: input,
            outputTokens: output,
            accruedCredit: accrued,
            finalCredit: final,
            status: status,
            creditedAt: status == .credited ? start.addingTimeInterval(5_400) : nil,
            updatedAt: start.addingTimeInterval(5_400),
            allowanceMonth: Self.allowanceMonth,
            nodeMonthlyCapUSD: 20,
            cursor: cursor
        )
    }

    /// The current month's bucket, so the summary panel's heading matches the
    /// month a reviewer is looking at the capture in.
    static var allowanceMonth: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-01"
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.string(from: Date())
    }

    static var account: ShareComputeLedgerAccount {
        ShareComputeLedgerAccount(
            unit: "usd_api_credit",
            note: "Amounts are QuickSilver API credit (spendable platform balance), not a cash payout. 1.0 == $1 of API usage.",
            caps: ShareComputeLedgerCaps(nodeMonthlyUSD: 20, poolMonthlyUSD: 100),
            windows: [
                window(cursor: 48, node: "qspnode-34f07f8fade84d42", model: "qwen3.8-27b",
                       minutesAgo: 30, requests: 7, input: 100, output: 250,
                       accrued: Decimal(string: "0.0123"), final: Decimal(string: "0.0123")!,
                       status: .pending),
                window(cursor: 47, node: "qspnode-34f07f8fade84d42", model: "qwen3.6-35b",
                       minutesAgo: 90, requests: 31, input: 184_200, output: 41_100,
                       accrued: Decimal(string: "0.4821"), final: Decimal(string: "0.4821")!,
                       status: .credited),
                // Unknown accrued amount — renders `—`, never `$0.00`.
                window(cursor: 46, node: "qspnode-b31480ca21d7", model: "nemotron-3.5-lightning",
                       minutesAgo: 150, requests: 12, input: 40_900, output: 9_800,
                       accrued: nil, final: Decimal(string: "0.1042")!,
                       status: .credited),
                // Deleted node/model record — neutral label, no guess.
                window(cursor: 45, node: "qspnode-b31480ca21d7", model: nil,
                       minutesAgo: 210, requests: 4, input: 8_100, output: 2_400,
                       accrued: Decimal(string: "0.0210"), final: Decimal(string: "0.0210")!,
                       status: .credited),
                window(cursor: 44, node: "qspnode-34f07f8fade84d42", model: "qwen3.8-27b",
                       minutesAgo: 270, requests: 0, input: 0, output: 0,
                       accrued: Decimal(0), final: Decimal(0), status: .zero),
                // An unknown future status must render, not crash or blank.
                window(cursor: 43, node: "qspnode-34f07f8fade84d42", model: "qwen3.8-27b",
                       minutesAgo: 330, requests: 2, input: 900, output: 300,
                       accrued: Decimal(string: "0.0031"), final: Decimal(0),
                       status: .unknown("void")),
            ],
            isTruncated: false
        )
    }

    static var emptyAccount: ShareComputeLedgerAccount {
        ShareComputeLedgerAccount(
            unit: "usd_api_credit",
            note: nil,
            caps: ShareComputeLedgerCaps(nodeMonthlyUSD: 20, poolMonthlyUSD: 100),
            windows: [],
            isTruncated: false
        )
    }

    /// The page state this fixture stands for.
    var state: ShareComputeLedgerState {
        switch self {
        case .noReadKey: return .noReadKey
        case .loaded: return .loaded(Self.account)
        case .empty: return .loadedEmpty(Self.emptyAccount)
        case .revoked: return .unauthorized
        case .stale: return .refreshFailed(Self.account, .serviceUnavailable)
        }
    }

    /// The redacted key label the summary panel should show, if any.
    var savedKeyLabel: String? {
        switch self {
        case .noReadKey, .revoked: return nil
        case .loaded, .empty, .stale: return Self.keyLabel
        }
    }
}
